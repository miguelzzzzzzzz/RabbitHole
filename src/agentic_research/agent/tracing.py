"""Build a run's trace (ADR-0008) from what the agent loop observes.

`TraceRecorder` is driven by `AgentLoop.run_traced`; nothing else writes
spans. Times are seconds since the run started on the loop's clock (the
executor's clock), clamped so a span never ends before it starts.

Span layout:

- ``run`` (id ``run``): the root, written last. Its status, answer, error,
  budget usage/limits and ``parse_failures`` come from the final `RunState`.
  ``budget_exhausted`` and ``abstained`` runs carry an event naming the
  budget kind or the abstain reason, because the v1 run span has no field
  for them.
- ``step`` (id ``step-<i>``): one loop iteration that reached the policy.
  The index matches `RunState.steps`; an iteration that ends the run
  without recording a step (a policy failure, a deadline hit while
  deciding) is still a step span, and always the last one. A run stopped by
  a budget before the policy was asked has no span for that iteration.
- ``provider_call`` (``step-<i>.provider-<k>``): each provider call the
  policy reported (`Decision.calls` / `PolicyFailure.calls`). Its timing is
  the whole policy call; M2 policies make at most one call per step. A
  policy call that times out reports nothing, so it has no such span.
- ``tool_call`` (``step-<i>.tool-<attempt>``): one span per executor
  attempt, retries included, timed from `AttemptRecord`. A call the executor
  rejected before running it (``invalid_args``, or no time left before the
  deadline) is a zero-length attempt 1 carrying that error.

Guardrail refusals (``loop_detected``) and verifier rejections
(``answer_rejected``) are events on their step.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Final

from agentic_research import __version__
from agentic_research.agent.actions import ToolCall
from agentic_research.agent.policy import ProviderCallInfo
from agentic_research.agent.state import Budgets, RunState
from agentic_research.agent.state import RunStatus as LoopStatus
from agentic_research.tools.base import ErrorClass, ToolResult
from agentic_research.tools.executor import ToolExecution
from agentic_research.trace.schema import (
    MAX_MESSAGE_CHARS,
    BudgetLimits,
    BudgetUsage,
    ProviderCallSpan,
    RunSpan,
    RunStatus,
    Span,
    SpanError,
    SpanEvent,
    StepSpan,
    ToolCallSpan,
    ToolFormat,
    Trace,
    TraceHeader,
)

GENERATOR: Final = f"agentic_research {__version__}"
RUN_SPAN_ID: Final = "run"

_STATUS: Final = {
    LoopStatus.SUCCEEDED: RunStatus.ANSWERED,
    LoopStatus.ABSTAINED: RunStatus.ABSTAINED,
    LoopStatus.BUDGET_EXHAUSTED: RunStatus.BUDGET_EXHAUSTED,
    LoopStatus.FAILED: RunStatus.FAILED,
}

#: Stored instead of tool arguments/outputs that JSON cannot represent.
UNREPRESENTABLE: Final = "_unrepresentable"


def json_object(value: dict[str, Any]) -> dict[str, Any]:
    """``value`` as plain JSON data, or a marker if it has NaN/inf or non-JSON types.

    A model can send ``NaN`` in arguments (Python's JSON parser accepts it);
    the trace must still be written, so such a value is replaced rather than
    allowed to fail the run.
    """
    try:
        loaded = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        return {UNREPRESENTABLE: f"{type(exc).__name__}: not representable as JSON"}
    assert isinstance(loaded, dict)
    return loaded


class TraceRecorder:
    """Collects spans for one run, in the order they end."""

    def __init__(self, start: float) -> None:
        self._start = start
        self._spans: list[Span] = []
        self._step: int | None = None
        self._step_start = 0.0
        self._step_events: list[tuple[str, float | None, dict[str, Any]]] = []
        self._failure: tuple[ErrorClass, str] | None = None

    def rel(self, at: float) -> float:
        """Clock reading ``at`` as seconds since the run started (never negative)."""
        return max(0.0, at - self._start)

    @property
    def parse_failures(self) -> int:
        return sum(
            1
            for s in self._spans
            if isinstance(s, ProviderCallSpan) and s.parse_failure is not None
        )

    def begin_step(self, index: int, at: float) -> None:
        if self._step is not None:
            raise RuntimeError(f"step {self._step} is still open")
        self._step, self._step_start, self._step_events = index, self.rel(at), []

    def _step_id(self) -> str:
        if self._step is None:
            raise RuntimeError("no open step")
        return f"step-{self._step}"

    def provider_calls(self, calls: tuple[ProviderCallInfo, ...], start: float, end: float) -> None:
        parent = self._step_id()
        s, e = self.rel(start), self.rel(end)
        for k, call in enumerate(calls, start=1):
            self._spans.append(
                ProviderCallSpan(
                    span_id=f"{parent}.provider-{k}",
                    parent_id=parent,
                    start_s=s,
                    end_s=max(s, e),
                    **call.model_dump(),
                )
            )

    def tool_execution(self, action: ToolCall, execution: ToolExecution, at: float) -> None:
        """One span per attempt; ``at`` is when the loop handed the call to the executor."""
        parent = self._step_id()
        arguments = json_object(action.args)
        result = execution.result
        if not execution.attempts:  # rejected before anything ran
            s = self.rel(at)
            assert result.error is not None
            self._spans.append(self._tool_span(parent, action.tool, 1, s, s, arguments, result))
            return
        for attempt in execution.attempts:
            s = self.rel(attempt.started_at)
            if attempt.error is None:
                outcome = result  # only the final attempt can succeed
            else:
                outcome = ToolResult.failure(action.tool, attempt.error)
            span = self._tool_span(
                parent, action.tool, attempt.attempt, s, s + attempt.duration_s, arguments, outcome
            )
            self._spans.append(span)
        last = execution.attempts[-1].error
        if result.error is not None and result.error != last:
            # A retry was due but the deadline left no time to start it.
            self.step_event(
                "tool_not_retried", at=None, reason=result.error.reason, tool=action.tool
            )

    @staticmethod
    def _tool_span(
        parent: str,
        tool: str,
        attempt: int,
        start_s: float,
        end_s: float,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> ToolCallSpan:
        return ToolCallSpan(
            span_id=f"{parent}.tool-{attempt}",
            parent_id=parent,
            start_s=start_s,
            end_s=max(start_s, end_s),
            tool=tool,
            attempt=attempt,
            arguments=arguments,
            output=None if result.output is None else json_object(result.output),
            error=None if result.error is None else SpanError.from_tool_error(result.error),
        )

    def step_event(self, name: str, *, at: float | None, **attributes: Any) -> None:
        """An event on the open step; ``at=None`` places it when the step ends."""
        self._step_id()
        when = None if at is None else self.rel(at)
        self._step_events.append((name, when, attributes))

    def end_step(self, at: float) -> None:
        parent = self._step_id()
        end = max(self._step_start, self.rel(at))
        events = [
            SpanEvent(
                name=name,
                at_s=end if when is None else min(max(when, self._step_start), end),
                attributes=json_object(attributes),
            )
            for name, when, attributes in self._step_events
        ]
        assert self._step is not None
        self._spans.append(
            StepSpan(
                span_id=parent,
                parent_id=RUN_SPAN_ID,
                start_s=self._step_start,
                end_s=end,
                index=self._step,
                events=events,
            )
        )
        self._step, self._step_events = None, []

    @property
    def step_open(self) -> bool:
        return self._step is not None

    def fail(self, error_class: ErrorClass, message: str) -> None:
        """Classify the failure that is about to end the run (for the run span's error)."""
        self._failure = (error_class, message)

    def build(
        self,
        state: RunState,
        budgets: Budgets,
        *,
        at: float,
        run_id: str,
        started_at: datetime,
        provider: str,
        model: str | None,
        tool_format: ToolFormat | None,
    ) -> Trace:
        """The complete trace once ``state`` is terminal."""
        if not state.terminal:
            raise ValueError("the run has not finished")
        if self._step is not None:
            self.end_step(at)
        end = self.rel(at)
        for span in self._spans:  # children never outlast the run
            end = max(end, span.end_s)
        events: list[SpanEvent] = []
        error: SpanError | None = None
        detail = state.detail or ""
        if state.status is LoopStatus.BUDGET_EXHAUSTED:
            events.append(SpanEvent(name="budget_exhausted", at_s=end, attributes={"kind": detail}))
        elif state.status is LoopStatus.ABSTAINED:
            events.append(SpanEvent(name="abstained", at_s=end, attributes={"reason": detail}))
        elif state.status is LoopStatus.FAILED:
            error_class, message = self._failure or (ErrorClass.PERMANENT, detail)
            error = SpanError(
                error_class=error_class, reason=detail, message=message[:MAX_MESSAGE_CHARS]
            )
        counters = state.counters
        run = RunSpan(
            span_id=RUN_SPAN_ID,
            start_s=0.0,
            end_s=end,
            events=events,
            question=state.task.question,
            task_id=state.task.task_id,
            provider=provider,
            model=model,
            tool_format=tool_format,
            status=_STATUS[state.status],
            answer=state.answer,
            error=error,
            budget_used=BudgetUsage(
                steps=counters.steps,
                tool_calls=counters.tool_calls,
                tokens=counters.tokens,
                time_s=counters.elapsed_s,
            ),
            budget_limits=BudgetLimits(
                steps=budgets.max_steps,
                tool_calls=budgets.max_tool_calls,
                tokens=budgets.max_tokens,
                time_s=budgets.deadline_s,
            ),
            parse_failures=self.parse_failures,
        )
        header = TraceHeader(run_id=run_id, started_at=started_at, generator=GENERATOR)
        return Trace(header=header, spans=[*self._spans, run])
