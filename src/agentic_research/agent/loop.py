"""The budgeted agent loop (ADR-0001, ADR-0006).

One iteration is: budget check, policy (bounded by the step timeout and the
run deadline), then either the verifier hook (`FinalAnswer`), abstention
(`GiveUp`) or, for a `ToolCall`, the budget re-check, the loop guard and the
tool executor. Each recorded step is then shaped into an observation for the
policy (`agentic_research.agent.observations`). The loop owns every state
transition and never raises for a policy, verifier or tool failure: each run
ends in a terminal `RunState`.

Budget rules:

- Before each policy call, `Budgets.exhausted` is checked; the first
  exhausted budget ends the run as ``budget_exhausted:<kind>``.
- A `ToolCall` is checked again (now including the per-tool limit and any
  tokens or time the policy just used) before it runs. A refused call is
  recorded as a step without a result.
- ``max_tokens`` is soft: usage is charged after the policy returns, so the
  counter can pass the limit. A `FinalAnswer` or `GiveUp` from that call is
  still honoured (it costs nothing more); a `ToolCall` is refused.
- The run deadline is passed to the executor as an absolute time on the
  executor's clock, so tool timeouts are shortened to it and a retry whose
  backoff would cross it is skipped. Attempt durations and backoff sleeps
  are charged to ``elapsed_s`` (plus ``tool_attempts``/``backoff_s``).
- A rejected answer costs one repair; the run ends as
  ``budget_exhausted:repairs`` once repairs exceed ``max_repairs``.

Time: budgets and tool deadlines use the executor's clock, while policy and
tool timeouts run on the event loop's clock via ``asyncio.timeout``. In
production both are monotonic seconds; tests inject a fake executor clock.

Loop detection (`agentic_research.agent.guards`): a `ToolCall` that passed
the budget re-check but repeats an identical call that already ran
``max_identical_calls`` times is not run; it is recorded with a
``loop_detected`` guardrail (one step, no tool-call budget). Repeating it once
more ends the run as ``failed:loop``. ``loop_guard=None`` disables this.

A policy that raises `PolicyFailure` ends the run as ``failed:<reason>``
with its reason code; any other policy exception is ``failed:policy_error``.

Tracing (`agentic_research.agent.tracing`, ADR-0008): `AgentLoop.run_traced`
returns the final state plus a trace with a span for the run, every step,
every provider call the policy reports and every tool attempt; `run` is the
same loop without returning the trace.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from agentic_research.agent.actions import FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.guards import LOOP, LOOP_DETECTED, LoopCheck, LoopGuard
from agentic_research.agent.observations import ObservationShaper
from agentic_research.agent.policy import Decision, Policy, PolicyFailure, ProviderCallInfo
from agentic_research.agent.state import (
    BudgetKind,
    Budgets,
    Guardrail,
    RunState,
    RunStatus,
    Task,
    Verdict,
)
from agentic_research.agent.tracing import TraceRecorder
from agentic_research.agent.verifier import AcceptAll, Verifier
from agentic_research.tools.base import ErrorClass, ToolContext, ToolResult
from agentic_research.tools.executor import ToolExecutor
from agentic_research.trace.schema import ToolFormat, Trace

POLICY_TIMEOUT = "policy_timeout"
POLICY_ERROR = "policy_error"
INVALID_ACTION = "invalid_action"
VERIFIER_ERROR = "verifier_error"
DEFAULT_LOOP_GUARD = LoopGuard()
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


@dataclass(frozen=True)
class _Stop:
    status: RunStatus
    detail: str | BudgetKind
    error_class: ErrorClass = ErrorClass.PERMANENT
    message: str = ""


@dataclass(frozen=True)
class TracedRun:
    """A finished run and its trace."""

    state: RunState
    trace: Trace


class AgentLoop:
    """Runs one task to a terminal state under `Budgets`.

    ``max_steps`` is required so that every run terminates whatever the
    policy does. ``step_timeout_s`` bounds each policy call; the deadline
    bounds it too when it is closer.
    """

    def __init__(
        self,
        policy: Policy,
        executor: ToolExecutor,
        budgets: Budgets,
        *,
        verifier: Verifier | None = None,
        step_timeout_s: float | None = None,
        run_id: str = "run",
        loop_guard: LoopGuard | None = DEFAULT_LOOP_GUARD,
        shaper: ObservationShaper | None = None,
    ) -> None:
        if budgets.max_steps is None:
            raise ValueError("AgentLoop needs budgets.max_steps so every run terminates")
        valid_timeout = step_timeout_s is None or (
            math.isfinite(step_timeout_s) and step_timeout_s > 0
        )
        if not valid_timeout:
            raise ValueError("step_timeout_s must be a finite number > 0")
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id must be 1-64 characters from [A-Za-z0-9_.:-] (a trace id)")
        self._policy = policy
        self._executor = executor
        self._budgets = budgets
        self._verifier: Verifier = verifier if verifier is not None else AcceptAll()
        self._step_timeout_s = step_timeout_s
        self._run_id = run_id
        self._loop_guard = loop_guard
        self._shaper = shaper if shaper is not None else ObservationShaper()

    async def run(self, task: Task) -> RunState:
        """Run ``task`` to a terminal state (the trace is built and dropped)."""
        return (await self.run_traced(task)).state

    async def run_traced(
        self,
        task: Task,
        *,
        provider: str = "none",
        model: str | None = None,
        tool_format: ToolFormat | None = None,
        started_at: datetime | None = None,
    ) -> TracedRun:
        """Run ``task`` and return the final state with its trace.

        ``provider``, ``model`` and ``tool_format`` describe the policy on the
        run span (``none`` for policies that call no model); ``started_at``
        is the header's wall-clock start (now, in UTC, by default).
        """
        started_at = started_at if started_at is not None else datetime.now(UTC)
        state = RunState(task=task)
        clock = self._executor.clock
        start = clock()
        deadline = None if self._budgets.deadline_s is None else start + self._budgets.deadline_s
        rec = TraceRecorder(start)

        def tick() -> None:
            state.counters.elapsed_s = max(0.0, clock() - start)

        while not state.terminal:
            tick()
            kind = self._budgets.exhausted(state.counters)
            if kind is not None:
                state.finish(RunStatus.BUDGET_EXHAUSTED, detail=kind)
                break
            policy_start = clock()
            rec.begin_step(len(state.steps), policy_start)
            decision, calls = await self._decide(state, deadline)
            rec.provider_calls(calls, policy_start, clock())
            tick()
            if isinstance(decision, _Stop):
                if decision.status is RunStatus.FAILED:
                    rec.fail(decision.error_class, decision.message)
                state.finish(decision.status, detail=decision.detail)
                break
            state.counters.tokens += decision.tokens
            action = decision.action
            if isinstance(action, GiveUp):
                self._record(state, action)
                state.finish(RunStatus.ABSTAINED, detail=action.reason)
            elif isinstance(action, FinalAnswer):
                await self._answer(state, action, rec)
            else:
                await self._call_tool(state, action, deadline, rec)
            tick()
            rec.end_step(clock())
        trace = rec.build(
            state,
            self._budgets,
            at=clock(),
            run_id=self._run_id,
            started_at=started_at,
            provider=provider,
            model=model,
            tool_format=tool_format,
        )
        return TracedRun(state=state, trace=trace)

    async def _decide(
        self, state: RunState, deadline: float | None
    ) -> tuple[Decision | _Stop, tuple[ProviderCallInfo, ...]]:
        """The policy's decision (or why the run stops) and the provider calls it reported."""
        limit = self._step_timeout_s
        deadline_binds = False
        if deadline is not None:
            remaining = deadline - self._executor.clock()
            if remaining <= 0:
                return _Stop(RunStatus.BUDGET_EXHAUSTED, BudgetKind.DEADLINE), ()
            if limit is None or remaining < limit:
                limit, deadline_binds = remaining, True
        try:
            async with asyncio.timeout(limit) as scope:
                raw = await self._policy.next_action(state.model_copy(deep=True))
        except TimeoutError:
            if not scope.expired():  # the policy raised its own TimeoutError
                return _Stop(
                    RunStatus.FAILED, POLICY_ERROR, message="policy raised TimeoutError"
                ), ()
            if deadline_binds:
                return _Stop(RunStatus.BUDGET_EXHAUSTED, BudgetKind.DEADLINE), ()
            message = f"policy did not decide within {limit:g}s"
            return _Stop(RunStatus.FAILED, POLICY_TIMEOUT, ErrorClass.TIMEOUT, message), ()
        except PolicyFailure as exc:  # the policy's own classified failure
            return _Stop(
                RunStatus.FAILED, exc.reason, _failure_class(exc), exc.message or exc.reason
            ), exc.calls
        except Exception as exc:  # a broken policy must not crash the loop
            message = f"policy raised {type(exc).__name__}"
            return _Stop(RunStatus.FAILED, POLICY_ERROR, message=message), ()
        if isinstance(raw, Decision):
            return raw, raw.calls
        if isinstance(raw, ToolCall | FinalAnswer | GiveUp):
            return Decision(action=raw), ()
        message = f"policy returned {type(raw).__name__}, not an action"
        return _Stop(RunStatus.FAILED, INVALID_ACTION, message=message), ()

    async def _answer(self, state: RunState, action: FinalAnswer, rec: TraceRecorder) -> None:
        try:
            verdict = await self._verifier.verify(action, state.model_copy(deep=True))
        except Exception as exc:
            self._record(state, action)
            rec.fail(ErrorClass.PERMANENT, f"verifier raised {type(exc).__name__}")
            state.finish(RunStatus.FAILED, detail=VERIFIER_ERROR)
            return
        if not isinstance(verdict, Verdict):
            self._record(state, action)
            rec.fail(ErrorClass.PERMANENT, f"verifier returned {type(verdict).__name__}")
            state.finish(RunStatus.FAILED, detail=VERIFIER_ERROR)
            return
        self._record(state, action, verdict=verdict)
        if verdict.accepted:
            state.finish(RunStatus.SUCCEEDED, answer=action.answer)
        else:
            rec.step_event("answer_rejected", at=None, reason=verdict.reason)
            state.counters.repairs += 1

    async def _call_tool(
        self, state: RunState, action: ToolCall, deadline: float | None, rec: TraceRecorder
    ) -> None:
        kind = self._budgets.exhausted(state.counters, tool=action.tool)
        if kind is not None:
            self._record(state, action)
            state.finish(RunStatus.BUDGET_EXHAUSTED, detail=kind)
            return
        if self._loop_guard is not None:
            check = self._loop_guard.check(state.steps, action)
            if check is not LoopCheck.OK:
                guardrail = Guardrail(code=LOOP_DETECTED, message=self._loop_guard.message(action))
                self._record(state, action, guardrail=guardrail)
                rec.step_event(LOOP_DETECTED, at=None, tool=action.tool)
                if check is LoopCheck.STOP:
                    rec.fail(ErrorClass.PERMANENT, guardrail.message or LOOP)
                    state.finish(RunStatus.FAILED, detail=LOOP)
                return
        ctx = ToolContext(run_id=self._run_id, step=len(state.steps))
        handed_over = self._executor.clock()
        execution = await self._executor.execute(action.tool, action.args, ctx, deadline=deadline)
        rec.tool_execution(action, execution, handed_over)
        counters = state.counters
        counters.count_tool_call(action.tool)
        counters.tool_attempts += len(execution.attempts)
        counters.backoff_s += execution.total_backoff_s
        self._record(state, action, result=execution.result)

    def _record(
        self,
        state: RunState,
        action: ToolCall | FinalAnswer | GiveUp,
        *,
        result: ToolResult | None = None,
        verdict: Verdict | None = None,
        guardrail: Guardrail | None = None,
    ) -> None:
        record = state.add_step(action, result=result, verdict=verdict, guardrail=guardrail)
        state.counters.steps += 1
        observation = self._shaper.observe(record)
        if observation is not None:
            self._shaper.add(state.observations, observation)


def _failure_class(exc: PolicyFailure) -> ErrorClass:
    """The run error's class for a policy failure, from the last call it reported."""
    if exc.calls:
        last = exc.calls[-1]
        if last.error is not None:
            return last.error.error_class
        if last.parse_failure is not None:
            return ErrorClass.BAD_OUTPUT
    return ErrorClass.PERMANENT
