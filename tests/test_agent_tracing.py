"""AgentLoop.run_traced: spans for the run, steps, provider calls and tool attempts.

A fake clock is shared by the executor and the loop, so span times are exact.
Model-backed runs use `LLMPolicy` over `ScriptedProvider` (no model calls).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from agentic_research.agent import (
    AgentLoop,
    Budgets,
    Decision,
    FinalAnswer,
    GiveUp,
    LoopGuard,
    PolicyFailure,
    RunState,
    Task,
    ToolCall,
    Verdict,
)
from agentic_research.agent.policy import ProviderCallInfo
from agentic_research.agent.state import RunStatus as LoopStatus
from agentic_research.agent.tracing import UNREPRESENTABLE, TraceRecorder, json_object
from agentic_research.policies import LLMPolicy
from agentic_research.policies.llm import provider_error_class
from agentic_research.providers import (
    LocalServerUnavailableError,
    ProviderError,
    ProviderHTTPError,
    ScriptedError,
    ScriptedFunctionCall,
    ScriptedProvider,
    ScriptedTurn,
    TokenUsage,
)
from agentic_research.providers.errors import ProviderResponseError
from agentic_research.tools import (
    ErrorClass,
    RetryPolicy,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolExecutor,
    ToolFailure,
    ToolOutput,
    ToolRegistry,
)
from agentic_research.tools.builtin import default_registry
from agentic_research.trace import (
    ParseFailure,
    ProviderCallSpan,
    RunStatus,
    SpanError,
    StepSpan,
    ToolCallSpan,
    Trace,
    dumps_trace,
    loads_trace,
)

TASK = Task(question="What is 6*7?", task_id="t-1")
NO_JITTER = RetryPolicy(
    max_attempts=4, initial_backoff_s=0.5, multiplier=2.0, max_backoff_s=8.0, jitter=0.0
)
STARTED = datetime(2026, 11, 14, 1, 0, tzinfo=UTC)


class FakeTime:
    def __init__(self, *, oversleep: float = 1.0) -> None:
        self.now = 500.0
        self._oversleep = oversleep

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds * self._oversleep


class NoArgs(ToolArgs):
    pass


class Out(ToolOutput):
    value: int


class CostTool:
    """Each attempt advances the fake clock by ``cost``; fails the first ``failures`` times."""

    description = "fake tool"
    args_model: type[ToolArgs] = NoArgs
    output_model: type[ToolOutput] = Out
    timeout_s = 30.0
    idempotent = True

    def __init__(self, name: str, time_: FakeTime, *, cost: float = 0.25, failures: int = 0):
        self.name = name
        self.retry = NO_JITTER
        self._time = time_
        self._cost = cost
        self._failures = failures
        self.runs = 0

    async def run(self, args: NoArgs, ctx: ToolContext) -> Out:
        self.runs += 1
        self._time.now += self._cost
        if self.runs <= self._failures:
            raise ToolFailure(
                ToolError(error_class=ErrorClass.TRANSIENT, reason="reset", message="x")
            )
        return Out(value=42)


Item = ToolCall | FinalAnswer | GiveUp | Decision | Exception


class Scripted:
    """Returns (or raises) the items in order, advancing the clock by ``think`` per call."""

    def __init__(self, items: Sequence[Item], time_: FakeTime, think: float = 0.0) -> None:
        self._items = list(items)
        self._time = time_
        self._think = think
        self.calls = 0

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp | Decision:
        item = self._items[self.calls]
        self.calls += 1
        self._time.now += self._think
        if isinstance(item, Exception):
            raise item
        return item


def run(
    policy: object,
    *,
    budgets: Budgets | None = None,
    tools: Sequence[CostTool] = (),
    time_: FakeTime,
    registry: ToolRegistry | None = None,
    **kwargs: object,
) -> tuple[RunState, Trace]:
    executor = ToolExecutor(
        registry if registry is not None else ToolRegistry(tools),
        clock=time_.clock,
        sleeper=time_.sleep,
    )
    loop = AgentLoop(
        policy,  # type: ignore[arg-type]
        executor,
        budgets or Budgets(max_steps=6),
        run_id="r-trace",
        **kwargs,  # type: ignore[arg-type]
    )
    traced = asyncio.run(loop.run_traced(TASK, started_at=STARTED))
    # Every trace must survive the JSONL round trip unchanged.
    assert loads_trace(dumps_trace(traced.trace)) == traced.trace
    return traced.state, traced.trace


def spans(trace: Trace, kind: type) -> list:  # type: ignore[type-arg]
    return [s for s in trace.spans if isinstance(s, kind)]


def tool_turn(arguments: dict[str, object] | str | None, **extra: object) -> ScriptedTurn:
    call = ScriptedFunctionCall(name="calculator", arguments=arguments)
    return ScriptedTurn(tool_calls=(call,), **extra)  # type: ignore[arg-type]


# --- scripted model run ---------------------------------------------------------


def test_two_step_llm_run_traces_every_span() -> None:
    time_ = FakeTime()
    provider = ScriptedProvider(
        [
            tool_turn(
                {"expression": "6*7"}, usage=TokenUsage(prompt_tokens=120, completion_tokens=18)
            ),
            ScriptedTurn(content="42", usage=TokenUsage(total_tokens=9)),
        ]
    )
    policy = LLMPolicy(provider, default_registry(), model="qwen")
    state, trace = run(policy, time_=time_, registry=default_registry())

    assert state.terminal_label == "succeeded"
    assert [s.span_id for s in trace.spans] == [
        "step-0.provider-1",
        "step-0.tool-1",
        "step-0",
        "step-1.provider-1",
        "step-1",
        "run",
    ]
    first, second = spans(trace, ProviderCallSpan)
    assert (first.provider, first.model, first.tool_format) == ("scripted", "qwen", "native")
    assert (first.input_tokens, first.output_tokens, first.finish_reason) == (120, 18, "tool_calls")
    assert (second.input_tokens, second.output_tokens) == (None, None)  # only total reported
    (tool,) = spans(trace, ToolCallSpan)
    assert (tool.tool, tool.attempt, tool.arguments, tool.output) == (
        "calculator",
        1,
        {"expression": "6*7"},
        {"value": 42},
    )
    run_span = trace.run
    assert run_span.status is RunStatus.ANSWERED and run_span.answer == "42"
    assert (run_span.task_id, run_span.provider, run_span.model) == ("t-1", "none", None)
    assert run_span.budget_used.steps == 2 and run_span.budget_used.tokens == 138 + 9
    assert run_span.budget_limits.steps == 6 and run_span.parse_failures == 0
    assert trace.header.run_id == "r-trace" and trace.header.started_at == STARTED
    assert [s.index for s in spans(trace, StepSpan)] == [0, 1]


def test_parse_failure_is_on_the_provider_span_and_counted_on_the_run() -> None:
    time_ = FakeTime()
    provider = ScriptedProvider([tool_turn('{"expression": ')])
    state, trace = run(LLMPolicy(provider, default_registry()), time_=time_)

    assert state.status.value == "failed" and len(state.steps) == 0
    (step,) = spans(trace, StepSpan)  # the failing iteration still has a step span
    (call,) = spans(trace, ProviderCallSpan)
    assert call.parent_id == step.span_id and call.parse_failure is not None
    assert call.parse_failure.reason == state.detail and call.error is None
    assert call.model == "scripted"  # from the reply
    assert trace.run.parse_failures == 1
    assert trace.run.error is not None
    assert trace.run.error.error_class is ErrorClass.BAD_OUTPUT
    assert trace.run.error.reason == state.detail


@pytest.mark.parametrize(
    ("error", "error_class", "reason"),
    [
        (ScriptedError(kind="unavailable"), ErrorClass.TRANSIENT, "local_server_unavailable"),
        (ScriptedError(kind="http", status_code=503), ErrorClass.TRANSIENT, "http_error"),
        (ScriptedError(kind="http", status_code=400), ErrorClass.PERMANENT, "http_error"),
    ],
)
def test_provider_error_is_on_the_provider_span_and_the_run(
    error: ScriptedError, error_class: ErrorClass, reason: str
) -> None:
    provider = ScriptedProvider([ScriptedTurn(error=error)])
    policy = LLMPolicy(provider, default_registry(), model="qwen")
    state, trace = run(policy, time_=FakeTime())

    assert state.terminal_label == f"failed:{reason}"
    (call,) = spans(trace, ProviderCallSpan)
    assert call.error is not None and call.parse_failure is None
    assert (call.error.error_class, call.error.reason) == (error_class, reason)
    assert (call.model, call.input_tokens, call.finish_reason) == ("qwen", None, None)
    assert trace.run.error is not None
    assert (trace.run.error.error_class, trace.run.error.message) == (
        error_class,
        "scripted provider failure",
    )
    assert trace.run.parse_failures == 0


def test_script_exhaustion_is_a_permanent_provider_error() -> None:
    provider = ScriptedProvider([tool_turn({"expression": "1+1"})])
    state, trace = run(LLMPolicy(provider, default_registry()), time_=FakeTime())
    assert state.terminal_label == "failed:script_exhausted"
    calls = spans(trace, ProviderCallSpan)
    assert [c.span_id for c in calls] == ["step-0.provider-1", "step-1.provider-1"]
    assert calls[1].error is not None and calls[1].error.error_class is ErrorClass.PERMANENT


def test_model_label_falls_back_to_unknown() -> None:
    class Bare:
        name = "bare"

        async def complete(self, request: object) -> object:
            raise ProviderError("down", reason="bare_down")

    state, trace = run(LLMPolicy(Bare(), default_registry()), time_=FakeTime())  # type: ignore[arg-type]
    (call,) = spans(trace, ProviderCallSpan)
    assert (call.provider, call.model) == ("bare", "unknown")
    assert state.terminal_label == "failed:bare_down"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (LocalServerUnavailableError("x"), ErrorClass.TRANSIENT),
        (ProviderHTTPError("x", status_code=429), ErrorClass.TRANSIENT),
        (ProviderHTTPError("x", status_code=500), ErrorClass.TRANSIENT),
        (ProviderHTTPError("x", status_code=404), ErrorClass.PERMANENT),
        (ProviderResponseError("x"), ErrorClass.BAD_OUTPUT),
        (ProviderError("x", reason="script_exhausted"), ErrorClass.PERMANENT),
    ],
)
def test_provider_error_class(exc: ProviderError, expected: ErrorClass) -> None:
    assert provider_error_class(exc) is expected


def test_provider_call_info_cannot_have_both_outcomes() -> None:
    with pytest.raises(ValueError, match="no reply to fail parsing"):
        ProviderCallInfo(
            provider="p",
            model="m",
            tool_format="json",
            parse_failure=ParseFailure(reason="x"),
            error=SpanError(error_class=ErrorClass.PERMANENT, reason="y", message=""),
        )


def test_policy_failure_carries_its_calls() -> None:
    info = ProviderCallInfo(provider="p", model="m", tool_format="native")
    assert PolicyFailure("nope", calls=[info]).calls == (info,)
    assert PolicyFailure("nope").calls == ()


# --- tool attempts --------------------------------------------------------------


def test_each_retry_is_its_own_tool_span_with_exact_timing() -> None:
    time_ = FakeTime()
    flaky = CostTool("flaky", time_, cost=0.25, failures=2)
    policy = Scripted([ToolCall(tool="flaky"), FinalAnswer(answer="42")], time_, think=0.125)
    state, trace = run(policy, tools=[flaky], time_=time_)

    assert state.terminal_label == "succeeded"
    tools = spans(trace, ToolCallSpan)
    assert [t.span_id for t in tools] == ["step-0.tool-1", "step-0.tool-2", "step-0.tool-3"]
    assert [t.attempt for t in tools] == [1, 2, 3]
    assert [t.error.reason if t.error else None for t in tools] == ["reset", "reset", None]
    assert tools[-1].output == {"value": 42}
    # think 0.125, attempt 0.25, backoff 0.5, attempt 0.25, backoff 1.0, attempt 0.25
    assert [(t.start_s, t.end_s) for t in tools] == [
        (0.125, 0.375),
        (0.875, 1.125),
        (2.125, 2.375),
    ]
    step0, step1 = spans(trace, StepSpan)
    assert (step0.start_s, step0.end_s) == (0.0, 2.375)
    assert (step1.start_s, step1.end_s) == (2.375, 2.5)
    assert trace.run.end_s == 2.5 and trace.run.budget_used.time_s == 2.5


def test_a_call_rejected_before_running_is_a_zero_length_attempt() -> None:
    time_ = FakeTime()
    policy = Scripted(
        [ToolCall(tool="calculator", args={"nope": 1}), GiveUp(reason="stuck")], time_
    )
    state, trace = run(policy, time_=time_, registry=default_registry())

    assert state.terminal_label == "abstained:stuck"
    (tool,) = spans(trace, ToolCallSpan)
    assert tool.start_s == tool.end_s and tool.attempt == 1
    assert tool.error is not None and tool.error.error_class is ErrorClass.INVALID_ARGS
    assert tool.arguments == {"nope": 1}
    assert trace.run.status is RunStatus.ABSTAINED
    assert [(e.name, e.attributes) for e in trace.run.events] == [
        ("abstained", {"reason": "stuck"})
    ]


def test_unrepresentable_arguments_are_replaced_not_fatal() -> None:
    time_ = FakeTime()
    policy = Scripted(
        [ToolCall(tool="calculator", args={"x": float("nan")}), FinalAnswer(answer="42")], time_
    )
    state, trace = run(policy, time_=time_, registry=default_registry())
    assert state.terminal_label == "succeeded"
    (tool,) = spans(trace, ToolCallSpan)
    assert set(tool.arguments) == {UNREPRESENTABLE}


def test_json_object_helper() -> None:
    mixed: dict[Any, Any] = {"a": [1, 2.5, None], 3: "x"}  # non-string key
    assert json_object(mixed) == {"a": [1, 2.5, None], "3": "x"}
    assert UNREPRESENTABLE in json_object({"a": float("inf")})
    assert UNREPRESENTABLE in json_object({"a": object()})


def test_a_retry_the_deadline_cut_off_is_a_step_event() -> None:
    # The sleeper overshoots, so after the backoff no time is left for attempt 2.
    time_ = FakeTime(oversleep=10.0)
    flaky = CostTool("flaky", time_, cost=0.25, failures=1)
    policy = Scripted([ToolCall(tool="flaky")], time_)
    state, trace = run(
        policy, budgets=Budgets(max_steps=3, deadline_s=2.0), tools=[flaky], time_=time_
    )

    assert state.terminal_label == "budget_exhausted:deadline"
    (tool,) = spans(trace, ToolCallSpan)
    assert tool.error is not None and tool.error.reason == "reset"
    step = spans(trace, StepSpan)[0]
    assert [(e.name, e.attributes) for e in step.events] == [
        ("tool_not_retried", {"reason": "deadline_exceeded", "tool": "flaky"})
    ]
    assert [(e.name, e.attributes) for e in trace.run.events] == [
        ("budget_exhausted", {"kind": "deadline"})
    ]


# --- loop outcomes ---------------------------------------------------------------


def test_budget_stop_before_the_policy_has_no_step_span() -> None:
    time_ = FakeTime()
    a = CostTool("a", time_)
    policy = Scripted([ToolCall(tool="a"), ToolCall(tool="a")], time_)
    state, trace = run(policy, budgets=Budgets(max_steps=2), tools=[a], time_=time_)
    assert state.terminal_label == "budget_exhausted:steps"
    assert [s.index for s in spans(trace, StepSpan)] == [0, 1]
    assert trace.run.events[0].attributes == {"kind": "steps"}
    assert trace.run.events[0].at_s == trace.run.end_s


def test_deadline_used_up_by_a_tool_stops_before_another_step_span() -> None:
    time_ = FakeTime()
    a = CostTool("a", time_, cost=1.0)
    policy = Scripted([ToolCall(tool="a"), ToolCall(tool="a")], time_)
    # The tool call uses the whole second; the budget check sees 1.0 >= 1.0.
    state, trace = run(policy, budgets=Budgets(max_steps=4, deadline_s=1.0), tools=[a], time_=time_)
    assert state.terminal_label == "budget_exhausted:deadline"
    assert len(spans(trace, StepSpan)) == 1


def test_verifier_rejection_is_a_step_event() -> None:
    class RejectWrong:
        async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
            if answer.answer == "wrong":
                return Verdict.reject("bad_answer", "try again")
            return Verdict.accept()

    time_ = FakeTime()
    policy = Scripted([FinalAnswer(answer="wrong"), FinalAnswer(answer="42")], time_)
    state, trace = run(policy, time_=time_, verifier=RejectWrong())
    assert state.terminal_label == "succeeded"
    step0, step1 = spans(trace, StepSpan)
    assert [(e.name, e.attributes) for e in step0.events] == [
        ("answer_rejected", {"reason": "bad_answer"})
    ]
    assert step1.events == []


@pytest.mark.parametrize("broken", ["raises", "returns"])
def test_verifier_failures_explain_themselves(broken: str) -> None:
    class Broken:
        async def verify(self, answer: FinalAnswer, state: RunState) -> object:
            if broken == "raises":
                raise KeyError("x")
            return "yes"

    time_ = FakeTime()
    state, trace = run(Scripted([FinalAnswer(answer="42")], time_), time_=time_, verifier=Broken())
    assert state.terminal_label == "failed:verifier_error"
    assert trace.run.error is not None
    expected = "verifier raised KeyError" if broken == "raises" else "verifier returned str"
    assert trace.run.error.message == expected


def test_loop_guard_events_and_failure_message() -> None:
    time_ = FakeTime()
    a = CostTool("a", time_)
    policy = Scripted([ToolCall(tool="a")] * 3, time_)
    state, trace = run(policy, tools=[a], time_=time_, loop_guard=LoopGuard(max_identical_calls=1))
    assert state.terminal_label == "failed:loop"
    events = [e.name for s in spans(trace, StepSpan) for e in s.events]
    assert events == ["loop_detected", "loop_detected"]
    assert trace.run.error is not None and "already" in trace.run.error.message
    assert trace.run.error.error_class is ErrorClass.PERMANENT


@pytest.mark.parametrize(
    ("item", "label", "message"),
    [
        (KeyError("x"), "failed:policy_error", "policy raised KeyError"),
        (TimeoutError(), "failed:policy_error", "policy raised TimeoutError"),
        (PolicyFailure("my_reason", "because"), "failed:my_reason", "because"),
        (PolicyFailure("my_reason"), "failed:my_reason", "my_reason"),
    ],
)
def test_policy_failures_are_on_the_run_span(item: Exception, label: str, message: str) -> None:
    time_ = FakeTime()
    state, trace = run(Scripted([item], time_), time_=time_)
    assert state.terminal_label == label
    assert trace.run.error is not None
    assert (trace.run.error.message, trace.run.error.error_class) == (
        message,
        ErrorClass.PERMANENT,
    )
    assert spans(trace, ProviderCallSpan) == []


def test_policy_failure_after_a_clean_call_is_permanent() -> None:
    time_ = FakeTime()
    clean = ProviderCallInfo(provider="p", model="m", tool_format="native", input_tokens=3)
    state, trace = run(Scripted([PolicyFailure("odd", calls=[clean])], time_), time_=time_)
    assert state.terminal_label == "failed:odd"
    assert trace.run.error is not None and trace.run.error.error_class is ErrorClass.PERMANENT
    (call,) = spans(trace, ProviderCallSpan)
    assert call.input_tokens == 3 and call.parse_failure is None and call.error is None


def test_invalid_action_message() -> None:
    time_ = FakeTime()
    state, trace = run(Scripted(["nope"], time_), time_=time_)  # type: ignore[list-item]
    assert state.terminal_label == "failed:invalid_action"
    assert trace.run.error is not None
    assert trace.run.error.message == "policy returned str, not an action"


async def test_policy_timeout_is_a_timeout_error_with_no_provider_span() -> None:
    class Slow:
        async def next_action(self, state: RunState) -> FinalAnswer:
            await asyncio.sleep(5)
            return FinalAnswer(answer="late")  # pragma: no cover

    loop = AgentLoop(
        Slow(), ToolExecutor(ToolRegistry()), Budgets(max_steps=2), step_timeout_s=0.01
    )
    traced = await loop.run_traced(TASK)
    assert traced.state.terminal_label == "failed:policy_timeout"
    error = traced.trace.run.error
    assert error is not None and error.error_class is ErrorClass.TIMEOUT
    assert error.message == "policy did not decide within 0.01s"
    assert spans(traced.trace, ProviderCallSpan) == []
    started = traced.trace.header.started_at
    assert started.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - started) < timedelta(minutes=5)


def test_run_metadata_and_run_matches_run_traced() -> None:
    time_ = FakeTime()
    loop = AgentLoop(
        Scripted([FinalAnswer(answer="42")] * 2, time_),
        ToolExecutor(ToolRegistry(), clock=time_.clock),
        Budgets(max_steps=2, max_tool_calls=3, max_tokens=50, deadline_s=9.0),
    )
    started = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=8)))
    traced = asyncio.run(
        loop.run_traced(
            TASK, provider="scripted", model="m", tool_format="json", started_at=started
        )
    )
    run_span = traced.trace.run
    assert (run_span.provider, run_span.model, run_span.tool_format) == ("scripted", "m", "json")
    limits = run_span.budget_limits
    assert (limits.steps, limits.tool_calls, limits.tokens, limits.time_s) == (2, 3, 50, 9.0)
    assert traced.trace.header.started_at == started
    assert asyncio.run(loop.run(TASK)) == traced.state


@pytest.mark.parametrize("run_id", ["", "has space", "a/b", "x" * 65])
def test_run_id_must_be_a_valid_trace_id(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        AgentLoop(
            Scripted([], FakeTime()),
            ToolExecutor(ToolRegistry()),
            Budgets(max_steps=1),
            run_id=run_id,
        )


# --- TraceRecorder guards ---------------------------------------------------------


def test_recorder_refuses_misuse() -> None:
    rec = TraceRecorder(10.0)
    assert rec.rel(9.0) == 0.0 and rec.rel(10.5) == 0.5
    with pytest.raises(RuntimeError, match="no open step"):
        rec.end_step(11.0)
    rec.begin_step(0, 10.0)
    assert rec.step_open
    with pytest.raises(RuntimeError, match="still open"):
        rec.begin_step(1, 10.0)
    with pytest.raises(ValueError, match="not finished"):
        rec.build(
            RunState(task=TASK),
            Budgets(max_steps=1),
            at=11.0,
            run_id="r",
            started_at=STARTED,
            provider="none",
            model=None,
            tool_format=None,
        )


def test_recorder_clamps_backwards_time_and_early_events() -> None:
    rec = TraceRecorder(0.0)
    rec.begin_step(0, 5.0)
    rec.step_event("early", at=1.0)
    rec.provider_calls((ProviderCallInfo(provider="p", model="m", tool_format="native"),), 5.0, 4.0)
    rec.end_step(3.0)  # a clock that went backwards still gives a valid span
    state = RunState(task=TASK)
    state.finish(LoopStatus.ABSTAINED, detail="stuck")
    trace = rec.build(
        state,
        Budgets(max_steps=1),
        at=2.0,
        run_id="r",
        started_at=STARTED,
        provider="none",
        model=None,
        tool_format=None,
    )
    step = spans(trace, StepSpan)[0]
    call = spans(trace, ProviderCallSpan)[0]
    assert (step.start_s, step.end_s, step.events[0].at_s) == (5.0, 5.0, 5.0)
    assert (call.start_s, call.end_s) == (5.0, 5.0)
    assert trace.run.end_s == 5.0  # the run covers its children
