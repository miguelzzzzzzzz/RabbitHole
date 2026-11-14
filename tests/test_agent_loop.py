"""AgentLoop: one test per budget kind, deadline-aware retries, failure paths.

Time is a fake clock shared by the executor and the loop: tools advance it by
a fixed cost per attempt and the sleeper advances it by each backoff, so no
test waits for backoff in real time. Only the policy-timeout tests use real
(short) waits, because ``asyncio.timeout`` runs on the event loop's clock.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentic_research.agent import (
    AgentLoop,
    BudgetKind,
    Budgets,
    Decision,
    FinalAnswer,
    GiveUp,
    LoopGuard,
    ObservationShaper,
    Policy,
    RunState,
    RunStatus,
    Task,
    ToolCall,
    Verdict,
    Verifier,
)
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
from agentic_research.trace import StepSpan, ToolCallSpan, dumps_trace, loads_trace

TASK = Task(question="What is 6*7?")
NO_JITTER = RetryPolicy(
    max_attempts=4, initial_backoff_s=0.5, multiplier=2.0, max_backoff_s=8.0, jitter=0.0
)


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


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

    def __init__(
        self,
        name: str,
        time_: FakeTime,
        *,
        cost: float = 0.1,
        failures: int = 0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.retry = retry if retry is not None else NO_JITTER
        self._time = time_
        self._cost = cost
        self._failures = failures
        self.contexts: list[ToolContext] = []

    async def run(self, args: NoArgs, ctx: ToolContext) -> Out:
        self.contexts.append(ctx)
        self._time.now += self._cost
        if len(self.contexts) <= self._failures:
            raise ToolFailure(
                ToolError(error_class=ErrorClass.TRANSIENT, reason="reset", message="x")
            )
        return Out(value=42)


Scripted = ToolCall | FinalAnswer | GiveUp | Decision


class ScriptedPolicy:
    """Returns the scripted items in order; raises IndexError when it runs out."""

    def __init__(self, items: Sequence[Scripted]) -> None:
        self._items = list(items)
        self.seen: list[RunState] = []

    async def next_action(self, state: RunState) -> Scripted:
        self.seen.append(state)
        return self._items[len(self.seen) - 1]


class RejectWrong:
    """Rejects the answer ``"wrong"``; accepts anything else."""

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        if answer.answer == "wrong":
            return Verdict.reject("bad_answer", "try again")
        return Verdict.accept()


DEFAULT_GUARD = LoopGuard()


def make_loop(
    items: Sequence[Scripted],
    budgets: Budgets,
    *,
    tools: Sequence[CostTool] = (),
    time_: FakeTime | None = None,
    verifier: Verifier | None = None,
    step_timeout_s: float | None = None,
    loop_guard: LoopGuard | None = DEFAULT_GUARD,
    shaper: ObservationShaper | None = None,
) -> tuple[AgentLoop, ScriptedPolicy, FakeTime]:
    time_ = time_ or FakeTime()
    executor = ToolExecutor(ToolRegistry(tools), clock=time_.clock, sleeper=time_.sleep)
    policy = ScriptedPolicy(items)
    loop = AgentLoop(
        policy,
        executor,
        budgets,
        verifier=verifier,
        step_timeout_s=step_timeout_s,
        run_id="r-test",
        loop_guard=loop_guard,
        shaper=shaper,
    )
    return loop, policy, time_


CALL = ToolCall(tool="a")
ANSWER = FinalAnswer(answer="42")


# ---- happy paths ------------------------------------------------------------------


async def test_tool_call_then_answer_succeeds() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, policy, _ = make_loop([CALL, ANSWER], Budgets(max_steps=5), tools=[tool], time_=time_)
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.answer == "42"
    assert [s.action for s in state.steps] == [CALL, ANSWER]
    first = state.steps[0].result
    assert first is not None and first.output == {"value": 42}
    assert state.steps[1].verdict == Verdict.accept()
    c = state.counters
    assert (c.steps, c.tool_calls, c.per_tool, c.tool_attempts) == (2, 1, {"a": 1}, 1)
    assert c.elapsed_s == pytest.approx(0.1)
    assert [ctx.run_id for ctx in tool.contexts] == ["r-test"]
    assert [ctx.step for ctx in tool.contexts] == [0]
    # The policy saw the tool result before answering.
    assert policy.seen[1].steps[0].result == first


async def test_give_up_abstains_with_its_reason() -> None:
    loop, _, _ = make_loop([GiveUp(reason="no_evidence")], Budgets(max_steps=3))
    state = await loop.run(TASK)
    assert state.terminal_label == "abstained:no_evidence"
    assert state.counters.steps == 1


async def test_unknown_tool_is_an_observation_not_a_crash() -> None:
    loop, _, _ = make_loop([ToolCall(tool="nope"), ANSWER], Budgets(max_steps=3))
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    result = state.steps[0].result
    assert result is not None and result.error is not None
    assert result.error.reason == "unknown_tool"
    assert (state.counters.tool_calls, state.counters.tool_attempts) == (1, 0)


async def test_policy_gets_a_copy_it_cannot_corrupt() -> None:
    class Vandal:
        async def next_action(self, state: RunState) -> Scripted:
            state.counters.steps = 99
            state.steps.clear()
            return ANSWER

    time_ = FakeTime()
    loop = AgentLoop(
        Vandal(), ToolExecutor(ToolRegistry(), clock=time_.clock), Budgets(max_steps=2)
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.counters.steps == 1
    assert len(state.steps) == 1


# ---- one test per budget kind ------------------------------------------------------


async def test_steps_budget() -> None:
    time_ = FakeTime()
    loop, policy, _ = make_loop(
        [CALL] * 5, Budgets(max_steps=3), tools=[CostTool("a", time_)], time_=time_
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:steps"
    assert state.counters.steps == 3
    assert len(policy.seen) == 3


async def test_tool_calls_budget() -> None:
    time_ = FakeTime()
    tools = [CostTool("a", time_), CostTool("b", time_)]
    script = [CALL, ToolCall(tool="b"), CALL]
    loop, _, _ = make_loop(
        script, Budgets(max_steps=10, max_tool_calls=2), tools=tools, time_=time_
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:tool_calls"
    assert state.counters.tool_calls == 2
    # The third call was refused before it ran and is recorded without a result.
    assert state.steps[-1].action == CALL and state.steps[-1].result is None
    assert len(tools[0].contexts) == 1


async def test_answer_after_the_last_allowed_tool_call_succeeds() -> None:
    time_ = FakeTime()
    budgets = Budgets(max_steps=10, max_tool_calls=1, max_tool_calls_per_tool=1)
    loop, _, _ = make_loop([CALL, ANSWER], budgets, tools=[CostTool("a", time_)], time_=time_)
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.counters.tool_calls == 1


async def test_tool_calls_per_tool_budget() -> None:
    time_ = FakeTime()
    tools = [CostTool("a", time_), CostTool("b", time_)]
    script = [CALL, ToolCall(tool="b"), CALL, ToolCall(tool="b"), CALL]
    budgets = Budgets(max_steps=10, max_tool_calls_per_tool=2)
    loop, _, _ = make_loop(script, budgets, tools=tools, time_=time_)
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:tool_calls_per_tool"
    assert state.counters.per_tool == {"a": 2, "b": 2}
    assert len(tools[0].contexts) == 2


async def test_deadline_budget() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_, cost=1.0)
    loop, _, _ = make_loop(
        [CALL] * 10, Budgets(max_steps=10, deadline_s=2.5), tools=[tool], time_=time_
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:deadline"
    assert state.counters.tool_calls == 3
    assert state.counters.elapsed_s == pytest.approx(3.0)


async def test_tokens_budget() -> None:
    script: list[Scripted] = [Decision(action=CALL, tokens=60)] * 3
    time_ = FakeTime()
    loop, _, _ = make_loop(
        script, Budgets(max_steps=10, max_tokens=100), tools=[CostTool("a", time_)], time_=time_
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:tokens"
    # Soft: the second call's usage passes the limit, so its tool call is refused.
    assert state.counters.tokens == 120
    assert state.counters.tool_calls == 1
    assert state.steps[1].result is None


async def test_answer_is_honoured_even_when_its_tokens_pass_the_limit() -> None:
    loop, _, _ = make_loop(
        [Decision(action=ANSWER, tokens=500)], Budgets(max_steps=3, max_tokens=100)
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.counters.tokens == 500


async def test_repairs_budget() -> None:
    wrong = FinalAnswer(answer="wrong")
    loop, policy, _ = make_loop(
        [wrong] * 5, Budgets(max_steps=10, max_repairs=2), verifier=RejectWrong()
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:repairs"
    # max_repairs is the number of repairs allowed: three rejections end the run.
    assert state.counters.repairs == 3
    assert len(policy.seen) == 3
    assert all(s.verdict == Verdict.reject("bad_answer", "try again") for s in state.steps)


async def test_rejected_answer_is_repaired() -> None:
    loop, policy, _ = make_loop(
        [FinalAnswer(answer="wrong"), ANSWER], Budgets(max_steps=5), verifier=RejectWrong()
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.counters.repairs == 1
    # The policy saw the rejection verdict when asked again.
    verdict = policy.seen[1].steps[0].verdict
    assert verdict is not None and verdict.reason == "bad_answer"


# ---- Chad's check 2: retries and backoff count against the run budget -------------


async def test_retry_attempts_and_backoff_are_charged_to_the_run() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_, cost=0.1, failures=2)
    loop, _, _ = make_loop([CALL, ANSWER], Budgets(max_steps=5), tools=[tool], time_=time_)
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    c = state.counters
    assert c.tool_calls == 1  # one call the policy asked for...
    assert c.tool_attempts == 3  # ...that ran three times
    assert c.backoff_s == pytest.approx(0.5 + 1.0)
    assert time_.sleeps == pytest.approx([0.5, 1.0])
    assert c.elapsed_s == pytest.approx(3 * 0.1 + 1.5)


async def test_run_deadline_is_passed_to_the_executor() -> None:
    # Backoffs would be 0.5, 1.0, 2.0. With a 2.0 s run deadline the third
    # backoff (ending at 1.8 + 2.0) would cross it, so the executor stops
    # after three attempts instead of the policy's four.
    time_ = FakeTime()
    tool = CostTool("a", time_, cost=0.1, failures=10)
    loop, _, _ = make_loop(
        [CALL, ANSWER], Budgets(max_steps=5, deadline_s=2.0), tools=[tool], time_=time_
    )
    state = await loop.run(TASK)
    result = state.steps[0].result
    assert result is not None and result.error is not None
    assert result.error.reason == "reset"
    assert len(tool.contexts) == 3
    assert state.counters.tool_attempts == 3
    assert state.counters.backoff_s == pytest.approx(1.5)
    assert state.terminal_label == "succeeded"

    # Without a deadline the same tool uses all four attempts.
    time2 = FakeTime()
    tool2 = CostTool("a", time2, cost=0.1, failures=10)
    loop2, _, _ = make_loop([CALL, ANSWER], Budgets(max_steps=5), tools=[tool2], time_=time2)
    await loop2.run(TASK)
    assert len(tool2.contexts) == 4


async def test_retry_time_alone_can_exhaust_the_deadline() -> None:
    # One tool call: attempt (1.0 s) + backoff (0.5 s) + attempt (1.0 s) = 2.5 s,
    # which uses up the 2.5 s deadline before the policy is asked again.
    time_ = FakeTime()
    tool = CostTool("a", time_, cost=1.0, failures=10)
    loop, policy, _ = make_loop(
        [CALL, ANSWER], Budgets(max_steps=5, deadline_s=2.5), tools=[tool], time_=time_
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:deadline"
    assert state.counters.tool_calls == 1
    assert state.counters.tool_attempts == 2
    assert state.counters.elapsed_s == pytest.approx(2.5)
    assert len(policy.seen) == 1


# ---- policy and verifier failures ------------------------------------------------


class Hang:
    async def next_action(self, state: RunState) -> Scripted:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_policy_step_timeout_fails_the_run() -> None:
    loop = AgentLoop(
        Hang(), ToolExecutor(ToolRegistry()), Budgets(max_steps=3), step_timeout_s=0.02
    )
    started = time.monotonic()
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:policy_timeout"
    assert time.monotonic() - started < 5


async def test_step_timeout_shorter_than_the_deadline_still_fails_the_run() -> None:
    loop = AgentLoop(
        Hang(),
        ToolExecutor(ToolRegistry()),
        Budgets(max_steps=3, deadline_s=60.0),
        step_timeout_s=0.02,
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:policy_timeout"


async def test_policy_hanging_past_the_deadline_exhausts_it() -> None:
    loop = AgentLoop(
        Hang(),
        ToolExecutor(ToolRegistry()),
        Budgets(max_steps=3, deadline_s=0.05),
        step_timeout_s=60,
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:deadline"


async def test_no_time_left_before_the_policy_exhausts_the_deadline() -> None:
    # The policy itself uses up the deadline; the next step cannot start.
    time_ = FakeTime()

    class Slow:
        async def next_action(self, state: RunState) -> Scripted:
            time_.now += 5.0
            return CALL

    loop = AgentLoop(
        Slow(),
        ToolExecutor(ToolRegistry([CostTool("a", time_)]), clock=time_.clock),
        Budgets(max_steps=3, deadline_s=1.0),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:deadline"
    # The call chosen after the deadline passed is refused, not run.
    assert state.steps[0].result is None


async def test_deadline_rechecked_between_budget_check_and_policy() -> None:
    # A clock that jumps past the deadline between the loop's budget check and
    # the policy call (as a real clock can) ends the run without asking the policy.
    times = iter([0.0, 0.0, 10.0])
    executor = ToolExecutor(ToolRegistry(), clock=lambda: next(times, 10.0))
    policy = ScriptedPolicy([ANSWER])
    loop = AgentLoop(policy, executor, Budgets(max_steps=3, deadline_s=1.0))
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:deadline"
    assert policy.seen == []


@pytest.mark.parametrize(
    ("exc", "label"),
    [
        (RuntimeError("boom"), "failed:policy_error"),
        (TimeoutError("own timeout"), "failed:policy_error"),
        (IndexError("script ran out"), "failed:policy_error"),
    ],
)
async def test_policy_exception_fails_the_run(exc: Exception, label: str) -> None:
    class Raises:
        async def next_action(self, state: RunState) -> Scripted:
            raise exc

    loop = AgentLoop(Raises(), ToolExecutor(ToolRegistry()), Budgets(max_steps=3))
    state = await loop.run(TASK)
    assert state.terminal_label == label
    assert state.steps == []


async def test_policy_returning_garbage_fails_the_run() -> None:
    class Garbage:
        async def next_action(self, state: RunState) -> Any:
            return {"kind": "final_answer", "answer": "42"}

    loop = AgentLoop(Garbage(), ToolExecutor(ToolRegistry()), Budgets(max_steps=3))
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:invalid_action"


async def test_verifier_exception_fails_the_run() -> None:
    class Broken:
        async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
            raise RuntimeError("verifier bug")

    loop, _, _ = make_loop([ANSWER], Budgets(max_steps=3), verifier=Broken())
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:verifier_error"
    assert state.steps[0].verdict is None


async def test_verifier_returning_garbage_fails_the_run() -> None:
    class Garbage:
        async def verify(self, answer: FinalAnswer, state: RunState) -> Any:
            return True

    loop, _, _ = make_loop([ANSWER], Budgets(max_steps=3), verifier=Garbage())
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:verifier_error"


async def test_outer_cancellation_propagates() -> None:
    loop = AgentLoop(Hang(), ToolExecutor(ToolRegistry()), Budgets(max_steps=3))
    task = asyncio.ensure_future(loop.run(TASK))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---- construction -----------------------------------------------------------------


def test_max_steps_is_required() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        AgentLoop(ScriptedPolicy([]), ToolExecutor(ToolRegistry()), Budgets(max_tool_calls=3))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_step_timeout_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="step_timeout_s"):
        AgentLoop(
            ScriptedPolicy([]),
            ToolExecutor(ToolRegistry()),
            Budgets(max_steps=1),
            step_timeout_s=bad,
        )


def test_run_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="run_id"):
        AgentLoop(ScriptedPolicy([]), ToolExecutor(ToolRegistry()), Budgets(max_steps=1), run_id="")


def test_scripted_policy_satisfies_the_protocol() -> None:
    _p: Policy = ScriptedPolicy([])
    _v: Verifier = RejectWrong()


# ---- property: counters never pass their limits; every run ends terminal ----------

_ACTIONS: st.SearchStrategy[Scripted] = st.one_of(
    st.sampled_from(["a", "b", "flaky", "missing"]).map(lambda t: ToolCall(tool=t)),
    st.just(FinalAnswer(answer="wrong")),
    st.just(ANSWER),
    st.just(GiveUp(reason="stuck")),
)
_SCRIPTED = st.one_of(
    _ACTIONS,
    st.builds(Decision, action=_ACTIONS, tokens=st.integers(min_value=0, max_value=50)),
)
_OPT = st.none() | st.integers(min_value=1, max_value=6)
_BUDGETS = st.builds(
    Budgets,
    max_steps=st.integers(min_value=1, max_value=12),
    max_tool_calls=_OPT,
    max_tool_calls_per_tool=_OPT,
    deadline_s=st.none() | st.floats(min_value=0.05, max_value=5.0),
    max_tokens=st.none() | st.integers(min_value=1, max_value=120),
    max_repairs=st.none() | st.integers(min_value=0, max_value=3),
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    script=st.lists(_SCRIPTED, max_size=15),
    budgets=_BUDGETS,
    max_identical=st.none() | st.integers(min_value=1, max_value=4),
)
def test_property_counters_stay_within_limits(
    script: list[Scripted], budgets: Budgets, max_identical: int | None
) -> None:
    time_ = FakeTime()
    tools = [
        CostTool("a", time_, cost=0.3),
        CostTool("b", time_, cost=0.05),
        CostTool("flaky", time_, cost=0.2, failures=2),
    ]
    guard = None if max_identical is None else LoopGuard(max_identical_calls=max_identical)
    loop, policy, _ = make_loop(
        script, budgets, tools=tools, time_=time_, verifier=RejectWrong(), loop_guard=guard
    )
    traced = asyncio.run(loop.run_traced(TASK))
    state, trace = traced.state, traced.trace

    assert state.terminal
    c = state.counters
    assert c.steps == len(state.steps)
    assert budgets.max_steps is not None and c.steps <= budgets.max_steps
    if budgets.max_tool_calls is not None:
        assert c.tool_calls <= budgets.max_tool_calls
    if budgets.max_tool_calls_per_tool is not None:
        assert all(n <= budgets.max_tool_calls_per_tool for n in c.per_tool.values())
    if budgets.max_repairs is not None:
        assert c.repairs <= budgets.max_repairs + 1
    assert c.tool_attempts == sum(len(t.contexts) for t in tools)
    assert c.tool_calls == sum(1 for s in state.steps if s.result is not None)
    if state.status is RunStatus.BUDGET_EXHAUSTED:
        assert state.detail is not None
        limit = {
            BudgetKind.STEPS: budgets.max_steps,
            BudgetKind.TOOL_CALLS: budgets.max_tool_calls,
            BudgetKind.TOOL_CALLS_PER_TOOL: budgets.max_tool_calls_per_tool,
            BudgetKind.DEADLINE: budgets.deadline_s,
            BudgetKind.TOKENS: budgets.max_tokens,
            BudgetKind.REPAIRS: budgets.max_repairs,
        }[BudgetKind(state.detail)]
        assert limit is not None
    for tool in tools:  # no identical call runs more often than the guard allows
        if max_identical is not None:
            assert len([s for s in state.steps if s.result and s.result.tool == tool.name]) <= (
                max_identical
            )
    observed = [o.step for o in state.observations]
    assert observed == sorted(set(observed))
    assert sum(not o.elided for o in state.observations) <= 3
    if state.status is RunStatus.FAILED:
        # Besides the script running out, the only failure is an ignored loop guardrail.
        if state.detail == "loop":
            assert max_identical is not None
            last = state.steps[-1]
            assert last.guardrail is not None and last.result is None
        else:
            assert state.detail == "policy_error"
            assert len(policy.seen) == len(script) + 1

    # The trace (already validated by `Trace`) agrees with the state.
    step_spans = [s for s in trace.spans if isinstance(s, StepSpan)]
    assert len(step_spans) - len(state.steps) in (0, 1)
    assert len(step_spans) == len(policy.seen)  # one step span per policy call
    tool_parents = {s.parent_id for s in trace.spans if isinstance(s, ToolCallSpan)}
    ran = {f"step-{s.index}" for s in state.steps if s.result is not None}
    assert tool_parents == ran
    attempts = [s for s in trace.spans if isinstance(s, ToolCallSpan)]
    assert len(attempts) >= c.tool_attempts
    used = trace.run.budget_used
    assert (used.steps, used.tool_calls, used.tokens) == (c.steps, c.tool_calls, c.tokens)
    assert all(s.end_s <= trace.run.end_s for s in trace.spans)
    assert loads_trace(dumps_trace(trace)) == trace
    assert RunState.from_json(state.to_json()) == state


# ---- loop detection ----------------------------------------------------------------


class XArgs(ToolArgs):
    x: float


class EchoX:
    """Takes one number; counts its runs."""

    name = "echo"
    description = "echo x"
    args_model: type[ToolArgs] = XArgs
    output_model: type[ToolOutput] = Out
    timeout_s = 5.0
    idempotent = True
    retry = NO_JITTER

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, args: XArgs, ctx: ToolContext) -> Out:
        self.runs += 1
        return Out(value=int(args.x))


async def test_identical_call_past_the_threshold_gets_a_guardrail_then_can_answer() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, policy, _ = make_loop(
        [CALL, CALL, CALL, ANSWER],
        Budgets(max_steps=10),
        tools=[tool],
        time_=time_,
        loop_guard=LoopGuard(max_identical_calls=2),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert len(tool.contexts) == 2  # the third identical call never ran
    refused = state.steps[2]
    assert refused.result is None
    assert refused.guardrail is not None and refused.guardrail.code == "loop_detected"
    c = state.counters
    assert (c.steps, c.tool_calls, c.per_tool, c.tool_attempts) == (4, 2, {"a": 2}, 2)
    # The policy saw the guardrail as its newest observation before answering.
    last = policy.seen[3].observations[-1]
    assert (last.step, last.source) == (2, "guardrail")
    assert "already ran 2 time(s)" in last.text


async def test_ignoring_the_guardrail_fails_the_run_as_loop() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, _, _ = make_loop(
        [CALL] * 6,
        Budgets(max_steps=10),
        tools=[tool],
        time_=time_,
        loop_guard=LoopGuard(max_identical_calls=2),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:loop"
    assert len(tool.contexts) == 2
    assert [s.result is not None for s in state.steps] == [True, True, False, False]
    assert [s.guardrail is not None for s in state.steps] == [False, False, True, True]
    assert state.counters.steps == 4
    assert RunState.from_json(state.to_json()) == state


async def test_default_guard_allows_three_identical_calls() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, _, _ = make_loop([CALL] * 5, Budgets(max_steps=10), tools=[tool], time_=time_)
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:loop"
    assert len(tool.contexts) == 3


async def test_a_cycle_between_two_calls_is_caught() -> None:
    time_ = FakeTime()
    tools = [CostTool("a", time_), CostTool("b", time_)]
    b = ToolCall(tool="b")
    loop, _, _ = make_loop(
        [CALL, b, CALL, b, CALL, b, CALL],
        Budgets(max_steps=20),
        tools=tools,
        time_=time_,
        loop_guard=LoopGuard(max_identical_calls=2),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:loop"
    assert [len(t.contexts) for t in tools] == [2, 2]
    assert [s.guardrail is not None for s in state.steps] == [False] * 4 + [True] * 3


async def test_normalized_args_count_as_identical_and_new_args_run() -> None:
    tool = EchoX()
    script: list[Scripted] = [
        ToolCall(tool="echo", args={"x": 2}),
        ToolCall(tool="echo", args={"x": 2.0}),  # identical after normalization
        ToolCall(tool="echo", args={"x": 3}),  # different: runs
        FinalAnswer(answer="2"),
    ]
    time_ = FakeTime()
    executor = ToolExecutor(ToolRegistry([tool]), clock=time_.clock, sleeper=time_.sleep)
    loop = AgentLoop(
        ScriptedPolicy(script),
        executor,
        Budgets(max_steps=10),
        loop_guard=LoopGuard(max_identical_calls=1),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert tool.runs == 2
    assert [s.guardrail is not None for s in state.steps] == [False, True, False, False]


async def test_loop_guard_none_disables_detection() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, _, _ = make_loop(
        [CALL] * 5 + [ANSWER], Budgets(max_steps=10), tools=[tool], time_=time_, loop_guard=None
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert len(tool.contexts) == 5


async def test_budget_recheck_comes_before_the_loop_guard() -> None:
    time_ = FakeTime()
    tool = CostTool("a", time_)
    loop, _, _ = make_loop(
        [CALL] * 4,
        Budgets(max_steps=10, max_tool_calls_per_tool=2),
        tools=[tool],
        time_=time_,
        loop_guard=LoopGuard(max_identical_calls=2),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:tool_calls_per_tool"
    assert state.steps[-1].guardrail is None


# ---- observation shaping in the loop -----------------------------------------------


async def test_policy_sees_shaped_observations_with_old_ones_elided() -> None:
    time_ = FakeTime()
    tools = [CostTool("a", time_), CostTool("b", time_)]
    script: list[Scripted] = [CALL, ToolCall(tool="b"), ToolCall(tool="nope"), ANSWER]
    loop, policy, _ = make_loop(
        script,
        Budgets(max_steps=10),
        tools=tools,
        time_=time_,
        shaper=ObservationShaper(keep_recent=2),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert policy.seen[0].observations == []
    assert [o.text for o in policy.seen[1].observations] == ['a ok: {"value":42}']
    seen = policy.seen[3].observations
    assert [(o.step, o.elided) for o in seen] == [(0, True), (1, False), (2, False)]
    assert seen[0].text == "step 0: a ok, 12 chars elided"
    assert seen[2].text.startswith("nope error invalid_args/unknown_tool: ")
    # The accepted answer adds no observation; full results stay in the steps.
    assert state.observations == seen
    first = state.steps[0].result
    assert first is not None and first.output == {"value": 42}


async def test_rejected_answer_becomes_a_feedback_observation() -> None:
    loop, policy, _ = make_loop(
        [FinalAnswer(answer="wrong"), ANSWER], Budgets(max_steps=5), verifier=RejectWrong()
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    (feedback,) = policy.seen[1].observations
    assert (feedback.source, feedback.text) == ("verifier", "answer rejected bad_answer: try again")


async def test_long_tool_output_is_truncated_for_the_policy_only() -> None:
    class Big(ToolOutput):
        text: str

    class BigTool:
        name = "big"
        description = "big"
        args_model: type[ToolArgs] = NoArgs
        output_model: type[ToolOutput] = Big
        timeout_s = 5.0
        idempotent = True
        retry = NO_JITTER

        async def run(self, args: NoArgs, ctx: ToolContext) -> Big:
            return Big(text="z" * 500)

    time_ = FakeTime()
    executor = ToolExecutor(ToolRegistry([BigTool()]), clock=time_.clock, sleeper=time_.sleep)
    policy = ScriptedPolicy([ToolCall(tool="big"), ANSWER])
    loop = AgentLoop(policy, executor, Budgets(max_steps=5), shaper=ObservationShaper(max_chars=40))
    state = await loop.run(TASK)
    (obs,) = policy.seen[1].observations
    assert obs.text.startswith('big ok: {"text":"zzz')
    assert obs.truncated_chars == len('big ok: {"text":"' + "z" * 500 + '"}') - 40
    assert obs.text.endswith(f"...[truncated {obs.truncated_chars} chars]")
    result = state.steps[0].result
    assert result is not None and result.output == {"text": "z" * 500}
