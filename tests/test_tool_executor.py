"""Behaviour tests for ToolExecutor: no real backoff sleeping, real cancellation."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
from agentic_research.tools.executor import DEADLINE_REASON

CTX = ToolContext(run_id="r1", step=3)


class NoArgs(ToolArgs):
    pass


class Out(ToolOutput):
    value: int


class FakeTime:
    """A clock that only moves when told to; its sleeper records and advances."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


Step = Callable[[ToolContext], Awaitable[object]]


class ScriptTool:
    """Each call runs the next scripted step (the last one repeats)."""

    description = "scripted"
    args_model: type[ToolArgs] = NoArgs
    output_model: type[ToolOutput] = Out

    def __init__(
        self,
        steps: Sequence[Step],
        *,
        name: str = "script",
        timeout_s: float = 1.0,
        retry: RetryPolicy | None = None,
        idempotent: bool = True,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retry = retry if retry is not None else RetryPolicy()
        self.idempotent = idempotent
        self._steps = list(steps)
        self.contexts: list[ToolContext] = []

    @property
    def calls(self) -> int:
        return len(self.contexts)

    async def run(self, args: NoArgs, ctx: ToolContext) -> Any:
        self.contexts.append(ctx)
        step = self._steps[min(len(self.contexts), len(self._steps)) - 1]
        return await step(ctx)


def ok(value: int = 1) -> Step:
    async def step(ctx: ToolContext) -> object:
        return Out(value=value)

    return step


def fail(error_class: ErrorClass, reason: str = "boom") -> Step:
    async def step(ctx: ToolContext) -> object:
        raise ToolFailure(ToolError(error_class=error_class, reason=reason, message="x"))

    return step


def raising(exc: BaseException) -> Step:
    async def step(ctx: ToolContext) -> object:
        raise exc

    return step


def make(
    tool: ScriptTool, time_: FakeTime | None = None, seed: int = 0
) -> tuple[ToolExecutor, FakeTime]:
    time_ = time_ or FakeTime()
    executor = ToolExecutor(ToolRegistry([tool]), seed=seed, clock=time_.clock, sleeper=time_.sleep)
    return executor, time_


NO_JITTER = RetryPolicy(
    max_attempts=4, initial_backoff_s=0.5, multiplier=2.0, max_backoff_s=8.0, jitter=0.0
)


async def test_success_first_try_does_not_sleep() -> None:
    tool = ScriptTool([ok(7)])
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert run.result.ok
    assert run.result.output == {"value": 7}
    assert [a.attempt for a in run.attempts] == [1]
    assert run.attempts[0].ok
    assert time_.sleeps == []
    assert run.total_backoff_s == 0.0


async def test_transient_failures_retry_with_exact_backoff_schedule() -> None:
    tool = ScriptTool(
        [fail(ErrorClass.TRANSIENT), fail(ErrorClass.TRANSIENT), ok(5)], retry=NO_JITTER
    )
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert run.result.output == {"value": 5}
    assert tool.calls == 3
    assert time_.sleeps == [0.5, 1.0]
    assert [a.backoff_s for a in run.attempts] == [0.5, 1.0, 0.0]
    assert [a.error.error_class if a.error else None for a in run.attempts] == [
        ErrorClass.TRANSIENT,
        ErrorClass.TRANSIENT,
        None,
    ]
    assert run.total_backoff_s == 1.5


async def test_tool_sees_incrementing_attempt_number() -> None:
    tool = ScriptTool([fail(ErrorClass.TRANSIENT), fail(ErrorClass.TRANSIENT), ok()])
    executor, _ = make(tool)
    await executor.execute("script", {}, CTX)
    assert [c.attempt for c in tool.contexts] == [1, 2, 3]
    assert {(c.run_id, c.step) for c in tool.contexts} == {("r1", 3)}


async def test_jittered_backoff_is_reproducible_from_the_seed() -> None:
    policy = RetryPolicy(
        max_attempts=4, initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=10.0, jitter=0.5
    )

    async def schedule(seed: int) -> list[float]:
        tool = ScriptTool([fail(ErrorClass.TRANSIENT)], retry=policy)
        executor, time_ = make(tool, seed=seed)
        await executor.execute("script", {}, CTX)
        return time_.sleeps

    rng = random.Random(42)
    expected = [1.0 * (1 - 0.5 * rng.random()), 2.0 * (1 - 0.5 * rng.random())]
    expected.append(4.0 * (1 - 0.5 * rng.random()))
    assert await schedule(42) == expected
    assert await schedule(42) == expected
    assert await schedule(43) != expected


async def test_rng_stream_continues_across_calls_on_one_executor() -> None:
    policy = RetryPolicy(max_attempts=2, initial_backoff_s=1.0, jitter=1.0)
    tool = ScriptTool([fail(ErrorClass.TRANSIENT)], retry=policy)
    executor, time_ = make(tool, seed=5)
    await executor.execute("script", {}, CTX)
    await executor.execute("script", {}, CTX)
    rng = random.Random(5)
    assert time_.sleeps == [1.0 * (1 - rng.random()), 1.0 * (1 - rng.random())]
    assert time_.sleeps[0] != time_.sleeps[1]


async def test_attempts_are_capped_by_policy_and_last_error_is_returned() -> None:
    tool = ScriptTool([fail(ErrorClass.TRANSIENT, "reset")], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert tool.calls == 4
    assert len(run.attempts) == 4
    assert time_.sleeps == [0.5, 1.0, 2.0]
    assert run.attempts[-1].backoff_s == 0.0  # no sleep after the final failure
    assert run.result.error is not None
    assert (run.result.error.error_class, run.result.error.reason) == (
        ErrorClass.TRANSIENT,
        "reset",
    )


@pytest.mark.parametrize(
    "error_class", [ErrorClass.PERMANENT, ErrorClass.INVALID_ARGS, ErrorClass.BAD_OUTPUT]
)
async def test_non_retryable_classes_run_exactly_once(error_class: ErrorClass) -> None:
    tool = ScriptTool([fail(error_class)], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert tool.calls == 1
    assert time_.sleeps == []
    assert run.result.error is not None
    assert run.result.error.error_class is error_class


async def test_policy_retry_on_is_respected() -> None:
    policy = RetryPolicy(max_attempts=3, retry_on=frozenset({ErrorClass.TRANSIENT}))
    timeout_tool = ScriptTool([fail(ErrorClass.TIMEOUT, "slow")], retry=policy)
    executor, _ = make(timeout_tool)
    await executor.execute("script", {}, CTX)
    assert timeout_tool.calls == 1

    transient_tool = ScriptTool([fail(ErrorClass.TRANSIENT)], retry=policy)
    executor, _ = make(transient_tool)
    await executor.execute("script", {}, CTX)
    assert transient_tool.calls == 3


async def test_invalid_arguments_never_run_the_tool() -> None:
    tool = ScriptTool([ok()], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {"surprise": 1}, CTX)
    assert tool.calls == 0
    assert run.attempts == ()
    assert run.result.error is not None
    assert run.result.error.error_class is ErrorClass.INVALID_ARGS
    assert run.result.error.reason == "schema_violation"
    assert time_.sleeps == []


async def test_unknown_tool_and_bad_json_are_invalid_args() -> None:
    executor, _ = make(ScriptTool([ok()]))
    unknown = await executor.execute("nope", {}, CTX)
    assert unknown.result.error is not None
    assert unknown.result.error.reason == "unknown_tool"
    assert unknown.result.tool == "nope"
    bad = await executor.execute("script", "{not json", CTX)
    assert bad.result.error is not None
    assert bad.result.error.reason == "invalid_json"


async def test_arguments_may_be_a_json_string() -> None:
    executor, _ = make(ScriptTool([ok(3)]))
    run = await executor.execute("script", "{}", CTX)
    assert run.result.output == {"value": 3}


async def test_bad_output_is_not_retried() -> None:
    async def wrong(ctx: ToolContext) -> object:
        return {"value": "not an int"}

    tool = ScriptTool([wrong], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert tool.calls == 1
    assert time_.sleeps == []
    assert run.result.error is not None
    assert run.result.error.error_class is ErrorClass.BAD_OUTPUT


async def test_timeout_cancels_a_hanging_tool() -> None:
    state = {"started": False, "cancelled": False, "finished": False}

    async def hang(ctx: ToolContext) -> object:
        state["started"] = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return Out(value=1)

    tool = ScriptTool([hang], timeout_s=0.02, retry=RetryPolicy.no_retry())
    executor, _ = make(tool)
    loop = asyncio.get_running_loop()
    began = loop.time()
    run = await executor.execute("script", {}, CTX)
    assert loop.time() - began < 5  # nowhere near the tool's 30s sleep
    assert state == {"started": True, "cancelled": True, "finished": False}
    assert run.result.error is not None
    assert (run.result.error.error_class, run.result.error.reason) == (
        ErrorClass.TIMEOUT,
        "tool_timeout",
    )
    assert "0.02s" in run.result.error.message


async def test_timeouts_are_retried_then_succeed() -> None:
    async def hang(ctx: ToolContext) -> object:
        await asyncio.sleep(30)
        return Out(value=0)  # pragma: no cover

    tool = ScriptTool([hang, ok(9)], timeout_s=0.02, retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert run.result.output == {"value": 9}
    assert time_.sleeps == [0.5]
    first = run.attempts[0].error
    assert first is not None
    assert first.error_class is ErrorClass.TIMEOUT


async def test_timeout_error_raised_by_the_tool_is_upstream_not_our_timeout() -> None:
    tool = ScriptTool([raising(TimeoutError("db")), ok()], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert run.result.ok
    assert run.attempts[0].error is not None
    assert run.attempts[0].error.reason == "upstream_timeout"
    assert run.attempts[0].error.error_class is ErrorClass.TIMEOUT
    assert time_.sleeps == [0.5]


async def test_crash_is_permanent_and_does_not_leak_the_exception_text() -> None:
    tool = ScriptTool([raising(RuntimeError("password=hunter2"))], retry=NO_JITTER)
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX)
    assert tool.calls == 1
    assert time_.sleeps == []
    assert run.result.error is not None
    assert (run.result.error.error_class, run.result.error.reason) == (
        ErrorClass.PERMANENT,
        "tool_crashed",
    )
    assert "RuntimeError" in run.result.error.message
    assert "hunter2" not in run.result.error.message


async def test_outer_cancellation_is_not_swallowed() -> None:
    entered = asyncio.Event()

    async def hang(ctx: ToolContext) -> object:
        entered.set()
        await asyncio.sleep(30)
        return Out(value=0)  # pragma: no cover

    tool = ScriptTool([hang], timeout_s=60, retry=NO_JITTER)
    executor, time_ = make(tool)
    task = asyncio.ensure_future(executor.execute("script", {}, CTX))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.calls == 1
    assert time_.sleeps == []


async def test_durations_are_measured_on_the_injected_clock() -> None:
    time_ = FakeTime()

    async def slow(ctx: ToolContext) -> object:
        time_.now += 2.5
        raise ToolFailure(ToolError(error_class=ErrorClass.TRANSIENT, reason="reset", message="x"))

    tool = ScriptTool([slow, ok()], retry=NO_JITTER)
    executor, _ = make(tool, time_)
    run = await executor.execute("script", {}, CTX)
    assert run.attempts[0].duration_s == 2.5
    assert run.attempts[1].duration_s == 0.0


async def test_attempt_start_times_are_read_from_the_injected_clock() -> None:
    time_ = FakeTime()
    start = time_.now

    async def slow(ctx: ToolContext) -> object:
        time_.now += 2.5
        raise ToolFailure(ToolError(error_class=ErrorClass.TRANSIENT, reason="reset", message="x"))

    tool = ScriptTool([slow, ok()], retry=NO_JITTER)
    executor, _ = make(tool, time_)
    run = await executor.execute("script", {}, CTX)
    # attempt 2 starts after attempt 1's 2.5s and the 0.5s backoff
    assert [a.started_at for a in run.attempts] == [start, start + 3.0]


async def test_deadline_already_passed_runs_nothing() -> None:
    tool = ScriptTool([ok()])
    executor, time_ = make(tool)
    run = await executor.execute("script", {}, CTX, deadline=time_.now)
    assert tool.calls == 0
    assert run.attempts == ()
    assert run.result.error is not None
    assert run.result.error.reason == DEADLINE_REASON
    assert run.result.error.error_class is ErrorClass.TIMEOUT


async def test_backoff_that_would_cross_the_deadline_skips_the_retry() -> None:
    tool = ScriptTool([fail(ErrorClass.TRANSIENT, "reset")], retry=NO_JITTER)
    executor, time_ = make(tool)
    # first backoff is 0.5s; only 0.4s remain
    run = await executor.execute("script", {}, CTX, deadline=time_.now + 0.4)
    assert tool.calls == 1
    assert time_.sleeps == []
    assert run.attempts[0].backoff_s == 0.0
    assert run.result.error is not None
    assert run.result.error.reason == "reset"  # the real failure, not a deadline error


async def test_retries_continue_while_the_deadline_has_room_then_stop() -> None:
    tool = ScriptTool([fail(ErrorClass.TRANSIENT, "reset")], retry=NO_JITTER)
    executor, time_ = make(tool)
    # backoffs 0.5, 1.0, 2.0: the third would end at +3.5 > deadline +3.2
    run = await executor.execute("script", {}, CTX, deadline=time_.now + 3.2)
    assert tool.calls == 3
    assert time_.sleeps == [0.5, 1.0]
    assert len(run.attempts) == 3


async def test_timeout_is_shortened_to_the_deadline_and_not_retried() -> None:
    async def hang(ctx: ToolContext) -> object:
        await asyncio.sleep(30)
        return Out(value=0)  # pragma: no cover

    tool = ScriptTool([hang], timeout_s=60, retry=NO_JITTER)
    executor, time_ = make(tool)
    loop = asyncio.get_running_loop()
    began = loop.time()
    run = await executor.execute("script", {}, CTX, deadline=time_.now + 0.02)
    assert loop.time() - began < 5
    assert tool.calls == 1
    assert time_.sleeps == []
    assert run.result.error is not None
    assert run.result.error.reason == DEADLINE_REASON


async def test_real_builtin_tool_through_the_executor() -> None:
    executor = ToolExecutor(default_registry())
    good = await executor.execute("calculator", {"expression": "2 + 3 * 4"}, CTX)
    assert good.result.ok
    assert good.result.output is not None
    assert good.result.output["value"] == 14
    bad = await executor.execute("calculator", {"expression": "__import__('os')"}, CTX)
    assert bad.result.error is not None
    assert bad.result.error.error_class is ErrorClass.INVALID_ARGS
    assert len(bad.attempts) == 1


@given(
    failures=st.integers(min_value=0, max_value=6),
    max_attempts=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**16),
    jitter=st.floats(min_value=0.0, max_value=1.0),
)
def test_property_attempts_and_sleeps_match_the_policy(
    failures: int, max_attempts: int, seed: int, jitter: float
) -> None:
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_s=0.25,
        multiplier=2.0,
        max_backoff_s=2.0,
        jitter=jitter,
    )
    steps = [fail(ErrorClass.TRANSIENT)] * failures + [ok()]
    tool = ScriptTool(steps, retry=policy)
    executor, time_ = make(tool, seed=seed)
    run = asyncio.run(executor.execute("script", {}, CTX))
    succeeded = failures < max_attempts
    expected_calls = failures + 1 if succeeded else max_attempts
    assert tool.calls == expected_calls
    assert run.result.ok is succeeded
    assert len(time_.sleeps) == expected_calls - 1
    for index, delay in enumerate(time_.sleeps):
        cap = min(0.25 * 2**index, 2.0)
        assert cap * (1 - jitter) - 1e-12 <= delay <= cap + 1e-12
