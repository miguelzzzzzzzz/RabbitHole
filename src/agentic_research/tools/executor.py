"""Tool executor: validation, per-call timeout, and seeded backoff retries.

The executor is the only place a tool is actually run. It never raises for a
tool's own failure: every outcome is a :class:`ToolResult` plus a per-attempt
record the loop can put into the trace (ADR-0006).

* Arguments are validated once, before the first attempt; an ``invalid_args``
  call is never run and never retried.
* Each attempt runs under ``asyncio.timeout`` (the tool's ``timeout_s``,
  shortened to the time left before an optional ``deadline``). A tool that
  overruns is cancelled and reported as ``timeout``.
* Only ``transient`` and ``timeout`` failures that the tool's
  :class:`RetryPolicy` allows are retried. Backoff delays come from a seeded
  RNG, so a run is reproducible, and pass through an injectable sleeper; the
  clock is injectable too, so tests need no real waiting for backoff.
* A retry is skipped when its backoff would end at or after the deadline.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_research.tools.base import (
    ErrorClass,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolOutput,
    ToolResult,
)
from agentic_research.tools.registry import ToolRegistry

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]

DEADLINE_REASON = "deadline_exceeded"


class AttemptRecord(BaseModel):
    """One run of a tool. ``backoff_s`` is the delay slept *after* it (0 if none).

    ``started_at`` is the executor clock's reading when the attempt began, so
    the agent loop can place each attempt in the run's trace.
    """

    model_config = ConfigDict(frozen=True)

    attempt: int = Field(ge=1)
    started_at: float = Field(allow_inf_nan=False)
    error: ToolError | None = None
    duration_s: float = Field(ge=0.0)
    backoff_s: float = Field(default=0.0, ge=0.0)

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolExecution(BaseModel):
    """The final result of a call and every attempt that led to it."""

    model_config = ConfigDict(frozen=True)

    result: ToolResult
    attempts: tuple[AttemptRecord, ...] = ()

    @property
    def total_backoff_s(self) -> float:
        return sum(a.backoff_s for a in self.attempts)


def _failed(tool: str, error: ToolError) -> ToolExecution:
    return ToolExecution(result=ToolResult.failure(tool, error))


class ToolExecutor:
    """Runs registered tools with timeouts and retries."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        seed: int = 0,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._registry = registry
        self._rng = random.Random(seed)
        self._clock = clock
        self._sleeper = sleeper

    @property
    def clock(self) -> Clock:
        """The clock deadlines are measured on; the agent loop shares it."""
        return self._clock

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | str | None,
        ctx: ToolContext,
        *,
        deadline: float | None = None,
    ) -> ToolExecution:
        """Run tool ``name``; ``deadline`` is an absolute time on this executor's clock."""
        parsed = self._registry.validate_args(name, arguments)
        if isinstance(parsed, ToolError):
            return _failed(name, parsed)
        tool = self._registry.get(name)
        attempts: list[AttemptRecord] = []
        while True:
            attempt = len(attempts) + 1
            started = self._clock()
            limit = tool.timeout_s
            clamped = False
            if deadline is not None:
                remaining = deadline - started
                if remaining <= 0:
                    # Nothing ran, so this is not an attempt.
                    return ToolExecution(
                        result=ToolResult.failure(name, _deadline_error(name)),
                        attempts=tuple(attempts),
                    )
                if remaining < limit:
                    limit, clamped = remaining, True
            outcome = await self._run_once(
                name, parsed, ctx.model_copy(update={"attempt": attempt}), limit, clamped
            )
            duration = max(0.0, self._clock() - started)
            if not isinstance(outcome, ToolError):
                attempts.append(
                    AttemptRecord(attempt=attempt, started_at=started, duration_s=duration)
                )
                return ToolExecution(
                    result=ToolResult.success(name, outcome), attempts=tuple(attempts)
                )
            policy = tool.retry
            delay = 0.0
            retry = policy.should_retry(outcome, attempt) and outcome.reason != DEADLINE_REASON
            if retry:
                delay = policy.backoff_s(attempt, self._rng)
                if deadline is not None and self._clock() + delay >= deadline:
                    retry = False
                    delay = 0.0
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    started_at=started,
                    error=outcome,
                    duration_s=duration,
                    backoff_s=delay,
                )
            )
            if not retry:
                return ToolExecution(
                    result=ToolResult.failure(name, outcome), attempts=tuple(attempts)
                )
            await self._sleeper(delay)

    async def _run_once(
        self,
        name: str,
        args: ToolArgs,
        ctx: ToolContext,
        limit: float,
        clamped: bool,
    ) -> ToolOutput | ToolError:
        """One attempt: run under the timeout and classify whatever happens."""
        tool = self._registry.get(name)
        try:
            async with asyncio.timeout(limit) as scope:
                raw = await tool.run(args, ctx)
        except TimeoutError:
            if scope.expired():
                if clamped:
                    return _deadline_error(name)
                return ToolError(
                    error_class=ErrorClass.TIMEOUT,
                    reason="tool_timeout",
                    message=f"tool {name!r} did not finish within {tool.timeout_s:g}s",
                )
            return ToolError(
                error_class=ErrorClass.TIMEOUT,
                reason="upstream_timeout",
                message=f"tool {name!r} reported a timeout from a service it called",
            )
        except ToolFailure as failure:
            return failure.error
        except Exception as exc:  # a crashing tool must not crash the agent loop
            return ToolError(
                error_class=ErrorClass.PERMANENT,
                reason="tool_crashed",
                message=f"tool {name!r} raised {type(exc).__name__}",
            )
        return self._registry.validate_output(name, raw)


def _deadline_error(name: str) -> ToolError:
    return ToolError(
        error_class=ErrorClass.TIMEOUT,
        reason=DEADLINE_REASON,
        message=f"no time left before the run deadline for tool {name!r}",
    )
