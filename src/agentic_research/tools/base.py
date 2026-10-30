"""Tool contracts: the protocol every tool implements, and its result types.

Failures are data, not exceptions: the registry and (from M2) the executor
turn every failure into a `ToolError` with an `ErrorClass`, so the agent loop
can decide what to do without a ``try`` around each call (ADR-0006).
"""

from __future__ import annotations

import random
from enum import StrEnum
from typing import Any, Protocol, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorClass(StrEnum):
    """How a tool failure should be handled."""

    TRANSIENT = "transient"  # may succeed on retry (e.g. connection reset)
    TIMEOUT = "timeout"  # exceeded the tool's timeout; may succeed on retry
    PERMANENT = "permanent"  # will fail again with the same input
    INVALID_ARGS = "invalid_args"  # the call itself is wrong; the model must fix it
    BAD_OUTPUT = "bad_output"  # the tool returned data that violates its output schema


#: Only these classes may ever be retried (ADR-0006).
RETRYABLE: frozenset[ErrorClass] = frozenset({ErrorClass.TRANSIENT, ErrorClass.TIMEOUT})


class ToolArgs(BaseModel):
    """Base class for tool argument models.

    Unknown keys are rejected rather than silently dropped, so a model that
    invents an argument gets an ``invalid_args`` observation it can act on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolOutput(BaseModel):
    """Base class for tool output models (strict, immutable)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolError(BaseModel):
    """A classified tool failure.

    ``reason`` is a short machine-readable code (for example ``unknown_tool``
    or ``schema_violation``); ``message`` is human-readable detail that is
    safe to show the model.
    """

    model_config = ConfigDict(frozen=True)

    error_class: ErrorClass
    reason: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    message: str

    @property
    def retryable(self) -> bool:
        return self.error_class in RETRYABLE


class ToolResult(BaseModel):
    """Outcome of one tool call: exactly one of ``output`` or ``error``."""

    model_config = ConfigDict(frozen=True)

    tool: str
    output: dict[str, Any] | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("ToolResult needs exactly one of output or error")
        return self

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, tool: str, output: ToolOutput) -> ToolResult:
        return cls(tool=tool, output=output.model_dump(mode="json"))

    @classmethod
    def failure(cls, tool: str, error: ToolError) -> ToolResult:
        return cls(tool=tool, error=error)


class RetryPolicy(BaseModel):
    """Attempts, exponential backoff with jitter, and which errors retry.

    ``retry_on`` may only contain retryable classes (``transient``,
    ``timeout``); retrying a permanent or invalid call wastes budget and hides
    bugs (ADR-0006). Jitter is drawn from a caller-supplied RNG so that runs
    are reproducible from a seed.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_s: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    multiplier: float = Field(default=2.0, ge=1.0, allow_inf_nan=False)
    max_backoff_s: float = Field(default=8.0, ge=0.0, allow_inf_nan=False)
    jitter: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    retry_on: frozenset[ErrorClass] = RETRYABLE

    @model_validator(mode="after")
    def _check(self) -> Self:
        not_retryable = self.retry_on - RETRYABLE
        if not_retryable:
            names = ", ".join(sorted(c.value for c in not_retryable))
            raise ValueError(f"retry_on may only contain transient/timeout, got: {names}")
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("max_backoff_s must be >= initial_backoff_s")
        return self

    @classmethod
    def no_retry(cls) -> RetryPolicy:
        return cls(max_attempts=1, retry_on=frozenset())

    def should_retry(self, error: ToolError, attempt: int) -> bool:
        """Whether to try again after ``attempt`` (1-based) failed with ``error``."""
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        return attempt < self.max_attempts and error.error_class in self.retry_on

    def backoff_s(self, attempt: int, rng: random.Random) -> float:
        """Delay before attempt ``attempt + 1``, after ``attempt`` (1-based) failed.

        The base delay is ``initial * multiplier**(attempt - 1)`` capped at
        ``max_backoff_s``; jitter then removes up to ``jitter`` of it, so the
        result lies in ``[base * (1 - jitter), base]``.
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        # Cap the exponent's effect before multiplying to avoid float overflow.
        base = self.initial_backoff_s
        for _ in range(attempt - 1):
            base *= self.multiplier
            if base >= self.max_backoff_s:
                break
        base = min(base, self.max_backoff_s)
        return base * (1.0 - self.jitter * rng.random())


class ToolContext(BaseModel):
    """Per-call context passed to a tool."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    attempt: int = Field(default=1, ge=1)


ArgsT = TypeVar("ArgsT", bound=ToolArgs, contravariant=True)
OutT = TypeVar("OutT", bound=ToolOutput, covariant=True)


class Tool(Protocol[ArgsT, OutT]):
    """A typed, asynchronous tool.

    ``args_model`` is exported as the JSON schema the model sees;
    ``output_model`` validates what the tool returns. Timeouts and retries
    are enforced by the executor (M2), not by the tool.
    """

    # Read-only members, so plain class attributes satisfy the protocol.
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def args_model(self) -> type[ToolArgs]: ...
    @property
    def output_model(self) -> type[ToolOutput]: ...
    @property
    def timeout_s(self) -> float: ...
    @property
    def retry(self) -> RetryPolicy: ...

    async def run(self, args: ArgsT, ctx: ToolContext) -> OutT: ...
