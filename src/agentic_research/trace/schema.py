"""Trace schema v1: the records in one run's JSONL trace (ADR-0008).

A trace file is one JSON object per line. The first line is a
`TraceHeader` carrying ``schema_version``; every following line is a span:

- `RunSpan` (exactly one, the root): the question, provider/model, final
  status, budget usage and limits, and the run's ``parse_failures`` count.
- `StepSpan`: one agent-loop step, child of the run.
- `ProviderCallSpan`: one model call, child of a step. A reply that could
  not be parsed into an action carries a `ParseFailure` with a reason code.
- `ToolCallSpan`: one tool attempt, child of a step: arguments plus exactly
  one of ``output`` or ``error``.

Spans are written when they end, so children usually precede their parent
and the run span is the last line. Timings (``start_s``/``end_s``) are
seconds since the run started, read from a monotonic clock; the only
wall-clock value is ``TraceHeader.started_at``.

`Trace` checks the cross-span invariants (one root, unique ids, parents
exist and have the right kind, contiguous step indexes, and the run's
``parse_failures`` equals the number of provider calls with a parse
failure). Any change to these models that an older reader would reject is a
new schema version.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from agentic_research.tools.base import ErrorClass, ToolError

#: The trace schema version this code writes and reads.
SCHEMA_VERSION: Final = 1

#: Upper bound on free-text messages and details stored in a span.
MAX_MESSAGE_CHARS = 2000

_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"
_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"

SpanId = Annotated[str, Field(pattern=_ID_PATTERN)]
ReasonCode = Annotated[str, Field(min_length=1, max_length=64, pattern=_CODE_PATTERN)]
Seconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]
JsonObject = dict[str, JsonValue]
ToolFormat = Literal["native", "json"]


def _require_finite_json(value: JsonObject) -> JsonObject:
    """Reject values JSON cannot round-trip (NaN/inf would be written as null)."""
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as exc:
        raise ValueError("must be JSON-serializable without NaN or infinity") from exc
    return value


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TraceHeader(_Record):
    """First line of every trace file."""

    kind: Literal["trace"] = "trace"
    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: SpanId
    started_at: AwareDatetime
    generator: str = Field(min_length=1, max_length=200)


class SpanError(_Record):
    """A classified failure recorded on a span (same classes as tool errors)."""

    error_class: ErrorClass
    reason: ReasonCode
    message: str = Field(max_length=MAX_MESSAGE_CHARS)

    @classmethod
    def from_tool_error(cls, error: ToolError) -> SpanError:
        """Copy a tool error, truncating its message to the span limit."""
        message = error.message[:MAX_MESSAGE_CHARS]
        return cls(error_class=error.error_class, reason=error.reason, message=message)


class SpanEvent(_Record):
    """A point-in-time event on a span (for example a guardrail flag)."""

    name: ReasonCode
    at_s: Seconds
    attributes: JsonObject = Field(default_factory=dict)

    @field_validator("attributes")
    @classmethod
    def _finite_attributes(cls, value: JsonObject) -> JsonObject:
        return _require_finite_json(value)


class ParseFailure(_Record):
    """Why a model reply could not be turned into an action."""

    reason: ReasonCode
    detail: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class _Span(_Record):
    span_id: SpanId
    parent_id: SpanId | None
    start_s: Seconds
    end_s: Seconds
    events: list[SpanEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_timing(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError(f"span {self.span_id}: end_s < start_s")
        for event in self.events:
            if not self.start_s <= event.at_s <= self.end_s:
                raise ValueError(f"span {self.span_id}: event {event.name} outside the span")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class RunStatus(StrEnum):
    """How a run ended."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


class BudgetUsage(_Record):
    """What a run consumed."""

    steps: Count = 0
    tool_calls: Count = 0
    tokens: Count = 0
    time_s: Seconds = 0.0


class BudgetLimits(_Record):
    """Behaviour limits of the agent loop; ``None`` means unlimited."""

    steps: Annotated[int, Field(ge=1)] | None = None
    tool_calls: Count | None = None
    tokens: Annotated[int, Field(ge=1)] | None = None
    time_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None = None


class RunSpan(_Span):
    """The root span: one per trace."""

    kind: Literal["run"] = "run"
    parent_id: None = None
    question: str = Field(min_length=1)
    task_id: str | None = None
    provider: str = Field(min_length=1)
    model: str | None = None
    tool_format: ToolFormat | None = None
    status: RunStatus
    answer: str | None = None
    error: SpanError | None = None
    budget_used: BudgetUsage = Field(default_factory=BudgetUsage)
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    parse_failures: Count = 0

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status is RunStatus.FAILED and self.error is None:
            raise ValueError("a failed run needs an error")
        if self.status is not RunStatus.FAILED and self.error is not None:
            raise ValueError(f"a run with status {self.status.value} cannot carry an error")
        if self.status is RunStatus.ANSWERED and self.answer is None:
            raise ValueError("an answered run needs an answer")
        return self


class StepSpan(_Span):
    """One iteration of the agent loop."""

    kind: Literal["step"] = "step"
    parent_id: SpanId
    index: Count


class ProviderCallSpan(_Span):
    """One chat-completion call (one attempt)."""

    kind: Literal["provider_call"] = "provider_call"
    parent_id: SpanId
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tool_format: ToolFormat
    attempt: Annotated[int, Field(ge=1)] = 1
    input_tokens: Count | None = None
    output_tokens: Count | None = None
    finish_reason: str | None = None
    parse_failure: ParseFailure | None = None
    error: SpanError | None = None

    @model_validator(mode="after")
    def _check_outcome(self) -> Self:
        if self.error is not None and self.parse_failure is not None:
            raise ValueError("a failed provider call has no reply to fail parsing")
        return self


class ToolCallSpan(_Span):
    """One tool attempt: exactly one of ``output`` or ``error``."""

    kind: Literal["tool_call"] = "tool_call"
    parent_id: SpanId
    tool: str = Field(min_length=1)
    attempt: Annotated[int, Field(ge=1)] = 1
    arguments: JsonObject
    output: JsonObject | None = None
    error: SpanError | None = None

    @field_validator("arguments")
    @classmethod
    def _finite_arguments(cls, value: JsonObject) -> JsonObject:
        return _require_finite_json(value)

    @field_validator("output")
    @classmethod
    def _finite_output(cls, value: JsonObject | None) -> JsonObject | None:
        return None if value is None else _require_finite_json(value)

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("a tool call span needs exactly one of output or error")
        return self


Span = Annotated[RunSpan | StepSpan | ProviderCallSpan | ToolCallSpan, Field(discriminator="kind")]

#: Which span kind each kind's parent must be.
_PARENT_KIND: dict[str, str] = {
    "step": "run",
    "provider_call": "step",
    "tool_call": "step",
}


class Trace(_Record):
    """A complete, internally consistent run trace."""

    header: TraceHeader
    spans: list[Span]

    @model_validator(mode="after")
    def _check_tree(self) -> Self:
        by_id: dict[str, Any] = {}
        for span in self.spans:
            if span.span_id in by_id:
                raise ValueError(f"duplicate span_id {span.span_id!r}")
            by_id[span.span_id] = span

        runs = [s for s in self.spans if isinstance(s, RunSpan)]
        if len(runs) != 1:
            raise ValueError(f"a trace needs exactly one run span, found {len(runs)}")

        for span in self.spans:
            if isinstance(span, RunSpan):
                continue
            parent = by_id.get(span.parent_id)
            if parent is None:
                raise ValueError(f"span {span.span_id!r}: parent {span.parent_id!r} not found")
            expected = _PARENT_KIND[span.kind]
            if parent.kind != expected:
                raise ValueError(
                    f"span {span.span_id!r} ({span.kind}): parent must be a {expected} span, "
                    f"got {parent.kind}"
                )

        indexes = sorted(s.index for s in self.spans if isinstance(s, StepSpan))
        if indexes != list(range(len(indexes))):
            raise ValueError(f"step indexes must be 0..n-1 without gaps, got {indexes}")

        failures = sum(
            1 for s in self.spans if isinstance(s, ProviderCallSpan) and s.parse_failure is not None
        )
        if runs[0].parse_failures != failures:
            raise ValueError(
                f"run.parse_failures is {runs[0].parse_failures} but {failures} provider "
                "call(s) have a parse_failure"
            )
        return self

    @property
    def run(self) -> RunSpan:
        return next(s for s in self.spans if isinstance(s, RunSpan))

    def children(self, span_id: str) -> list[RunSpan | StepSpan | ProviderCallSpan | ToolCallSpan]:
        """Direct children of ``span_id``, in file order."""
        return [s for s in self.spans if s.parent_id == span_id]
