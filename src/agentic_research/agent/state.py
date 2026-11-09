"""Serializable run state, budgets and terminal statuses.

The loop (next commit) owns all transitions; this module only defines the
data and the invariants that keep it consistent:

- ``status`` is ``running`` until `RunState.finish` is called once; a
  terminal state never changes again.
- ``budget_exhausted`` always names the `BudgetKind` that ran out, and
  ``failed`` always carries a reason code, so `RunState.terminal_label`
  gives the ``budget_exhausted:<kind>`` / ``failed:<reason>`` strings from the
  plan.
- A state round-trips through JSON (`to_json` / `from_json`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.agent.actions import Action
from agentic_research.providers.types import ChatMessage
from agentic_research.tools.base import ToolResult

_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"
Count = Annotated[int, Field(ge=0)]
Seconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class RunStatus(StrEnum):
    """Where a run is. Everything but ``running`` is terminal."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    ABSTAINED = "abstained"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self is not RunStatus.RUNNING


class BudgetKind(StrEnum):
    """Which budget ran out."""

    STEPS = "steps"
    TOOL_CALLS = "tool_calls"
    TOOL_CALLS_PER_TOOL = "tool_calls_per_tool"
    DEADLINE = "deadline"
    TOKENS = "tokens"
    REPAIRS = "repairs"


class Budgets(BaseModel):
    """Limits for one run; ``None`` means unlimited.

    A budget is *exhausted* once its counter has reached the limit: with
    ``max_steps=3`` the fourth step is not allowed. Two exceptions: ``max_repairs``
    is the number of repairs *allowed*, so it is exhausted once the counter
    goes past it; and ``max_tokens`` is soft, because usage is only known after
    a model call returns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: Annotated[int, Field(ge=1)] | None = None
    max_tool_calls: Annotated[int, Field(ge=1)] | None = None
    max_tool_calls_per_tool: Annotated[int, Field(ge=1)] | None = None
    deadline_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] | None = None
    max_tokens: Annotated[int, Field(ge=1)] | None = None
    max_repairs: Annotated[int, Field(ge=0)] | None = None

    def exhausted(self, counters: BudgetCounters, tool: str | None = None) -> BudgetKind | None:
        """The first exhausted budget, or ``None``.

        ``tool`` is the tool about to be called, if any; it enables the
        per-tool check. The order is fixed so a run that hits two limits at
        once gets a deterministic status.
        """
        if self.max_steps is not None and counters.steps >= self.max_steps:
            return BudgetKind.STEPS
        if self.deadline_s is not None and counters.elapsed_s >= self.deadline_s:
            return BudgetKind.DEADLINE
        if self.max_tokens is not None and counters.tokens >= self.max_tokens:
            return BudgetKind.TOKENS
        if self.max_repairs is not None and counters.repairs > self.max_repairs:
            return BudgetKind.REPAIRS
        if self.max_tool_calls is not None and counters.tool_calls >= self.max_tool_calls:
            return BudgetKind.TOOL_CALLS
        if (
            tool is not None
            and self.max_tool_calls_per_tool is not None
            and counters.per_tool.get(tool, 0) >= self.max_tool_calls_per_tool
        ):
            return BudgetKind.TOOL_CALLS_PER_TOOL
        return None


class BudgetCounters(BaseModel):
    """What a run has consumed so far."""

    model_config = ConfigDict(extra="forbid")

    steps: Count = 0
    tool_calls: Count = 0
    per_tool: dict[str, Count] = Field(default_factory=dict)
    tokens: Count = 0
    repairs: Count = 0
    elapsed_s: Seconds = 0.0

    @model_validator(mode="after")
    def _per_tool_adds_up(self) -> Self:
        if sum(self.per_tool.values()) != self.tool_calls:
            raise ValueError("per_tool counts must add up to tool_calls")
        return self

    def count_tool_call(self, tool: str) -> None:
        self.tool_calls += 1
        self.per_tool[tool] = self.per_tool.get(tool, 0) + 1


class Task(BaseModel):
    """The question a run is trying to answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    task_id: str | None = None


class Evidence(BaseModel):
    """One piece of evidence gathered by a tool, addressable by ``id``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str


class StepRecord(BaseModel):
    """One loop step: the chosen action and, for a tool call, its result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: Count
    action: Action
    result: ToolResult | None = None

    @model_validator(mode="after")
    def _result_only_for_tool_calls(self) -> Self:
        if self.result is not None and self.action.kind != "tool_call":
            raise ValueError("only a tool_call step can carry a result")
        return self


class RunState(BaseModel):
    """Everything the loop and a policy need to know about a run."""

    model_config = ConfigDict(extra="forbid")

    task: Task
    plan: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    counters: BudgetCounters = Field(default_factory=BudgetCounters)
    status: RunStatus = RunStatus.RUNNING
    detail: Annotated[str, Field(pattern=_CODE_PATTERN, max_length=64)] | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        status, detail = self.status, self.detail
        if status is RunStatus.BUDGET_EXHAUSTED:
            if detail not in {k.value for k in BudgetKind}:
                raise ValueError("budget_exhausted needs a BudgetKind detail")
        elif status is RunStatus.FAILED:
            if detail is None:
                raise ValueError("failed needs a reason code")
        elif status is RunStatus.ABSTAINED:
            pass  # an optional reason code
        elif detail is not None:
            raise ValueError(f"status {status.value} cannot carry a detail")
        if (self.answer is not None) != (status is RunStatus.SUCCEEDED):
            raise ValueError("answer is set exactly when the run succeeded")
        if [s.index for s in self.steps] != list(range(len(self.steps))):
            raise ValueError("step indexes must be contiguous from 0")
        return self

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    @property
    def terminal_label(self) -> str:
        """``succeeded``, ``abstained[:why]``, ``budget_exhausted:<kind>`` or ``failed:<why>``."""
        if self.detail is None:
            return self.status.value
        return f"{self.status.value}:{self.detail}"

    def add_step(self, action: Action, result: ToolResult | None = None) -> StepRecord:
        """Append a step record with the next index. Not allowed once terminal."""
        self._require_running()
        record = StepRecord(index=len(self.steps), action=action, result=result)
        self.steps.append(record)
        return record

    def finish(
        self,
        status: RunStatus,
        *,
        detail: str | BudgetKind | None = None,
        answer: str | None = None,
    ) -> None:
        """Move to a terminal status exactly once; invalid combinations raise."""
        self._require_running()
        if not status.terminal:
            raise ValueError("finish needs a terminal status")
        # Build and validate the terminal state first so a bad combination
        # raises before anything on ``self`` changes.
        candidate = RunState.model_validate(
            self.model_dump()
            | {
                "status": status,
                "detail": detail.value if isinstance(detail, BudgetKind) else detail,
                "answer": answer,
            }
        )
        self.status, self.detail, self.answer = candidate.status, candidate.detail, candidate.answer

    def _require_running(self) -> None:
        if self.terminal:
            raise ValueError(f"run already finished ({self.terminal_label})")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | bytes) -> RunState:
        return cls.model_validate_json(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
