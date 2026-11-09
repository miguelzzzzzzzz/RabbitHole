"""The actions a policy can take: call a tool, answer, or give up.

`Action` is a tagged union on ``kind`` so a stored step round-trips through
JSON into the right class. This `ToolCall` is the agent's *intent* (tool name
plus arguments); it is distinct from the wire-level
`agentic_research.providers.types.ToolCall` a provider returns.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(_Action):
    """Run ``tool`` with ``args``. Validation of ``args`` is the registry's job."""

    kind: Literal["tool_call"] = "tool_call"
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(_Action):
    """Answer the task. ``evidence_ids`` name entries in the run's evidence store."""

    kind: Literal["final_answer"] = "final_answer"
    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class GiveUp(_Action):
    """Abstain: the policy decides it cannot answer. ``reason`` is a short code."""

    kind: Literal["give_up"] = "give_up"
    reason: str = Field(pattern=_CODE_PATTERN, max_length=64)
    message: str = ""


Action = Annotated[ToolCall | FinalAnswer | GiveUp, Field(discriminator="kind")]
