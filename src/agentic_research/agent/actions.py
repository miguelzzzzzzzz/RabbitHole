"""The actions a policy can take: call a tool, answer, or give up.

`Action` is a tagged union on ``kind`` so a stored step round-trips through
JSON into the right class. This `ToolCall` is the agent's *intent* (tool name
plus arguments); it is distinct from the wire-level
`agentic_research.providers.types.ToolCall` a provider returns.

A `FinalAnswer` carries structured ``claims``, each backed by `Citation`s
into the normalized stored text of a corpus document (PLAN M3). The models
only check shape; whether a citation is valid, fresh and grounded is the
citation verifier's job (`agentic_research.citations`).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.tools.base import Offset

_CODE_PATTERN = r"^[a-z][a-z0-9_]*$"


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(_Action):
    """Run ``tool`` with ``args``. Validation of ``args`` is the registry's job."""

    kind: Literal["tool_call"] = "tool_call"
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """``quote`` is claimed to be ``text[start:end]`` of document ``doc_id``.

    Offsets index the document's *normalized* stored text (CitationNeeded
    index format 2), the same offsets ``search_corpus`` and ``read_document``
    return. ``content_hash``, when given, names the document version the
    quote was taken from. Offsets refuse booleans and floats (``True``,
    ``0.0``) and non-integer strings (``"0.0"``), which pydantic's lax mode
    would read as integers; a string of digits such as ``"0"`` is still read
    as an integer (`agentic_research.tools.base.Offset`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(
        min_length=1, max_length=64, description="Id of the cited document, as the tools gave it."
    )
    start: Offset = Field(ge=0, description="Start offset (inclusive) in the document text.")
    end: Offset = Field(ge=1, description="End offset (exclusive); must be greater than start.")
    quote: str = Field(min_length=1, description="The document text at [start, end), exactly.")
    content_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="The document's content_hash from the tool result, if given.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class Claim(BaseModel):
    """One statement of the answer and the citations that support it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, description="One statement of the answer.")
    citations: tuple[Citation, ...] = Field(
        default=(), description="Passages that support the statement."
    )


class FinalAnswer(_Action):
    """Answer the task; ``claims`` back the answer with citations.

    An answer without claims (or a claim without citations) is well formed
    here; the citation verifier rejects it as ``no_citation``.
    """

    kind: Literal["final_answer"] = "final_answer"
    answer: str = Field(min_length=1)
    claims: tuple[Claim, ...] = ()


class GiveUp(_Action):
    """Abstain: the policy decides it cannot answer. ``reason`` is a short code."""

    kind: Literal["give_up"] = "give_up"
    reason: str = Field(pattern=_CODE_PATTERN, max_length=64)
    message: str = ""


Action = Annotated[ToolCall | FinalAnswer | GiveUp, Field(discriminator="kind")]
