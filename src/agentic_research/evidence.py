"""The per-run evidence store: which spans of which documents a run has seen.

The citation verifier (next M3 slice) counts a cited span as *grounded* only
if it lies inside text this run observed. The store is built from the run's
recorded steps, not filled in by the tools, so it holds exactly the tool
results the loop accepted and handed to the policy. A search that timed out
(its worker thread may still have finished), was refused by a guardrail or
budget, or failed holds no evidence.

* A successful ``search_corpus`` result contributes one span per passage; a
  successful ``read_document`` result contributes its slice. Other tools
  contribute nothing.
* Spans are keyed by ``(doc_id, content_hash)``: text seen in one version of a
  document does not ground a citation of another version.
* `EvidenceStore.covers` asks whether ``[start, end)`` lies inside the union of
  the observed spans of that document version (overlapping or touching spans
  merge, since together they show contiguous text).

Known limit: a span counts as observed when its tool result was recorded,
even if the observation shaper later truncated the text shown to the policy
(``Observation.truncated_chars``). ``read_document`` caps slices below the
shaper's default limit; the verifier slice decides how truncated observations
count.

This is separate from ``RunState.evidence`` (an M2 placeholder the loop does
not fill), which the ``final_answer`` slice will replace or remove.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.agent.state import StepRecord
from agentic_research.tools.base import ToolResult
from agentic_research.tools.read import READ_DOCUMENT, ReadDocumentOutput
from agentic_research.tools.search import SEARCH_CORPUS, SearchCorpusOutput

__all__ = ["EVIDENCE_TOOLS", "EvidenceSpan", "EvidenceStore"]

#: Tools whose successful results are evidence.
EVIDENCE_TOOLS: Final = frozenset({SEARCH_CORPUS, READ_DOCUMENT})


class EvidenceSpan(BaseModel):
    """``[start_char, end_char)`` of one document version, seen at ``step`` via ``tool``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    tool: str = Field(min_length=1)
    doc_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        return self


class EvidenceStore:
    """Observed spans of one run, in the order they were seen."""

    def __init__(self) -> None:
        self._spans: list[EvidenceSpan] = []
        # (doc_id, content_hash) -> sorted, merged, disjoint [start, end) intervals.
        self._merged: dict[tuple[str, str], list[tuple[int, int]]] = {}

    @classmethod
    def from_steps(cls, steps: Iterable[StepRecord]) -> EvidenceStore:
        """The evidence in a run's recorded steps (``RunState.steps``)."""
        store = cls()
        for step in steps:
            if step.result is not None:
                store.add_result(step.index, step.result)
        return store

    @property
    def spans(self) -> tuple[EvidenceSpan, ...]:
        return tuple(self._spans)

    def add_result(self, step: int, result: ToolResult) -> int:
        """Record the spans of one tool result; returns how many were added.

        Failed results and tools outside `EVIDENCE_TOOLS` add nothing. A
        successful evidence result is re-validated against its tool's output
        model; a result that does not fit raises ``ValidationError``
        (the executor validates outputs, so that is a bug, not evidence).
        """
        if not result.ok or result.output is None or result.tool not in EVIDENCE_TOOLS:
            return 0
        if result.tool == SEARCH_CORPUS:
            found = SearchCorpusOutput.model_validate(result.output)
            ranges = [(p.doc_id, p.content_hash, p.start_char, p.end_char) for p in found.passages]
        else:
            read = ReadDocumentOutput.model_validate(result.output)
            ranges = [(read.doc_id, read.content_hash, read.start_char, read.end_char)]
        for doc_id, content_hash, start, end in ranges:
            self._add(
                EvidenceSpan(
                    step=step,
                    tool=result.tool,
                    doc_id=doc_id,
                    content_hash=content_hash,
                    start_char=start,
                    end_char=end,
                )
            )
        return len(ranges)

    def observed(self, doc_id: str, content_hash: str) -> tuple[tuple[int, int], ...]:
        """Merged observed ``[start, end)`` intervals of one document version."""
        return tuple(self._merged.get((doc_id, content_hash), ()))

    def covers(self, doc_id: str, content_hash: str, start: int, end: int) -> bool:
        """Whether ``[start, end)`` (non-empty) lies inside observed text of that version."""
        if start < 0 or end <= start:
            return False
        return any(
            lo <= start and end <= hi for lo, hi in self._merged.get((doc_id, content_hash), ())
        )

    def _add(self, span: EvidenceSpan) -> None:
        self._spans.append(span)
        key = (span.doc_id, span.content_hash)
        intervals = [*self._merged.get(key, []), (span.start_char, span.end_char)]
        intervals.sort()
        merged: list[tuple[int, int]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        self._merged[key] = merged
