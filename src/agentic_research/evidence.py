"""The per-run evidence store: which spans of which documents a run has seen.

The citation verifier (next M3 slice) counts a cited span as *grounded* only
if it lies inside text this run observed. The store is built from the run's
recorded steps, not filled in by the tools, so it holds exactly the tool
results the loop accepted and handed to the policy. A search that timed out
(its worker thread may still have finished), was refused by a guardrail or
budget, or failed holds no evidence.

* A successful ``search_corpus`` result contributes one span per passage
  the policy was shown; a successful ``read_document`` result contributes
  its slice if it was shown untruncated (below). Other tools contribute
  nothing.
* Spans are keyed by ``(doc_id, content_hash)``: text seen in one version of a
  document does not ground a citation of another version.
* `EvidenceStore.covers` asks whether ``[start, end)`` lies inside the union of
  the observed spans of that document version (overlapping or touching spans
  merge, since together they show contiguous text).

Only text the policy was shown in full grounds anything (ADR-0004,
"Truncated observations and groundedness"). The store therefore reads each
step's `Observation` next to its result:

* ``search_corpus``: the shaper cuts at passage boundaries and records
  ``shown_passages``; only those leading passages become spans. The dropped
  ones are counted in `EvidenceStore.dropped`.
* ``read_document``: its slice counts only if the observation was not
  truncated (``truncated_chars == 0``). The 1500-character cap keeps plain
  prose with an ordinary ``source`` path below the shaper's default 2000
  (tested), but JSON escaping (quotes,
  newlines, control characters) can still push a slice over; such a slice
  adds nothing. There is no partial-span credit.
* A step with a result but no observation adds nothing.
* Later elision does not remove evidence: an elided observation keeps
  ``truncated_chars`` and ``shown_passages``, so text shown in full once
  stays observed.

This is separate from ``RunState.evidence`` (an M2 placeholder the loop does
not fill), which the ``final_answer`` slice will replace or remove.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.agent.state import Observation, RunState, StepRecord
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
        self._dropped: dict[int, int] = {}
        # (doc_id, content_hash) -> sorted, merged, disjoint [start, end) intervals.
        self._merged: dict[tuple[str, str], list[tuple[int, int]]] = {}

    @classmethod
    def from_steps(
        cls, steps: Iterable[StepRecord], observations: Iterable[Observation]
    ) -> EvidenceStore:
        """The evidence in a run's steps and observations (``RunState.steps`` and
        ``RunState.observations``)."""
        by_step = {o.step: o for o in observations}
        store = cls()
        for step in steps:
            store.add_step(step, by_step.get(step.index))
        return store

    @classmethod
    def from_state(cls, state: RunState) -> EvidenceStore:
        """`from_steps` over one run's state."""
        return cls.from_steps(state.steps, state.observations)

    @property
    def spans(self) -> tuple[EvidenceSpan, ...]:
        return tuple(self._spans)

    @property
    def dropped(self) -> Mapping[int, int]:
        """Step index -> spans its evidence result returned but the policy was
        not shown in full (only steps that dropped at least one)."""
        return MappingProxyType(self._dropped)

    def add_step(self, step: StepRecord, observation: Observation | None) -> int:
        """Record the fully shown spans of one step; returns how many were added.

        Steps without a successful result from a tool in `EVIDENCE_TOOLS` add
        nothing. ``observation`` must be the one the shaper made for this step
        (``ValueError`` otherwise). A successful evidence result is
        re-validated against its tool's output model; a result that does not
        fit raises ``ValidationError`` (the executor validates outputs, so
        that is a bug, not evidence).
        """
        result = step.result
        if result is None or not result.ok or result.output is None:
            return 0
        if result.tool not in EVIDENCE_TOOLS:
            return 0
        if observation is not None and (
            observation.step != step.index or observation.source != "tool"
        ):
            raise ValueError(
                f"observation of step {observation.step} does not match step {step.index}"
            )
        if result.tool == SEARCH_CORPUS:
            found = SearchCorpusOutput.model_validate(result.output)
            ranges = [(p.doc_id, p.content_hash, p.start_char, p.end_char) for p in found.passages]
            shown = _shown_passages(observation, len(ranges))
        else:
            read = ReadDocumentOutput.model_validate(result.output)
            ranges = [(read.doc_id, read.content_hash, read.start_char, read.end_char)]
            shown = 1 if observation is not None and observation.truncated_chars == 0 else 0
        for doc_id, content_hash, start, end in ranges[:shown]:
            self._add(
                EvidenceSpan(
                    step=step.index,
                    tool=result.tool,
                    doc_id=doc_id,
                    content_hash=content_hash,
                    start_char=start,
                    end_char=end,
                )
            )
        if shown < len(ranges):
            self._dropped[step.index] = self._dropped.get(step.index, 0) + len(ranges) - shown
        return shown

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


def _shown_passages(observation: Observation | None, total: int) -> int:
    """How many leading passages of a ``total``-passage search the policy saw in full."""
    if observation is None:
        return 0
    if observation.shown_passages is not None:
        return min(observation.shown_passages, total)
    # Not shaped at passage boundaries: all or nothing.
    return total if observation.truncated_chars == 0 else 0
