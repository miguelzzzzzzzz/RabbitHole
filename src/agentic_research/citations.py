"""`CitationVerifier`: checks a `FinalAnswer`'s citations against the corpus (PLAN M3).

Plugged into the loop's verifier hook, it accepts an answer only if it has at
least one claim, every claim has at least one citation, and every citation
passes all of these checks, in this order (the first failure is the
citation's issue):

1. **Known source.** ``doc_id`` names a document of the backend's corpus;
   otherwise ``invalid_citation``.
2. **Freshness.** A ``content_hash`` on the citation equals the stored
   document's; otherwise ``stale_source`` (the source changed, or the quote
   came from another document). A citation without ``content_hash`` is
   checked against the stored version.
3. **Validity.** ``quote`` equals the stored *normalized* text at
   ``[start, end)`` exactly (offsets inside the document); otherwise
   ``invalid_citation``. Off-by-one spans fail here.
4. **Groundedness.** ``[start, end)`` lies inside text the policy was shown
   in full during this run, for that document version: the run's
   `EvidenceStore` (`agentic_research.agent.evidence`, ADR-0004). A passage
   omitted from a cut search observation is located by the omission marker
   but was never shown, so citing it is ``ungrounded_citation`` until the
   policy reads it with ``read_document``. A span shown in full stays
   grounded after its observation is elided.

An answer with no claims, or a claim with no citations, is ``no_citation``.

A rejection's ``reason`` is the first issue's code (claims and citations in
order) and its ``message`` lists the issues by position
(``claims[0].citations[1]``) with offsets only, never document text, so the
feedback the policy sees is not itself evidence. The loop charges one repair
per rejection; past ``max_repairs`` the run ends ``budget_exhausted:repairs``
(graded as not abstaining, ADR-0004). Whether the answer is *correct* is the
graders' job (M4), not the verifier's: there is no entailment check, so a
valid, grounded citation of an irrelevant passage passes, and "verified" does
not mean "correct" (ADR-0004 amendment, 2026-11-22).

Document lookups run in a worker thread so the loop's step timeout and run
deadline still bound a slow backend. A backend failure other than an unknown
``doc_id`` propagates and ends the run as ``failed:verifier_error``.
"""

from __future__ import annotations

import asyncio
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from agentic_research.agent.actions import Citation, FinalAnswer
from agentic_research.agent.evidence import EvidenceStore
from agentic_research.agent.state import RunState, Verdict
from agentic_research.backends.base import SearchBackend, StoredDocument, UnknownDocumentError

__all__ = [
    "INVALID_CITATION",
    "NO_CITATION",
    "STALE_SOURCE",
    "UNGROUNDED_CITATION",
    "CitationIssue",
    "CitationVerifier",
    "check_citations",
    "feedback",
]

NO_CITATION: Final = "no_citation"
INVALID_CITATION: Final = "invalid_citation"
STALE_SOURCE: Final = "stale_source"
UNGROUNDED_CITATION: Final = "ungrounded_citation"

#: Observed ranges listed in an ``ungrounded_citation`` message.
_MAX_RANGES: Final = 5


class CitationIssue(BaseModel):
    """Why one claim or citation fails; ``citation`` is None for a claim-level issue
    and ``claim`` is None when the answer has no claims at all."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    claim: int | None = Field(default=None, ge=0)
    citation: int | None = Field(default=None, ge=0)
    message: str = Field(min_length=1)

    @property
    def where(self) -> str:
        if self.claim is None:
            return "answer"
        if self.citation is None:
            return f"claims[{self.claim}]"
        return f"claims[{self.claim}].citations[{self.citation}]"


def check_citations(
    answer: FinalAnswer, backend: SearchBackend, evidence: EvidenceStore
) -> tuple[CitationIssue, ...]:
    """Every issue with ``answer``'s claims and citations, in order (empty: valid)."""
    if not answer.claims:
        return (CitationIssue(code=NO_CITATION, message="the answer has no claims with citations"),)
    issues: list[CitationIssue] = []
    documents: dict[str, StoredDocument | None] = {}
    for c, claim in enumerate(answer.claims):
        if not claim.citations:
            issues.append(
                CitationIssue(code=NO_CITATION, claim=c, message="the claim has no citations")
            )
            continue
        for i, citation in enumerate(claim.citations):
            if citation.doc_id not in documents:
                try:
                    documents[citation.doc_id] = backend.document(citation.doc_id)
                except UnknownDocumentError:
                    documents[citation.doc_id] = None
            problem = _check(citation, documents[citation.doc_id], evidence)
            if problem is not None:
                code, message = problem
                issues.append(CitationIssue(code=code, claim=c, citation=i, message=message))
    return tuple(issues)


def _check(
    citation: Citation, document: StoredDocument | None, evidence: EvidenceStore
) -> tuple[str, str] | None:
    """``(code, message)`` for the first check ``citation`` fails, or None."""
    span = f"[{citation.start}, {citation.end})"
    if document is None:
        return INVALID_CITATION, f"unknown doc_id {citation.doc_id!r}"
    if citation.content_hash is not None and citation.content_hash != document.content_hash:
        return (
            STALE_SOURCE,
            f"content_hash {citation.content_hash[:12]}... is not the stored version "
            f"{document.content_hash[:12]}... of {document.doc_id}",
        )
    length = len(document.text)
    if citation.end > length:
        return (
            INVALID_CITATION,
            f"{span} ends past the end of {document.doc_id} ({length} chars)",
        )
    stored = document.text[citation.start : citation.end]
    if stored != citation.quote:
        return (
            INVALID_CITATION,
            f"quote ({len(citation.quote)} chars) is not the stored text at {span} of "
            f"{document.doc_id} ({len(stored)} chars; first difference at quote offset "
            f"{_first_difference(citation.quote, stored)})",
        )
    if not evidence.covers(document.doc_id, document.content_hash, citation.start, citation.end):
        seen = evidence.observed(document.doc_id, document.content_hash)
        if seen:
            listed = ", ".join(f"[{lo}, {hi})" for lo, hi in seen[:_MAX_RANGES])
            if len(seen) > _MAX_RANGES:
                listed += f", +{len(seen) - _MAX_RANGES} more"
            shown = f"shown in full: {listed}"
        else:
            shown = "no text of this document version was shown in full"
        return (
            UNGROUNDED_CITATION,
            f"{span} of {document.doc_id} is not inside text shown in full in this run ({shown})",
        )
    return None


def _first_difference(a: str, b: str) -> int:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return min(len(a), len(b))


def feedback(issues: tuple[CitationIssue, ...], *, max_listed: int = 10) -> str:
    """The repair message for ``issues`` (non-empty): one line per issue, by position."""
    if not issues:
        raise ValueError("feedback needs at least one issue")
    noun = "problem" if len(issues) == 1 else "problems"
    lines = [
        f"{len(issues)} citation {noun}. Every claim needs a citation whose quote is the exact "
        "stored text at [start, end) of text shown in full in this run."
    ]
    lines += [f"{i.where}: {i.code}: {i.message}" for i in issues[:max_listed]]
    if len(issues) > max_listed:
        lines.append(f"+{len(issues) - max_listed} more")
    return "\n".join(lines)


class CitationVerifier:
    """The `Verifier` that enforces valid, fresh and grounded citations."""

    def __init__(self, backend: SearchBackend) -> None:
        self._backend = backend

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        evidence = EvidenceStore.from_state(state)
        issues = await asyncio.to_thread(check_citations, answer, self._backend, evidence)
        if not issues:
            return Verdict.accept()
        return Verdict.reject(issues[0].code, feedback(issues))
