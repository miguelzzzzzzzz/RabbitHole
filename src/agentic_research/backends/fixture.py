"""`FixtureSearchBackend`: a tiny keyword backend over in-memory documents.

Unit tests use it so the default suite needs neither CitationNeeded's index
build nor any model. It honours the same contract as `RagEngineBackend`
(format-2 doc ids, ``content_hash``, offsets into the stored text), so a test
written against it holds for the real backend too (see the shared contract
tests).

Passages are paragraphs: maximal runs of non-blank lines, trimmed of
surrounding whitespace. A passage's score is the number of distinct query terms
(lowercased ``\\w+`` tokens) it contains; passages without a query term are not
returned. Ties break on ``source`` then ``start_char``, so results are
deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from agentic_research.backends.base import (
    Passage,
    StoredDocument,
    UnknownDocumentError,
    make_doc_id,
    sha256_hex,
    validate_search_args,
)

__all__ = ["FixtureSearchBackend"]

# One or more lines that each hold a non-space character (unambiguous, so linear).
_PARAGRAPH_RE: Final = re.compile(r"(?:[^\S\n]*\S[^\n]*(?:\n|$))+")
_TOKEN_RE: Final = re.compile(r"\w+")


def _terms(text: str) -> frozenset[str]:
    return frozenset(match.group().lower() for match in _TOKEN_RE.finditer(text))


class FixtureSearchBackend:
    """Keyword search over ``{relative POSIX path: text}`` documents."""

    def __init__(self, corpus_id: str, documents: Mapping[str, str]) -> None:
        self._corpus_id = corpus_id
        self._documents: dict[str, StoredDocument] = {}
        # (source, start, end, doc_id, chunk_id, terms) per paragraph.
        self._paragraphs: list[tuple[str, int, int, str, str, frozenset[str]]] = []
        for source in sorted(documents):
            text = documents[source]
            doc_id = make_doc_id(corpus_id, source)
            self._documents[doc_id] = StoredDocument(
                doc_id=doc_id, source=source, content_hash=sha256_hex(text), text=text
            )
            for index, match in enumerate(_PARAGRAPH_RE.finditer(text)):
                block = match.group()
                start = match.start() + (len(block) - len(block.lstrip()))
                end = match.start() + len(block.rstrip())
                self._paragraphs.append(
                    (source, start, end, doc_id, f"{doc_id}:{index}", _terms(text[start:end]))
                )

    @classmethod
    def from_directory(cls, root: str | Path, corpus_id: str) -> FixtureSearchBackend:
        """Load every regular file under ``root`` (UTF-8) as one document."""
        base = Path(root)
        documents = {
            path.relative_to(base).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(base.rglob("*"))
            if path.is_file()
        }
        return cls(corpus_id, documents)

    @property
    def corpus_id(self) -> str:
        return self._corpus_id

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        validate_search_args(query, k)
        wanted = _terms(query)
        scored: list[tuple[float, str, int, int, str, str]] = []
        for source, start, end, doc_id, chunk_id, terms in self._paragraphs:
            if filters and not _matches({"doc_id": doc_id, "source": source}, filters):
                continue
            score = float(len(wanted & terms))
            if score > 0:
                scored.append((score, source, start, end, doc_id, chunk_id))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        passages: list[Passage] = []
        for rank, (score, source, start, end, doc_id, chunk_id) in enumerate(scored[:k], 1):
            document = self._documents[doc_id]
            passages.append(
                Passage(
                    rank=rank,
                    score=score,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    content_hash=document.content_hash,
                    start_char=start,
                    end_char=end,
                    text=document.text[start:end],
                )
            )
        return passages

    def document(self, doc_id: str) -> StoredDocument:
        try:
            return self._documents[doc_id]
        except KeyError:
            raise UnknownDocumentError(doc_id) from None


def _matches(fields: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    """Exact match on every filter key; a key the backend does not know matches nothing."""
    return all(fields.get(key) == value for key, value in filters.items())
