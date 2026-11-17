"""`RagEngineBackend`: the `SearchBackend` adapter over a CitationNeeded index.

Consumes the CitationNeeded ``v1.0.0`` contract through its in-process API
(``load_index`` / ``load_documents`` and the retrievers), never its HTTP
service or ``answer`` command:

* The index must be format 2 (``index.json`` ``format_version``), checked here
  as well as by ``load_index``, which also re-checks every chunk offset against
  ``documents.jsonl`` and so refuses a tampered index.
* Every stored document must belong to the expected ``corpus_id``; RabbitHole
  always names the corpus explicitly and never relies on CitationNeeded's
  folder-name fallback.
* Hits go through the same payload as ``rag-engine search --json`` and are read
  by `parse_search_payload`, so the adapter gates on ``schema_version``, not on
  ``rag_engine.__version__``.

``mode="lexical"`` (BM25) is the default. ``mode="hybrid"`` fuses the index's
dense and BM25 retrievers with reciprocal rank fusion; with a fastembed index
that needs fastembed and a model download (slow tests only), with a hashing
index it runs offline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from rag_engine.cli import SEARCH_JSON_SCHEMA_VERSION
from rag_engine.retrieval import HybridRetriever, Retriever, load_index
from rag_engine.retrieval.index_store import load_documents

from agentic_research.backends.base import (
    BackendError,
    Passage,
    StoredDocument,
    UnknownDocumentError,
    parse_search_payload,
    validate_search_args,
)

__all__ = ["REQUIRED_INDEX_FORMAT", "RagEngineBackend", "RetrievalMode"]

#: CitationNeeded index format with corpus-scoped doc ids and ``documents.jsonl``.
REQUIRED_INDEX_FORMAT: Final = 2

RetrievalMode = Literal["lexical", "hybrid"]
_MODES: Final = ("lexical", "hybrid")


class RagEngineBackend:
    """Searches one CitationNeeded format-2 index directory."""

    def __init__(
        self, index_dir: str | Path, *, corpus_id: str, mode: RetrievalMode = "lexical"
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {_MODES}")
        path = Path(index_dir)
        _check_format(path)
        try:
            dense, lexical = load_index(path)
            documents = load_documents(path)
        except (OSError, ValueError) as exc:
            raise BackendError(f"cannot load index {path}: {exc}") from exc
        foreign = sorted({doc.corpus_id for doc in documents} - {corpus_id})
        if foreign:
            raise BackendError(
                f"index {path} holds documents from corpus {', '.join(foreign)}, "
                f"expected only {corpus_id!r}"
            )
        self._corpus_id = corpus_id
        self._mode: RetrievalMode = mode
        self._documents = {
            doc.doc_id: StoredDocument(
                doc_id=doc.doc_id, source=doc.source, content_hash=doc.content_hash, text=doc.text
            )
            for doc in documents
        }
        self._retriever: Retriever = (
            lexical if mode == "lexical" else HybridRetriever([dense, lexical])
        )

    @property
    def corpus_id(self) -> str:
        return self._corpus_id

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        validate_search_args(query, k)
        hits = self._retriever.retrieve(query, k, dict(filters) if filters else None)
        payload: dict[str, Any] = {
            "schema_version": SEARCH_JSON_SCHEMA_VERSION,
            "query": query,
            "mode": self._mode,
            "rerank": False,
            "hits": [hit.to_dict() for hit in hits],
        }
        return parse_search_payload(payload)

    def document(self, doc_id: str) -> StoredDocument:
        try:
            return self._documents[doc_id]
        except KeyError:
            raise UnknownDocumentError(doc_id) from None


def _check_format(path: Path) -> None:
    """Refuse anything but a format-2 index before handing it to CitationNeeded."""
    manifest_path = path / "index.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BackendError(f"not a CitationNeeded index directory: {path}") from None
    except (OSError, ValueError) as exc:
        raise BackendError(f"cannot read {manifest_path}: {exc}") from exc
    version = manifest.get("format_version") if isinstance(manifest, dict) else None
    if isinstance(version, bool) or version != REQUIRED_INDEX_FORMAT:
        raise BackendError(
            f"index {path} has format_version {version!r}; RabbitHole needs format "
            f"{REQUIRED_INDEX_FORMAT} (rebuild it with CitationNeeded v1.0.0 `rag-engine index`)"
        )
