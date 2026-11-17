"""Search backend contract: `Passage`, `StoredDocument` and the `SearchBackend` protocol.

The research tools (``search_corpus``, ``read_document``) and the citation
verifier only see this contract, so the agent can run over CitationNeeded's
index (`RagEngineBackend`) or over in-memory fixtures (`FixtureSearchBackend`)
without knowing which.

Citation invariants every backend guarantees (PLAN M3, CitationNeeded index
format 2):

* ``doc_id == make_doc_id(corpus_id, source)``, where ``source`` is the
  document's POSIX path relative to the corpus root.
* ``StoredDocument.content_hash == sha256(text)`` of the *normalized* text, and
  a passage's ``content_hash`` is its document's.
* Offsets index the normalized text: ``passage.text ==
  document(passage.doc_id).text[passage.start_char:passage.end_char]``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "SUPPORTED_SEARCH_SCHEMA_VERSION",
    "BackendError",
    "Passage",
    "SearchBackend",
    "StoredDocument",
    "UnknownDocumentError",
    "make_doc_id",
    "parse_search_payload",
    "sha256_hex",
    "validate_search_args",
]

#: The CitationNeeded ``search --json`` payload version this adapter reads
#: (the value at both ``v0.1.0`` and ``v1.0.0``).
SUPPORTED_SEARCH_SCHEMA_VERSION: Final = 1

_DOC_ID_RE: Final = r"^[0-9a-f]{16}$"
_SHA256_RE: Final = r"^[0-9a-f]{64}$"
_CORPUS_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class BackendError(Exception):
    """A backend cannot serve the request (bad index, unknown payload version, ...)."""


class UnknownDocumentError(BackendError):
    """The requested ``doc_id`` is not in the backend's corpus."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(f"unknown doc_id {doc_id!r}")
        self.doc_id = doc_id


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(corpus_id: str, source: str) -> str:
    """``sha256("<corpus_id>:<source>")[:16]``, CitationNeeded's format-2 doc id."""
    if _CORPUS_ID_RE.fullmatch(corpus_id) is None:
        raise ValueError(f"invalid corpus_id {corpus_id!r}")
    parts = source.split("/")
    if (
        not source
        or "\\" in source
        or source.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"source must be a normalized relative POSIX path, got {source!r}")
    return sha256_hex(f"{corpus_id}:{source}")[:16]


class StoredDocument(BaseModel):
    """A corpus document's normalized text, the text every citation offset indexes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(pattern=_DOC_ID_RE)
    source: str = Field(min_length=1)
    content_hash: str = Field(pattern=_SHA256_RE)
    text: str

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        if self.content_hash != sha256_hex(self.text):
            raise ValueError("content_hash does not match sha256 of the document text")
        return self


class Passage(BaseModel):
    """One search hit with everything a citation needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    score: float
    doc_id: str = Field(pattern=_DOC_ID_RE)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content_hash: str = Field(pattern=_SHA256_RE)
    heading_path: tuple[str, ...] = ()
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _check_fields(self) -> Self:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("text length must equal end_char - start_char")
        if (self.page is None) != (self.page_end is None):
            raise ValueError("page and page_end must both be set or both be None")
        if self.page is not None and self.page_end is not None and self.page_end < self.page:
            raise ValueError("page_end must be >= page")
        return self


class SearchBackend(Protocol):
    """What the research tools and the citation verifier need from a corpus."""

    @property
    def corpus_id(self) -> str: ...

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        """Top-``k`` passages, ranks 1..n, scores non-increasing.

        ``filters`` are exact-match constraints on passage metadata (``doc_id``,
        ``source``); a key the backend does not know matches nothing. A lexical
        backend returns only passages that share a term with the query; a
        hybrid one may also return dense neighbours that do not. Raises
        ``ValueError`` for a blank query or ``k < 1``.
        """
        ...

    def document(self, doc_id: str) -> StoredDocument:
        """The stored normalized document; `UnknownDocumentError` if absent."""
        ...


def validate_search_args(query: str, k: int) -> None:
    """The argument checks every backend applies before searching."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if k < 1:
        raise ValueError("k must be >= 1")


_PASSAGE_FIELDS: Final = tuple(Passage.model_fields)


def parse_search_payload(payload: object) -> list[Passage]:
    """Read a CitationNeeded ``search --json`` payload into passages.

    Gates on ``schema_version`` (an unknown or missing version is an error, never
    an empty hit list), reads results from ``hits`` and requires the format-2
    citation fields on every hit (``content_hash``, normalized offsets,
    ``page``/``page_end``). Keys RabbitHole does not use (``retriever``,
    ``components``, ``ranks``) are ignored.
    """
    if not isinstance(payload, Mapping):
        raise BackendError(f"search payload must be a JSON object, got {type(payload).__name__}")
    if "schema_version" not in payload:
        raise BackendError("search payload has no schema_version")
    version = payload["schema_version"]
    # Exactly the JSON integer: not a bool, a float such as 1.0, or a string.
    if type(version) is not int or version != SUPPORTED_SEARCH_SCHEMA_VERSION:
        raise BackendError(
            f"unsupported search payload schema_version {version!r} "
            f"(this adapter reads {SUPPORTED_SEARCH_SCHEMA_VERSION})"
        )
    hits = payload.get("hits")
    if not isinstance(hits, Sequence) or isinstance(hits, str | bytes):
        raise BackendError("search payload 'hits' must be a list")
    passages: list[Passage] = []
    for position, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            raise BackendError(f"search hit {position} must be an object")
        missing = [name for name in _PASSAGE_FIELDS if name not in hit]
        if missing:
            raise BackendError(f"search hit {position} is missing {', '.join(missing)}")
        fields: dict[str, Any] = {name: hit[name] for name in _PASSAGE_FIELDS}
        try:
            passages.append(Passage.model_validate(fields))
        except ValidationError as exc:
            raise BackendError(f"search hit {position} is invalid: {exc}") from exc
    return passages
