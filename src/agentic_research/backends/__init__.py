"""Search backends behind the research tools (M3)."""

from agentic_research.backends.base import (
    SUPPORTED_SEARCH_SCHEMA_VERSION,
    BackendError,
    Passage,
    SearchBackend,
    StoredDocument,
    UnknownDocumentError,
    make_doc_id,
    parse_search_payload,
    sha256_hex,
    validate_search_args,
)
from agentic_research.backends.fixture import FixtureSearchBackend

__all__ = [
    "SUPPORTED_SEARCH_SCHEMA_VERSION",
    "BackendError",
    "FixtureSearchBackend",
    "Passage",
    "SearchBackend",
    "StoredDocument",
    "UnknownDocumentError",
    "make_doc_id",
    "parse_search_payload",
    "sha256_hex",
    "validate_search_args",
]
