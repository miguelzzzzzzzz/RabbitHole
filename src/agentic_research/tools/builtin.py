"""The built-in tool sets the agent (and `arp tools schema`) uses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from agentic_research.backends.base import BackendError, Passage, SearchBackend, StoredDocument
from agentic_research.tools.answer import final_answer_schema
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.read import ReadDocumentTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import SearchCorpusTool

__all__ = [
    "NO_CORPUS_ID",
    "NoCorpusBackend",
    "catalog_registry",
    "catalog_schemas",
    "default_registry",
    "research_registry",
]

#: ``corpus_id`` of `NoCorpusBackend` (never a real corpus: real ids are given explicitly).
NO_CORPUS_ID: Final = "no-corpus"


def default_registry() -> ToolRegistry:
    """A fresh registry with every built-in tool that needs no corpus (``calculator``).

    The research tools need a `SearchBackend`; see `research_registry`.
    ``fetch_url`` joins later in M3. ``final_answer`` is never registered: it
    is the answer action (`agentic_research.tools.answer`).
    """
    return ToolRegistry([CalculatorTool()])


def research_registry(backend: SearchBackend) -> ToolRegistry:
    """``calculator`` plus ``search_corpus`` and ``read_document`` over ``backend``."""
    return ToolRegistry([CalculatorTool(), SearchCorpusTool(backend), ReadDocumentTool(backend)])


class NoCorpusBackend:
    """A `SearchBackend` with no corpus behind it: every call raises `BackendError`.

    It lets the research tools be listed (their schemas do not depend on the
    corpus) without loading an index. A tool over it fails every call as
    ``permanent/backend_error``, so it can never produce evidence.
    """

    @property
    def corpus_id(self) -> str:
        return NO_CORPUS_ID

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        raise BackendError("no corpus is loaded (pass --index DIR --corpus-id ID to arp run)")

    def document(self, doc_id: str) -> StoredDocument:
        raise BackendError("no corpus is loaded (pass --index DIR --corpus-id ID to arp run)")


def catalog_registry() -> ToolRegistry:
    """Every built-in tool, for schema export (``arp tools schema``).

    The research tools are bound to `NoCorpusBackend`; runs use
    `default_registry` or `research_registry` instead.
    """
    return research_registry(NoCorpusBackend())


def catalog_schemas() -> list[dict[str, Any]]:
    """Every built-in tool schema, then ``final_answer`` (``arp tools schema``).

    ``final_answer`` is listed for the model but is not in any registry: a
    call to it ends the run as a `FinalAnswer` (see `agentic_research.tools.answer`).
    """
    return [*catalog_registry().openai_schemas(), final_answer_schema()]
