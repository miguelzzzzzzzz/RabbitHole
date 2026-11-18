"""The built-in tool sets the agent (and `arp tools schema`) uses."""

from __future__ import annotations

from agentic_research.backends.base import SearchBackend
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.read import ReadDocumentTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import SearchCorpusTool


def default_registry() -> ToolRegistry:
    """A fresh registry with every built-in tool that needs no corpus (``calculator``).

    The research tools need a `SearchBackend`; see `research_registry`.
    ``fetch_url`` and ``final_answer`` join later in M3.
    """
    return ToolRegistry([CalculatorTool()])


def research_registry(backend: SearchBackend) -> ToolRegistry:
    """``calculator`` plus ``search_corpus`` and ``read_document`` over ``backend``."""
    return ToolRegistry([CalculatorTool(), SearchCorpusTool(backend), ReadDocumentTool(backend)])
