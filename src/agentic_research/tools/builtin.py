"""The built-in tool set the agent (and `arp tools schema`) uses by default."""

from __future__ import annotations

from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry


def default_registry() -> ToolRegistry:
    """A fresh registry with every built-in tool.

    M1 ships ``calculator``; the research tools (``search_corpus``,
    ``read_document``, ``fetch_url``, ``final_answer``) join in M3.
    """
    return ToolRegistry([CalculatorTool()])
