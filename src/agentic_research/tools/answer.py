"""``final_answer``: the answer action as a tool schema (PLAN M3).

A policy ends a run by answering. For a model, answering looks like one more
function in its ``tools`` list, so ``final_answer`` has an OpenAI-format
schema like the other tools (`final_answer_schema`, printed by
``arp tools schema``). It is *not* a registered tool: the executor never runs
it. `ToolRegistry` reserves the name (it refuses to register a tool called
``final_answer`` and answers a request to run one with
``invalid_args/reserved_tool_name``), so an answer can only end a run as a
`FinalAnswer` action, which the loop hands to its verifier.

`parse_final_answer` turns a model's ``final_answer`` call (a JSON string or a
mapping, as native ``tool_calls`` carry them) into that `FinalAnswer`, or into
an ``invalid_args`` `ToolError` the policy can report. Argument parsing is the
registry's (`parse_arguments`), so malformed JSON, duplicate keys and schema
violations get the same reason codes as any tool call.

The schema checks shape only. ``claims`` is required so the model is told to
cite, but an empty list (or a claim without citations) is well formed: the
citation verifier rejects it as ``no_citation`` with repair feedback, which a
schema rejection would not give. Wiring this into the LLM policy's native and
JSON-in-text reply formats is M5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field

from agentic_research.agent.actions import Claim, FinalAnswer
from agentic_research.tools.base import ToolArgs, ToolError
from agentic_research.tools.registry import parse_arguments

__all__ = [
    "FINAL_ANSWER_DESCRIPTION",
    "FINAL_ANSWER_TOOL",
    "FinalAnswerArgs",
    "final_answer_schema",
    "parse_final_answer",
]

FINAL_ANSWER_TOOL: Final = "final_answer"

FINAL_ANSWER_DESCRIPTION: Final = (
    "Finish the task with your answer. Back the answer with claims; each claim cites "
    "passages you were shown in this run by doc_id and the start/end character offsets "
    "from search_corpus or read_document, with quote equal to the document text at "
    "[start, end) exactly. Citations are checked; an answer with an invalid, stale or "
    "unshown citation, or with no citation, is rejected."
)


class FinalAnswerArgs(ToolArgs):
    """Arguments of ``final_answer``: the answer text and the claims that back it."""

    answer: str = Field(
        min_length=1,
        pattern=r"\S",
        description="The answer to the task, in plain prose.",
    )
    claims: tuple[Claim, ...] = Field(
        description="Statements that support the answer, each with its citations.",
    )


def final_answer_schema() -> dict[str, Any]:
    """``final_answer`` in the OpenAI chat-completions ``tools`` format.

    The same shape as `ToolRegistry.openai_schemas` entries. JSON Schema
    cannot express ``end > start``; `Citation` checks it when the call is
    parsed (``schema_violation``).
    """
    return {
        "type": "function",
        "function": {
            "name": FINAL_ANSWER_TOOL,
            "description": FINAL_ANSWER_DESCRIPTION,
            "parameters": FinalAnswerArgs.model_json_schema(),
        },
    }


def parse_final_answer(arguments: Mapping[str, Any] | str | None) -> FinalAnswer | ToolError:
    """A model's ``final_answer`` arguments as a `FinalAnswer`, or ``invalid_args``.

    ``answer`` is kept verbatim (a whitespace-only answer is a
    ``schema_violation``). Whether the citations are valid, fresh and grounded
    is left to the verifier.
    """
    parsed = parse_arguments(FINAL_ANSWER_TOOL, FinalAnswerArgs, arguments)
    if isinstance(parsed, ToolError):
        return parsed
    return FinalAnswer(answer=parsed.answer, claims=parsed.claims)
