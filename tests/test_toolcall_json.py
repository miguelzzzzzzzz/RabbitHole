"""Table-driven tests for the JSON-in-text tool-call parser."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agentic_research.providers.toolcall_json import (
    REASON_ARGUMENTS_NOT_OBJECT,
    REASON_EMPTY_REPLY,
    REASON_INVALID_ARGUMENTS,
    REASON_INVALID_JSON,
    REASON_LEADING_CONTENT,
    REASON_MISSING_KEY,
    REASON_MULTIPLE_OBJECTS,
    REASON_NOT_AN_OBJECT,
    REASON_THINK_BLOCK_PRESENT,
    REASON_THINK_BLOCK_UNCLOSED,
    REASON_TRAILING_CONTENT,
    REASON_UNCLOSED_FENCE,
    REASON_UNKNOWN_ACTION,
    REASON_UNKNOWN_KEY,
    REASON_UNKNOWN_TOOL,
    REASON_UNSUPPORTED_FENCE,
    ParsedFinalAnswer,
    ParsedToolCall,
    ParseFailureCounter,
    parse_toolcall_json,
)
from agentic_research.tools.calculator import CalculatorArgs


class _StubSearchArgs(BaseModel):
    """Minimal args model so unknown-tool / multi-tool cases stay local."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)


TOOLS: dict[str, type[BaseModel]] = {
    "calculator": CalculatorArgs,
    "search_corpus": _StubSearchArgs,
}


def _tool_call_json(tool: str = "calculator", expression: str = "1+1") -> str:
    return (
        '{"action": "tool_call", "tool": "'
        + tool
        + '", "arguments": {"expression": "'
        + expression
        + '"}}'
    )


def _final_json(answer: str = "42") -> str:
    return '{"action": "final_answer", "answer": "' + answer + '"}'


def _fence(payload: str, *, lang: str = "json") -> str:
    tag = f"```{lang}" if lang else "```"
    return f"{tag}\n{payload}\n```"


# --- successes ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expect_tool", "expect_args", "empty_think"),
    [
        (_tool_call_json(), "calculator", {"expression": "1+1"}, False),
        (
            _fence(_tool_call_json()),
            "calculator",
            {"expression": "1+1"},
            False,
        ),
        (
            _fence(_tool_call_json(), lang=""),
            "calculator",
            {"expression": "1+1"},
            False,
        ),
        (
            "  \n" + _tool_call_json() + "\n  ",
            "calculator",
            {"expression": "1+1"},
            False,
        ),
        (
            "<think></think>" + _tool_call_json(),
            "calculator",
            {"expression": "1+1"},
            True,
        ),
        (
            "<think>  \n  </think>\n" + _fence(_tool_call_json()),
            "calculator",
            {"expression": "1+1"},
            True,
        ),
        (
            '{\n  "action": "tool_call",\n  "tool": "search_corpus",\n'
            '  "arguments": {"query": "rabbits"}\n}',
            "search_corpus",
            {"query": "rabbits"},
            False,
        ),
        (
            # Unicode in arguments
            '{"action":"tool_call","tool":"calculator","arguments":{"expression":"π"}}'.replace(
                "π", "pi"
            ),
            "calculator",
            {"expression": "pi"},
            False,
        ),
    ],
)
def test_valid_tool_call(
    text: str,
    expect_tool: str,
    expect_args: dict[str, Any],
    empty_think: bool,
) -> None:
    failures = ParseFailureCounter()
    result = parse_toolcall_json(text, TOOLS, failures=failures)
    assert result.ok
    assert result.failure is None
    assert failures.total == 0
    assert result.empty_think_stripped is empty_think
    assert isinstance(result.action, ParsedToolCall)
    assert result.action.tool == expect_tool
    assert result.action.arguments == expect_args


@pytest.mark.parametrize(
    ("text", "expect_answer", "empty_think"),
    [
        (_final_json("hello"), "hello", False),
        (_fence(_final_json("done")), "done", False),
        ("<think></think>" + _final_json(""), "", True),
        (_fence(_final_json("ok"), lang=""), "ok", False),
    ],
)
def test_valid_final_answer(text: str, expect_answer: str, empty_think: bool) -> None:
    failures = ParseFailureCounter()
    result = parse_toolcall_json(text, TOOLS, failures=failures)
    assert result.ok
    assert failures.total == 0
    assert result.empty_think_stripped is empty_think
    assert isinstance(result.action, ParsedFinalAnswer)
    assert result.action.answer == expect_answer


# --- rejections ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", REASON_EMPTY_REPLY),
        ("   \n\t  ", REASON_EMPTY_REPLY),
        ("<think></think>", REASON_EMPTY_REPLY),
        ("<think></think>\n  ", REASON_EMPTY_REPLY),
        ("<think>reasoning here</think>" + _tool_call_json(), REASON_THINK_BLOCK_PRESENT),
        ("<think>x</think>", REASON_THINK_BLOCK_PRESENT),
        ("<think>still open", REASON_THINK_BLOCK_UNCLOSED),
        ("not json at all", REASON_LEADING_CONTENT),
        ("Sure!\n" + _tool_call_json(), REASON_LEADING_CONTENT),
        (_tool_call_json() + "\nThanks!", REASON_TRAILING_CONTENT),
        (_tool_call_json() + _final_json(), REASON_MULTIPLE_OBJECTS),
        (_tool_call_json() + "\n" + _tool_call_json(), REASON_MULTIPLE_OBJECTS),
        ('{"action": "tool_call", "tool": "calculator", "arguments": {', REASON_INVALID_JSON),
        ("[1, 2, 3]", REASON_NOT_AN_OBJECT),
        ('"just a string"', REASON_LEADING_CONTENT),  # does not start with {
        ("null", REASON_LEADING_CONTENT),
        ("42", REASON_LEADING_CONTENT),
        (_fence("[1, 2]"), REASON_NOT_AN_OBJECT),
        (_fence(""), REASON_EMPTY_REPLY),
        ("```python\n" + _tool_call_json() + "\n```", REASON_UNSUPPORTED_FENCE),
        ("```json " + _tool_call_json() + "```", REASON_UNSUPPORTED_FENCE),  # inline, no newline
        ("```json\n" + _tool_call_json(), REASON_UNCLOSED_FENCE),
        (
            _fence(_tool_call_json()) + "\nextra",
            REASON_TRAILING_CONTENT,
        ),
        (
            '{"action": "tool_call", "tool": "calculator",'
            ' "arguments": {"expression": "1"}, "extra": true}',
            REASON_UNKNOWN_KEY,
        ),
        (
            '{"action": "final_answer", "answer": "x", "extra": 1}',
            REASON_UNKNOWN_KEY,
        ),
        (
            '{"action": "give_up"}',
            REASON_UNKNOWN_ACTION,
        ),
        (
            '{"tool": "calculator", "arguments": {"expression": "1"}}',
            REASON_MISSING_KEY,
        ),
        (
            '{"action": "tool_call", "arguments": {"expression": "1"}}',
            REASON_MISSING_KEY,
        ),
        (
            '{"action": "final_answer"}',
            REASON_MISSING_KEY,
        ),
        (
            '{"action": "tool_call", "tool": "not_a_tool", "arguments": {"expression": "1"}}',
            REASON_UNKNOWN_TOOL,
        ),
        (
            '{"action": "tool_call", "tool": "calculator", "arguments": {"expression": 1}}',
            REASON_INVALID_ARGUMENTS,
        ),
        (
            '{"action": "tool_call", "tool": "calculator", "arguments": {}}',
            REASON_INVALID_ARGUMENTS,
        ),
        (
            '{"action": "tool_call", "tool": "calculator",'
            ' "arguments": {"expression": "1", "unit": "m"}}',
            REASON_INVALID_ARGUMENTS,
        ),
        (
            '{"action": "tool_call", "tool": "calculator", "arguments": "1+1"}',
            REASON_ARGUMENTS_NOT_OBJECT,
        ),
        (
            '{"action": "tool_call", "tool": "calculator", "arguments": [1, 2]}',
            REASON_ARGUMENTS_NOT_OBJECT,
        ),
        (
            # Prose after a fence
            "```json\n" + _tool_call_json() + "\n```\nMore text",
            REASON_TRAILING_CONTENT,
        ),
        (
            # Two fenced blocks → trailing content after first close
            _fence(_tool_call_json()) + "\n" + _fence(_final_json()),
            REASON_TRAILING_CONTENT,
        ),
    ],
)
def test_rejection_reason_and_counter(text: str, reason: str) -> None:
    failures = ParseFailureCounter()
    result = parse_toolcall_json(text, TOOLS, failures=failures)
    assert not result.ok
    assert result.action is None
    assert result.failure is not None
    assert result.failure.reason == reason
    assert failures.total == 1
    assert failures.by_reason == {reason: 1}


def test_nonempty_think_does_not_parse_following_json() -> None:
    """A non-empty think block fails even when the remainder is valid JSON."""
    text = "<think>I should call the calculator</think>\n" + _tool_call_json()
    result = parse_toolcall_json(text, TOOLS)
    assert result.failure is not None
    assert result.failure.reason == REASON_THINK_BLOCK_PRESENT
    assert result.empty_think_stripped is False


def test_empty_think_strip_flag_on_later_failure() -> None:
    text = "<think></think>\nnot an object"
    result = parse_toolcall_json(text, TOOLS)
    assert result.failure is not None
    assert result.failure.reason == REASON_LEADING_CONTENT
    assert result.empty_think_stripped is True


def test_never_extracts_json_from_prose() -> None:
    """A JSON object buried in prose is not recovered."""
    text = f"Here is the call you want: {_tool_call_json()} — use it."
    result = parse_toolcall_json(text, TOOLS)
    assert result.failure is not None
    assert result.failure.reason == REASON_LEADING_CONTENT


def test_counter_aggregates_across_calls() -> None:
    failures = ParseFailureCounter()
    parse_toolcall_json("nope", TOOLS, failures=failures)
    parse_toolcall_json(_tool_call_json(tool="missing"), TOOLS, failures=failures)
    parse_toolcall_json("nope", TOOLS, failures=failures)
    assert failures.total == 3
    assert failures.by_reason[REASON_LEADING_CONTENT] == 2
    assert failures.by_reason[REASON_UNKNOWN_TOOL] == 1
    failures.clear()
    assert failures.total == 0


def test_fenced_object_with_trailing_ws_inside_ok() -> None:
    text = "```json\n" + _tool_call_json() + "  \n```\n  "
    result = parse_toolcall_json(text, TOOLS)
    assert result.ok
    assert isinstance(result.action, ParsedToolCall)


def test_crlf_fenced_tool_call() -> None:
    text = "```json\r\n" + _tool_call_json() + "\r\n```\r\n"
    result = parse_toolcall_json(text, TOOLS)
    assert result.ok


def test_only_one_leading_think_is_considered() -> None:
    """A second think block after an empty one is surrounding text."""
    text = "<think></think><think></think>" + _tool_call_json()
    result = parse_toolcall_json(text, TOOLS)
    # After stripping the first empty think, remainder starts with <think>...
    assert result.failure is not None
    assert result.failure.reason == REASON_LEADING_CONTENT
    assert result.empty_think_stripped is True
