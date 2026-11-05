"""Strict JSON-in-text tool-call protocol (M1 fallback; ADR-0003).

The model replies with exactly one JSON object, optionally inside a single
fenced block (```` ```json ```` or bare ```` ``` ````)::

    {"action": "tool_call", "tool": "<name>", "arguments": {...}}
    {"action": "final_answer", "answer": "..."}

Nothing is guessed out of surrounding prose. The public entry point is
:func:`parse_toolcall_json`; callers pass an allowlist of tool names mapped to
pydantic argument models (typically the registry's ``args_model`` classes) so
the probe can score replies without importing the agent loop.

Think-block policy (Qwen3, PLAN M1 + 2026-11-04 probe decision)
---------------------------------------------------------------
Qwen3 may emit a leading ``<think>...</think>`` even in non-thinking mode.

* **Empty / whitespace-only body**: stripped as allowed preprocessing. The
  parse continues; ``ToolCallJsonResult.empty_think_stripped`` is ``True`` so
  counters can still observe the strip. This is **not** a parse failure
  (PLAN: "strips one leading think block").
* **Non-empty body**: rejected with reason ``think_block_present``. The
  remainder is not parsed — a stray reasoning block is its own failure type
  for probe scoring (2026-11-04 decision), even if valid JSON follows.
* Only **one leading** think block is considered. An unclosed ``<think>`` is
  ``think_block_unclosed``. A think block that is not leading is treated as
  ordinary surrounding text / invalid JSON.

Every rejection yields a :class:`~agentic_research.trace.ParseFailure` with a
stable reason code and, when a :class:`ParseFailureCounter` is supplied,
increments that counter by one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_research.trace.schema import ParseFailure

#: Stable reason codes emitted by this parser (``ReasonCode`` pattern).
REASON_EMPTY_REPLY: Final = "empty_reply"
REASON_THINK_BLOCK_PRESENT: Final = "think_block_present"
REASON_THINK_BLOCK_UNCLOSED: Final = "think_block_unclosed"
REASON_INVALID_JSON: Final = "invalid_json"
REASON_NOT_AN_OBJECT: Final = "not_an_object"
REASON_MULTIPLE_OBJECTS: Final = "multiple_objects"
REASON_LEADING_CONTENT: Final = "leading_content"
REASON_TRAILING_CONTENT: Final = "trailing_content"
REASON_UNSUPPORTED_FENCE: Final = "unsupported_fence"
REASON_UNCLOSED_FENCE: Final = "unclosed_fence"
REASON_UNKNOWN_ACTION: Final = "unknown_action"
REASON_UNKNOWN_KEY: Final = "unknown_key"
REASON_MISSING_KEY: Final = "missing_key"
REASON_INVALID_FIELD: Final = "invalid_field"
REASON_UNKNOWN_TOOL: Final = "unknown_tool"
REASON_ARGUMENTS_NOT_OBJECT: Final = "arguments_not_object"
REASON_INVALID_ARGUMENTS: Final = "invalid_arguments"

_THINK_OPEN: Final = "<think>"
_THINK_CLOSE: Final = "</think>"

_FENCE_OPEN_LINE_RE: Final = re.compile(r"\A```([^\n`]*)\Z")
_FENCE_CLOSE_LINE_RE: Final = re.compile(r"\A```[ \t]*\Z")

_DECODER: Final = json.JSONDecoder()


class ParsedToolCall(BaseModel):
    """A schema-valid JSON-in-text tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    arguments: dict[str, Any]


class ParsedFinalAnswer(BaseModel):
    """A schema-valid JSON-in-text final answer (M1 shape; M3 may extend)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str


class ToolCallJsonResult(BaseModel):
    """Outcome of :func:`parse_toolcall_json`.

    Exactly one of ``action`` (success) or ``failure`` (rejection) is set.
    ``empty_think_stripped`` is orthogonal: it is ``True`` when an empty
    leading think block was removed during preprocessing before the rest of
    the parse (success or a later failure). A ``think_block_present`` /
    ``think_block_unclosed`` failure leaves it ``False``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ParsedToolCall | ParsedFinalAnswer | None = None
    failure: ParseFailure | None = None
    empty_think_stripped: bool = False

    @property
    def ok(self) -> bool:
        return self.failure is None and self.action is not None


class ParseFailureCounter:
    """Mutable tallies for ``parse_failures`` aggregation (call / run / report)."""

    def __init__(self) -> None:
        self._by_reason: dict[str, int] = {}

    @property
    def total(self) -> int:
        return sum(self._by_reason.values())

    @property
    def by_reason(self) -> Mapping[str, int]:
        return dict(self._by_reason)

    def record(self, failure: ParseFailure) -> None:
        """Increment the count for ``failure.reason`` by one."""
        reason = failure.reason
        self._by_reason[reason] = self._by_reason.get(reason, 0) + 1

    def clear(self) -> None:
        self._by_reason.clear()


class _ToolCallEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["tool_call"]
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]


class _FinalAnswerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["final_answer"]
    answer: str


def _fail(
    reason: str,
    detail: str = "",
    *,
    empty_think_stripped: bool = False,
    failures: ParseFailureCounter | None,
) -> ToolCallJsonResult:
    failure = ParseFailure(reason=reason, detail=detail[:2000])
    if failures is not None:
        failures.record(failure)
    return ToolCallJsonResult(
        action=None,
        failure=failure,
        empty_think_stripped=empty_think_stripped,
    )


def _strip_leading_think(text: str) -> tuple[str, bool, str | None, str]:
    """Return ``(remainder, empty_stripped, fail_reason, fail_detail)``."""
    stripped_left = text.lstrip()
    if not stripped_left.startswith(_THINK_OPEN):
        return text, False, None, ""

    after_open = stripped_left[len(_THINK_OPEN) :]
    close_at = after_open.find(_THINK_CLOSE)
    if close_at < 0:
        return text, False, REASON_THINK_BLOCK_UNCLOSED, "leading <think> has no closing tag"

    body = after_open[:close_at]
    remainder = after_open[close_at + len(_THINK_CLOSE) :]
    if body.strip():
        return (
            text,
            False,
            REASON_THINK_BLOCK_PRESENT,
            "leading <think> body is not empty",
        )
    return remainder, True, None, ""


def _extract_fenced_payload(
    body: str,
    *,
    empty_think_stripped: bool,
    failures: ParseFailureCounter | None,
) -> tuple[str | None, ToolCallJsonResult | None]:
    """``body`` starts with ````` ``. Return ``(json_text, None)`` or failure."""
    lines = body.splitlines()
    if not lines:
        return None, _fail(
            REASON_UNSUPPORTED_FENCE,
            "opening fence must be ``` or ```json on its own line",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    open_match = _FENCE_OPEN_LINE_RE.fullmatch(lines[0])
    if open_match is None:
        return None, _fail(
            REASON_UNSUPPORTED_FENCE,
            "opening fence must be ``` or ```json on its own line",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )
    lang = open_match.group(1).strip().lower()
    if lang not in ("", "json"):
        return None, _fail(
            REASON_UNSUPPORTED_FENCE,
            f"fence language {lang!r} is not supported (use json or bare)",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    close_idx: int | None = None
    for index in range(1, len(lines)):
        if _FENCE_CLOSE_LINE_RE.fullmatch(lines[index]):
            close_idx = index
            break
    if close_idx is None:
        return None, _fail(
            REASON_UNCLOSED_FENCE,
            "closing ``` fence not found on its own line",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    after_lines = lines[close_idx + 1 :]
    if any(line.strip() for line in after_lines):
        return None, _fail(
            REASON_TRAILING_CONTENT,
            "non-whitespace content after closing fence",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    json_text = "\n".join(lines[1:close_idx])
    return json_text, None


def _extract_fenced_or_bare(
    text: str,
    *,
    empty_think_stripped: bool,
    failures: ParseFailureCounter | None,
) -> tuple[str | None, ToolCallJsonResult | None]:
    """Return ``(json_text, None)`` on success, or ``(None, failure_result)``."""
    if text.strip() == "":
        return None, _fail(
            REASON_EMPTY_REPLY,
            "reply is empty after preprocessing",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    body = text.lstrip()
    # Trailing whitespace on the whole reply is fine; keep it out of the
    # "leading content" check by only looking at the left-stripped body.

    if body.startswith("```"):
        return _extract_fenced_payload(
            body.rstrip(),
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    if not body.startswith("{"):
        if body.lstrip().startswith("["):
            return None, _fail(
                REASON_NOT_AN_OBJECT,
                "top-level JSON value is an array, not an object",
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )
        return None, _fail(
            REASON_LEADING_CONTENT,
            "non-whitespace text before the JSON object",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    return body, None


def _trailing_reason(rest: str) -> str:
    peek = rest.lstrip()
    if peek.startswith("{") or peek.startswith("["):
        return REASON_MULTIPLE_OBJECTS
    return REASON_TRAILING_CONTENT


def _decode_single_object(
    json_text: str,
    *,
    empty_think_stripped: bool,
    failures: ParseFailureCounter | None,
) -> tuple[dict[str, Any] | None, ToolCallJsonResult | None]:
    candidate = json_text.strip()
    if candidate == "":
        return None, _fail(
            REASON_EMPTY_REPLY,
            "no JSON object inside fence",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    try:
        value, end = _DECODER.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        return None, _fail(
            REASON_INVALID_JSON,
            f"{exc.msg} (line {exc.lineno} column {exc.colno})",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    if not isinstance(value, dict):
        return None, _fail(
            REASON_NOT_AN_OBJECT,
            f"top-level JSON value is {type(value).__name__}, not an object",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )

    rest = candidate[end:]
    if rest.strip():
        return None, _fail(
            _trailing_reason(rest),
            "extra content after the JSON object",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )
    return value, None


def _validation_reason(exc: ValidationError) -> tuple[str, str]:
    """Map a pydantic ``ValidationError`` to a stable reason + detail."""
    errors = exc.errors(include_url=False, include_input=False)
    if not errors:
        return REASON_INVALID_FIELD, "validation failed"
    err = errors[0]
    loc = ".".join(str(item) for item in err["loc"]) or "<root>"
    detail = f"{loc}: {err['msg']}"
    err_type = err["type"]
    if err_type == "extra_forbidden":
        return REASON_UNKNOWN_KEY, detail
    if err_type == "missing":
        return REASON_MISSING_KEY, detail
    if err_type == "literal_error" and loc == "action":
        return REASON_UNKNOWN_ACTION, detail
    return REASON_INVALID_FIELD, detail


def _validate_action(
    obj: dict[str, Any],
    tools: Mapping[str, type[BaseModel]],
    *,
    empty_think_stripped: bool,
    failures: ParseFailureCounter | None,
) -> ToolCallJsonResult:
    action = obj.get("action")
    if action == "tool_call":
        if "arguments" in obj and not isinstance(obj["arguments"], dict):
            return _fail(
                REASON_ARGUMENTS_NOT_OBJECT,
                f"arguments must be a JSON object (got {type(obj['arguments']).__name__})",
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )
        try:
            envelope = _ToolCallEnvelope.model_validate(obj)
        except ValidationError as exc:
            reason, detail = _validation_reason(exc)
            return _fail(
                reason,
                detail,
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )

        if envelope.tool not in tools:
            return _fail(
                REASON_UNKNOWN_TOOL,
                f"tool {envelope.tool!r} is not in the allowlist",
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )
        args_model = tools[envelope.tool]
        try:
            validated = args_model.model_validate(envelope.arguments)
        except ValidationError as exc:
            _reason, detail = _validation_reason(exc)
            return _fail(
                REASON_INVALID_ARGUMENTS,
                detail,
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )
        return ToolCallJsonResult(
            action=ParsedToolCall(tool=envelope.tool, arguments=validated.model_dump()),
            failure=None,
            empty_think_stripped=empty_think_stripped,
        )

    if action == "final_answer":
        try:
            final = _FinalAnswerEnvelope.model_validate(obj)
        except ValidationError as exc:
            reason, detail = _validation_reason(exc)
            return _fail(
                reason,
                detail,
                empty_think_stripped=empty_think_stripped,
                failures=failures,
            )
        return ToolCallJsonResult(
            action=ParsedFinalAnswer(answer=final.answer),
            failure=None,
            empty_think_stripped=empty_think_stripped,
        )

    if action is None and "action" not in obj:
        return _fail(
            REASON_MISSING_KEY,
            "action: Field required",
            empty_think_stripped=empty_think_stripped,
            failures=failures,
        )
    return _fail(
        REASON_UNKNOWN_ACTION,
        f"action must be 'tool_call' or 'final_answer' (got {action!r})",
        empty_think_stripped=empty_think_stripped,
        failures=failures,
    )


def parse_toolcall_json(
    text: str,
    tools: Mapping[str, type[BaseModel]],
    *,
    failures: ParseFailureCounter | None = None,
) -> ToolCallJsonResult:
    """Parse a model reply under the JSON-in-text tool protocol.

    Parameters
    ----------
    text:
        Raw assistant message content (may be empty).
    tools:
        Allowlist of tool name → pydantic argument model. Unknown names are
        rejected with ``unknown_tool``; argument objects that fail the model
        are ``invalid_arguments``. Models should use ``extra="forbid"``.
    failures:
        Optional counter; incremented by one on every rejection.

    Returns
    -------
    ToolCallJsonResult
        ``ok`` is true only when ``action`` is set and ``failure`` is None.
    """
    remainder, empty_think_stripped, think_reason, think_detail = _strip_leading_think(text)
    if think_reason is not None:
        return _fail(think_reason, think_detail, failures=failures)

    json_text, early_fail = _extract_fenced_or_bare(
        remainder,
        empty_think_stripped=empty_think_stripped,
        failures=failures,
    )
    if early_fail is not None:
        return early_fail
    assert json_text is not None

    obj, decode_fail = _decode_single_object(
        json_text,
        empty_think_stripped=empty_think_stripped,
        failures=failures,
    )
    if decode_fail is not None:
        return decode_fail
    assert obj is not None

    return _validate_action(
        obj,
        tools,
        empty_think_stripped=empty_think_stripped,
        failures=failures,
    )
