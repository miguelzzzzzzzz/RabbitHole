"""`ScriptedProvider`: a `ChatProvider` that replays canned completions.

Used for deterministic agent runs with no model: tests and CI now, and the
planned ``arp run --provider scripted --script FILE`` (not wired yet). Each
call to `complete` consumes the next `ScriptedTurn`:

- a *message* turn (``content``, ``tool_calls``, ``reasoning_content``,
  ``finish_reason``, ``usage``) is rendered into an OpenAI chat-completions
  body and parsed by the same `parse_chat_completion` the real provider uses,
  so malformed tool-call arguments (a JSON string that does not parse, a
  missing ``arguments`` field) reach the caller exactly as they would from a
  server;
- a ``raw`` turn is a whole chat-completions body passed to the parser
  verbatim, for replies that are malformed beyond the arguments (no choices,
  a non-list ``tool_calls``, ...): the parser raises
  `ProviderResponseError` as it would for a server;
- an ``error`` turn raises the matching provider error
  (`LocalServerUnavailableError` or `ProviderHTTPError`).

Running past the last turn raises `ProviderError` with reason
``script_exhausted``. Every request is kept in `ScriptedProvider.requests`.
``latency_ms`` is always 0.0: nothing is timed, so replies are identical
from run to run.

A script file is JSON: ``{"schema_version": 1, "question": "...", "turns":
[...]}`` (``question`` is optional; it gives ``arp run`` a task).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.providers.errors import (
    LocalServerUnavailableError,
    ProviderError,
    ProviderHTTPError,
)
from agentic_research.providers.openai_compat import parse_chat_completion
from agentic_research.providers.types import ChatRequest, ChatResponse, TokenUsage

SCRIPT_SCHEMA_VERSION: Final = 1
SCRIPTED_MODEL: Final = "scripted"
SCRIPT_EXHAUSTED: Final = "script_exhausted"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScriptedFunctionCall(_Model):
    """One tool call in a scripted reply.

    ``arguments`` given as an object is sent as its JSON string (as OpenAI
    does); given as a string it is sent verbatim, so it may be malformed;
    omitted, the ``arguments`` field is left out of the payload.
    """

    name: str
    arguments: dict[str, Any] | str | None = None
    id: str | None = None


class ScriptedError(_Model):
    """A provider failure to raise instead of replying."""

    kind: Literal["unavailable", "http"]
    message: str = "scripted provider failure"
    status_code: Annotated[int, Field(ge=400, le=599)] = 500


class ScriptedTurn(_Model):
    """One canned reply: a message, a raw body, or an error (exactly one kind)."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ScriptedFunctionCall, ...] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    raw: dict[str, Any] | None = None
    error: ScriptedError | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        message_set = bool(self.model_fields_set - {"raw", "error"})
        kinds = message_set + (self.raw is not None) + (self.error is not None)
        if kinds > 1:
            raise ValueError("a turn is exactly one of: message fields, raw, or error")
        return self

    def payload(self, model: str) -> dict[str, Any]:
        """The chat-completions body for a message turn (or the raw body)."""
        if self.raw is not None:
            return dict(self.raw)
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            calls = []
            for index, call in enumerate(self.tool_calls):
                function: dict[str, Any] = {"name": call.name}
                if isinstance(call.arguments, dict):
                    function["arguments"] = json.dumps(call.arguments, separators=(",", ":"))
                elif call.arguments is not None:
                    function["arguments"] = call.arguments
                calls.append(
                    {"id": call.id or f"call_{index}", "type": "function", "function": function}
                )
            message["tool_calls"] = calls
        finish = self.finish_reason
        if finish is None:
            finish = "tool_calls" if self.tool_calls else "stop"
        body: dict[str, Any] = {
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if self.usage is not None:
            body["usage"] = self.usage.model_dump(exclude_none=True)
        return body


class Script(_Model):
    """A script file: optional task question plus the turns to replay."""

    schema_version: Literal[1] = SCRIPT_SCHEMA_VERSION
    question: Annotated[str, Field(min_length=1)] | None = None
    turns: tuple[ScriptedTurn, ...] = Field(min_length=1)


def load_script(path: str | Path) -> Script:
    """Read and validate a JSON script file (raises ``pydantic.ValidationError``)."""
    return Script.model_validate_json(Path(path).read_bytes())


class ScriptedProvider:
    """Replays `ScriptedTurn` values in order; see the module docstring."""

    name = "scripted"

    def __init__(self, turns: Sequence[ScriptedTurn], *, model: str = SCRIPTED_MODEL) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.model = model
        self._turns = tuple(turns)
        self._next = 0
        self.requests: list[ChatRequest] = []

    @classmethod
    def from_script(cls, script: Script, *, model: str = SCRIPTED_MODEL) -> ScriptedProvider:
        return cls(script.turns, model=model)

    @property
    def remaining(self) -> int:
        """Turns not yet replayed."""
        return len(self._turns) - self._next

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._next >= len(self._turns):
            raise ProviderError(
                f"script has no turn left for call {self._next + 1} ({len(self._turns)} scripted)",
                reason=SCRIPT_EXHAUSTED,
            )
        turn = self._turns[self._next]
        self._next += 1
        if turn.error is not None:
            if turn.error.kind == "unavailable":
                raise LocalServerUnavailableError(turn.error.message)
            raise ProviderHTTPError(turn.error.message, status_code=turn.error.status_code)
        model = request.model or self.model
        return parse_chat_completion(turn.payload(model), latency_ms=0.0, fallback_model=model)
