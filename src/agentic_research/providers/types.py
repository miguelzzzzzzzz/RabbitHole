"""ChatProvider protocol and OpenAI chat-completions shaped request/response models.

Enough for the M1 tool-calling probe. Keep the surface minimal; HostedProvider,
scripted/replay providers, retries, and recording arrive later.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    """One function invocation inside a native ``tool_calls`` entry.

    ``arguments`` is the parsed object when the payload's ``arguments`` field
    was valid JSON (object or JSON string). When parsing fails the probe gets
    ``arguments=None`` plus ``parse_error`` and the original ``arguments_raw``;
    nothing is guessed from prose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    arguments: dict[str, Any] | None = None
    arguments_raw: str = ""
    parse_error: str | None = None

    @property
    def arguments_valid(self) -> bool:
        return self.parse_error is None and self.arguments is not None


class ToolCall(BaseModel):
    """One native OpenAI-shaped tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """One chat message (request or assistant reply).

    ``reasoning_content`` captures llama.cpp / DeepSeek-style hidden
    thinking when the server puts thoughts there (``--reasoning-format
    auto|deepseek``). Empty or absent is fine; a non-empty value is scored
    as ``think_block_present`` by the probe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class TokenUsage(BaseModel):
    """Token counts from a chat-completions ``usage`` object.

    Missing fields stay ``None`` (never coerced to 0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatRequest(BaseModel):
    """Input to ``ChatProvider.complete``.

    Defaults favour the M1 probe: temperature 0, fixed seed, non-thinking
    mode (``chat_template_kwargs.enable_thinking = false`` for Qwen3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    tools: tuple[dict[str, Any], ...] | None = None
    model: str | None = None
    temperature: float | None = 0.0
    seed: int | None = 0
    max_tokens: int | None = None
    enable_thinking: bool = False


class ChatResponse(BaseModel):
    """Parsed reply from a chat-completions call.

    ``raw`` is the decoded JSON body (no auth headers). ``tool_calls`` on the
    message are already structured: valid argument objects, or
    ``parse_error`` set when arguments were malformed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: ChatMessage
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str
    latency_ms: float
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ChatProvider(Protocol):
    """OpenAI chat-completions shaped provider (ADR-0003)."""

    name: str

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run one chat completion and return a structured reply."""
        ...
