"""Minimal async OpenAI-compatible chat provider (local llama-server / Ollama).

Talks to ``{base_url}/chat/completions`` with httpx. Enough for the M1 probe:
optional ``tools``, usage capture, non-thinking mode, and structured native
``tool_calls`` parsing. Retries, runaway guard, HostedProvider, and recording
are M5.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any, Final

import httpx

from agentic_research.providers.errors import (
    LocalServerUnavailableError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseError,
)
from agentic_research.providers.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FunctionCall,
    TokenUsage,
    ToolCall,
)

#: Default OpenAI-compatible base URL (matches ``ARP_LLM_BASE_URL`` / preflight).
DEFAULT_BASE_URL: Final = "http://127.0.0.1:8080/v1"
#: Default served model id (``ARP_LOCAL_MODEL`` / ``--alias``).
DEFAULT_MODEL: Final = "Qwen3-1.7B-Q4_K_M"
#: Default request timeout in seconds.
DEFAULT_TIMEOUT_S: Final = 120.0

_REDACTED: Final = "***"


def base_url_from_environ(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get("ARP_LLM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL).rstrip("/")


def model_from_environ(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get("ARP_LOCAL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def parse_function_arguments(raw: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    """Parse a tool-call ``arguments`` field into ``(object, raw_string, error)``.

    OpenAI sends a JSON string; some servers (and recorded payloads) send an
    object. Malformed JSON yields ``(None, raw_string, reason)`` so the probe
    can score the call without guessing.
    """
    if raw is None:
        return None, "", "missing_arguments"
    if isinstance(raw, dict):
        # Already an object; keep a stable raw form for traces/scoring.
        try:
            raw_str = json.dumps(raw, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            return None, repr(raw), f"unserializable_arguments: {exc}"
        return dict(raw), raw_str, None
    if isinstance(raw, str):
        if raw.strip() == "":
            return None, raw, "empty_arguments"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, raw, f"invalid_json: {exc.msg}"
        if not isinstance(parsed, dict):
            return None, raw, f"arguments_not_object: {type(parsed).__name__}"
        return parsed, raw, None
    return None, repr(raw), f"unsupported_arguments_type: {type(raw).__name__}"


def parse_tool_calls(raw_tool_calls: Any) -> tuple[ToolCall, ...] | None:
    """Turn a ``message.tool_calls`` field into structured ``ToolCall`` values.

    ``None`` means the field was absent (content-only reply). An empty list
    becomes an empty tuple. Missing call ids become ``call_<index>``. Bad
    entry shapes raise ``ProviderResponseError``; argument problems stay on
    ``FunctionCall.parse_error`` so the probe can score them without guessing.
    """
    if raw_tool_calls is None:
        return None
    if not isinstance(raw_tool_calls, list):
        raise ProviderResponseError(
            f"tool_calls must be a list or null (got {type(raw_tool_calls).__name__})",
            reason="malformed_tool_calls",
        )
    parsed: list[ToolCall] = []
    for index, entry in enumerate(raw_tool_calls):
        if not isinstance(entry, dict):
            raise ProviderResponseError(
                f"tool_calls[{index}] must be an object (got {type(entry).__name__})",
                reason="malformed_tool_calls",
            )
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        fn_raw = entry.get("function")
        if not isinstance(fn_raw, dict):
            raise ProviderResponseError(
                f"tool_calls[{index}].function must be an object",
                reason="malformed_tool_calls",
            )
        name = fn_raw.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderResponseError(
                f"tool_calls[{index}].function.name must be a non-empty string",
                reason="malformed_tool_calls",
            )
        arguments, arguments_raw, parse_error = parse_function_arguments(fn_raw.get("arguments"))
        parsed.append(
            ToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=arguments,
                    arguments_raw=arguments_raw,
                    parse_error=parse_error,
                ),
            )
        )
    return tuple(parsed)


def parse_usage(raw: Any) -> TokenUsage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None

    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    return TokenUsage(
        prompt_tokens=_int_or_none(raw.get("prompt_tokens")),
        completion_tokens=_int_or_none(raw.get("completion_tokens")),
        total_tokens=_int_or_none(raw.get("total_tokens")),
    )


def parse_chat_completion(
    payload: Mapping[str, Any],
    *,
    latency_ms: float,
    fallback_model: str,
) -> ChatResponse:
    """Parse an OpenAI-shaped chat-completions JSON body into ``ChatResponse``."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderResponseError("choices[0] must be an object")
    message_raw = choice.get("message")
    if not isinstance(message_raw, dict):
        raise ProviderResponseError("choices[0].message must be an object")
    role = message_raw.get("role", "assistant")
    if role != "assistant":
        raise ProviderResponseError(f"expected assistant message (got role={role!r})")
    content = message_raw.get("content")
    if content is not None and not isinstance(content, str):
        raise ProviderResponseError(
            f"message.content must be a string or null (got {type(content).__name__})"
        )
    reasoning_content = message_raw.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ProviderResponseError(
            "message.reasoning_content must be a string or null "
            f"(got {type(reasoning_content).__name__})"
        )
    if isinstance(reasoning_content, str) and reasoning_content == "":
        reasoning_content = None
    tool_calls = parse_tool_calls(message_raw.get("tool_calls"))
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        model = fallback_model
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        ),
        finish_reason=finish_reason,
        usage=parse_usage(payload.get("usage")),
        model=model,
        latency_ms=latency_ms,
        raw=dict(payload),
    )


class OpenAICompatibleProvider:
    """Async httpx client for a local OpenAI-compatible ``/v1`` server.

    Optional ``api_key`` is sent as a Bearer token when set (hosted path later;
    local llama-server needs none). The key never appears in ``repr``, logs, or
    raised error messages.
    """

    name: str = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        temperature: float | None = 0.0,
        seed: int | None = 0,
        enable_thinking: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0 (got {timeout_s})")
        self.base_url = (base_url if base_url is not None else base_url_from_environ()).rstrip("/")
        self.model = model if model is not None else model_from_environ()
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.enable_thinking = enable_thinking
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
        )

    def __repr__(self) -> str:
        key_state = "set" if self._api_key else "unset"
        return (
            f"OpenAICompatibleProvider(name={self.name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout_s={self.timeout_s!r}, "
            f"temperature={self.temperature!r}, seed={self.seed!r}, "
            f"enable_thinking={self.enable_thinking!r}, api_key={key_state})"
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        model = request.model or self.model
        temperature = self.temperature if request.temperature is None else request.temperature
        seed = self.seed if request.seed is None else request.seed
        enable_thinking = request.enable_thinking
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            entry: dict[str, Any] = {"role": message.role}
            if message.content is not None:
                entry["content"] = message.content
            if message.name is not None:
                entry["name"] = message.name
            if message.tool_call_id is not None:
                entry["tool_call_id"] = message.tool_call_id
            if message.tool_calls is not None:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments_raw
                            if call.function.arguments_raw
                            else json.dumps(call.function.arguments or {}),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(entry)
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools is not None:
            body["tools"] = list(request.tools)
        # Qwen3 chat template: non-thinking mode for the probe / live runs.
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, _REDACTED)
        return text

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """POST ``/chat/completions`` and return a structured ``ChatResponse``."""
        body = self._build_body(request)
        url = "/chat/completions"
        started = time.perf_counter()
        try:
            response = await self._client.post(url, json=body, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LocalServerUnavailableError(
                self._redact(f"timeout talking to {self.base_url}{url}: {exc}")
            ) from exc
        except httpx.RequestError as exc:
            raise LocalServerUnavailableError(
                self._redact(f"cannot reach {self.base_url}{url}: {exc}")
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            detail = self._redact(response.text[:500])
            raise ProviderHTTPError(
                f"HTTP {response.status_code} from {self.base_url}{url}: {detail}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(self._redact(f"response is not JSON: {exc.msg}")) from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"response JSON must be an object (got {type(payload).__name__})"
            )
        try:
            return parse_chat_completion(
                payload,
                latency_ms=latency_ms,
                fallback_model=request.model or self.model,
            )
        except ProviderError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ProviderResponseError(self._redact(f"cannot parse response: {exc}")) from exc
