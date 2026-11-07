"""OpenAICompatibleProvider: MockTransport unit tests + one slow live Qwen call."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from agentic_research.providers import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ChatMessage,
    ChatRequest,
    FunctionCall,
    LocalServerUnavailableError,
    OpenAICompatibleProvider,
    ProviderHTTPError,
    ProviderResponseError,
    ToolCall,
    base_url_from_environ,
    model_from_environ,
    parse_function_arguments,
    parse_tool_calls,
    parse_usage,
)
from agentic_research.providers.openai_compat import parse_chat_completion

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local_runtime.sh"
RUNTIME_DIR = Path(os.environ.get("ARP_RUNTIME_DIR", "/workspace/portfolio/runtime"))


# --- helpers ------------------------------------------------------------------


def _completion_payload(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    model: str = DEFAULT_MODEL,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _tool_call(
    *,
    call_id: str = "call_1",
    name: str = "calculator",
    arguments: Any = '{"expression":"1+1"}',
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _mock_provider(
    handler: Any,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=base_url)
    return OpenAICompatibleProvider(
        base_url=base_url,
        model=DEFAULT_MODEL,
        api_key=api_key,
        client=client,
    )


# --- pure parsers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_args", "expected_error"),
    [
        ('{"expression":"2+2"}', {"expression": "2+2"}, None),
        ({"expression": "2+2"}, {"expression": "2+2"}, None),
        ("", None, "empty_arguments"),
        ("{", None, "invalid_json"),
        ("[1,2]", None, "arguments_not_object"),
        (None, None, "missing_arguments"),
        (12, None, "unsupported_arguments_type"),
    ],
)
def test_parse_function_arguments(
    raw: Any, expected_args: dict[str, Any] | None, expected_error: str | None
) -> None:
    args, raw_str, err = parse_function_arguments(raw)
    assert args == expected_args
    if expected_error is None:
        assert err is None
        assert raw_str
    else:
        assert err is not None
        assert err.startswith(expected_error)


def test_parse_tool_calls_none_means_content_only() -> None:
    assert parse_tool_calls(None) is None


def test_parse_tool_calls_empty_list() -> None:
    assert parse_tool_calls([]) == ()


def test_parse_tool_calls_arguments_as_object_and_string() -> None:
    calls = parse_tool_calls(
        [
            _tool_call(call_id="a", arguments={"expression": "1+1"}),
            _tool_call(call_id="b", arguments='{"expression":"2+2"}'),
        ]
    )
    assert calls is not None
    assert len(calls) == 2
    assert calls[0].function.arguments == {"expression": "1+1"}
    assert calls[0].function.arguments_valid
    assert calls[1].function.arguments == {"expression": "2+2"}
    assert calls[1].function.arguments_raw == '{"expression":"2+2"}'


def test_parse_tool_calls_malformed_arguments() -> None:
    calls = parse_tool_calls([_tool_call(arguments="{not-json")])
    assert calls is not None
    assert calls[0].function.arguments is None
    assert calls[0].function.parse_error is not None
    assert calls[0].function.parse_error.startswith("invalid_json")
    assert calls[0].function.arguments_raw == "{not-json"
    assert not calls[0].function.arguments_valid


def test_parse_tool_calls_rejects_non_list() -> None:
    with pytest.raises(ProviderResponseError, match="tool_calls must be a list"):
        parse_tool_calls({"id": "x"})


# --- MockTransport happy path / tools -----------------------------------------


@pytest.mark.asyncio
async def test_complete_happy_path_captures_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_completion_payload(content="hi there"))

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
        )
    finally:
        await provider.aclose()

    assert response.message.content == "hi there"
    assert response.message.tool_calls is None
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5
    assert response.model == DEFAULT_MODEL
    assert response.latency_ms >= 0.0
    assert seen["url"].endswith("/chat/completions")
    body = seen["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["temperature"] == 0.0
    assert body["seed"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_complete_with_tools_and_tool_calls_json_string() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["tools"] == list(tools)
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[_tool_call(arguments='{"expression":"1+1"}')],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            ),
        )

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(
                messages=(ChatMessage(role="user", content="what is 1+1?"),),
                tools=tools,
            )
        )
    finally:
        await provider.aclose()

    assert response.message.content is None
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    call = response.message.tool_calls[0]
    assert call.function.name == "calculator"
    assert call.function.arguments == {"expression": "1+1"}
    assert call.function.arguments_valid
    assert response.finish_reason == "tool_calls"
    assert response.usage is not None
    assert response.usage.total_tokens == 18


@pytest.mark.asyncio
async def test_complete_tool_calls_arguments_as_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[_tool_call(arguments={"expression": "3*3"})],
                finish_reason="tool_calls",
            ),
        )

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(messages=(ChatMessage(role="user", content="3*3"),))
        )
    finally:
        await provider.aclose()

    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].function.arguments == {"expression": "3*3"}


@pytest.mark.asyncio
async def test_complete_malformed_tool_call_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[_tool_call(arguments="{bad")],
                finish_reason="tool_calls",
            ),
        )

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(messages=(ChatMessage(role="user", content="x"),))
        )
    finally:
        await provider.aclose()

    call = response.message.tool_calls[0]  # type: ignore[index]
    assert call.function.arguments is None
    assert call.function.parse_error is not None
    assert call.function.arguments_raw == "{bad"


@pytest.mark.asyncio
async def test_complete_empty_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_payload(content="no tools", tool_calls=[]),
        )

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(messages=(ChatMessage(role="user", content="x"),))
        )
    finally:
        await provider.aclose()

    assert response.message.content == "no tools"
    assert response.message.tool_calls == ()


@pytest.mark.asyncio
async def test_complete_content_only_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload(content="just text"))

    provider = _mock_provider(handler)
    try:
        response = await provider.complete(
            ChatRequest(messages=(ChatMessage(role="user", content="x"),))
        )
    finally:
        await provider.aclose()

    assert response.message.content == "just text"
    assert response.message.tool_calls is None


# --- error paths --------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_failure_is_local_server_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _mock_provider(handler)
    try:
        with pytest.raises(LocalServerUnavailableError) as exc_info:
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
        assert exc_info.value.reason == "local_server_unavailable"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_http_error_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    provider = _mock_provider(handler)
    try:
        with pytest.raises(ProviderHTTPError) as exc_info:
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
        assert exc_info.value.status_code == 503
        assert "overloaded" in exc_info.value.message
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json{")

    provider = _mock_provider(handler)
    try:
        with pytest.raises(ProviderResponseError):
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_missing_choices_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "model": DEFAULT_MODEL})

    provider = _mock_provider(handler)
    try:
        with pytest.raises(ProviderResponseError, match="no choices"):
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
    finally:
        await provider.aclose()


def test_parse_chat_completion_rejects_non_assistant_role() -> None:
    payload = _completion_payload()
    payload["choices"][0]["message"]["role"] = "user"
    with pytest.raises(ProviderResponseError, match="assistant"):
        parse_chat_completion(payload, latency_ms=1.0, fallback_model=DEFAULT_MODEL)


# --- api key never leaks ------------------------------------------------------


SENTINEL_KEY = "test-sentinel-api-key-DO-NOT-LEAK-9f3a2b"


@pytest.mark.asyncio
async def test_api_key_never_appears_in_repr_or_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"
        raise httpx.ConnectError(f"boom including {SENTINEL_KEY}", request=request)

    provider = _mock_provider(handler, api_key=SENTINEL_KEY)
    try:
        text = repr(provider)
        assert SENTINEL_KEY not in text
        assert "api_key='set'" in text or 'api_key="set"' in text or "api_key=set" in text
        with pytest.raises(LocalServerUnavailableError) as exc_info:
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
        assert SENTINEL_KEY not in str(exc_info.value)
        assert SENTINEL_KEY not in exc_info.value.message
        assert "***" in exc_info.value.message or "cannot reach" in exc_info.value.message
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_api_key_redacted_in_http_error_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {SENTINEL_KEY}")

    provider = _mock_provider(handler, api_key=SENTINEL_KEY)
    try:
        with pytest.raises(ProviderHTTPError) as exc_info:
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
        assert SENTINEL_KEY not in exc_info.value.message
        assert "***" in exc_info.value.message
    finally:
        await provider.aclose()


def test_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=())


def test_function_call_arguments_valid_property() -> None:
    ok = FunctionCall(name="calculator", arguments={"expression": "1"}, arguments_raw="{}")
    assert ok.arguments_valid
    bad = FunctionCall(
        name="calculator",
        arguments=None,
        arguments_raw="{",
        parse_error="invalid_json",
    )
    assert not bad.arguments_valid
    assert isinstance(
        ToolCall(id="c1", function=ok),
        ToolCall,
    )


# --- extra coverage -----------------------------------------------------------


def test_environ_helpers_defaults_and_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARP_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ARP_LOCAL_MODEL", raising=False)
    assert base_url_from_environ() == DEFAULT_BASE_URL
    assert model_from_environ() == DEFAULT_MODEL
    assert base_url_from_environ({"ARP_LLM_BASE_URL": " http://example.test/v1/ "}) == (
        "http://example.test/v1"
    )
    assert base_url_from_environ({"ARP_LLM_BASE_URL": "   "}) == DEFAULT_BASE_URL
    assert model_from_environ({"ARP_LOCAL_MODEL": "  "}) == DEFAULT_MODEL
    assert model_from_environ({"ARP_LOCAL_MODEL": " other "}) == "other"


def test_parse_function_arguments_unserializable_dict() -> None:
    class Boom:
        pass

    args, raw_str, err = parse_function_arguments({"x": Boom()})
    assert args is None
    assert err is not None
    assert err.startswith("unserializable_arguments")
    assert "Boom" in raw_str


def test_parse_tool_calls_bad_entries() -> None:
    with pytest.raises(ProviderResponseError, match="must be an object"):
        parse_tool_calls(["nope"])
    with pytest.raises(ProviderResponseError, match="function must be an object"):
        parse_tool_calls([{"id": "c1", "function": "x"}])
    with pytest.raises(ProviderResponseError, match="name must be a non-empty"):
        parse_tool_calls([{"id": "c1", "function": {"name": "", "arguments": "{}"}}])
    calls = parse_tool_calls([{"function": {"name": "calculator", "arguments": "{}"}}])
    assert calls is not None
    assert calls[0].id == "call_0"


def test_parse_usage_variants() -> None:
    assert parse_usage(None) is None
    assert parse_usage("nope") is None
    usage = parse_usage(
        {
            "prompt_tokens": None,
            "completion_tokens": True,
            "total_tokens": "x",
            "extra": 9,
        }
    )
    assert usage is not None
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_parse_chat_completion_shape_errors() -> None:
    with pytest.raises(ProviderResponseError, match=r"choices\[0\] must be an object"):
        parse_chat_completion(
            {"choices": ["x"], "model": DEFAULT_MODEL},
            latency_ms=1.0,
            fallback_model=DEFAULT_MODEL,
        )
    with pytest.raises(ProviderResponseError, match="message must be an object"):
        parse_chat_completion(
            {"choices": [{"message": "x"}], "model": DEFAULT_MODEL},
            latency_ms=1.0,
            fallback_model=DEFAULT_MODEL,
        )
    with pytest.raises(ProviderResponseError, match="content must be a string"):
        parse_chat_completion(
            {
                "choices": [{"message": {"role": "assistant", "content": 12}}],
                "model": DEFAULT_MODEL,
            },
            latency_ms=1.0,
            fallback_model=DEFAULT_MODEL,
        )
    response = parse_chat_completion(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": 0,
                }
            ],
            "model": "",
            "usage": None,
        },
        latency_ms=2.5,
        fallback_model="fallback-model",
    )
    assert response.model == "fallback-model"
    assert response.finish_reason == "0"
    assert response.usage is None


def test_timeout_s_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        OpenAICompatibleProvider(timeout_s=0)


@pytest.mark.asyncio
async def test_context_manager_closes_owned_client() -> None:
    provider = OpenAICompatibleProvider(base_url="http://127.0.0.1:9/v1", model=DEFAULT_MODEL)
    assert provider._owns_client is True
    async with provider as entered:
        assert entered is provider
    # Second close is harmless after owned client closed.
    await provider.aclose()


@pytest.mark.asyncio
async def test_aclose_skips_injected_client() -> None:
    closed = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload())

    transport = httpx.MockTransport(handler)

    class TrackingClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            closed["n"] += 1
            await super().aclose()

    client = TrackingClient(transport=transport, base_url=DEFAULT_BASE_URL)
    provider = OpenAICompatibleProvider(client=client, base_url=DEFAULT_BASE_URL)
    assert provider._owns_client is False
    await provider.aclose()
    assert closed["n"] == 0
    await client.aclose()
    assert closed["n"] == 1


@pytest.mark.asyncio
async def test_build_body_message_and_request_overrides() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_completion_payload())

    provider = _mock_provider(handler)
    tool = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="calculator",
            arguments={"expression": "1+1"},
            arguments_raw="",
        ),
    )
    try:
        await provider.complete(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content="sys"),
                    ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=(tool,),
                    ),
                    ChatMessage(
                        role="tool",
                        content="2",
                        name="calculator",
                        tool_call_id="call_1",
                    ),
                ),
                model="override-model",
                temperature=None,
                seed=None,
                max_tokens=32,
                enable_thinking=True,
            )
        )
    finally:
        await provider.aclose()

    body = seen["body"]
    assert body["model"] == "override-model"
    assert body["temperature"] == 0.0  # provider default when request leaves None
    assert body["seed"] == 0
    assert body["max_tokens"] == 32
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "1+1"
    }
    assert body["messages"][2]["tool_call_id"] == "call_1"
    assert body["messages"][2]["name"] == "calculator"


@pytest.mark.asyncio
async def test_timeout_exception_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = _mock_provider(handler)
    try:
        with pytest.raises(LocalServerUnavailableError, match="timeout"):
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_non_object_json_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    provider = _mock_provider(handler)
    try:
        with pytest.raises(ProviderResponseError, match="must be an object"):
            await provider.complete(ChatRequest(messages=(ChatMessage(role="user", content="x"),)))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_default_temperature_seed_none_omitted() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_completion_payload())

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=DEFAULT_BASE_URL)
    provider = OpenAICompatibleProvider(
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        temperature=None,
        seed=None,
        client=client,
    )
    try:
        await provider.complete(
            ChatRequest(
                messages=(ChatMessage(role="user", content="x"),),
                temperature=None,
                seed=None,
            )
        )
    finally:
        await provider.aclose()
    assert "temperature" not in seen["body"]
    assert "seed" not in seen["body"]


# --- slow live test -----------------------------------------------------------


def _runtime_assets_present() -> bool:
    gguf = RUNTIME_DIR / "models" / "Qwen3-1.7B-Q4_K_M.gguf"
    server = RUNTIME_DIR / "llama-b12032" / "llama-server"
    return SCRIPT.is_file() and gguf.is_file() and server.is_file()


def _wait_for_model(base_url: str, model_id: str, *, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url.rstrip('/')}/models")
                response.raise_for_status()
                payload = response.json()
            ids = [item["id"] for item in payload.get("data", [])]
            if model_id in ids:
                return
            last_error = RuntimeError(f"model {model_id!r} not in {ids!r}")
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"server at {base_url} not ready: {last_error}")


@pytest.fixture
def live_server() -> Iterator[str]:
    """Start the pinned llama-server, yield base URL, always stop it."""
    if not _runtime_assets_present():
        pytest.skip(f"runtime assets missing under {RUNTIME_DIR}")
    base_url = os.environ.get("ARP_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    # Refuse to start if something is already listening (do not fight it).
    try:
        with httpx.Client(timeout=1.0) as client:
            client.get(f"{base_url}/models")
        pytest.skip(f"something already answers at {base_url}; not starting another server")
    except (httpx.HTTPError, OSError):
        pass

    env = os.environ.copy()
    env["ARP_RUNTIME_DIR"] = str(RUNTIME_DIR)
    proc = subprocess.Popen(
        ["bash", str(SCRIPT), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    try:
        _wait_for_model(base_url, DEFAULT_MODEL)
        yield base_url
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


@pytest.mark.slow
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_live_qwen_non_thinking_content_reply(live_server: str) -> None:
    """One tiny non-thinking completion against the real local Qwen; never in CI."""
    async with OpenAICompatibleProvider(
        base_url=live_server,
        model=DEFAULT_MODEL,
        timeout_s=120.0,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
    ) as provider:
        response = await provider.complete(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="user",
                        content="Reply with exactly the word pong and nothing else.",
                    ),
                ),
                max_tokens=16,
            )
        )
    assert response.message.content is not None
    assert response.message.content.strip()
    assert "<think>" not in response.message.content
    assert response.model
    assert response.latency_ms > 0.0


# --- recorded llama-server fixtures (MINOR-7) + reasoning_content (MAJOR-2) ---


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_recorded_llama_server_content_fixture_parses() -> None:
    payload = json.loads(
        (FIXTURES / "llama_server_chat_completion_content.json").read_text(encoding="utf-8")
    )
    response = parse_chat_completion(payload, latency_ms=1.0, fallback_model=DEFAULT_MODEL)
    assert response.model == DEFAULT_MODEL
    assert response.finish_reason == "stop"
    assert response.message.content is not None
    assert response.usage is not None
    assert response.usage.total_tokens == 21


@pytest.mark.asyncio
async def test_recorded_llama_server_tool_reasoning_fixture_via_mock() -> None:
    payload = json.loads(
        (FIXTURES / "llama_server_chat_completion_tool_reasoning.json").read_text(encoding="utf-8")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider = _mock_provider(handler)
    response = await provider.complete(
        ChatRequest(messages=(ChatMessage(role="user", content="1+1"),))
    )
    await provider.aclose()
    assert response.message.reasoning_content == "I should use the calculator."
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].function.name == "calculator"
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_mock_transport_reasoning_content_shaped_like_llama_cpp() -> None:
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "need calculator",
                    "tool_calls": [_tool_call()],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    provider = _mock_provider(handler)
    response = await provider.complete(
        ChatRequest(messages=(ChatMessage(role="user", content="add"),))
    )
    await provider.aclose()
    assert response.message.reasoning_content == "need calculator"
    assert response.message.tool_calls is not None
