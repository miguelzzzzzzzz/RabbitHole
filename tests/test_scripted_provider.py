"""ScriptedProvider: canned replies through the real parser, malformed ones included."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_research.providers import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    LocalServerUnavailableError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseError,
    Script,
    ScriptedError,
    ScriptedFunctionCall,
    ScriptedProvider,
    ScriptedTurn,
    TokenUsage,
    load_script,
)

FIXTURE = Path(__file__).parent / "fixtures" / "scripts" / "two_step.json"
REQUEST = ChatRequest(messages=(ChatMessage(role="user", content="hi"),))


def test_is_a_chat_provider() -> None:
    assert isinstance(ScriptedProvider([ScriptedTurn(content="x")]), ChatProvider)


async def test_replays_turns_in_order_and_records_requests() -> None:
    provider = ScriptedProvider([ScriptedTurn(content="one"), ScriptedTurn(content="two")])
    assert provider.remaining == 2
    first = await provider.complete(REQUEST)
    second = await provider.complete(REQUEST)
    assert (first.message.content, second.message.content) == ("one", "two")
    assert first.finish_reason == "stop"
    assert first.model == "scripted"
    assert first.latency_ms == 0.0
    assert first.usage is None
    assert provider.requests == [REQUEST, REQUEST]
    assert provider.remaining == 0


async def test_running_past_the_script_raises_script_exhausted() -> None:
    provider = ScriptedProvider([ScriptedTurn(content="only")])
    await provider.complete(REQUEST)
    with pytest.raises(ProviderError) as info:
        await provider.complete(REQUEST)
    assert info.value.reason == "script_exhausted"
    assert "call 2" in info.value.message
    assert len(provider.requests) == 2


async def test_tool_call_with_object_arguments_is_sent_as_json_string() -> None:
    turn = ScriptedTurn(
        tool_calls=(ScriptedFunctionCall(name="calculator", arguments={"expression": "1+1"}),),
        usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    assert turn.payload("m")["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression":"1+1"}',
    }
    reply = await ScriptedProvider([turn]).complete(REQUEST)
    assert reply.finish_reason == "tool_calls"
    assert reply.usage == TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    (call,) = reply.message.tool_calls or ()
    assert call.id == "call_0"
    assert call.function.arguments == {"expression": "1+1"}
    assert call.function.arguments_valid


@pytest.mark.parametrize(
    ("arguments", "error_prefix"),
    [
        ('{"expression": "1+', "invalid_json"),
        ("", "empty_arguments"),
        ("[1, 2]", "arguments_not_object"),
        (None, "missing_arguments"),
    ],
)
async def test_malformed_arguments_reach_the_caller_unrepaired(
    arguments: str | None, error_prefix: str
) -> None:
    turn = ScriptedTurn(tool_calls=(ScriptedFunctionCall(name="calculator", arguments=arguments),))
    reply = await ScriptedProvider([turn]).complete(REQUEST)
    (call,) = reply.message.tool_calls or ()
    assert call.function.arguments is None
    assert call.function.parse_error is not None
    assert call.function.parse_error.startswith(error_prefix)
    assert call.function.arguments_raw == (arguments or "")


async def test_reasoning_content_and_explicit_finish_reason_pass_through() -> None:
    turn = ScriptedTurn(content="a", reasoning_content="thinking", finish_reason="length")
    reply = await ScriptedProvider([turn]).complete(REQUEST)
    assert reply.message.reasoning_content == "thinking"
    assert reply.finish_reason == "length"


async def test_request_model_overrides_the_scripted_model_name() -> None:
    provider = ScriptedProvider([ScriptedTurn(content="a")], model="fake-1")
    assert (await provider.complete(REQUEST)).model == "fake-1"
    provider = ScriptedProvider([ScriptedTurn(content="a")], model="fake-1")
    request = REQUEST.model_copy(update={"model": "other"})
    assert (await provider.complete(request)).model == "other"


async def test_raw_turn_is_parsed_verbatim() -> None:
    body = {"model": "raw-model", "choices": [{"message": {"content": "hey"}}]}
    reply = await ScriptedProvider([ScriptedTurn(raw=body)]).complete(REQUEST)
    assert reply.message.content == "hey"
    assert reply.model == "raw-model"
    assert reply.raw == body


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"choices": []}, "malformed_response"),
        ({"choices": [{"message": {"role": "user", "content": "x"}}]}, "malformed_response"),
        ({"choices": [{"message": {"tool_calls": {"not": "a list"}}}]}, "malformed_tool_calls"),
    ],
)
async def test_malformed_raw_bodies_raise_like_a_server_reply(
    body: dict[str, object], reason: str
) -> None:
    with pytest.raises(ProviderResponseError) as info:
        await ScriptedProvider([ScriptedTurn(raw=body)]).complete(REQUEST)
    assert info.value.reason == reason


async def test_error_turns_raise_typed_provider_errors() -> None:
    provider = ScriptedProvider(
        [
            ScriptedTurn(error=ScriptedError(kind="unavailable", message="down")),
            ScriptedTurn(error=ScriptedError(kind="http", status_code=503)),
            ScriptedTurn(content="recovered"),
        ]
    )
    with pytest.raises(LocalServerUnavailableError) as down:
        await provider.complete(REQUEST)
    assert down.value.message == "down"
    with pytest.raises(ProviderHTTPError) as http:
        await provider.complete(REQUEST)
    assert http.value.status_code == 503
    assert (await provider.complete(REQUEST)).message.content == "recovered"


@pytest.mark.parametrize(
    "fields",
    [
        {"content": "x", "raw": {"choices": []}},
        {"content": "x", "error": {"kind": "http"}},
        {"raw": {"choices": []}, "error": {"kind": "http"}},
        {"usage": {"total_tokens": 1}, "error": {"kind": "unavailable"}},
    ],
)
def test_a_turn_has_exactly_one_kind(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        ScriptedTurn.model_validate(fields)


def test_empty_turn_is_a_valid_empty_reply() -> None:
    message = ScriptedTurn().payload("m")["choices"][0]["message"]
    assert message == {"role": "assistant", "content": None}


def test_script_validation() -> None:
    with pytest.raises(ValidationError):
        Script.model_validate({"turns": []})
    with pytest.raises(ValidationError):
        Script.model_validate({"schema_version": 2, "turns": [{"content": "x"}]})
    with pytest.raises(ValidationError):
        Script.model_validate({"turns": [{"content": "x", "surprise": 1}]})
    with pytest.raises(ValidationError):
        ScriptedError.model_validate({"kind": "http", "status_code": 200})
    with pytest.raises(ValueError, match="model"):
        ScriptedProvider([], model="")


async def test_load_script_fixture(tmp_path: Path) -> None:
    script = load_script(FIXTURE)
    assert script.question == "What is 6*7?"
    assert len(script.turns) == 2
    provider = ScriptedProvider.from_script(script)
    first = await provider.complete(REQUEST)
    assert first.message.tool_calls is not None
    assert first.message.tool_calls[0].id == "call_a"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"turns": [{"error": {"kind": "nope"}}]}))
    with pytest.raises(ValidationError):
        load_script(bad)
