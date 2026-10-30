from __future__ import annotations

import asyncio
import json
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field

from agentic_research.tools import (
    ErrorClass,
    RetryPolicy,
    Tool,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolOutput,
    ToolRegistrationError,
    ToolRegistry,
)
from agentic_research.tools.registry import MAX_ERROR_MESSAGE_CHARS


class EchoArgs(ToolArgs):
    text: str = Field(description="Text to repeat.")
    times: int = Field(default=1, ge=1, le=5)


class EchoOutput(ToolOutput):
    text: str


class EchoTool:
    name = "echo"
    description = "Repeat a text a few times."
    args_model: type[ToolArgs] = EchoArgs
    output_model: type[ToolOutput] = EchoOutput
    timeout_s = 1.0
    retry = RetryPolicy.no_retry()

    async def run(self, args: EchoArgs, ctx: ToolContext) -> EchoOutput:
        return EchoOutput(text=args.text * args.times)


class NoArgs(ToolArgs):
    pass


class ClockOutput(ToolOutput):
    now: str


class ClockTool:
    name = "clock"
    description = "Return a fixed time."
    args_model: type[ToolArgs] = NoArgs
    output_model: type[ToolOutput] = ClockOutput
    timeout_s = 2
    retry = RetryPolicy()

    async def run(self, args: NoArgs, ctx: ToolContext) -> ClockOutput:
        return ClockOutput(now="2026-10-30T00:00:00Z")


# Static check (mypy runs on tests): the sample tools satisfy the Tool protocol.
_ECHO: Tool[EchoArgs, EchoOutput] = EchoTool()
_CLOCK: Tool[NoArgs, ClockOutput] = ClockTool()


def make_tool(**overrides: Any) -> Any:
    """An EchoTool-like object with some attributes replaced (or deleted with None)."""

    class Custom(EchoTool):
        pass

    for key, value in overrides.items():
        setattr(Custom, key, value)
    return Custom()


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([_ECHO, _CLOCK])


def _error(result: object) -> ToolError:
    assert isinstance(result, ToolError), result
    return result


# --- registration -----------------------------------------------------------


def test_registry_keeps_registration_order(registry: ToolRegistry) -> None:
    assert registry.names == ["echo", "clock"]
    assert len(registry) == 2
    assert [tool.name for tool in registry] == ["echo", "clock"]
    assert "echo" in registry
    assert "missing" not in registry
    assert isinstance(registry.get("clock"), ClockTool)


def test_empty_registry() -> None:
    registry = ToolRegistry()
    assert len(registry) == 0
    assert registry.openai_schemas() == []


def test_duplicate_name_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolRegistrationError, match="duplicate tool name: 'echo'"):
        registry.register(EchoTool())
    assert len(registry) == 2


def test_get_unknown_tool_raises_key_error(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="unknown tool: 'nope'"):
        registry.get("nope")


@pytest.mark.parametrize("name", ["", "has space", "dots.not.allowed", "x" * 65, 7, None])
def test_invalid_tool_names_are_rejected(name: object) -> None:
    with pytest.raises(ToolRegistrationError, match="must match"):
        ToolRegistry([make_tool(name=name)])


def test_longest_allowed_name_is_accepted() -> None:
    registry = ToolRegistry([make_tool(name="a-b_" + "x" * 60)])
    assert registry.names == ["a-b_" + "x" * 60]


@pytest.mark.parametrize("description", ["", "   ", None])
def test_description_is_required(description: object) -> None:
    with pytest.raises(ToolRegistrationError, match="non-empty description"):
        ToolRegistry([make_tool(description=description)])


class PlainArgs(BaseModel):
    text: str


class LooseArgs(ToolArgs):
    model_config = ConfigDict(extra="ignore")
    text: str


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"args_model": PlainArgs}, "args_model must subclass ToolArgs"),
        ({"args_model": EchoArgs(text="x")}, "args_model must subclass ToolArgs"),
        ({"args_model": LooseArgs}, "must forbid extra keys"),
        ({"output_model": PlainArgs}, "output_model must subclass ToolOutput"),
        ({"output_model": None}, "output_model must subclass ToolOutput"),
        ({"timeout_s": 0}, "timeout_s must be a finite number > 0"),
        ({"timeout_s": -1.0}, "timeout_s must be a finite number > 0"),
        ({"timeout_s": float("nan")}, "timeout_s must be a finite number > 0"),
        ({"timeout_s": float("inf")}, "timeout_s must be a finite number > 0"),
        ({"timeout_s": True}, "timeout_s must be a finite number > 0"),
        ({"timeout_s": "5"}, "timeout_s must be a finite number > 0"),
        ({"retry": None}, "retry must be a RetryPolicy"),
        ({"retry": {"max_attempts": 2}}, "retry must be a RetryPolicy"),
        ({"run": None}, "has no run"),
        ({"run": lambda self, args, ctx: None}, "run\\(\\) must be async"),
    ],
)
def test_invalid_tool_definitions_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ToolRegistrationError, match=message):
        ToolRegistry([make_tool(**overrides)])


def test_failed_registration_leaves_registry_unchanged(registry: ToolRegistry) -> None:
    with pytest.raises(ToolRegistrationError):
        registry.register(make_tool(name="other", timeout_s=0))
    assert registry.names == ["echo", "clock"]


# --- schema export ----------------------------------------------------------


def test_openai_schemas_use_the_function_tool_format(registry: ToolRegistry) -> None:
    schemas = registry.openai_schemas()
    assert [s["function"]["name"] for s in schemas] == ["echo", "clock"]
    echo = schemas[0]
    assert echo["type"] == "function"
    assert echo["function"]["description"] == "Repeat a text a few times."
    params = echo["function"]["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["text"]
    assert params["additionalProperties"] is False
    assert params["properties"]["text"]["description"] == "Text to repeat."
    assert params["properties"]["times"]["maximum"] == 5
    # The export is plain JSON.
    assert json.loads(json.dumps(schemas)) == schemas


def test_exported_parameters_are_valid_json_schemas(registry: ToolRegistry) -> None:
    for schema in registry.openai_schemas():
        params = schema["function"]["parameters"]
        jsonschema.Draft202012Validator.check_schema(params)


def test_exported_schema_agrees_with_argument_validation(registry: ToolRegistry) -> None:
    params = registry.openai_schemas()[0]["function"]["parameters"]
    validator = jsonschema.Draft202012Validator(params)
    for arguments in ({"text": "a"}, {"text": "a", "times": 5}):
        assert validator.is_valid(arguments)
        assert isinstance(registry.validate_args("echo", arguments), EchoArgs)
    for arguments in ({}, {"text": "a", "times": 6}, {"text": "a", "extra": 1}):
        assert not validator.is_valid(arguments)
        assert isinstance(registry.validate_args("echo", arguments), ToolError)


# --- argument validation ----------------------------------------------------


def test_valid_mapping_arguments_parse_to_the_args_model(registry: ToolRegistry) -> None:
    args = registry.validate_args("echo", {"text": "hi", "times": 2})
    assert args == EchoArgs(text="hi", times=2)


def test_json_string_arguments_are_accepted(registry: ToolRegistry) -> None:
    assert registry.validate_args("echo", '{"text": "hi"}') == EchoArgs(text="hi", times=1)


@pytest.mark.parametrize("arguments", [None, "", "  ", {}, "{}"])
def test_missing_arguments_mean_an_empty_object(
    registry: ToolRegistry, arguments: dict[str, Any] | str | None
) -> None:
    assert registry.validate_args("clock", arguments) == NoArgs()
    error = _error(registry.validate_args("echo", arguments))
    assert error.reason == "schema_violation"
    assert "text: Field required" in error.message


def test_unknown_tool_is_an_invalid_args_observation(registry: ToolRegistry) -> None:
    error = _error(registry.validate_args("search_web", {"q": "x"}))
    assert error.error_class is ErrorClass.INVALID_ARGS
    assert error.reason == "unknown_tool"
    assert "available tools: echo, clock" in error.message
    assert not error.retryable


def test_unknown_tool_in_an_empty_registry() -> None:
    error = _error(ToolRegistry().validate_args("echo", {}))
    assert "available tools: <none>" in error.message


@pytest.mark.parametrize(
    ("arguments", "reason", "fragment"),
    [
        ('{"text": "hi"', "invalid_json", "not valid JSON"),
        ("text=hi", "invalid_json", "not valid JSON"),
        ('["hi"]', "not_an_object", "got list"),
        ('"hi"', "not_an_object", "got str"),
        ("null", "not_an_object", "got NoneType"),
        ('{"text": "a", "text": "b"}', "duplicate_key", "repeat the key 'text'"),
        ('{"text": "a", "times": {"x": 1, "x": 2}}', "duplicate_key", "repeat the key 'x'"),
        ("[" * 100_000 + "]" * 100_000, "invalid_json", "nested too deeply"),
    ],
)
def test_malformed_json_arguments(
    registry: ToolRegistry, arguments: str, reason: str, fragment: str
) -> None:
    error = _error(registry.validate_args("echo", arguments))
    assert error.error_class is ErrorClass.INVALID_ARGS
    assert error.reason == reason
    assert fragment in error.message


def test_non_mapping_python_arguments_are_rejected(registry: ToolRegistry) -> None:
    error = _error(registry.validate_args("echo", ["hi"]))  # type: ignore[arg-type]
    assert error.reason == "not_an_object"


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"text": "hi", "loud": True}, "loud: Extra inputs are not permitted"),
        ({"text": "hi", "times": 9}, "times: Input should be less than or equal to 5"),
        ({"text": "hi", "times": "many"}, "times: Input should be a valid integer"),
        ({"text": 3}, "text: Input should be a valid string"),
        ({"times": 2}, "text: Field required"),
    ],
)
def test_schema_violations_become_structured_errors(
    registry: ToolRegistry, arguments: dict[str, Any], fragment: str
) -> None:
    error = _error(registry.validate_args("echo", arguments))
    assert error.error_class is ErrorClass.INVALID_ARGS
    assert error.reason == "schema_violation"
    assert error.message.startswith("invalid arguments for 'echo': ")
    assert fragment in error.message


def test_error_messages_do_not_echo_argument_values(registry: ToolRegistry) -> None:
    error = _error(registry.validate_args("echo", {"text": "hi", "times": "secret-value"}))
    assert "secret-value" not in error.message


def test_long_error_messages_are_truncated(registry: ToolRegistry) -> None:
    arguments = {"text": "hi", **{f"k{i}_" + "x" * 50: 1 for i in range(100)}}
    error = _error(registry.validate_args("echo", arguments))
    assert len(error.message) == MAX_ERROR_MESSAGE_CHARS
    assert error.message.endswith("...")


# --- output validation ------------------------------------------------------


def test_valid_outputs_pass(registry: ToolRegistry) -> None:
    assert registry.validate_output("echo", EchoOutput(text="hi")) == EchoOutput(text="hi")
    assert registry.validate_output("echo", {"text": "hi"}) == EchoOutput(text="hi")


def test_unvalidated_instances_are_revalidated(registry: ToolRegistry) -> None:
    sneaky = EchoOutput.model_construct(text=123)
    error = _error(registry.validate_output("echo", sneaky))
    assert error.error_class is ErrorClass.BAD_OUTPUT
    assert error.reason == "schema_violation"
    assert "text: Input should be a valid string" in error.message
    assert not error.retryable


@pytest.mark.parametrize(
    ("output", "reason", "fragment"),
    [
        (ClockOutput(now="x"), "wrong_output_type", "returned ClockOutput, expected EchoOutput"),
        (["hi"], "wrong_output_type", "returned list, expected an object"),
        (None, "wrong_output_type", "returned NoneType, expected an object"),
        ({"text": "hi", "extra": 1}, "schema_violation", "extra: Extra inputs are not permitted"),
        ({}, "schema_violation", "text: Field required"),
    ],
)
def test_bad_outputs_become_bad_output_errors(
    registry: ToolRegistry, output: object, reason: str, fragment: str
) -> None:
    error = _error(registry.validate_output("echo", output))
    assert error.error_class is ErrorClass.BAD_OUTPUT
    assert error.reason == reason
    assert fragment in error.message


def test_validate_output_for_unknown_tool_is_a_programming_error(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError):
        registry.validate_output("nope", {"text": "hi"})


# --- end to end with a real tool --------------------------------------------


async def _call(registry: ToolRegistry, name: str, arguments: str) -> object:
    args = registry.validate_args(name, arguments)
    assert not isinstance(args, ToolError)
    tool = registry.get(name)
    raw = await tool.run(args, ToolContext(run_id="run-1", step=0))
    return registry.validate_output(name, raw)


def test_validated_call_runs_the_tool(registry: ToolRegistry) -> None:
    assert asyncio.run(_call(registry, "echo", '{"text": "ab", "times": 3}')) == EchoOutput(
        text="ababab"
    )
