"""Tool registry: registration, OpenAI-format schema export, and validation.

Validation never raises for bad model input. A malformed call (unknown tool,
arguments that are not a JSON object, schema violations) becomes a
``ToolError`` with class ``invalid_args``; a tool output that violates the
tool's output model becomes ``bad_output``. Programming errors (registering
an invalid tool, a duplicate name) do raise, because they are bugs in our
code rather than model behaviour.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from agentic_research.tools.base import (
    ErrorClass,
    RetryPolicy,
    Tool,
    ToolArgs,
    ToolError,
    ToolOutput,
)

#: OpenAI function names: 1-64 characters of letters, digits, ``_`` and ``-``.
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Upper bound on one error message shown to the model.
MAX_ERROR_MESSAGE_CHARS = 1000

AnyTool = Tool[Any, Any]


class ToolRegistrationError(ValueError):
    """A tool definition is invalid or its name is already registered."""


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(item) for item in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= MAX_ERROR_MESSAGE_CHARS:
        return text
    return text[: MAX_ERROR_MESSAGE_CHARS - 3] + "..."


class ToolRegistry:
    """An ordered set of tools, keyed by unique name."""

    def __init__(self, tools: Iterable[AnyTool] = ()) -> None:
        self._tools: dict[str, AnyTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AnyTool) -> None:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not TOOL_NAME_PATTERN.fullmatch(name):
            raise ToolRegistrationError(
                f"tool name {name!r} must match {TOOL_NAME_PATTERN.pattern}"
            )
        if name in self._tools:
            raise ToolRegistrationError(f"duplicate tool name: {name!r}")
        description = getattr(tool, "description", None)
        if not isinstance(description, str) or not description.strip():
            raise ToolRegistrationError(f"tool {name!r} needs a non-empty description")
        args_model = getattr(tool, "args_model", None)
        if not (isinstance(args_model, type) and issubclass(args_model, ToolArgs)):
            raise ToolRegistrationError(f"tool {name!r}: args_model must subclass ToolArgs")
        if args_model.model_config.get("extra") != "forbid":
            raise ToolRegistrationError(
                f"tool {name!r}: args_model must forbid extra keys (extra='forbid')"
            )
        output_model = getattr(tool, "output_model", None)
        if not (isinstance(output_model, type) and issubclass(output_model, ToolOutput)):
            raise ToolRegistrationError(f"tool {name!r}: output_model must subclass ToolOutput")
        timeout_s = getattr(tool, "timeout_s", None)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ToolRegistrationError(
                f"tool {name!r}: timeout_s must be a finite number > 0, got {timeout_s!r}"
            )
        retry = getattr(tool, "retry", None)
        if not isinstance(retry, RetryPolicy):
            raise ToolRegistrationError(f"tool {name!r}: retry must be a RetryPolicy")
        idempotent = getattr(tool, "idempotent", None)
        if not isinstance(idempotent, bool):
            raise ToolRegistrationError(
                f"tool {name!r}: idempotent must be a bool, got {idempotent!r}"
            )
        if ErrorClass.TIMEOUT in retry.retry_on and not idempotent:
            raise ToolRegistrationError(
                f"tool {name!r}: only an idempotent tool may retry on timeout; "
                "set idempotent = True or drop timeout from retry.retry_on"
            )
        run = getattr(tool, "run", None)
        if not callable(run):
            raise ToolRegistrationError(f"tool {name!r} has no run() method")
        if not inspect.iscoroutinefunction(run):
            raise ToolRegistrationError(f"tool {name!r}: run() must be async")
        self._tools[name] = tool

    def get(self, name: str) -> AnyTool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[AnyTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in the OpenAI chat-completions ``tools`` format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def validate_args(
        self, name: str, arguments: Mapping[str, Any] | str | None
    ) -> ToolArgs | ToolError:
        """Parse a model's tool call into the tool's argument model.

        ``arguments`` may be a mapping or a JSON string (native ``tool_calls``
        carry arguments as a string). ``None`` and an empty string mean no
        arguments.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(self._tools) or "<none>"
            return ToolError(
                error_class=ErrorClass.INVALID_ARGS,
                reason="unknown_tool",
                message=_truncate(f"unknown tool {name!r}; available tools: {known}"),
            )
        raw: object
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            raw = {}
        elif isinstance(arguments, str):
            try:
                raw = json.loads(arguments, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                return ToolError(
                    error_class=ErrorClass.INVALID_ARGS,
                    reason="invalid_json",
                    message=f"arguments for {name!r} are not valid JSON: {exc.msg}",
                )
            except _DuplicateKeyError as exc:
                return ToolError(
                    error_class=ErrorClass.INVALID_ARGS,
                    reason="duplicate_key",
                    message=_truncate(f"arguments for {name!r} repeat the key {exc.key!r}"),
                )
            except RecursionError:
                return ToolError(
                    error_class=ErrorClass.INVALID_ARGS,
                    reason="invalid_json",
                    message=f"arguments for {name!r} are nested too deeply",
                )
        else:
            raw = arguments
        if not isinstance(raw, Mapping):
            return ToolError(
                error_class=ErrorClass.INVALID_ARGS,
                reason="not_an_object",
                message=f"arguments for {name!r} must be a JSON object, got {type(raw).__name__}",
            )
        try:
            return tool.args_model.model_validate(dict(raw))
        except ValidationError as exc:
            return ToolError(
                error_class=ErrorClass.INVALID_ARGS,
                reason="schema_violation",
                message=_truncate(
                    f"invalid arguments for {name!r}: {_format_validation_error(exc)}"
                ),
            )

    def validate_output(self, name: str, output: object) -> ToolOutput | ToolError:
        """Check a tool's return value against its output model.

        ``name`` must be a registered tool (``KeyError`` otherwise): the
        executor only calls this after a successful ``validate_args``.

        The value is re-validated from its data even when it is already an
        instance of the output model, because pydantic models can be built
        without validation (``model_construct``).
        """
        tool = self.get(name)
        output_model = tool.output_model
        data: object
        if isinstance(output, output_model):
            # warnings=False: a model_construct()-built instance may hold bad types;
            # validation below reports them as bad_output instead.
            data = output.model_dump(warnings=False)
        elif isinstance(output, BaseModel):
            return ToolError(
                error_class=ErrorClass.BAD_OUTPUT,
                reason="wrong_output_type",
                message=(
                    f"tool {name!r} returned {type(output).__name__}, "
                    f"expected {output_model.__name__}"
                ),
            )
        else:
            data = output
        if not isinstance(data, Mapping):
            return ToolError(
                error_class=ErrorClass.BAD_OUTPUT,
                reason="wrong_output_type",
                message=f"tool {name!r} returned {type(data).__name__}, expected an object",
            )
        try:
            return output_model.model_validate(dict(data))
        except ValidationError as exc:
            return ToolError(
                error_class=ErrorClass.BAD_OUTPUT,
                reason="schema_violation",
                message=_truncate(
                    f"tool {name!r} returned invalid output: {_format_validation_error(exc)}"
                ),
            )
