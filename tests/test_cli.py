from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import jsonschema
import pytest

from agentic_research import __version__
from agentic_research.cli import main
from agentic_research.tools import ToolError
from agentic_research.tools.builtin import default_registry
from agentic_research.tools.registry import TOOL_NAME_PATTERN


def test_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"arp {__version__}"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: arp" in capsys.readouterr().out


def test_unknown_argument_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--no-such-flag"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# --- arp tools schema ---------------------------------------------------------


def _schema_output(capsys: pytest.CaptureFixture[str], *extra: str) -> list[dict[str, Any]]:
    assert main(["tools", "schema", *extra]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    out: list[dict[str, Any]] = json.loads(captured.out)
    return out


def test_tools_schema_prints_the_default_registry(capsys: pytest.CaptureFixture[str]) -> None:
    schemas = _schema_output(capsys)
    assert schemas == default_registry().openai_schemas()
    assert [s["function"]["name"] for s in schemas] == ["calculator"]


def test_tools_schema_is_valid_openai_format(capsys: pytest.CaptureFixture[str]) -> None:
    schemas = _schema_output(capsys)
    assert schemas
    for schema in schemas:
        assert set(schema) == {"type", "function"}
        assert schema["type"] == "function"
        function = schema["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert TOOL_NAME_PATTERN.fullmatch(function["name"])
        assert function["description"].strip()
        params = function["parameters"]
        # Valid against the JSON Schema 2020-12 meta-schema, and an object schema.
        jsonschema.Draft202012Validator.check_schema(params)
        assert params["type"] == "object"
        assert params["additionalProperties"] is False


def test_printed_calculator_schema_agrees_with_the_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    (schema,) = _schema_output(capsys, "--tool", "calculator")
    validator = jsonschema.Draft202012Validator(schema["function"]["parameters"])
    registry = default_registry()
    valid: list[dict[str, Any]] = [{"expression": "1 + 1"}, {"expression": "sqrt(2)"}]
    invalid: list[dict[str, Any]] = [{}, {"expression": 3}, {"expression": "1", "extra": True}]
    for arguments in valid:
        assert validator.is_valid(arguments)
        assert not isinstance(registry.validate_args("calculator", arguments), ToolError)
    for arguments in invalid:
        assert not validator.is_valid(arguments)
        assert isinstance(registry.validate_args("calculator", arguments), ToolError)


def test_tools_schema_unknown_tool(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools", "schema", "--tool", "nope"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown tool 'nope' (known: calculator)" in captured.err


def test_tools_without_subcommand_prints_tools_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "usage: arp tools" in out
    assert "schema" in out


def test_arp_module_entry_point_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "agentic_research.cli", "tools", "schema"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == default_registry().openai_schemas()
