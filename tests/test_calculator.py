from __future__ import annotations

import ast
import asyncio
import math
import warnings
from typing import Any

import jsonschema
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_research.tools import (
    ErrorClass,
    Tool,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolRegistry,
)
from agentic_research.tools.calculator import (
    ALLOWED_FUNCTIONS,
    MAX_DEPTH,
    MAX_EXPRESSION_CHARS,
    MAX_INT_BITS,
    MAX_NODES,
    CalculatorArgs,
    CalculatorError,
    CalculatorOutput,
    CalculatorTool,
    evaluate,
)

# Static check (mypy runs on tests): the calculator satisfies the Tool protocol.
_CALC: Tool[CalculatorArgs, CalculatorOutput] = CalculatorTool()

CTX = ToolContext(run_id="run-1", step=0)


def _reason(expression: str) -> str:
    with pytest.raises(CalculatorError) as info:
        evaluate(expression)
    return info.value.reason


# --- correctness on hand-computed cases --------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3),
        ("2 * (3 + 4)", 14),
        ("7 - 10", -3),
        ("10 / 4", 2.5),
        ("8 / 2", 4.0),  # true division always gives a float, as in Python
        ("7 // 2", 3),
        ("-7 // 2", -4),
        ("-7 % 3", 2),
        ("7.5 % 2", 1.5),
        ("2 ** 10", 1024),
        ("-2 ** 2", -4),  # ** binds tighter than unary minus
        ("(-2) ** 2", 4),
        ("2 ** -2", 0.25),
        ("2 ** 3 ** 2", 512),  # right-associative
        ("+5", 5),
        ("--5", 5),
        ("1e3 + 0.5", 1000.5),
        ("1_000 * 3", 3000),
        ("0x10 + 0b11", 19),
        ("pi", math.pi),
        ("e", math.e),
        ("abs(-3.5)", 3.5),
        ("round(2.5)", 2),  # banker's rounding, as in Python
        ("round(3.14159, 2)", 3.14),
        ("round(1234, -2)", 1200),
        ("min(3, 1, 2)", 1),
        ("max(3, 1.5)", 3),
        ("sqrt(16)", 4.0),
        ("exp(0)", 1.0),
        ("log(e)", 1.0),
        ("log(8, 2)", 3.0),
        ("log10(1000)", 3.0),
        ("log2(1024)", 10.0),
        ("sin(0)", 0.0),
        ("cos(0)", 1.0),
        ("tan(0)", 0.0),
        ("floor(-2.5)", -3),
        ("ceil(2.1)", 3),
        ("  (1 + 2) * 3  ", 9),
        ("2 ** 999", 2**999),  # 1000 bits: the largest accepted power of two
        ("-(2 ** 999)", -(2**999)),
    ],
)
def test_hand_computed_results(expression: str, expected: float) -> None:
    result = evaluate(expression)
    assert result == expected
    assert type(result) is type(expected)


def test_function_list_is_the_documented_allowlist() -> None:
    assert ALLOWED_FUNCTIONS == (
        "abs",
        "round",
        "min",
        "max",
        "sqrt",
        "exp",
        "log",
        "log10",
        "log2",
        "sin",
        "cos",
        "tan",
        "floor",
        "ceil",
    )


# --- table-driven rejections -------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        # names and attributes
        ("x + 1", "unknown_name"),
        ("__builtins__", "unknown_name"),
        ("math.pi", "disallowed_syntax"),
        ("().__class__", "disallowed_syntax"),
        ("(1).real", "disallowed_syntax"),
        # calls outside the allowlist, or in a disallowed form
        ("__import__('os')", "unknown_function"),
        ("eval('1')", "unknown_function"),
        ("exec('1')", "unknown_function"),
        ("open('f')", "unknown_function"),
        ("pow(2, 3)", "unknown_function"),
        ("factorial(5)", "unknown_function"),
        ("math.sqrt(4)", "disallowed_syntax"),
        ("(lambda: 1)()", "disallowed_syntax"),
        ("sqrt(x=4)", "disallowed_syntax"),
        ("max(*[1, 2])", "disallowed_syntax"),
        ("round(1.5, 1.0)", "bad_function_args"),
        ("round(1.5, 101)", "bad_function_args"),
        ("sqrt()", "bad_function_args"),
        ("sqrt(1, 2)", "bad_function_args"),
        ("log(1, 2, 3)", "bad_function_args"),
        ("min(1)", "bad_function_args"),
        # huge exponents and results
        ("2 ** 1000", "exponent_too_large"),
        ("10 ** 10 ** 10", "exponent_too_large"),
        ("(-3) ** 100000", "exponent_too_large"),
        ("2 ** 999 * 2", "result_too_large"),
        ("2 ** 999 + 2 ** 999", "result_too_large"),
        ("2.0 ** 10000", "result_too_large"),
        ("exp(1000)", "result_too_large"),
        ("1e308 * 10", "non_finite_result"),
        ("1e999", "non_finite_result"),
        ("9" * 302, "result_too_large"),
        # lambdas, comprehensions, and other non-arithmetic syntax
        ("lambda: 1", "disallowed_syntax"),
        ("[x for x in (1, 2)]", "disallowed_syntax"),
        ("{x for x in (1, 2)}", "disallowed_syntax"),
        ("{x: 1 for x in (1, 2)}", "disallowed_syntax"),
        ("sum(x for x in (1, 2))", "unknown_function"),
        ("(x := 1)", "disallowed_syntax"),
        ("1 if 1 else 2", "disallowed_syntax"),
        ("1 < 2", "disallowed_syntax"),
        ("1 and 2", "disallowed_syntax"),
        ("not 1", "disallowed_syntax"),
        ("~1", "disallowed_syntax"),
        ("1 << 2", "disallowed_syntax"),
        ("1 & 3", "disallowed_syntax"),
        ("1 | 3", "disallowed_syntax"),
        ("1 ^ 3", "disallowed_syntax"),
        ("(1, 2)", "disallowed_syntax"),
        ("[1, 2]", "disallowed_syntax"),
        ("{1: 2}", "disallowed_syntax"),
        ("[1][0]", "disallowed_syntax"),
        ("'abc'", "disallowed_syntax"),
        ("f'{1}'", "disallowed_syntax"),
        ("b'1'", "disallowed_syntax"),
        ("True + 1", "disallowed_syntax"),
        ("None", "disallowed_syntax"),
        ("...", "disallowed_syntax"),
        ("1j", "disallowed_syntax"),
        ("(-8) ** 0.5", "complex_result"),
        ("await x", "disallowed_syntax"),
        # arithmetic errors
        ("1 / 0", "division_by_zero"),
        ("1 // 0", "division_by_zero"),
        ("1 % 0", "division_by_zero"),
        ("0 ** -1", "division_by_zero"),
        ("log(8, 1)", "division_by_zero"),
        ("sqrt(-1)", "math_domain_error"),
        ("log(0)", "math_domain_error"),
        # malformed or oversized input
        ("", "syntax_error"),
        ("   ", "syntax_error"),
        ("1 +", "syntax_error"),
        ("import os", "syntax_error"),
        ("1; 2", "syntax_error"),
        ("1\x002", "syntax_error"),
        ("1" * (MAX_EXPRESSION_CHARS + 1), "too_long"),
        ("max(" + ",".join(["1"] * MAX_NODES) + ")", "too_complex"),
        ("-" * MAX_DEPTH + "1", "too_complex"),
    ],
)
def test_disallowed_and_failing_expressions_are_rejected(expression: str, reason: str) -> None:
    assert _reason(expression) == reason


def test_depth_and_node_limits_are_inclusive() -> None:
    assert evaluate("-" * (MAX_DEPTH - 1) + "1") == (-1 if (MAX_DEPTH - 1) % 2 else 1)
    # One Call node plus MAX_NODES - 1 argument nodes (the callee name is not counted).
    assert evaluate("max(" + ",".join(["1"] * (MAX_NODES - 1)) + ")") == 1
    # A left-nested chain of n additions is n + 1 levels deep.
    assert evaluate("+".join(["1"] * MAX_DEPTH)) == MAX_DEPTH
    assert _reason("+".join(["1"] * (MAX_DEPTH + 1))) == "too_complex"


@pytest.mark.parametrize(
    ("expression", "reason"),
    [("1 is 1", "disallowed_syntax"), ("'\\d'", "disallowed_syntax")],
)
def test_parser_syntax_warnings_are_silenced(expression: str, reason: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any leaked warning would fail the test
        assert _reason(expression) == reason


@pytest.mark.parametrize("exc", [ValueError("x"), RecursionError(), MemoryError()])
def test_parser_failures_become_syntax_errors(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    def failing_parse(*args: object, **kwargs: object) -> object:
        raise exc

    monkeypatch.setattr(ast, "parse", failing_parse)
    assert _reason("1 + 1") == "syntax_error"


def test_power_guard_refuses_before_computing() -> None:
    # Would be a ~3.3e9-bit integer; refused by the pre-check, not by computing it.
    assert _reason("10 ** 1000000000") == "exponent_too_large"
    # Bases 0, 1, -1 and negative exponents stay cheap and are allowed.
    assert evaluate("1 ** 1000000000") == 1
    assert evaluate("(-1) ** 1000000001") == -1
    assert evaluate("0 ** 1000000000") == 0
    assert evaluate("2 ** -1000000") == 0.0


def test_max_int_bits_bound_on_literals() -> None:
    assert evaluate(str(2**MAX_INT_BITS - 1)) == 2**MAX_INT_BITS - 1
    assert _reason(str(2**MAX_INT_BITS)) == "result_too_large"


def test_error_messages_are_specific() -> None:
    with pytest.raises(CalculatorError) as info:
        evaluate("foo(1)")
    assert "unknown function 'foo'" in info.value.message
    assert "sqrt" in info.value.message
    with pytest.raises(CalculatorError) as info:
        evaluate("max(1)")
    assert info.value.message == "max() takes 2 or more arguments, got 1"
    with pytest.raises(CalculatorError) as info:
        evaluate("log()")
    assert info.value.message == "log() takes 1-2 arguments, got 0"
    with pytest.raises(CalculatorError) as info:
        evaluate("sqrt(1, 2)")
    assert info.value.message == "sqrt() takes 1 arguments, got 2"


# --- property tests: agreement with Python arithmetic ------------------------

_INT = st.integers(min_value=-1000, max_value=1000).map(lambda n: f"({n})")
# Floats are 0 or at least 1e-3 in magnitude: subnormal bases would overflow
# under a negative exponent (``5e-324 ** -1``), which is a size rejection, not
# an agreement case (covered by the table above).
_FLOAT = (
    st.floats(min_value=-1000, max_value=1000, allow_nan=False)
    .filter(lambda x: x == 0 or abs(x) >= 1e-3)
    .map(lambda x: f"({x!r})")
)
# ** only on a literal base with a small literal exponent keeps every
# intermediate far inside the int and float bounds (at most 12 leaves of
# magnitude 1e-9..1e9), so any rejection other than division by zero is a bug.
_POW = st.tuples(st.one_of(_INT, _FLOAT), st.integers(min_value=-3, max_value=3)).map(
    lambda pair: f"({pair[0]} ** ({pair[1]}))"
)
_LEAF = st.one_of(_INT, _FLOAT, _POW)
_OPS = st.sampled_from(["+", "-", "*", "/", "//", "%"])


def _combine(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    binary = st.tuples(children, _OPS, children).map(lambda t: f"({t[0]} {t[1]} {t[2]})")
    unary = children.map(lambda c: f"(-{c})")
    return st.one_of(binary, unary)


_EXPRESSIONS = st.recursive(_LEAF, _combine, max_leaves=12)


def _python(expression: str) -> Any:
    # Reference semantics: CPython itself, on generated arithmetic-only strings.
    return eval(expression, {"__builtins__": {}}, {})


@settings(max_examples=400, deadline=None)
@given(_EXPRESSIONS)
def test_agrees_with_python_arithmetic(expression: str) -> None:
    try:
        expected = _python(expression)
    except ZeroDivisionError:
        assert _reason(expression) == "division_by_zero"
        return
    result = evaluate(expression)
    assert type(result) is type(expected)
    assert result == expected


def _assert_number_or_rejection(expression: str) -> None:
    try:
        result = evaluate(expression)
    except CalculatorError as exc:
        assert exc.reason
        return
    assert isinstance(result, int | float)
    assert not isinstance(result, bool)


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet="0123456789+-*/%().e ,abcdgilmnoqrstx_[]{}:'\"", max_size=60))
def test_arbitrary_input_never_crashes(expression: str) -> None:
    _assert_number_or_rejection(expression)


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=80))
def test_arbitrary_unicode_never_crashes(expression: str) -> None:
    _assert_number_or_rejection(expression)


# --- the tool, through the registry ------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool()])


def test_schema_export_is_valid_and_documents_limits(registry: ToolRegistry) -> None:
    (schema,) = registry.openai_schemas()
    assert schema["function"]["name"] == "calculator"
    params = schema["function"]["parameters"]
    jsonschema.Draft202012Validator.check_schema(params)
    assert params["required"] == ["expression"]
    assert params["additionalProperties"] is False
    expression = params["properties"]["expression"]
    assert expression["maxLength"] == MAX_EXPRESSION_CHARS
    assert "sqrt" in expression["description"]


def test_tool_evaluates_through_the_registry(registry: ToolRegistry) -> None:
    args = registry.validate_args("calculator", '{"expression": "(3.5 + 2) * 4"}')
    assert isinstance(args, CalculatorArgs)
    output = asyncio.run(registry.get("calculator").run(args, CTX))
    validated = registry.validate_output("calculator", output)
    assert isinstance(validated, CalculatorOutput)
    assert validated.value == 22.0
    assert validated.model_dump(mode="json") == {"value": 22.0}


def test_large_int_output_round_trips_exactly(registry: ToolRegistry) -> None:
    args = CalculatorArgs(expression="3 ** 600")
    output = asyncio.run(CalculatorTool().run(args, CTX))
    validated = registry.validate_output("calculator", output)
    assert isinstance(validated, CalculatorOutput)
    assert validated.value == 3**600
    assert isinstance(validated.value, int)


@pytest.mark.parametrize(
    ("expression", "reason"),
    [("1 / 0", "division_by_zero"), ("__import__('os')", "unknown_function")],
)
def test_tool_reports_rejections_as_invalid_args(expression: str, reason: str) -> None:
    with pytest.raises(ToolFailure) as info:
        asyncio.run(CalculatorTool().run(CalculatorArgs(expression=expression), CTX))
    error = info.value.error
    assert error.error_class is ErrorClass.INVALID_ARGS
    assert error.reason == reason
    assert not error.retryable
    assert info.value.__cause__ is None


@pytest.mark.parametrize(
    "arguments",
    [
        "{}",
        '{"expression": ""}',
        '{"expression": 12}',
        '{"expression": "1", "mode": "fast"}',
        '{"expression": "' + "1" * (MAX_EXPRESSION_CHARS + 1) + '"}',
    ],
)
def test_bad_arguments_are_rejected_before_running(registry: ToolRegistry, arguments: str) -> None:
    result = registry.validate_args("calculator", arguments)
    assert isinstance(result, ToolError)
    assert result.error_class is ErrorClass.INVALID_ARGS
    assert result.reason == "schema_violation"


def test_tool_attributes() -> None:
    tool = CalculatorTool()
    assert tool.retry.max_attempts == 1
    assert tool.idempotent is True
    assert tool.timeout_s == 1.0
