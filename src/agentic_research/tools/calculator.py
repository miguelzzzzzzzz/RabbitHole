"""``calculator``: a safe arithmetic evaluator over a whitelisted Python AST.

The expression is parsed with :func:`ast.parse` in ``eval`` mode and walked
node by node; nothing is ever passed to ``eval``/``exec``/``compile``.
Allowed: int and float literals, ``+ - * / // % **``, unary ``+``/``-``,
parentheses, the constants ``pi`` and ``e``, and calls to a small allowlist
of functions (``abs round min max sqrt exp log log10 log2 sin cos tan floor
ceil``) with positional arguments only. Everything else (names, attributes,
other calls, keywords, comparisons, boolean operators, lambdas,
comprehensions, subscripts, strings, complex literals, ...) is rejected.

The calculator is CPU-bound, so it cannot be cancelled by the executor's
timeout (ADR-0006); instead every input is bounded: expression length, AST
node count and depth, integer size (``MAX_INT_BITS``, checked after every
operation), and ``**`` is refused *before* computing when its integer result
would exceed that size. Results must be finite real numbers.

Every rejection is a :class:`CalculatorError` with a stable reason code; the
tool wraps it in a :class:`ToolFailure` with class ``invalid_args`` (the
model wrote an expression that cannot be evaluated and must change it).
"""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Callable
from typing import Final

from pydantic import Field

from agentic_research.tools.base import (
    ErrorClass,
    RetryPolicy,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolOutput,
)

Number = int | float

#: Longest accepted expression, in characters.
MAX_EXPRESSION_CHARS: Final = 500
#: Most AST nodes in one expression.
MAX_NODES: Final = 200
#: Deepest AST nesting.
MAX_DEPTH: Final = 100
#: Largest integer magnitude (bits) of any literal, intermediate, or result.
MAX_INT_BITS: Final = 1000
#: Largest ``|ndigits|`` accepted by ``round``.
MAX_ROUND_DIGITS: Final = 100

CONSTANTS: Final[dict[str, float]] = {"pi": math.pi, "e": math.e}


class CalculatorError(Exception):
    """An expression was rejected or could not be evaluated."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.message = message


def _round(x: Number, ndigits: Number | None = None) -> Number:
    if ndigits is None:
        return round(x)
    if not isinstance(ndigits, int):
        raise CalculatorError("bad_function_args", "round() ndigits must be an integer")
    if abs(ndigits) > MAX_ROUND_DIGITS:
        raise CalculatorError(
            "bad_function_args", f"round() ndigits must be within +/-{MAX_ROUND_DIGITS}"
        )
    return round(x, ndigits)


def _log(x: Number, base: Number | None = None) -> float:
    return math.log(x) if base is None else math.log(x, base)


# name -> (function, min positional args, max positional args or None for unbounded)
_FUNCTIONS: Final[dict[str, tuple[Callable[..., Number], int, int | None]]] = {
    "abs": (abs, 1, 1),
    "round": (_round, 1, 2),
    "min": (min, 2, None),
    "max": (max, 2, None),
    "sqrt": (math.sqrt, 1, 1),
    "exp": (math.exp, 1, 1),
    "log": (_log, 1, 2),
    "log10": (math.log10, 1, 1),
    "log2": (math.log2, 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
}

#: Names a model may call, in a stable order (shown in error messages and docs).
ALLOWED_FUNCTIONS: Final = tuple(_FUNCTIONS)

_BINARY_OPS: Final[dict[type[ast.operator], Callable[[Number, Number], Number]]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _check_number(value: object) -> Number:
    """Enforce the result invariants: a finite real int/float within bounds."""
    if isinstance(value, complex):
        raise CalculatorError("complex_result", "the result is a complex number")
    if isinstance(value, bool) or not isinstance(value, int | float):  # pragma: no cover
        raise CalculatorError("bad_result", f"unexpected result type {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("non_finite_result", "the result is infinite or not a number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculatorError("result_too_large", f"an integer exceeds {MAX_INT_BITS} bits")
    return value


def _check_pow(base: Number, exponent: Number) -> None:
    """Refuse ``base ** exponent`` before computing it when the int result is too big."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        # |base| ** exponent has at least (bit_length(|base|) - 1) * exponent + 1 bits.
        min_bits = (abs(base).bit_length() - 1) * exponent + 1
        if min_bits > MAX_INT_BITS:
            raise CalculatorError(
                "exponent_too_large", f"the power would exceed {MAX_INT_BITS} bits"
            )


class _Evaluator:
    def __init__(self) -> None:
        self.nodes = 0

    def eval(self, node: ast.AST, depth: int) -> Number:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise CalculatorError("too_complex", f"more than {MAX_NODES} syntax nodes")
        if depth > MAX_DEPTH:
            raise CalculatorError("too_complex", f"nested deeper than {MAX_DEPTH} levels")
        if isinstance(node, ast.Constant):
            return self._constant(node.value)
        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise CalculatorError("unknown_name", f"unknown name {node.id!r}; constants are: pi, e")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            operand = self.eval(node.operand, depth + 1)
            return _check_number(-operand if isinstance(node.op, ast.USub) else +operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left = self.eval(node.left, depth + 1)
            right = self.eval(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                _check_pow(left, right)
            return _check_number(_apply(_BINARY_OPS[type(node.op)], left, right))
        if isinstance(node, ast.Call):
            return self._call(node, depth)
        raise CalculatorError("disallowed_syntax", f"{type(node).__name__} is not allowed")

    @staticmethod
    def _constant(value: object) -> Number:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalculatorError(
                "disallowed_syntax", f"{type(value).__name__} literals are not allowed"
            )
        return _check_number(value)

    def _call(self, node: ast.Call, depth: int) -> Number:
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("disallowed_syntax", "only plain function names can be called")
        name = node.func.id
        spec = _FUNCTIONS.get(name)
        if spec is None:
            allowed = ", ".join(ALLOWED_FUNCTIONS)
            raise CalculatorError(
                "unknown_function", f"unknown function {name!r}; allowed: {allowed}"
            )
        if node.keywords:
            raise CalculatorError("disallowed_syntax", "keyword arguments are not allowed")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise CalculatorError("disallowed_syntax", "argument unpacking is not allowed")
        func, lo, hi = spec
        if len(node.args) < lo or (hi is not None and len(node.args) > hi):
            expected = f"{lo}" if lo == hi else f"{lo} or more" if hi is None else f"{lo}-{hi}"
            raise CalculatorError(
                "bad_function_args",
                f"{name}() takes {expected} arguments, got {len(node.args)}",
            )
        args = [self.eval(arg, depth + 1) for arg in node.args]
        return _check_number(_apply(func, *args))


def _apply(func: Callable[..., Number], *args: Number) -> Number:
    """Call ``func`` and map arithmetic exceptions to reason codes."""
    try:
        return func(*args)
    except ZeroDivisionError:
        raise CalculatorError("division_by_zero", "division or modulo by zero") from None
    except OverflowError:
        raise CalculatorError("result_too_large", "the result is too large") from None
    except ValueError as exc:
        # math domain errors, e.g. sqrt(-1) or log(0).
        raise CalculatorError("math_domain_error", str(exc) or "math domain error") from None


def evaluate(expression: str) -> Number:
    """Evaluate ``expression`` under the whitelist and bounds, or raise CalculatorError."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(
            "too_long", f"expression is longer than {MAX_EXPRESSION_CHARS} characters"
        )
    if not expression.strip():
        raise CalculatorError("syntax_error", "expression is empty")
    try:
        # The parser warns on some inputs (``1 is 1``; ``'\\d'`` is a
        # DeprecationWarning on 3.11 and a SyntaxWarning from 3.12). That is
        # noise for model-written input and, under ``-W error``, would turn into
        # a SyntaxError and change the outcome, so parse with warnings ignored.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("syntax_error", f"invalid syntax: {exc.msg}") from None
    except (ValueError, RecursionError, MemoryError):
        # e.g. null bytes, or a pathological nesting the parser itself refuses.
        raise CalculatorError("syntax_error", "the expression cannot be parsed") from None
    return _Evaluator().eval(tree.body, depth=1)


class CalculatorArgs(ToolArgs):
    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_CHARS,
        description=(
            "Arithmetic expression, e.g. '(3.5 + 2) * 4 / sqrt(2)'. Numbers, "
            "+ - * / // % **, parentheses, constants pi and e, and the functions "
            + ", ".join(ALLOWED_FUNCTIONS)
            + ". No variables."
        ),
    )


class CalculatorOutput(ToolOutput):
    value: int | float = Field(description="The exact int or float result.")


class CalculatorTool:
    """The ``calculator`` tool: evaluates one arithmetic expression."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression exactly. Use it for any computation "
        "instead of doing arithmetic yourself."
    )
    args_model: type[ToolArgs] = CalculatorArgs
    output_model: type[ToolOutput] = CalculatorOutput
    # Bounded by input limits, not cancellation (ADR-0006); 1 s is a generous ceiling.
    timeout_s = 1.0
    # Deterministic: the same expression fails the same way, so never retry.
    retry = RetryPolicy.no_retry()

    async def run(self, args: CalculatorArgs, ctx: ToolContext) -> CalculatorOutput:
        try:
            value = evaluate(args.expression)
        except CalculatorError as exc:
            raise ToolFailure(
                ToolError(
                    error_class=ErrorClass.INVALID_ARGS,
                    reason=exc.reason,
                    message=exc.message,
                )
            ) from None
        return CalculatorOutput(value=value)
