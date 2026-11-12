"""Loop detection: canonical argument keys and the identical-call guard."""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentic_research.agent import (
    FinalAnswer,
    Guardrail,
    LoopCheck,
    LoopGuard,
    StepRecord,
    ToolCall,
    call_key,
    canonical_args,
)
from agentic_research.agent.guards import LOOP_DETECTED
from agentic_research.tools.base import ToolResult

# ---- canonical_args ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),  # key order
        ({"x": 2}, {"x": 2.0}),  # integral float
        ({"x": 0}, {"x": -0.0}),
        ({"q": "  cats  "}, {"q": "cats"}),  # surrounding whitespace
        ({"q": "caf\u00e9"}, {"q": "cafe\u0301"}),  # NFC
        ({"n": {"b": [1.0, " x"], "a": None}}, {"n": {"a": None, "b": [1, "x"]}}),
        ({"t": (1, 2)}, {"t": [1, 2]}),
    ],
)
def test_equivalent_args_have_the_same_key(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert canonical_args(left) == canonical_args(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"l": [1, 2]}, {"l": [2, 1]}),  # list order matters
        ({"x": True}, {"x": 1}),  # bool is not an int here
        ({"x": 2.5}, {"x": 2}),
        ({"q": "Cats"}, {"q": "cats"}),  # case is kept
        ({"q": "a b"}, {"q": "a  b"}),  # inner whitespace is kept
        ({"x": "1"}, {"x": 1}),
        ({}, {"x": None}),
        ({"caf\u00e9": 1}, {"cafe\u0301": 1}),  # keys are kept exactly
        ({"d": {1: "a"}}, {"d": {"1": "a"}}),  # non-string keys keep their type
    ],
)
def test_different_args_have_different_keys(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert canonical_args(left) != canonical_args(right)


def test_canonical_args_is_compact_sorted_json() -> None:
    assert canonical_args({"b": "\u00e9", "a": [1.0, 2.5]}) == '{"a":[1,2.5],"b":"\u00e9"}'


def test_non_finite_floats_are_not_turned_into_ints() -> None:
    assert canonical_args({"x": math.inf}) == '{"x":Infinity}'
    assert canonical_args({"x": math.nan}) == '{"x":NaN}'


def test_non_json_values_compare_by_type_and_repr() -> None:
    assert canonical_args({"s": {3}}) == '{"s":"<set:{3}>"}'
    assert canonical_args({"s": {3}}) != canonical_args({"s": "{3}"})


def test_mixed_key_types_do_not_break_sorting() -> None:
    assert canonical_args({"d": {1: "a", "b": 2}}) == canonical_args({"d": {"b": 2, 1: "a"}})
    assert canonical_args({"d": {1: "a"}}) == '{"d":{"int:1":"a"}}'


def test_call_key_includes_the_tool() -> None:
    assert call_key(ToolCall(tool="a", args={"x": 1})) == ("a", '{"x":1}')
    assert call_key(ToolCall(tool="a")) != call_key(ToolCall(tool="b"))


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner, max_size=3)
    ),
    max_leaves=10,
)


@given(args=st.dictionaries(st.text(max_size=4), _JSON, max_size=4))
def test_key_ignores_insertion_order(args: dict[str, Any]) -> None:
    reversed_args = dict(reversed(list(args.items())))
    assert canonical_args(args) == canonical_args(reversed_args)


# ---- LoopGuard -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1])
def test_threshold_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError, match="max_identical_calls"):
        LoopGuard(max_identical_calls=bad)


@pytest.mark.parametrize("bad", [True, 2.0, "3"])
def test_threshold_must_be_an_int(bad: Any) -> None:
    with pytest.raises(TypeError, match="max_identical_calls"):
        LoopGuard(max_identical_calls=bad)


def _ran(i: int, call: ToolCall) -> StepRecord:
    return StepRecord(index=i, action=call, result=ToolResult(tool=call.tool, output={"v": 1}))


def _warned(i: int, call: ToolCall) -> StepRecord:
    return StepRecord(index=i, action=call, guardrail=Guardrail(code=LOOP_DETECTED, message="m"))


CALL = ToolCall(tool="a", args={"x": 1})
SAME = ToolCall(tool="a", args={"x": 1.0})
OTHER = ToolCall(tool="a", args={"x": 2})


def test_ok_until_threshold_then_warn_then_stop() -> None:
    guard = LoopGuard(max_identical_calls=2)
    steps: list[StepRecord] = []
    assert guard.check(steps, CALL) is LoopCheck.OK
    steps.append(_ran(0, CALL))
    assert guard.check(steps, SAME) is LoopCheck.OK
    steps.append(_ran(1, SAME))
    assert guard.check(steps, CALL) is LoopCheck.WARN
    assert guard.check(steps, OTHER) is LoopCheck.OK
    steps.append(_warned(2, CALL))
    assert guard.check(steps, CALL) is LoopCheck.STOP
    assert guard.check(steps, OTHER) is LoopCheck.OK


def test_only_calls_that_ran_count() -> None:
    guard = LoopGuard(max_identical_calls=1)
    budget_refused = StepRecord(index=0, action=CALL)  # no result, no guardrail
    other_guardrail = StepRecord(
        index=1, action=CALL, guardrail=Guardrail(code="something_else", message="")
    )
    answer = StepRecord(index=2, action=FinalAnswer(answer="x"))
    assert guard.check([budget_refused, other_guardrail, answer], CALL) is LoopCheck.OK


def test_a_warning_for_one_call_does_not_stop_another() -> None:
    guard = LoopGuard(max_identical_calls=1)
    steps = [_ran(0, CALL), _warned(1, CALL), _ran(2, OTHER)]
    assert guard.check(steps, OTHER) is LoopCheck.WARN


def test_guard_counts_a_non_consecutive_cycle() -> None:
    guard = LoopGuard(max_identical_calls=2)
    b = ToolCall(tool="b")
    steps = [_ran(0, CALL), _ran(1, b), _ran(2, CALL), _ran(3, b)]
    assert guard.check(steps, CALL) is LoopCheck.WARN
    assert guard.check(steps, b) is LoopCheck.WARN


def test_message_names_the_tool_on_one_line_without_args() -> None:
    guard = LoopGuard(max_identical_calls=3)
    message = guard.message(ToolCall(tool="evil\nname", args={"secret": "do not echo"}))
    assert "\n" not in message
    assert "evil name" in message
    assert "3 time(s)" in message
    assert "do not echo" not in message
    long = guard.message(ToolCall(tool="t" * 200))
    assert "t" * 61 + "..." in long
    assert "t" * 62 not in long
