"""RunState, Action, Budgets and the Policy protocol."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from agentic_research.agent import (
    Action,
    BudgetCounters,
    BudgetKind,
    Budgets,
    Evidence,
    FinalAnswer,
    GiveUp,
    Policy,
    RunState,
    RunStatus,
    ToolCall,
)
from agentic_research.agent.state import StepRecord, Task, Verdict
from agentic_research.providers.types import ChatMessage
from agentic_research.tools.base import ErrorClass, ToolError, ToolResult

ACTION = TypeAdapter(Action)


def _state() -> RunState:
    return RunState(task=Task(question="What is 2+2?", task_id="t1"))


# ---- actions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        ToolCall(tool="calculator", args={"expression": "2+2"}),
        FinalAnswer(answer="4", evidence_ids=("e1",)),
        GiveUp(reason="no_evidence", message="nothing found"),
    ],
)
def test_action_round_trips_to_the_same_class(action: Any) -> None:
    again = ACTION.validate_json(ACTION.dump_json(action))
    assert again == action
    assert type(again) is type(action)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "dance"},
        {"kind": "tool_call", "tool": ""},
        {"kind": "tool_call", "tool": "x", "extra": 1},
        {"kind": "final_answer", "answer": ""},
        {"kind": "give_up", "reason": "Not A Code"},
        {"tool": "x"},
    ],
)
def test_bad_actions_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ACTION.validate_python(payload)


def test_actions_are_immutable() -> None:
    with pytest.raises(ValidationError):
        ToolCall(tool="x").tool = "y"


# ---- budgets ----------------------------------------------------------------


def test_unlimited_budgets_never_exhaust() -> None:
    c = BudgetCounters(steps=10**6, tokens=10**9, elapsed_s=1e9, repairs=99)
    assert Budgets().exhausted(c) is None


@pytest.mark.parametrize(
    ("budgets", "counters", "tool", "kind"),
    [
        (Budgets(max_steps=3), {"steps": 3}, None, BudgetKind.STEPS),
        (
            Budgets(max_tool_calls=2),
            {"tool_calls": 2, "per_tool": {"a": 2}},
            "b",
            BudgetKind.TOOL_CALLS,
        ),
        (
            Budgets(max_tool_calls_per_tool=2),
            {"tool_calls": 2, "per_tool": {"a": 2}},
            "a",
            BudgetKind.TOOL_CALLS_PER_TOOL,
        ),
        (Budgets(deadline_s=5.0), {"elapsed_s": 5.0}, None, BudgetKind.DEADLINE),
        (Budgets(max_tokens=100), {"tokens": 150}, None, BudgetKind.TOKENS),
        (Budgets(max_repairs=1), {"repairs": 2}, None, BudgetKind.REPAIRS),
    ],
)
def test_each_budget_kind_reports_itself(
    budgets: Budgets, counters: dict[str, Any], tool: str | None, kind: BudgetKind
) -> None:
    assert budgets.exhausted(BudgetCounters(**counters), tool) is kind


def test_just_under_every_limit_is_fine() -> None:
    b = Budgets(max_steps=3, max_tool_calls=3, deadline_s=5.0, max_tokens=10, max_repairs=1)
    c = BudgetCounters(steps=2, tool_calls=2, per_tool={"a": 2}, tokens=9, elapsed_s=4.9, repairs=1)
    assert b.exhausted(c, "a") is None


def test_per_tool_budget_only_applies_to_the_named_tool() -> None:
    b = Budgets(max_tool_calls_per_tool=1)
    c = BudgetCounters(tool_calls=1, per_tool={"a": 1})
    assert b.exhausted(c, "b") is None
    assert b.exhausted(c) is None
    assert b.exhausted(c, "a") is BudgetKind.TOOL_CALLS_PER_TOOL


def test_tool_call_budgets_only_apply_to_a_call_about_to_run() -> None:
    b = Budgets(max_tool_calls=2, max_tool_calls_per_tool=1)
    c = BudgetCounters(tool_calls=2, per_tool={"a": 2})
    assert b.exhausted(c) is None  # a final answer is still allowed
    assert b.exhausted(c, "b") is BudgetKind.TOOL_CALLS


def test_simultaneous_exhaustion_uses_a_fixed_order() -> None:
    b = Budgets(max_steps=1, max_tokens=1, deadline_s=1.0)
    c = BudgetCounters(steps=1, tokens=1, elapsed_s=1.0)
    assert b.exhausted(c) is BudgetKind.STEPS


@pytest.mark.parametrize(
    "kwargs", [{"max_steps": 0}, {"deadline_s": 0.0}, {"max_tokens": 0}, {"bogus": 1}]
)
def test_invalid_budgets_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Budgets(**kwargs)


def test_counters_count_tool_calls_per_tool() -> None:
    c = BudgetCounters()
    c.count_tool_call("a")
    c.count_tool_call("a")
    c.count_tool_call("b")
    assert (c.tool_calls, c.per_tool) == (3, {"a": 2, "b": 1})


def test_counters_reject_inconsistent_per_tool() -> None:
    with pytest.raises(ValidationError, match="add up"):
        BudgetCounters(tool_calls=2, per_tool={"a": 1})


@given(
    st.lists(st.sampled_from(["a", "b", "c"]), max_size=30),
    st.integers(1, 10),
    st.integers(1, 10),
)
def test_exhausted_matches_the_definition(calls: list[str], total: int, per_tool: int) -> None:
    c = BudgetCounters()
    for tool in calls:
        c.count_tool_call(tool)
    b = Budgets(max_tool_calls=total, max_tool_calls_per_tool=per_tool)
    got = b.exhausted(c, "a")
    if len(calls) >= total:
        assert got is BudgetKind.TOOL_CALLS
    elif calls.count("a") >= per_tool:
        assert got is BudgetKind.TOOL_CALLS_PER_TOOL
    else:
        assert got is None


# ---- run state --------------------------------------------------------------


def test_new_state_is_running_with_empty_collections() -> None:
    s = _state()
    assert (s.status, s.terminal, s.terminal_label) == (RunStatus.RUNNING, False, "running")
    assert not (s.plan or s.messages or s.evidence or s.steps)


def test_add_step_assigns_contiguous_indexes() -> None:
    s = _state()
    ok = ToolResult(tool="calculator", output={"value": 4})
    s.add_step(ToolCall(tool="calculator", args={"expression": "2+2"}), ok)
    s.add_step(FinalAnswer(answer="4"))
    assert [r.index for r in s.steps] == [0, 1]


def test_only_tool_call_steps_carry_a_result() -> None:
    err = ToolError(error_class=ErrorClass.PERMANENT, reason="nope", message="x")
    with pytest.raises(ValidationError, match="only a tool_call"):
        StepRecord(index=0, action=GiveUp(reason="x"), result=ToolResult.failure("t", err))


@pytest.mark.parametrize(
    ("status", "detail", "answer", "label"),
    [
        (RunStatus.SUCCEEDED, None, "4", "succeeded"),
        (RunStatus.ABSTAINED, None, None, "abstained"),
        (RunStatus.ABSTAINED, "no_evidence", None, "abstained:no_evidence"),
        (RunStatus.BUDGET_EXHAUSTED, BudgetKind.TOKENS, None, "budget_exhausted:tokens"),
        (RunStatus.BUDGET_EXHAUSTED, "steps", None, "budget_exhausted:steps"),
        (RunStatus.FAILED, "loop", None, "failed:loop"),
    ],
)
def test_finish_sets_terminal_label(
    status: RunStatus, detail: Any, answer: str | None, label: str
) -> None:
    s = _state()
    s.finish(status, detail=detail, answer=answer)
    assert s.terminal
    assert s.terminal_label == label


@pytest.mark.parametrize(
    ("status", "detail", "answer"),
    [
        (RunStatus.RUNNING, None, None),
        (RunStatus.SUCCEEDED, None, None),
        (RunStatus.SUCCEEDED, "x", "4"),
        (RunStatus.ABSTAINED, None, "4"),
        (RunStatus.BUDGET_EXHAUSTED, None, None),
        (RunStatus.BUDGET_EXHAUSTED, "coffee", None),
        (RunStatus.FAILED, None, None),
        (RunStatus.FAILED, "Bad Code", None),
    ],
)
def test_invalid_finish_raises_and_leaves_state_running(
    status: RunStatus, detail: Any, answer: str | None
) -> None:
    s = _state()
    with pytest.raises(ValueError):
        s.finish(status, detail=detail, answer=answer)
    assert (s.status, s.detail, s.answer) == (RunStatus.RUNNING, None, None)


def test_a_terminal_state_never_changes_again() -> None:
    s = _state()
    s.finish(RunStatus.FAILED, detail="loop")
    with pytest.raises(ValueError, match="already finished"):
        s.finish(RunStatus.ABSTAINED)
    with pytest.raises(ValueError, match="already finished"):
        s.add_step(GiveUp(reason="late"))
    assert s.terminal_label == "failed:loop"


def test_state_rejects_noncontiguous_steps() -> None:
    s = _state().to_dict()
    s["steps"] = [{"index": 1, "action": {"kind": "give_up", "reason": "x"}}]
    with pytest.raises(ValidationError, match="contiguous"):
        RunState.model_validate(s)


def test_full_state_round_trips_through_json() -> None:
    s = _state()
    s.plan = ["look it up", "compute"]
    s.messages = [ChatMessage(role="user", content="What is 2+2?")]
    s.evidence["e1"] = Evidence(id="e1", source="calculator", text="4")
    s.counters.count_tool_call("calculator")
    s.counters.steps = 2
    s.counters.tokens = 31
    s.counters.elapsed_s = 0.25
    s.add_step(
        ToolCall(tool="calculator", args={"expression": "2+2"}),
        ToolResult(tool="calculator", output={"value": 4}),
    )
    s.add_step(FinalAnswer(answer="4", evidence_ids=("e1",)))
    s.finish(RunStatus.SUCCEEDED, answer="4")
    again = RunState.from_json(s.to_json())
    assert again == s
    assert isinstance(again.steps[1].action, FinalAnswer)
    assert again.to_dict() == s.to_dict()


def test_terminal_state_survives_json_validation() -> None:
    s = _state()
    s.finish(RunStatus.BUDGET_EXHAUSTED, detail=BudgetKind.DEADLINE)
    assert RunState.from_json(s.to_json()).terminal_label == "budget_exhausted:deadline"


# ---- policy protocol --------------------------------------------------------


class _Fixed:
    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp:
        return FinalAnswer(answer=f"echo: {state.task.question}")


def test_a_plain_class_satisfies_the_policy_protocol() -> None:
    policy: Policy = _Fixed()
    action = asyncio.run(policy.next_action(_state()))
    assert action == FinalAnswer(answer="echo: What is 2+2?")


# ---- verdicts ---------------------------------------------------------------


def test_verdict_constructors_and_invariants() -> None:
    assert Verdict.accept().accepted
    rejected = Verdict.reject("bad_answer", "cite a source")
    assert (rejected.accepted, rejected.reason, rejected.message) == (
        False,
        "bad_answer",
        "cite a source",
    )
    with pytest.raises(ValidationError, match="no reason or message"):
        Verdict(accepted=True, reason="x")
    with pytest.raises(ValidationError, match="no reason or message"):
        Verdict(accepted=True, message="fine")
    with pytest.raises(ValidationError, match="needs a reason code"):
        Verdict(accepted=False)
    with pytest.raises(ValidationError):
        Verdict.reject("Not A Code")


def test_only_a_final_answer_step_carries_a_verdict() -> None:
    record = StepRecord(index=0, action=FinalAnswer(answer="x"), verdict=Verdict.accept())
    assert record.verdict == Verdict.accept()
    with pytest.raises(ValidationError, match="only a final_answer step"):
        StepRecord(index=0, action=ToolCall(tool="t"), verdict=Verdict.accept())


def test_a_state_with_a_verdict_round_trips() -> None:
    s = _state()
    s.add_step(FinalAnswer(answer="x"), verdict=Verdict.reject("bad_answer", "no"))
    again = RunState.from_json(s.to_json())
    assert again.steps[0].verdict == Verdict.reject("bad_answer", "no")
