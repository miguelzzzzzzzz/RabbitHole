"""The M4 verifier gate (`agentic_research.trace.gate`) and the ``answer_accepted`` event.

Chad's 2026-11-24 spot-check: an accepted answer leaves a positive record in
the trace, and the scorer refuses traces without exactly one leading
``run_config`` whose verifier is ``citations``. Most traces here come from
real `AgentLoop` runs (no model calls); the shapes the loop never writes are
built by editing a real trace, with ``Trace.model_construct`` where `Trace`
validation would already refuse them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agentic_research.agent import (
    AgentLoop,
    Budgets,
    Citation,
    Claim,
    FinalAnswer,
    GiveUp,
    RunState,
    Task,
    ToolCall,
    Verdict,
)
from agentic_research.agent.verifier import AcceptAll
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolExecutor
from agentic_research.tools.builtin import default_registry
from agentic_research.trace import (
    ANSWER_ACCEPTED_EVENT,
    ANSWER_AFTER_DEADLINE_EVENT,
    RUN_CONFIG_EVENT,
    GateRefusal,
    RunStatus,
    Span,
    SpanEvent,
    StepSpan,
    Trace,
    TraceGateError,
    require_citations_verified,
)

TASK = Task(question="What is 6*7?")
CALC = ToolCall(tool="calculator", args={"expression": "6*7"})
CITATION = Citation(doc_id="d1", start=0, end=2, quote="42")
CITED = FinalAnswer(
    answer="42",
    claims=(
        Claim(text="It is 42.", citations=(CITATION, CITATION.model_copy(update={"end": 1}))),
        Claim(text="Still 42.", citations=(CITATION,)),
    ),
)
WRONG = FinalAnswer(answer="41")


class LabelledCitations:
    """Stands in for `CitationVerifier` (same label): rejects ``41``, accepts the rest."""

    label: ClassVar[str] = "citations"

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        if answer.answer == "41":
            return Verdict.reject("bad_answer", "try again")
        return Verdict.accept()


class Unlabelled(LabelledCitations):
    label: ClassVar[str] = ""


class LateCitations(LabelledCitations):
    """Accepts, but moves the loop's clock past the deadline while verifying."""

    def __init__(self, now: list[float]) -> None:
        self.now = now

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        self.now[0] += 5.0
        return Verdict.accept()


async def _run(
    actions: list[ToolCall | FinalAnswer | GiveUp],
    verifier: Any = None,
    *,
    budgets: Budgets | None = None,
    now: list[float] | None = None,
) -> Trace:
    clock = [0.0] if now is None else now
    loop = AgentLoop(
        OraclePolicy(actions),
        ToolExecutor(default_registry(), clock=lambda: clock[0]),
        budgets or Budgets(max_steps=4),
        verifier=LabelledCitations() if verifier is None else verifier,
        run_id="gate",
    )
    return (await loop.run_traced(TASK)).trace


def _steps(trace: Trace) -> list[StepSpan]:
    return sorted((s for s in trace.spans if isinstance(s, StepSpan)), key=lambda s: s.index)


def _events(step: StepSpan) -> list[tuple[str, dict[str, Any]]]:
    return [(e.name, dict(e.attributes)) for e in step.events]


def _edit(trace: Trace, span_id: str, *, validate: bool = True, **update: Any) -> Trace:
    spans: list[Span] = [
        s.model_copy(update=update) if s.span_id == span_id else s for s in trace.spans
    ]
    if validate:
        return Trace(header=trace.header, spans=spans)
    return Trace.model_construct(header=trace.header, spans=spans)


def _refusal(trace: Trace) -> GateRefusal:
    with pytest.raises(TraceGateError) as excinfo:
        require_citations_verified(trace)
    assert str(excinfo.value).startswith(f"{excinfo.value.reason.value}: ")
    return excinfo.value.reason


# --- answer_accepted ----------------------------------------------------------------


async def test_an_accepted_answer_is_a_step_event_with_claim_counts() -> None:
    trace = await _run([CALC, CITED])
    step0, step1 = _steps(trace)
    assert _events(step0) == []
    assert _events(step1) == [(ANSWER_ACCEPTED_EVENT, {"claims": 2, "citations": 3})]
    assert step1.events[0].at_s == step1.end_s
    config = require_citations_verified(trace)
    assert config.citations_verified and config.corpus is None


async def test_a_repaired_answer_is_rejected_then_accepted_and_passes_the_gate() -> None:
    trace = await _run([WRONG, CITED])
    step0, step1 = _steps(trace)
    assert _events(step0) == [("answer_rejected", {"reason": "bad_answer"})]
    assert [name for name, _ in _events(step1)] == [ANSWER_ACCEPTED_EVENT]
    assert trace.run.status is RunStatus.ANSWERED
    assert require_citations_verified(trace).verifier == "citations"


@pytest.mark.parametrize(
    ("actions", "status"),
    [
        ([CALC, GiveUp(reason="not_in_corpus")], RunStatus.ABSTAINED),
        ([WRONG, WRONG], RunStatus.BUDGET_EXHAUSTED),  # out of repairs
    ],
)
async def test_unanswered_citation_runs_pass_with_no_acceptance(
    actions: list[ToolCall | FinalAnswer | GiveUp], status: RunStatus
) -> None:
    # Graders score these by terminal status; nothing was accepted.
    trace = await _run(actions, budgets=Budgets(max_steps=4, max_repairs=1))
    assert trace.run.status is status
    assert all(e.name != ANSWER_ACCEPTED_EVENT for s in _steps(trace) for e in s.events)
    assert require_citations_verified(trace).citations_verified


async def test_an_answer_accepted_after_the_deadline_passes_as_not_answered() -> None:
    now = [0.0]
    trace = await _run(
        [CITED], LateCitations(now), budgets=Budgets(max_steps=4, deadline_s=1.0), now=now
    )
    assert trace.run.status is RunStatus.BUDGET_EXHAUSTED and trace.run.answer is None
    (step,) = _steps(trace)
    assert [name for name, _ in _events(step)] == [
        ANSWER_ACCEPTED_EVENT,
        ANSWER_AFTER_DEADLINE_EVENT,
    ]
    require_citations_verified(trace)


# --- run_config refusals ------------------------------------------------------------


@pytest.mark.parametrize("verifier", [AcceptAll(), Unlabelled()], ids=["accept-all", "custom"])
async def test_other_verifiers_are_refused(verifier: Any) -> None:
    trace = await _run([CITED], verifier)
    config = trace.run_config
    assert config is not None and config.verifier != "citations"
    assert _refusal(trace) is GateRefusal.NOT_CITATIONS_VERIFIED


async def test_a_trace_without_run_config_is_refused() -> None:
    # Written before 2026-11-24: Trace reads it, the gate never scores it.
    trace = await _run([CITED])
    old = _edit(trace, "run", events=[])
    assert old.run_config is None
    assert _refusal(old) is GateRefusal.NO_RUN_CONFIG


async def test_shapes_trace_validation_refuses_are_refused_by_the_gate_too() -> None:
    trace = await _run([CITED])
    (config,) = trace.run.events
    note = SpanEvent(name="note", at_s=0.0)
    garbled = config.model_copy(update={"attributes": {**config.attributes, "seed": "zero"}})
    cases = {
        GateRefusal.DUPLICATE_RUN_CONFIG: [config, config],
        GateRefusal.RUN_CONFIG_NOT_FIRST: [note, config],
        GateRefusal.INVALID_RUN_CONFIG: [garbled],
    }
    for reason, events in cases.items():
        with pytest.raises(ValueError, match=RUN_CONFIG_EVENT):
            _edit(trace, "run", events=events)  # Trace validation refuses it,
        unvalidated = _edit(trace, "run", validate=False, events=events)
        assert _refusal(unvalidated) is reason  # and so does the gate on its own


# --- acceptance refusals ------------------------------------------------------------


async def test_an_answered_run_without_answer_accepted_is_refused() -> None:
    # A trace from 2026-11-24 (run_config, but no answer_accepted yet).
    trace = await _run([CALC, CITED])
    last = _steps(trace)[-1]
    assert _refusal(_edit(trace, last.span_id, events=[])) is GateRefusal.ANSWER_NOT_ACCEPTED


async def test_acceptance_that_disagrees_with_the_run_is_refused() -> None:
    answered = await _run([CALC, CITED])
    step0, step1 = _steps(answered)
    accepted = step1.events[0]
    late = SpanEvent(name=ANSWER_AFTER_DEADLINE_EVENT, at_s=accepted.at_s)
    early = accepted.model_copy(update={"at_s": step0.end_s})

    on_earlier_step = _edit(_edit(answered, step1.span_id, events=[]), "step-0", events=[early])
    twice = _edit(answered, step1.span_id, events=[accepted, accepted])
    on_both = _edit(answered, "step-0", events=[early])
    answered_late = _edit(answered, step1.span_id, events=[accepted, late])
    for trace in (on_earlier_step, twice, on_both, answered_late):
        assert _refusal(trace) is GateRefusal.ACCEPTANCE_MISMATCH

    abstained = await _run([CALC, GiveUp(reason="not_in_corpus")])
    last = _steps(abstained)[-1]
    in_time = accepted.model_copy(update={"at_s": last.end_s})
    too_late = late.model_copy(update={"at_s": last.end_s})
    for events in ([in_time], [in_time, too_late]):  # late only explains budget_exhausted
        edited = _edit(abstained, last.span_id, events=events)
        assert _refusal(edited) is GateRefusal.ACCEPTANCE_MISMATCH
