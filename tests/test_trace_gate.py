"""The M4 verifier gate (`agentic_research.trace.gate`) and the ``answer_accepted`` event.

Chad's 2026-11-24 spot-check: an accepted answer leaves a positive record in
the trace, and the scorer refuses traces without exactly one leading
``run_config`` whose verifier is ``citations``. Since his 2026-11-25
spot-check only the real `CitationVerifier` is recorded as ``citations``, so
every trace the gate accepts here comes from a real `AgentLoop` run with the
real verifier over a one-document fixture corpus (no model calls); the
shapes the loop never writes are built by editing such a trace, with
``Trace.model_construct`` where `Trace` validation would already refuse them.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from agentic_research.backends import FixtureSearchBackend, Passage, StoredDocument
from agentic_research.backends.base import make_doc_id
from agentic_research.citations import NO_CITATION, CitationVerifier
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolExecutor
from agentic_research.tools.builtin import research_registry
from agentic_research.trace import (
    ANSWER_ACCEPTED_EVENT,
    ANSWER_AFTER_DEADLINE_EVENT,
    CITATION_VERIFIER,
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

CORPUS_ID = "gate"
TEXT = "42 is the answer."
DOC_ID = make_doc_id(CORPUS_ID, "answer.md")
BACKEND = FixtureSearchBackend(CORPUS_ID, {"answer.md": TEXT})
TASK = Task(question="What is the answer?")
# Shows the whole one-paragraph document in full, so it grounds citations into it.
SEARCH = ToolCall(tool="search_corpus", args={"query": "answer", "k": 1})
CITATION = Citation(doc_id=DOC_ID, start=0, end=2, quote="42")
CITED = FinalAnswer(
    answer="42",
    claims=(
        Claim(
            text="It is 42.",
            citations=(CITATION, CITATION.model_copy(update={"end": 1, "quote": "4"})),
        ),
        Claim(text="Still 42.", citations=(CITATION,)),
    ),
)
WRONG = FinalAnswer(answer="41")  # no claims: the citation verifier rejects it (no_citation)


class LabelledCitations:
    """Not the citation verifier, though it declares its label: accepts everything."""

    label: ClassVar[str] = CITATION_VERIFIER

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        return Verdict.accept()


class Unlabelled(LabelledCitations):
    label: ClassVar[str] = ""


class CitationSubclass(CitationVerifier):
    """Inherits ``label = "citations"`` and the real checks, but is not the real class."""


class SlowDocuments:
    """`BACKEND`, except each ``document`` lookup moves the loop's clock 5 s on.

    Only the verifier looks documents up in a search-then-answer run, so the
    answer is checked (and accepted) past a 1 s deadline.
    """

    def __init__(self, now: list[float]) -> None:
        self.now = now

    @property
    def corpus_id(self) -> str:
        return BACKEND.corpus_id

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        return BACKEND.search(query, k, filters)

    def document(self, doc_id: str) -> StoredDocument:
        self.now[0] += 5.0
        return BACKEND.document(doc_id)


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
        ToolExecutor(research_registry(BACKEND), clock=lambda: clock[0]),
        budgets or Budgets(max_steps=4),
        verifier=CitationVerifier(BACKEND) if verifier is None else verifier,
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
    trace = await _run([SEARCH, CITED])
    step0, step1 = _steps(trace)
    assert _events(step0) == []
    assert _events(step1) == [(ANSWER_ACCEPTED_EVENT, {"claims": 2, "citations": 3})]
    assert step1.events[0].at_s == step1.end_s
    config = require_citations_verified(trace)
    assert config.citations_verified and config.corpus is None


async def test_a_repaired_answer_is_rejected_then_accepted_and_passes_the_gate() -> None:
    trace = await _run([SEARCH, WRONG, CITED])
    _, step1, step2 = _steps(trace)
    assert _events(step1) == [("answer_rejected", {"reason": NO_CITATION})]
    assert [name for name, _ in _events(step2)] == [ANSWER_ACCEPTED_EVENT]
    assert trace.run.status is RunStatus.ANSWERED
    assert require_citations_verified(trace).verifier == "citations"


@pytest.mark.parametrize(
    ("actions", "status"),
    [
        ([SEARCH, GiveUp(reason="not_in_corpus")], RunStatus.ABSTAINED),
        ([SEARCH, WRONG, WRONG], RunStatus.BUDGET_EXHAUSTED),  # out of repairs
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
        [SEARCH, CITED],
        CitationVerifier(SlowDocuments(now)),
        budgets=Budgets(max_steps=4, deadline_s=1.0),
        now=now,
    )
    assert trace.run.status is RunStatus.BUDGET_EXHAUSTED and trace.run.answer is None
    _, step = _steps(trace)
    assert [name for name, _ in _events(step)] == [
        ANSWER_ACCEPTED_EVENT,
        ANSWER_AFTER_DEADLINE_EVENT,
    ]
    require_citations_verified(trace)


# --- run_config refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    ("verifier", "label"),
    [
        (AcceptAll(), "accept-all"),
        (Unlabelled(), "custom:test_trace_gate.Unlabelled"),
        (LabelledCitations(), "custom:test_trace_gate.LabelledCitations"),
        (CitationSubclass(BACKEND), "custom:test_trace_gate.CitationSubclass"),
    ],
    ids=["accept-all", "unlabelled", "declares-citations", "citation-verifier-subclass"],
)
async def test_other_verifiers_are_refused(verifier: Any, label: str) -> None:
    # The same run the real CitationVerifier passes (test above); only the class differs.
    trace = await _run([SEARCH, CITED], verifier)
    assert trace.run.status is RunStatus.ANSWERED
    config = trace.run_config
    assert config is not None and config.verifier == label
    assert _refusal(trace) is GateRefusal.NOT_CITATIONS_VERIFIED


async def test_a_trace_without_run_config_is_refused() -> None:
    # Written before 2026-11-24: Trace reads it, the gate never scores it.
    trace = await _run([SEARCH, CITED])
    old = _edit(trace, "run", events=[])
    assert old.run_config is None
    assert _refusal(old) is GateRefusal.NO_RUN_CONFIG


async def test_shapes_trace_validation_refuses_are_refused_by_the_gate_too() -> None:
    trace = await _run([SEARCH, CITED])
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
    trace = await _run([SEARCH, CITED])
    last = _steps(trace)[-1]
    assert _refusal(_edit(trace, last.span_id, events=[])) is GateRefusal.ANSWER_NOT_ACCEPTED


async def test_acceptance_that_disagrees_with_the_run_is_refused() -> None:
    answered = await _run([SEARCH, CITED])
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

    abstained = await _run([SEARCH, GiveUp(reason="not_in_corpus")])
    last = _steps(abstained)[-1]
    in_time = accepted.model_copy(update={"at_s": last.end_s})
    too_late = late.model_copy(update={"at_s": last.end_s})
    for events in ([in_time], [in_time, too_late]):  # late only explains budget_exhausted
        edited = _edit(abstained, last.span_id, events=events)
        assert _refusal(edited) is GateRefusal.ACCEPTANCE_MISMATCH
