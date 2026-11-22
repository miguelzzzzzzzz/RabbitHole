"""`CitationVerifier` (M3): valid, fresh and grounded citations, plugged into the loop."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from agentic_research.agent import (
    AgentLoop,
    Budgets,
    Citation,
    Claim,
    FinalAnswer,
    GiveUp,
    Observation,
    ObservationShaper,
    StepRecord,
    Task,
    ToolCall,
)
from agentic_research.agent.evidence import (
    DROPPED_NO_OBSERVATION,
    DROPPED_TRUNCATED,
    EvidenceStore,
    misconfigured_evidence_tools,
)
from agentic_research.agent.loop import EVIDENCE_DROPPED_TRUNCATED, EVIDENCE_UNREADABLE
from agentic_research.agent.policy import Decision
from agentic_research.agent.state import RunState
from agentic_research.backends import FixtureSearchBackend, Passage, SearchBackend, StoredDocument
from agentic_research.backends.base import BackendError, make_doc_id
from agentic_research.backends.rag_engine import RagEngineBackend
from agentic_research.citations import (
    INVALID_CITATION,
    NO_CITATION,
    STALE_SOURCE,
    UNGROUNDED_CITATION,
    CitationIssue,
    CitationVerifier,
    check_citations,
    feedback,
)
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.probe import probe_registry
from agentic_research.tools import ToolExecutor, ToolRegistry, ToolResult
from agentic_research.tools.base import (
    ErrorClass,
    RetryPolicy,
    ToolArgs,
    ToolContext,
    ToolOutput,
)
from agentic_research.tools.builtin import default_registry, research_registry
from agentic_research.tools.read import MAX_READ_CHARS
from agentic_research.tools.search import SearchCorpusOutput
from agentic_research.trace.schema import RunSpan, StepSpan, Trace
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID, build_index

VELMORA_DOC_ID = "7610eae52aac886e"
VELMORA_SOURCE = "bridges/velmora-bridge.md"
ILSE_SOURCE = "people/ilse-tarkand.md"
TASK = Task(question="Who designed the Velmora Bridge?")
DESIGNER = "The bridge was designed by the engineer Ilse Tarkand."
SHOWN = 200  # the setup run reads [0, SHOWN) of the Velmora document


def _fixture() -> FixtureSearchBackend:
    return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)


def _cite(
    doc: StoredDocument, start: int, end: int, *, quote: str | None = None, **extra: str
) -> Citation:
    text = doc.text[start:end] if quote is None else quote
    return Citation(doc_id=doc.doc_id, start=start, end=end, quote=text, **extra)


def _answer(*citations: Citation, text: str = "Ilse Tarkand") -> FinalAnswer:
    return FinalAnswer(answer=text, claims=(Claim(text=text, citations=citations),))


def _spans(trace: Trace) -> list[StepSpan]:
    return [s for s in trace.spans if isinstance(s, StepSpan)]


async def _read_state(backend: SearchBackend, start: int = 0, end: int = SHOWN) -> RunState:
    """A finished run that read ``[start, end)`` of the Velmora document in full."""
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(
                    tool="read_document",
                    args={"doc_id": VELMORA_DOC_ID, "start": start, "end": end},
                ),
                GiveUp(reason="done"),
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=3),
        run_id="r-read",
    )
    state = await loop.run(TASK)
    assert state.observations[0].truncated_chars == 0
    return state


# --- table-driven checks ---------------------------------------------------------


def _cases() -> list[tuple[str, object, str | None]]:
    """(case id, citation factory over (velmora, ilse), expected verdict reason)."""
    return [
        ("valid", lambda v, i: _cite(v, 2, 30), None),
        ("valid_with_hash", lambda v, i: _cite(v, 2, 30, content_hash=v.content_hash), None),
        ("whole_shown_text", lambda v, i: _cite(v, 0, SHOWN), None),
        # Off by one: the same quote one character later or longer no longer matches.
        ("start_off_by_one", lambda v, i: _cite(v, 3, 31, quote=v.text[2:30]), INVALID_CITATION),
        ("end_off_by_one", lambda v, i: _cite(v, 2, 31, quote=v.text[2:30]), INVALID_CITATION),
        (
            "quote_not_stored_text",
            lambda v, i: _cite(v, 2, 30, quote=v.text[2:30].upper()),
            INVALID_CITATION,
        ),
        (
            "past_end_of_document",
            lambda v, i: _cite(v, len(v.text) - 2, len(v.text) + 3, quote="x" * 5),
            INVALID_CITATION,
        ),
        (
            "unknown_doc_id",
            lambda v, i: Citation(doc_id="0" * 16, start=0, end=4, quote="text"),
            INVALID_CITATION,
        ),
        ("stale_hash", lambda v, i: _cite(v, 2, 30, content_hash="f" * 64), STALE_SOURCE),
        # Freshness is checked before validity: a stale citation is stale, whatever its quote.
        (
            "stale_and_wrong_quote",
            lambda v, i: _cite(v, 2, 30, quote="nonsense", content_hash="f" * 64),
            STALE_SOURCE,
        ),
        # Valid text the run never saw.
        ("unobserved_document", lambda v, i: _cite(i, 0, 14), UNGROUNDED_CITATION),
        ("outside_shown_range", lambda v, i: _cite(v, 300, 330), UNGROUNDED_CITATION),
        ("straddles_shown_end", lambda v, i: _cite(v, SHOWN - 5, SHOWN + 5), UNGROUNDED_CITATION),
    ]


@pytest.mark.parametrize(("case", "make", "reason"), _cases(), ids=[c[0] for c in _cases()])
async def test_citation_checks(case: str, make: object, reason: str | None) -> None:
    backend = _fixture()
    state = await _read_state(backend)
    velmora = backend.document(VELMORA_DOC_ID)
    ilse = backend.document(make_doc_id(FIXTURE_CORPUS_ID, ILSE_SOURCE))
    assert callable(make)
    citation = make(velmora, ilse)
    assert isinstance(citation, Citation)
    verdict = await CitationVerifier(backend).verify(_answer(citation), state)
    if reason is None:
        assert verdict.accepted, verdict.message
        return
    assert not verdict.accepted
    assert verdict.reason == reason
    assert "claims[0].citations[0]: " + reason in verdict.message


async def test_no_claims_and_uncited_claims_are_no_citation() -> None:
    backend = _fixture()
    state = await _read_state(backend)
    verifier = CitationVerifier(backend)
    bare = await verifier.verify(FinalAnswer(answer="Ilse Tarkand"), state)
    assert (bare.accepted, bare.reason) == (False, NO_CITATION)
    assert "answer: no_citation" in bare.message
    velmora = backend.document(VELMORA_DOC_ID)
    answer = FinalAnswer(
        answer="Ilse Tarkand, in 1911",
        claims=(
            Claim(text="completed in 1911", citations=(_cite(velmora, 2, 30),)),
            Claim(text="designed by Ilse Tarkand"),
        ),
    )
    partly = await verifier.verify(answer, state)
    assert (partly.accepted, partly.reason) == (False, NO_CITATION)
    assert "claims[1]: no_citation" in partly.message
    assert "claims[0]" not in partly.message  # the cited claim is fine


async def test_every_issue_is_listed_by_position_and_the_first_is_the_reason() -> None:
    backend = _fixture()
    state = await _read_state(backend)
    velmora = backend.document(VELMORA_DOC_ID)
    answer = FinalAnswer(
        answer="x",
        claims=(
            Claim(text="a", citations=(_cite(velmora, 2, 30), _cite(velmora, 300, 330))),
            Claim(text="b", citations=(_cite(velmora, 2, 30, content_hash="f" * 64),)),
        ),
    )
    issues = check_citations(answer, backend, EvidenceStore.from_state(state))
    assert [(i.where, i.code) for i in issues] == [
        ("claims[0].citations[1]", UNGROUNDED_CITATION),
        ("claims[1].citations[0]", STALE_SOURCE),
    ]
    verdict = await CitationVerifier(backend).verify(answer, state)
    assert verdict.reason == UNGROUNDED_CITATION
    lines = verdict.message.splitlines()
    assert lines[0].startswith("2 citation problems.")
    assert lines[1].startswith("claims[0].citations[1]: ungrounded_citation: [300, 330) of ")
    assert lines[1].endswith(f"(shown in full: [0, {SHOWN}))")
    assert lines[2].startswith("claims[1].citations[0]: stale_source: ")


async def test_feedback_names_offsets_never_document_text() -> None:
    # Rejection feedback is shown to the policy; it must not leak unseen text.
    backend = _fixture()
    state = await _read_state(backend)
    velmora = backend.document(VELMORA_DOC_ID)
    ilse = backend.document(make_doc_id(FIXTURE_CORPUS_ID, ILSE_SOURCE))
    answer = FinalAnswer(
        answer="x",
        claims=(
            Claim(
                text="a",
                citations=(
                    _cite(velmora, 300, 340),
                    _cite(velmora, 300, 340, quote="z" * 40),
                    _cite(ilse, 0, 40),
                ),
            ),
        ),
    )
    verdict = await CitationVerifier(backend).verify(answer, state)
    assert verdict.reason == UNGROUNDED_CITATION
    assert velmora.text[300:340] not in verdict.message
    assert ilse.text[0:40] not in verdict.message
    assert "first difference at quote offset 0" in verdict.message
    assert "no text of this document version was shown in full" in verdict.message


def test_feedback_caps_the_list() -> None:
    issues = tuple(
        CitationIssue(code=INVALID_CITATION, claim=0, citation=n, message="bad") for n in range(4)
    )
    text = feedback(issues, max_listed=2)
    assert text.splitlines()[1:] == [
        "claims[0].citations[0]: invalid_citation: bad",
        "claims[0].citations[1]: invalid_citation: bad",
        "+2 more",
    ]
    one = feedback(issues[:1])
    assert one.startswith("1 citation problem.")
    with pytest.raises(ValueError, match="at least one"):
        feedback(())


def test_observed_ranges_in_the_message_are_capped() -> None:
    backend = _fixture()
    velmora = backend.document(VELMORA_DOC_ID)
    steps: list[StepRecord] = []
    observations: list[Observation] = []
    shaper = ObservationShaper()
    for n in range(7):  # seven disjoint 5-char reads
        result = ToolResult(
            tool="read_document",
            output={
                "doc_id": velmora.doc_id,
                "source": velmora.source,
                "content_hash": velmora.content_hash,
                "start_char": 10 * n,
                "end_char": 10 * n + 5,
                "doc_length": len(velmora.text),
                "text": velmora.text[10 * n : 10 * n + 5],
            },
        )
        step = StepRecord(index=n, action=ToolCall(tool="read_document"), result=result)
        obs = shaper.observe(step)
        assert obs is not None
        steps.append(step)
        observations.append(obs)
    store = EvidenceStore.from_steps(steps, observations)
    (issue,) = check_citations(_answer(_cite(velmora, 100, 110)), backend, store)
    assert issue.message.endswith("[30, 35), [40, 45), +2 more)")


async def test_a_tampered_document_fails_the_original_citation() -> None:
    # The run read the original text; the stored copy changed afterwards.
    original = _fixture()
    state = await _read_state(original)
    velmora = original.document(VELMORA_DOC_ID)
    sources = {p.relative_to(FIXTURE_CORPUS).as_posix(): p for p in FIXTURE_CORPUS.rglob("*.md")}
    texts = {s: p.read_text(encoding="utf-8") for s, p in sources.items()}
    texts[VELMORA_SOURCE] = texts[VELMORA_SOURCE].replace("steel arch", "stone arch")
    tampered = FixtureSearchBackend(FIXTURE_CORPUS_ID, texts)
    at = velmora.text.index("steel arch")
    verifier = CitationVerifier(tampered)
    # With the version it was quoted from: that version is not the stored one.
    pinned = await verifier.verify(
        _answer(_cite(velmora, at, at + 10, content_hash=velmora.content_hash)), state
    )
    assert pinned.reason == STALE_SOURCE
    # Without a hash, the quote no longer matches the stored text at its offsets.
    unpinned = await verifier.verify(_answer(_cite(velmora, at, at + 10)), state)
    assert unpinned.reason == INVALID_CITATION
    # Unchanged text at the same offsets was seen only in the old version: not grounded.
    same = await verifier.verify(_answer(_cite(velmora, 2, 12)), state)
    assert same.reason == UNGROUNDED_CITATION


# --- in the loop -----------------------------------------------------------------


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


@pytest.fixture(params=["fixture", "rag_engine", "rag_engine_hybrid"])
def backend(request: pytest.FixtureRequest, index_dir: Path) -> SearchBackend:
    if request.param == "fixture":
        return _fixture()
    if request.param == "rag_engine_hybrid":
        return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="hybrid")
    return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID)


def _designer_citation(backend: SearchBackend) -> Citation:
    doc = backend.document(VELMORA_DOC_ID)
    at = doc.text.index(DESIGNER)
    return _cite(doc, at, at + len(DESIGNER), content_hash=doc.content_hash)


async def test_oracle_search_read_answer_is_accepted(backend: SearchBackend) -> None:
    # PLAN M3 acceptance: search, read, final_answer over 01's index; the verifier accepts.
    citation = _designer_citation(backend)
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(tool="search_corpus", args={"query": "Velmora Bridge designed", "k": 3}),
                ToolCall(
                    tool="read_document",
                    args={"doc_id": VELMORA_DOC_ID, "start": citation.start, "end": citation.end},
                ),
                _answer(citation),
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=4, max_repairs=0),
        verifier=CitationVerifier(backend),
        run_id="r-oracle",
    )
    traced = await loop.run_traced(TASK)
    state = traced.state
    assert state.terminal_label == "succeeded"
    assert state.answer == "Ilse Tarkand"
    verdict = state.steps[2].verdict
    assert verdict is not None and verdict.accepted
    assert state.counters.repairs == 0


async def test_an_unread_citation_is_repaired_by_reading_it() -> None:
    backend = _fixture()
    citation = _designer_citation(backend)
    answer = _answer(citation)
    read = ToolCall(
        tool="read_document",
        args={"doc_id": VELMORA_DOC_ID, "start": citation.start, "end": citation.end},
    )
    loop = AgentLoop(
        OraclePolicy([answer, read, answer]),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=4, max_repairs=1),
        verifier=CitationVerifier(backend),
        run_id="r-repair",
    )
    traced = await loop.run_traced(TASK)
    state = traced.state
    assert state.terminal_label == "succeeded"
    assert state.counters.repairs == 1
    rejected = state.steps[0].verdict
    assert rejected is not None and rejected.reason == UNGROUNDED_CITATION
    feedback_obs = state.observations[0]
    assert (feedback_obs.source, feedback_obs.step) == ("verifier", 0)
    assert feedback_obs.text.startswith("answer rejected ungrounded_citation: 1 citation problem.")
    events = [(e.name, e.attributes) for e in _spans(traced.trace)[0].events]
    assert events == [("answer_rejected", {"reason": UNGROUNDED_CITATION})]


async def test_omitted_passages_of_a_cut_search_do_not_ground(backend: SearchBackend) -> None:
    # ADR-0004: the marker locates omitted passages, but citing one is ungrounded.
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(
                    tool="search_corpus",
                    args={"query": "Velmora Bridge designed engineer", "k": 10},
                ),
                GiveUp(reason="done"),
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=3),
        run_id="r-cut",
    )
    traced = await loop.run_traced(TASK)
    state = traced.state
    result = state.steps[0].result
    assert result is not None
    passages = SearchCorpusOutput.model_validate(result.output).passages
    shown = state.observations[0].shown_passages
    assert shown is not None and 0 < shown < len(passages)
    store = EvidenceStore.from_state(state)

    def cite(p: Passage) -> FinalAnswer:
        return _answer(
            Citation(
                doc_id=p.doc_id,
                start=p.start_char,
                end=p.end_char,
                quote=p.text,
                content_hash=p.content_hash,
            )
        )

    verifier = CitationVerifier(backend)
    for p in passages[:shown]:
        assert (await verifier.verify(cite(p), state)).accepted
    unseen = [
        p
        for p in passages[shown:]
        if not store.covers(p.doc_id, p.content_hash, p.start_char, p.end_char)
    ]
    assert unseen
    for p in unseen:
        verdict = await verifier.verify(cite(p), state)
        assert verdict.reason == UNGROUNDED_CITATION
    # The trace says the search dropped passages.
    events = [(e.name, e.attributes) for e in _spans(traced.trace)[0].events]
    assert events == [
        (
            EVIDENCE_DROPPED_TRUNCATED,
            {
                "tool": "search_corpus",
                "shown": shown,
                "dropped": len(passages) - shown,
                "reason": DROPPED_TRUNCATED,
            },
        )
    ]


async def test_a_shown_then_elided_span_still_grounds() -> None:
    backend = _fixture()
    citation = _designer_citation(backend)
    calc = [ToolCall(tool="calculator", args={"expression": f"{n} + 1"}) for n in range(3)]
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(
                    tool="read_document",
                    args={"doc_id": VELMORA_DOC_ID, "start": citation.start, "end": citation.end},
                ),
                *calc,
                _answer(citation),
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=6, max_repairs=0),
        verifier=CitationVerifier(backend),
        shaper=ObservationShaper(keep_recent=3),
        run_id="r-elided",
    )
    state = await loop.run(TASK)
    assert state.observations[0].elided  # the read is now a one-line summary
    assert DESIGNER not in state.observations[0].text
    assert state.terminal_label == "succeeded"


class _AnswersNonsense:
    """Searches a nonsense query, then answers anyway: first with no citation,
    then citing text the run was not shown (it never gives up)."""

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp | Decision:
        if not state.steps:
            return ToolCall(tool="search_corpus", args={"query": "zyxxyq qwvvz", "k": 5})
        if len(state.steps) == 1:
            return FinalAnswer(answer="42")
        doc = _fixture().document(VELMORA_DOC_ID)
        store = EvidenceStore.from_state(state)
        start = next(  # the first 10 characters of the document nobody was shown
            s
            for s in range(len(doc.text) - 10)
            if not any(
                lo < s + 10 and s < hi for lo, hi in store.observed(doc.doc_id, doc.content_hash)
            )
        )
        return _answer(_cite(doc, start, start + 10), text="42")


async def test_answering_a_nonsense_hybrid_query_is_rejected_not_abstained(
    index_dir: Path,
) -> None:
    # ADR-0004: hybrid search returns k neighbours for nonsense; an answer instead
    # of a GiveUp is rejected, and running out of repairs is not an abstention.
    hybrid = RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="hybrid")
    loop = AgentLoop(
        _AnswersNonsense(),
        ToolExecutor(research_registry(hybrid)),
        Budgets(max_steps=6, max_repairs=1),
        verifier=CitationVerifier(hybrid),
        run_id="r-nonsense-answer",
    )
    state = await loop.run(TASK)
    result = state.steps[0].result
    assert result is not None
    assert len(SearchCorpusOutput.model_validate(result.output).passages) == 5
    reasons = [s.verdict.reason for s in state.steps[1:] if s.verdict is not None]
    assert reasons == [NO_CITATION, UNGROUNDED_CITATION]
    assert state.terminal_label == "budget_exhausted:repairs"
    assert state.answer is None


# --- trace: evidence dropped by truncation (Chad's spot-check item 3) ---------------


async def test_a_truncated_read_is_a_dropped_span_in_the_trace() -> None:
    docs = FixtureSearchBackend("c", {"quotes.md": '"' * 1600, "plain.md": "plain text " * 20})
    quotes_id, plain_id = make_doc_id("c", "quotes.md"), make_doc_id("c", "plain.md")
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(
                    tool="read_document",
                    args={"doc_id": quotes_id, "start": 0, "end": MAX_READ_CHARS},
                ),
                ToolCall(tool="read_document", args={"doc_id": plain_id, "start": 0, "end": 50}),
                GiveUp(reason="done"),
            ]
        ),
        ToolExecutor(research_registry(docs)),
        Budgets(max_steps=4),
        run_id="r-trunc",
    )
    traced = await loop.run_traced(TASK)
    first, second, _ = _spans(traced.trace)
    assert [(e.name, e.attributes) for e in first.events] == [
        (
            EVIDENCE_DROPPED_TRUNCATED,
            {"tool": "read_document", "shown": 0, "dropped": 1, "reason": DROPPED_TRUNCATED},
        )
    ]
    assert second.events == []  # shown in full: nothing dropped, no event


class _NoToolObservations(ObservationShaper):
    """A shaper that shows the policy nothing about tool results."""

    def observe(self, step: StepRecord) -> Observation | None:
        return None if step.result is not None else super().observe(step)


async def test_a_result_without_an_observation_is_dropped_and_traced() -> None:
    # Chad's item 3: a step with a result but no observation is not silently ignored.
    backend = _fixture()
    citation = _designer_citation(backend)
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(tool="search_corpus", args={"query": "Velmora Bridge designed", "k": 2}),
                ToolCall(
                    tool="read_document",
                    args={"doc_id": VELMORA_DOC_ID, "start": citation.start, "end": citation.end},
                ),
                _answer(citation),
                GiveUp(reason="unverifiable"),
            ]
        ),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=5, max_repairs=1),
        verifier=CitationVerifier(backend),
        shaper=_NoToolObservations(),
        run_id="r-no-obs",
    )
    traced = await loop.run_traced(TASK)
    state = traced.state
    search = state.steps[0].result
    assert search is not None
    n = len(SearchCorpusOutput.model_validate(search.output).passages)
    assert n == 2
    step0, step1, step2, _ = _spans(traced.trace)
    assert [(e.name, e.attributes) for e in step0.events] == [
        (
            EVIDENCE_DROPPED_TRUNCATED,
            {"tool": "search_corpus", "shown": 0, "dropped": n, "reason": DROPPED_NO_OBSERVATION},
        )
    ]
    assert [(e.name, e.attributes) for e in step1.events] == [
        (
            EVIDENCE_DROPPED_TRUNCATED,
            {"tool": "read_document", "shown": 0, "dropped": 1, "reason": DROPPED_NO_OBSERVATION},
        )
    ]
    assert EvidenceStore.from_state(state).dropped == {0: n, 1: 1}
    # Nothing was shown, so nothing grounds the answer.
    assert [(e.name, e.attributes) for e in step2.events] == [
        ("answer_rejected", {"reason": UNGROUNDED_CITATION})
    ]
    assert state.terminal_label == "abstained:unverifiable"


# --- verifier failure paths ----------------------------------------------------------


class _BlockingDocuments:
    """Search works; ``document`` blocks until released (or fails, if told to)."""

    corpus_id = FIXTURE_CORPUS_ID

    def __init__(self, *, fail: bool = False) -> None:
        self.release = threading.Event()
        self.fail = fail
        self._inner = _fixture()

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        return self._inner.search(query, k, filters)

    def document(self, doc_id: str) -> StoredDocument:
        if self.fail:
            raise BackendError("documents.jsonl unreadable")
        self.release.wait(5)
        return self._inner.document(doc_id)


async def test_a_slow_backend_is_bounded_by_the_step_timeout() -> None:
    slow = _BlockingDocuments()
    citation = _designer_citation(_fixture())
    loop = AgentLoop(
        OraclePolicy([_answer(citation)]),
        ToolExecutor(research_registry(_fixture())),
        Budgets(max_steps=3),
        verifier=CitationVerifier(slow),
        step_timeout_s=0.05,
        run_id="r-slow-verifier",
    )
    try:
        state = await loop.run(TASK)
    finally:
        slow.release.set()
    assert state.terminal_label == "failed:verifier_timeout"


async def test_a_backend_failure_is_a_verifier_error() -> None:
    citation = _designer_citation(_fixture())
    loop = AgentLoop(
        OraclePolicy([_answer(citation)]),
        ToolExecutor(research_registry(_fixture())),
        Budgets(max_steps=3),
        verifier=CitationVerifier(_BlockingDocuments(fail=True)),
        run_id="r-broken-verifier",
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "failed:verifier_error"


class _NotADocument(ToolOutput):
    value: int


class _ForeignRead:
    """A tool registered as ``read_document`` whose output is something else."""

    name = "read_document"
    description = "not the real read_document"
    args_model: type[ToolArgs] = ToolArgs
    output_model: type[ToolOutput] = _NotADocument
    timeout_s = 5.0
    idempotent = True

    retry = RetryPolicy()

    async def run(self, args: ToolArgs, ctx: ToolContext) -> _NotADocument:
        return _NotADocument(value=42)


class _WiderSearchOutput(SearchCorpusOutput):
    """A subclass with an extra field: its results do not fit `SearchCorpusOutput`."""

    note: str = "extra"


class _ForeignSearch(_ForeignRead):
    name = "search_corpus"
    output_model: type[ToolOutput] = _WiderSearchOutput


@pytest.mark.parametrize(
    ("tool", "expected", "got"),
    [
        (_ForeignRead(), "ReadDocumentOutput", "_NotADocument"),
        (_ForeignSearch(), "SearchCorpusOutput", "_WiderSearchOutput"),
    ],
)
def test_the_loop_refuses_a_foreign_tool_under_an_evidence_name(
    tool: _ForeignRead, expected: str, got: str
) -> None:
    # Chad's spot-check of 4444938: refuse the registry up front, so a run whose
    # evidence cannot be read never ends `abstained` because the policy gave up.
    with pytest.raises(ValueError, match="AgentLoop refuses the tool registry") as info:
        AgentLoop(
            OraclePolicy([GiveUp(reason="nothing_found")]),
            ToolExecutor(ToolRegistry([tool])),
            Budgets(max_steps=3),
            run_id="r-foreign",
        )
    assert f"tool {tool.name!r} must return {expected}" in str(info.value)
    assert f"got {got}" in str(info.value)


def test_misconfigured_evidence_tools_lists_only_wrong_evidence_tools() -> None:
    assert misconfigured_evidence_tools(research_registry(_fixture())) == []
    assert misconfigured_evidence_tools(default_registry()) == []
    # The frozen M1 probe registers a schema-only `search_corpus` stub, so its
    # registry can never drive the agent loop.
    [problem] = misconfigured_evidence_tools(probe_registry())
    assert problem.startswith("tool 'search_corpus' must return SearchCorpusOutput")

    class _Nameless:
        output_model = _NotADocument

    class _NoModel:
        name = "read_document"
        output_model = None

    assert misconfigured_evidence_tools([_Nameless()]) == []
    assert misconfigured_evidence_tools([_NoModel()]) == [
        "tool 'read_document' must return ReadDocumentOutput (its results are evidence), got None"
    ]


async def test_a_foreign_tool_registered_after_the_loop_fails_the_run() -> None:
    registry = ToolRegistry()
    loop = AgentLoop(
        OraclePolicy([ToolCall(tool="read_document"), GiveUp(reason="nothing_found")]),
        ToolExecutor(registry),
        Budgets(max_steps=3),
        verifier=CitationVerifier(_fixture()),
        run_id="r-foreign-late",
    )
    registry.register(_ForeignRead())  # after the loop checked the registry
    traced = await loop.run_traced(TASK)
    state = traced.state
    first = _spans(traced.trace)[0]
    assert [(e.name, e.attributes) for e in first.events] == [
        (EVIDENCE_UNREADABLE, {"tool": "read_document"})
    ]
    # Fail closed at once: the policy's GiveUp is never reached, so the run is
    # not `abstained`.
    assert len(state.steps) == 1
    assert state.terminal_label == "failed:evidence_unreadable"
    run = next(s for s in traced.trace.spans if isinstance(s, RunSpan))
    assert run.error is not None
    assert run.error.error_class == ErrorClass.PERMANENT
    assert run.error.reason == "evidence_unreadable"
    assert "'read_document'" in run.error.message
