"""The per-run evidence store (M3): spans a run observed, from its recorded steps."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_research.agent import (
    AgentLoop,
    Budgets,
    FinalAnswer,
    GiveUp,
    Observation,
    ObservationShaper,
    StepRecord,
    Task,
    ToolCall,
)
from agentic_research.backends import FixtureSearchBackend, Passage, SearchBackend, StoredDocument
from agentic_research.backends.base import make_doc_id
from agentic_research.backends.rag_engine import RagEngineBackend
from agentic_research.evidence import EvidenceSpan, EvidenceStore
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolError, ToolExecutor, ToolRegistry, ToolResult
from agentic_research.tools.base import ErrorClass
from agentic_research.tools.builtin import research_registry
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.read import MAX_READ_CHARS, ReadDocumentTool
from agentic_research.tools.search import SearchCorpusOutput, SearchCorpusTool
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID, build_index

VELMORA_DOC_ID = "7610eae52aac886e"
HASH_A = "a" * 64
HASH_B = "b" * 64
DOC = "0123456789abcdef"
TASK = Task(question="Who designed the Velmora Bridge?")


def _read(doc_id: str, content_hash: str, start: int, end: int) -> ToolResult:
    return ToolResult(
        tool="read_document",
        output={
            "doc_id": doc_id,
            "source": "x.md",
            "content_hash": content_hash,
            "start_char": start,
            "end_char": end,
            "doc_length": 1000,
            "text": "x" * (end - start),
        },
    )


def _step(index: int, result: ToolResult) -> StepRecord:
    return StepRecord(index=index, action=ToolCall(tool=result.tool), result=result)


def _shown(step: StepRecord, max_chars: int = 2000) -> Observation:
    obs = ObservationShaper(max_chars=max_chars).observe(step)
    assert obs is not None
    return obs


def _store(*ranges: tuple[int, int], content_hash: str = HASH_A) -> EvidenceStore:
    store = EvidenceStore()
    for index, (start, end) in enumerate(ranges):
        step = _step(index, _read(DOC, content_hash, start, end))
        assert store.add_step(step, _shown(step)) == 1
    return store


# --- interval logic -------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranges", "merged"),
    [
        ([(10, 20)], ((10, 20),)),
        ([(10, 20), (20, 30)], ((10, 30),)),  # touching spans show contiguous text
        ([(10, 20), (15, 30)], ((10, 30),)),
        ([(10, 40), (15, 30)], ((10, 40),)),  # nested
        ([(30, 40), (10, 20)], ((10, 20), (30, 40))),  # a gap stays a gap
        ([(30, 40), (10, 20), (20, 30)], ((10, 40),)),  # a later span bridges the gap
    ],
)
def test_observed_spans_merge(
    ranges: list[tuple[int, int]], merged: tuple[tuple[int, int]]
) -> None:
    assert _store(*ranges).observed(DOC, HASH_A) == merged


@pytest.mark.parametrize(
    ("start", "end", "covered"),
    [
        (10, 20, True),
        (12, 18, True),
        (15, 25, True),  # straddles two touching spans
        (9, 20, False),  # one character before the observed text
        (10, 31, False),  # one character past it
        (35, 45, False),  # inside a gap
        (25, 35, False),  # crosses into the gap
        (12, 12, False),  # empty
        (14, 12, False),
        (-1, 5, False),
    ],
)
def test_covers(start: int, end: int, covered: bool) -> None:
    store = _store((10, 20), (20, 30), (50, 60))
    assert store.covers(DOC, HASH_A, start, end) is covered


def test_text_seen_in_one_version_does_not_ground_another() -> None:
    store = _store((0, 100))
    assert store.covers(DOC, HASH_A, 10, 20)
    assert not store.covers(DOC, HASH_B, 10, 20)
    assert not store.covers("fedcba9876543210", HASH_A, 10, 20)
    assert store.observed(DOC, HASH_B) == ()


def test_spans_keep_order_and_provenance() -> None:
    store = _store((30, 40), (10, 20))
    assert store.spans == (
        EvidenceSpan(
            step=0,
            tool="read_document",
            doc_id=DOC,
            content_hash=HASH_A,
            start_char=30,
            end_char=40,
        ),
        EvidenceSpan(
            step=1,
            tool="read_document",
            doc_id=DOC,
            content_hash=HASH_A,
            start_char=10,
            end_char=20,
        ),
    )


def test_failed_and_non_evidence_results_add_nothing() -> None:
    store = EvidenceStore()
    failed = ToolResult.failure(
        "search_corpus",
        ToolError(error_class=ErrorClass.TIMEOUT, reason="tool_timeout", message="slow"),
    )
    for step in (
        _step(0, failed),
        _step(1, ToolResult(tool="calculator", output={"value": 4})),
        StepRecord(index=2, action=ToolCall(tool="search_corpus")),  # refused, no result
    ):
        assert store.add_step(step, ObservationShaper().observe(step)) == 0
    assert store.spans == ()
    assert store.dropped == {}


def test_an_evidence_result_that_breaks_its_output_model_is_an_error() -> None:
    bad = ToolResult(tool="read_document", output={"doc_id": DOC})
    with pytest.raises(ValidationError):
        EvidenceStore().add_step(_step(0, bad), None)


def test_evidence_span_rejects_an_empty_range() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(
            step=0, tool="read_document", doc_id=DOC, content_hash=HASH_A, start_char=5, end_char=5
        )


# --- from a real run ------------------------------------------------------------


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


@pytest.fixture(params=["fixture", "rag_engine", "rag_engine_hybrid"])
def backend(request: pytest.FixtureRequest, index_dir: Path) -> SearchBackend:
    if request.param == "fixture":
        return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)
    if request.param == "rag_engine_hybrid":
        return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="hybrid")
    return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID)


async def test_evidence_from_an_oracle_run(backend: SearchBackend) -> None:
    trajectory: list[ToolCall | FinalAnswer | GiveUp] = [
        ToolCall(tool="search_corpus", args={"query": "Velmora Bridge designed engineer", "k": 3}),
        ToolCall(tool="read_document", args={"doc_id": VELMORA_DOC_ID, "start": 0, "end": 40}),
        ToolCall(tool="calculator", args={"expression": "1976 - 1911"}),
        ToolCall(tool="read_document", args={"doc_id": "0000000000000000", "start": 0, "end": 5}),
        FinalAnswer(answer="Ilse Tarkand"),
    ]
    loop = AgentLoop(
        OraclePolicy(trajectory),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=6),
        run_id="r-evidence",
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    store = EvidenceStore.from_state(state)

    first = state.steps[0].result
    assert first is not None
    passages = SearchCorpusOutput.model_validate(first.output).passages
    assert len(passages) == 3
    velmora = backend.document(VELMORA_DOC_ID)
    # One span per passage, then the read slice; calculator and the failed read add nothing.
    assert [(s.step, s.tool) for s in store.spans] == [
        (0, "search_corpus"),
        (0, "search_corpus"),
        (0, "search_corpus"),
        (1, "read_document"),
    ]
    for passage in passages:
        assert store.covers(
            passage.doc_id, passage.content_hash, passage.start_char, passage.end_char
        )
    assert store.covers(VELMORA_DOC_ID, velmora.content_hash, 0, 40)
    # The designer sentence was only seen if a passage covering it was returned.
    designer = velmora.text.index("Ilse Tarkand")
    seen = any(
        p.doc_id == VELMORA_DOC_ID and p.start_char <= designer < p.end_char for p in passages
    )
    assert seen
    assert store.covers(VELMORA_DOC_ID, velmora.content_hash, designer, designer + 12)
    assert not store.covers(VELMORA_DOC_ID, velmora.content_hash, 0, len(velmora.text))


class _SlowBackend:
    """Blocks ``search`` until released; ``finished`` is set when it returns."""

    corpus_id = FIXTURE_CORPUS_ID

    def __init__(self) -> None:
        self.release = threading.Event()
        self.finished = threading.Event()
        self._inner = FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        try:
            self.release.wait(5)
            return self._inner.search(query, k, filters)
        finally:
            self.finished.set()

    def document(self, doc_id: str) -> StoredDocument:
        return self._inner.document(doc_id)


async def test_a_timed_out_search_is_not_evidence_even_after_its_thread_returns() -> None:
    slow = _SlowBackend()
    registry = ToolRegistry(
        [CalculatorTool(), SearchCorpusTool(slow, timeout_s=0.05), ReadDocumentTool(slow)]
    )
    loop = AgentLoop(
        OraclePolicy(
            [ToolCall(tool="search_corpus", args={"query": "bridge"}), GiveUp(reason="no_time")]
        ),
        ToolExecutor(registry),
        Budgets(max_steps=3),
        run_id="r-slow",
    )
    try:
        state = await loop.run(TASK)
    finally:
        slow.release.set()
    assert slow.finished.wait(5)
    result = state.steps[0].result
    assert result is not None
    assert result.error is not None
    assert result.error.reason == "tool_timeout"
    assert EvidenceStore.from_state(state).spans == ()


async def test_a_refused_call_has_no_result_and_no_evidence(backend: SearchBackend) -> None:
    # The per-tool budget refuses the second search before it runs.
    search = ToolCall(tool="search_corpus", args={"query": "Harrow Vale"})
    loop = AgentLoop(
        OraclePolicy([search, ToolCall(tool="search_corpus", args={"query": "tram"})]),
        ToolExecutor(research_registry(backend)),
        Budgets(max_steps=5, max_tool_calls_per_tool=1),
        run_id="r-budget",
    )
    state = await loop.run(TASK)
    assert state.terminal_label.startswith("budget_exhausted")
    assert {s.step for s in EvidenceStore.from_state(state).spans} == {0}


# --- only text shown in full grounds anything (ADR-0004) ------------------------


def _search_result(*lengths: int) -> ToolResult:
    passages = [
        Passage(
            rank=rank,
            score=1.0,
            doc_id=DOC,
            chunk_id=f"{DOC}:{rank}",
            source="x.md",
            content_hash=HASH_A,
            start_char=1000 * rank,
            end_char=1000 * rank + n,
            text="t" * n,
        )
        for rank, n in enumerate(lengths, 1)
    ]
    return ToolResult.success("search_corpus", SearchCorpusOutput(corpus_id="c", passages=passages))


def test_only_the_shown_passages_of_a_cut_search_are_evidence() -> None:
    step = _step(4, _search_result(50, 600, 50))
    obs = _shown(step, max_chars=600)  # room for the first passage only
    assert obs.shown_passages == 1
    store = EvidenceStore()
    assert store.add_step(step, obs) == 1
    assert store.covers(DOC, HASH_A, 1000, 1050)
    assert not store.covers(DOC, HASH_A, 2000, 2600)  # omitted: never shown
    assert not store.covers(DOC, HASH_A, 3000, 3050)  # omitted, though it would have fit
    assert store.dropped == {4: 2}


def test_a_truncated_read_is_not_evidence() -> None:
    step = _step(2, _read(DOC, HASH_A, 0, 400))
    obs = _shown(step, max_chars=300)
    assert obs.truncated_chars > 0
    store = EvidenceStore()
    assert store.add_step(step, obs) == 0
    assert store.spans == ()
    assert store.dropped == {2: 1}


def test_a_result_without_an_observation_is_not_evidence() -> None:
    store = EvidenceStore.from_steps([_step(0, _search_result(10, 10))], [])
    assert store.spans == ()
    assert store.dropped == {0: 2}


def test_an_elided_observation_still_grounds() -> None:
    shaper = ObservationShaper(keep_recent=1)
    steps = [_step(0, _read(DOC, HASH_A, 0, 40)), _step(1, _read(DOC, HASH_A, 100, 140))]
    observations: list[Observation] = []
    for step in steps:
        obs = shaper.observe(step)
        assert obs is not None
        shaper.add(observations, obs)
    assert observations[0].elided
    store = EvidenceStore.from_steps(steps, observations)
    assert store.covers(DOC, HASH_A, 0, 40)
    assert store.covers(DOC, HASH_A, 100, 140)


def test_an_observation_of_another_step_is_refused() -> None:
    step = _step(0, _read(DOC, HASH_A, 0, 40))
    other = _shown(_step(1, _read(DOC, HASH_A, 0, 40)))
    with pytest.raises(ValueError, match="step 1 does not match step 0"):
        EvidenceStore().add_step(step, other)


def test_from_steps_pairs_each_step_with_its_own_observation() -> None:
    steps = [_step(0, _read(DOC, HASH_A, 0, 400)), _step(1, _read(DOC, HASH_A, 500, 540))]
    observations = [_shown(steps[0], max_chars=300), _shown(steps[1], max_chars=300)]
    store = EvidenceStore.from_steps(steps, observations)
    assert store.observed(DOC, HASH_A) == ((500, 540),)
    assert store.dropped == {0: 1}


@pytest.mark.parametrize(("truncated", "added"), [(0, 2), (7, 0)])
def test_a_search_observation_without_a_passage_count_is_all_or_nothing(
    truncated: int, added: int
) -> None:
    step = _step(0, _search_result(10, 10))
    obs = Observation(step=0, source="tool", text="t", summary="s", truncated_chars=truncated)
    assert EvidenceStore().add_step(step, obs) == added


async def test_nothing_past_the_cut_of_a_k10_search_counts(backend: SearchBackend) -> None:
    # k=10 returns more passage text than the default 2000-char observation.
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
        run_id="r-k10",
    )
    state = await loop.run(TASK)
    result = state.steps[0].result
    assert result is not None
    passages = SearchCorpusOutput.model_validate(result.output).passages
    obs = state.observations[0]
    assert obs.shown_passages is not None
    shown = obs.shown_passages
    assert 0 < shown < len(passages)
    assert obs.text.endswith(f"[{len(passages) - shown} passages omitted]")
    store = EvidenceStore.from_state(state)
    assert [(s.doc_id, s.start_char, s.end_char) for s in store.spans] == [
        (p.doc_id, p.start_char, p.end_char) for p in passages[:shown]
    ]
    assert store.dropped == {0: len(passages) - shown}
    for p in passages[:shown]:
        assert json.dumps(p.text, ensure_ascii=False) in obs.text  # shown in full
    unseen = [
        p
        for p in passages[shown:]
        if not any(
            q.doc_id == p.doc_id and q.start_char <= p.start_char and p.end_char <= q.end_char
            for q in passages[:shown]
        )
    ]
    assert unseen  # some omitted passage is not inside a shown one
    for p in unseen:
        assert not store.covers(p.doc_id, p.content_hash, p.start_char, p.end_char)


async def test_hybrid_search_has_hits_for_nonsense_so_abstaining_is_a_give_up(
    index_dir: Path,
) -> None:
    hybrid = RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="hybrid")
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(tool="search_corpus", args={"query": "zyxxyq qwvvz", "k": 5}),
                GiveUp(reason="no_relevant_passage"),
            ]
        ),
        ToolExecutor(research_registry(hybrid)),
        Budgets(max_steps=3),
        run_id="r-nonsense",
    )
    state = await loop.run(TASK)
    result = state.steps[0].result
    assert result is not None
    # "No hits" never happens in hybrid mode: five weak neighbours come back...
    assert len(SearchCorpusOutput.model_validate(result.output).passages) == 5
    # ...so the run abstains only because the policy gave up.
    assert state.terminal_label == "abstained:no_relevant_passage"


PROSE = ("The Velmora Bridge spans the Harrow river in eleven arches. " * 40)[:1600]


async def test_a_full_length_prose_read_fits_the_default_observation() -> None:
    # MAX_READ_CHARS of plain prose plus the JSON envelope (realistic source) < 2000.
    source = "bridges/velmora-bridge-construction-history.md"
    docs = FixtureSearchBackend("c", {source: PROSE, "quotes.md": '"' * 1600})
    prose_id = make_doc_id("c", source)
    quotes_id = make_doc_id("c", "quotes.md")
    loop = AgentLoop(
        OraclePolicy(
            [
                ToolCall(
                    tool="read_document",
                    args={"doc_id": prose_id, "start": 0, "end": MAX_READ_CHARS},
                ),
                ToolCall(
                    tool="read_document",
                    args={"doc_id": quotes_id, "start": 0, "end": MAX_READ_CHARS},
                ),
                GiveUp(reason="done"),
            ]
        ),
        ToolExecutor(research_registry(docs)),
        Budgets(max_steps=4),
        run_id="r-read-cap",
    )
    state = await loop.run(TASK)
    prose_obs, quotes_obs = state.observations
    assert prose_obs.truncated_chars == 0
    # Escaping doubles every quote: the same cap no longer fits, so no credit.
    assert quotes_obs.truncated_chars > 0
    store = EvidenceStore.from_state(state)
    content_hash = docs.document(prose_id).content_hash
    assert store.observed(prose_id, content_hash) == ((0, MAX_READ_CHARS),)
    assert store.observed(quotes_id, docs.document(quotes_id).content_hash) == ()
    assert store.dropped == {1: 1}
