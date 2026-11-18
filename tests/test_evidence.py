"""The per-run evidence store (M3): spans a run observed, from its recorded steps."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_research.agent import AgentLoop, Budgets, FinalAnswer, GiveUp, Task, ToolCall
from agentic_research.backends import FixtureSearchBackend, Passage, SearchBackend, StoredDocument
from agentic_research.backends.rag_engine import RagEngineBackend
from agentic_research.evidence import EvidenceSpan, EvidenceStore
from agentic_research.policies.oracle import OraclePolicy
from agentic_research.tools import ToolError, ToolExecutor, ToolRegistry, ToolResult
from agentic_research.tools.base import ErrorClass
from agentic_research.tools.builtin import research_registry
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.read import ReadDocumentTool
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


def _store(*ranges: tuple[int, int], content_hash: str = HASH_A) -> EvidenceStore:
    store = EvidenceStore()
    for step, (start, end) in enumerate(ranges):
        assert store.add_result(step, _read(DOC, content_hash, start, end)) == 1
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
    assert store.add_result(0, failed) == 0
    assert store.add_result(1, ToolResult(tool="calculator", output={"value": 4})) == 0
    assert store.spans == ()


def test_an_evidence_result_that_breaks_its_output_model_is_an_error() -> None:
    bad = ToolResult(tool="read_document", output={"doc_id": DOC})
    with pytest.raises(ValidationError):
        EvidenceStore().add_result(0, bad)


def test_evidence_span_rejects_an_empty_range() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(
            step=0, tool="read_document", doc_id=DOC, content_hash=HASH_A, start_char=5, end_char=5
        )


# --- from a real run ------------------------------------------------------------


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


@pytest.fixture(params=["fixture", "rag_engine"])
def backend(request: pytest.FixtureRequest, index_dir: Path) -> SearchBackend:
    if request.param == "fixture":
        return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)
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
    store = EvidenceStore.from_steps(state.steps)

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
    assert EvidenceStore.from_steps(state.steps).spans == ()


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
    assert {s.step for s in EvidenceStore.from_steps(state.steps).spans} == {0}
