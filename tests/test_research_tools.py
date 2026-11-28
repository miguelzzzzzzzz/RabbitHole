"""`search_corpus` and `read_document` (M3) over every `SearchBackend`.

Tools run through the real registry and executor, so argument validation,
classified failures and output validation are what the agent loop sees.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from agentic_research.backends import (
    BackendError,
    FixtureSearchBackend,
    Passage,
    SearchBackend,
    StoredDocument,
    make_doc_id,
)
from agentic_research.backends.rag_engine import RagEngineBackend, RetrievalMode
from agentic_research.probe import probe_registry
from agentic_research.tools import ToolContext, ToolExecutor, ToolRegistry, ToolResult
from agentic_research.tools.builtin import research_registry
from agentic_research.tools.read import MAX_READ_CHARS, ReadDocumentOutput, ReadDocumentTool
from agentic_research.tools.search import SearchCorpusArgs, SearchCorpusOutput, SearchCorpusTool
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID, build_index

VELMORA = "bridges/velmora-bridge.md"
VELMORA_DOC_ID = "7610eae52aac886e"
CTX = ToolContext(run_id="r-tools", step=0)


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


@pytest.fixture(params=["fixture", "rag_engine-lexical", "rag_engine-hybrid"])
def backend(request: pytest.FixtureRequest, index_dir: Path) -> SearchBackend:
    if request.param == "fixture":
        return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)
    mode: RetrievalMode = "lexical" if request.param == "rag_engine-lexical" else "hybrid"
    return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode=mode)


async def _call(registry: ToolRegistry, tool: str, args: Mapping[str, Any] | str) -> ToolResult:
    return (await ToolExecutor(registry).execute(tool, args, CTX)).result


def _error(result: ToolResult) -> tuple[str, str]:
    assert result.error is not None, result
    return result.error.error_class.value, result.error.reason


class _Probe:
    """A backend double: records the calling thread, can block, fail or crash."""

    corpus_id = FIXTURE_CORPUS_ID

    def __init__(
        self,
        *,
        raise_exc: Exception | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.raise_exc = raise_exc
        self.release = release
        self.finished = threading.Event()
        self.threads: list[int] = []
        self._inner = FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)

    def _enter(self) -> None:
        self.threads.append(threading.get_ident())
        if self.release is not None:
            self.release.wait(5)
        if self.raise_exc is not None:
            raise self.raise_exc

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Passage]:
        try:
            self._enter()
            return self._inner.search(query, k, filters)
        finally:
            self.finished.set()

    def document(self, doc_id: str) -> StoredDocument:
        self._enter()
        return self._inner.document(doc_id)


# --- search_corpus ------------------------------------------------------------


async def test_search_returns_the_backend_passages(backend: SearchBackend) -> None:
    result = await _call(
        research_registry(backend), "search_corpus", {"query": "the Velmora Bridge"}
    )
    assert result.ok, result
    output = SearchCorpusOutput.model_validate(result.output)
    assert output.corpus_id == FIXTURE_CORPUS_ID
    # Default k is 5; the tool adds nothing to and drops nothing from the backend's hits.
    assert output.passages == backend.search("the Velmora Bridge", 5)
    assert len(output.passages) == 5
    for passage in output.passages:
        document = backend.document(passage.doc_id)
        assert document.text[passage.start_char : passage.end_char] == passage.text
        assert passage.content_hash == document.content_hash


async def test_search_passes_k_and_filters(backend: SearchBackend) -> None:
    result = await _call(
        research_registry(backend),
        "search_corpus",
        json.dumps({"query": "Harrow Vale", "k": 2, "filters": {"doc_id": VELMORA_DOC_ID}}),
    )
    output = SearchCorpusOutput.model_validate(result.output)
    assert 1 <= len(output.passages) <= 2
    assert {p.doc_id for p in output.passages} == {VELMORA_DOC_ID}


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"query": ""},
        {"query": "bridge", "k": 0},
        {"query": "bridge", "k": 51},
        {"query": "bridge", "k": "five"},
        {"query": "bridge", "filters": {"doc_id": 3}},
        {"query": "bridge", "top_k": 3},
    ],
)
async def test_search_schema_violations_are_invalid_args(args: dict[str, Any]) -> None:
    probe = _Probe()
    result = await _call(research_registry(probe), "search_corpus", args)
    assert _error(result)[0] == "invalid_args"
    assert probe.threads == []  # rejected before the backend is touched


async def test_blank_query_is_invalid_args_without_a_backend_call() -> None:
    probe = _Probe()
    result = await _call(research_registry(probe), "search_corpus", {"query": "  \n "})
    assert _error(result) == ("invalid_args", "blank_query")
    assert probe.threads == []


async def test_backend_error_is_permanent() -> None:
    probe = _Probe(raise_exc=BackendError("index unreadable"))
    result = await _call(research_registry(probe), "search_corpus", {"query": "bridge"})
    assert _error(result) == ("permanent", "backend_error")
    assert result.error is not None
    assert "index unreadable" in result.error.message


async def test_unexpected_backend_exception_is_a_crash_not_bad_args() -> None:
    # Even a ValueError from inside the backend is a bug, not the model's fault.
    probe = _Probe(raise_exc=ValueError("boom"))
    result = await _call(research_registry(probe), "search_corpus", {"query": "bridge"})
    assert _error(result) == ("permanent", "tool_crashed")


async def test_search_and_read_run_off_the_event_loop_thread() -> None:
    probe = _Probe()
    registry = research_registry(probe)
    assert (await _call(registry, "search_corpus", {"query": "bridge"})).ok
    read = {"doc_id": VELMORA_DOC_ID, "start": 0, "end": 10}
    assert (await _call(registry, "read_document", read)).ok
    assert len(probe.threads) == 2
    assert threading.get_ident() not in probe.threads


async def test_a_blocked_search_times_out_and_its_thread_finishes_later() -> None:
    release = threading.Event()
    probe = _Probe(release=release)
    registry = ToolRegistry([SearchCorpusTool(probe, timeout_s=0.05)])
    try:
        result = await _call(registry, "search_corpus", {"query": "bridge"})
    finally:
        release.set()
    assert _error(result) == ("timeout", "tool_timeout")
    assert probe.finished.wait(5)  # the worker thread finishes in the background


def test_search_tool_constructor_sets_timeout() -> None:
    probe = _Probe()
    assert SearchCorpusTool(probe).timeout_s == 30.0
    assert ReadDocumentTool(probe).timeout_s == 10.0
    assert ReadDocumentTool(probe, timeout_s=2.5).timeout_s == 2.5


def test_research_registry_names() -> None:
    registry = research_registry(_Probe())
    assert list(registry.names) == ["calculator", "search_corpus", "read_document"]
    assert all(registry.get(name).idempotent for name in registry.names)
    for schema in registry.openai_schemas():
        jsonschema.Draft202012Validator.check_schema(schema["function"]["parameters"])


# --- probe stub schema: frozen, and the real tool accepts the same arguments ----

#: sha256 of the probe's ``search_corpus`` tool schema as sent in probes/tool_calls.v1.
PROBE_SEARCH_SCHEMA_SHA256 = "8096ccd813dfd2df68629eedb71dc807401329651d79d914131cd192493c91ff"


def _schema(registry: ToolRegistry, name: str) -> dict[str, Any]:
    (schema,) = [s for s in registry.openai_schemas() if s["function"]["name"] == name]
    return schema


def test_probe_search_schema_is_unchanged() -> None:
    schema = _schema(probe_registry(), "search_corpus")
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == PROBE_SEARCH_SCHEMA_SHA256


def test_real_search_args_match_the_probe_stub() -> None:
    stub = _schema(probe_registry(), "search_corpus")["function"]["parameters"]
    real = SearchCorpusArgs.model_json_schema()
    # Only the model docstring differs; fields, defaults, bounds and strictness match.
    for key in ("properties", "required", "additionalProperties", "type"):
        assert real[key] == stub[key], key
    assert real["description"] != stub["description"]


# --- read_document --------------------------------------------------------------


async def test_read_returns_the_exact_slice(backend: SearchBackend) -> None:
    document = backend.document(VELMORA_DOC_ID)
    result = await _call(
        research_registry(backend),
        "read_document",
        {"doc_id": VELMORA_DOC_ID, "start": 17, "end": 60},
    )
    output = ReadDocumentOutput.model_validate(result.output)
    assert output.text == document.text[17:60]
    assert (output.start_char, output.end_char) == (17, 60)
    assert output.doc_length == len(document.text)
    assert output.content_hash == document.content_hash
    assert output.source == VELMORA


async def test_read_clamps_end_to_the_document_length(backend: SearchBackend) -> None:
    length = len(backend.document(VELMORA_DOC_ID).text)
    start = length - 7
    result = await _call(
        research_registry(backend),
        "read_document",
        {"doc_id": VELMORA_DOC_ID, "start": start, "end": start + 100},
    )
    output = ReadDocumentOutput.model_validate(result.output)
    assert (output.start_char, output.end_char) == (start, length)
    assert len(output.text) == 7


async def test_read_allows_exactly_the_maximum_range() -> None:
    text = "x" * (MAX_READ_CHARS + 10)
    backend = FixtureSearchBackend(FIXTURE_CORPUS_ID, {"long.txt": text})
    doc_id = make_doc_id(FIXTURE_CORPUS_ID, "long.txt")
    result = await _call(
        research_registry(backend),
        "read_document",
        {"doc_id": doc_id, "start": 5, "end": 5 + MAX_READ_CHARS},
    )
    assert len(ReadDocumentOutput.model_validate(result.output).text) == MAX_READ_CHARS


@pytest.mark.parametrize(
    "args",
    [
        {"doc_id": VELMORA_DOC_ID, "start": 10, "end": 10},
        {"doc_id": VELMORA_DOC_ID, "start": 10, "end": 9},
        {"doc_id": VELMORA_DOC_ID, "start": 0, "end": MAX_READ_CHARS + 1},
        {"doc_id": VELMORA_DOC_ID, "start": -1, "end": 5},
        {"doc_id": VELMORA_DOC_ID, "start": 0},
        {"doc_id": "VELMORA", "start": 0, "end": 5},
        {"doc_id": VELMORA_DOC_ID.upper(), "start": 0, "end": 5},
        {"doc_id": VELMORA_DOC_ID, "start": 0, "end": 5, "extra": 1},
        # Offsets refuse what lax mode would coerce (`Offset`, Chad's 2026-11-27 note).
        {"doc_id": VELMORA_DOC_ID, "start": False, "end": 5},
        {"doc_id": VELMORA_DOC_ID, "start": 0, "end": True},
        {"doc_id": VELMORA_DOC_ID, "start": 0.0, "end": 5},
        {"doc_id": VELMORA_DOC_ID, "start": 0, "end": 5.0},
        {"doc_id": VELMORA_DOC_ID, "start": "0.0", "end": 5},
        {"doc_id": VELMORA_DOC_ID, "start": 0, "end": "1_0"},
    ],
)
async def test_read_schema_violations_are_invalid_args(args: dict[str, Any]) -> None:
    probe = _Probe()
    result = await _call(research_registry(probe), "read_document", args)
    assert _error(result)[0] == "invalid_args"
    assert probe.threads == []


async def test_read_accepts_integer_strings_as_offsets(backend: SearchBackend) -> None:
    # Lax mode still reads a string of digits as an integer, as for every tool.
    document = backend.document(VELMORA_DOC_ID)
    result = await _call(
        research_registry(backend),
        "read_document",
        json.dumps({"doc_id": VELMORA_DOC_ID, "start": "17", "end": "60"}),
    )
    output = ReadDocumentOutput.model_validate(result.output)
    assert (output.start_char, output.end_char) == (17, 60)
    assert output.text == document.text[17:60]


async def test_read_start_past_the_end_is_out_of_range(backend: SearchBackend) -> None:
    length = len(backend.document(VELMORA_DOC_ID).text)
    result = await _call(
        research_registry(backend),
        "read_document",
        {"doc_id": VELMORA_DOC_ID, "start": length, "end": length + 1},
    )
    assert _error(result) == ("invalid_args", "start_out_of_range")


async def test_read_unknown_doc_id(backend: SearchBackend) -> None:
    result = await _call(
        research_registry(backend),
        "read_document",
        {"doc_id": "0000000000000000", "start": 0, "end": 5},
    )
    assert _error(result) == ("invalid_args", "unknown_doc_id")
    assert result.error is not None
    assert FIXTURE_CORPUS_ID in result.error.message


async def test_read_backend_error_is_permanent() -> None:
    probe = _Probe(raise_exc=BackendError("store offline"))
    result = await _call(
        research_registry(probe), "read_document", {"doc_id": VELMORA_DOC_ID, "start": 0, "end": 5}
    )
    assert _error(result) == ("permanent", "backend_error")


@pytest.mark.parametrize(
    "fields",
    [
        {"start_char": 5, "end_char": 5, "doc_length": 10, "text": ""},
        {"start_char": 0, "end_char": 11, "doc_length": 10, "text": "x" * 11},
        {"start_char": 0, "end_char": 5, "doc_length": 10, "text": "abc"},
    ],
)
def test_read_output_rejects_inconsistent_slices(fields: dict[str, Any]) -> None:
    base = {"doc_id": VELMORA_DOC_ID, "source": VELMORA, "content_hash": "0" * 64}
    with pytest.raises(ValueError):
        ReadDocumentOutput.model_validate({**base, **fields})
