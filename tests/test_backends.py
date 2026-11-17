"""Search backend contract (M3): the same suite runs on both backends.

`FixtureSearchBackend` reads the fixture corpus files directly;
`RagEngineBackend` reads a CitationNeeded format-2 index built from the same
files with CitationNeeded's own ingest/chunk/save code (hashing embedder, no
model, no network).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from rag_engine.cli import main as rag_engine_main
from rag_engine.models import make_doc_id as rag_engine_make_doc_id

from agentic_research.backends import (
    SUPPORTED_SEARCH_SCHEMA_VERSION,
    BackendError,
    FixtureSearchBackend,
    Passage,
    SearchBackend,
    StoredDocument,
    UnknownDocumentError,
    make_doc_id,
    parse_search_payload,
    sha256_hex,
)
from agentic_research.backends.rag_engine import (
    REQUIRED_INDEX_FORMAT,
    RagEngineBackend,
    RetrievalMode,
)
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID, build_index

VELMORA = "bridges/velmora-bridge.md"
TARKAND = "people/ilse-tarkand.md"
HARROW = "places/harrow-vale.md"
# Pinned: sha256("rabbithole-tasks-v1:bridges/velmora-bridge.md")[:16].
VELMORA_DOC_ID = "7610eae52aac886e"


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


@pytest.fixture(params=["fixture", "rag_engine-lexical", "rag_engine-hybrid"])
def backend(request: pytest.FixtureRequest, index_dir: Path) -> SearchBackend:
    if request.param == "fixture":
        return FixtureSearchBackend.from_directory(FIXTURE_CORPUS, FIXTURE_CORPUS_ID)
    mode: RetrievalMode = "lexical" if request.param == "rag_engine-lexical" else "hybrid"
    return RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode=mode)


def _copy_index(src: Path, dst: Path) -> Path:
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return dst


# --- contract: every backend -------------------------------------------------


def test_backend_reports_its_corpus_id(backend: SearchBackend) -> None:
    assert backend.corpus_id == FIXTURE_CORPUS_ID


@pytest.mark.parametrize(
    ("query", "source"),
    [
        ("Northmere Polytechnic graduated 1897", TARKAND),
        ("Velmora tram lines removed", VELMORA),
        ("census residents wool market", HARROW),
    ],
)
def test_distinctive_query_ranks_its_document_first(
    backend: SearchBackend, query: str, source: str
) -> None:
    passages = backend.search(query, 3)
    assert passages
    assert passages[0].source == source


@pytest.mark.parametrize("query", ["Velmora Bridge", "Harrow Vale river Osk", "engineer"])
def test_passages_carry_verifiable_citation_fields(backend: SearchBackend, query: str) -> None:
    passages = backend.search(query, 10)
    assert passages
    assert [p.rank for p in passages] == list(range(1, len(passages) + 1))
    scores = [p.score for p in passages]
    assert scores == sorted(scores, reverse=True)
    assert all(math.isfinite(score) for score in scores)
    for passage in passages:
        document = backend.document(passage.doc_id)
        # Offsets index the normalized stored text (CitationNeeded format 2).
        assert document.text[passage.start_char : passage.end_char] == passage.text
        assert passage.content_hash == document.content_hash == sha256_hex(document.text)
        assert passage.doc_id == make_doc_id(FIXTURE_CORPUS_ID, passage.source)
        assert document.source == passage.source
        # Markdown/text documents have no pages.
        assert passage.page is None
        assert passage.page_end is None


def test_k_bounds_the_number_of_passages(backend: SearchBackend) -> None:
    assert len(backend.search("the", 1)) == 1
    assert len(backend.search("the", 2)) == 2


def test_filters_restrict_results(backend: SearchBackend) -> None:
    by_doc = backend.search("Harrow Vale", 10, {"doc_id": VELMORA_DOC_ID})
    assert by_doc
    assert {p.doc_id for p in by_doc} == {VELMORA_DOC_ID}
    by_source = backend.search("Harrow Vale", 10, {"source": TARKAND})
    assert by_source
    assert {p.source for p in by_source} == {TARKAND}
    assert backend.search("Harrow Vale", 10, {"no_such_field": "x"}) == []


def test_query_without_a_matching_term(backend: SearchBackend) -> None:
    passages = backend.search("zyxxyq qwvvz", 5)
    if isinstance(backend, RagEngineBackend) and backend.mode == "hybrid":
        # The dense half ranks every chunk by vector similarity, so hybrid
        # search still returns (weak) neighbours; only lexical search is empty.
        assert len(passages) == 5
    else:
        assert passages == []


@pytest.mark.parametrize(("query", "k"), [("", 3), ("   ", 3), ("bridge", 0), ("bridge", -1)])
def test_bad_search_arguments_are_rejected(backend: SearchBackend, query: str, k: int) -> None:
    with pytest.raises(ValueError):
        backend.search(query, k)


def test_unknown_doc_id_is_a_clear_error(backend: SearchBackend) -> None:
    with pytest.raises(UnknownDocumentError, match="0000000000000000") as info:
        backend.document("0000000000000000")
    assert isinstance(info.value, BackendError)
    assert info.value.doc_id == "0000000000000000"


def test_every_fixture_file_is_a_document(backend: SearchBackend) -> None:
    for source in (VELMORA, TARKAND, HARROW, "README.txt"):
        document = backend.document(make_doc_id(FIXTURE_CORPUS_ID, source))
        assert document.source == source


# --- doc ids -----------------------------------------------------------------


def test_pinned_doc_id_under_the_fixture_corpus_id() -> None:
    expected = hashlib.sha256(f"{FIXTURE_CORPUS_ID}:{VELMORA}".encode()).hexdigest()[:16]
    assert expected == VELMORA_DOC_ID
    assert make_doc_id(FIXTURE_CORPUS_ID, VELMORA) == VELMORA_DOC_ID
    assert rag_engine_make_doc_id(FIXTURE_CORPUS_ID, VELMORA) == VELMORA_DOC_ID


def test_doc_id_depends_on_the_corpus_id() -> None:
    assert make_doc_id("rabbithole-hotpotqa", VELMORA) != VELMORA_DOC_ID


@pytest.mark.parametrize(
    ("corpus_id", "source"),
    [
        ("Bad Corpus", VELMORA),
        ("", VELMORA),
        ("a:b", VELMORA),
        (FIXTURE_CORPUS_ID, ""),
        (FIXTURE_CORPUS_ID, "/abs.md"),
        (FIXTURE_CORPUS_ID, "./a.md"),
        (FIXTURE_CORPUS_ID, "a//b.md"),
        (FIXTURE_CORPUS_ID, "a/../b.md"),
        (FIXTURE_CORPUS_ID, "a\\b.md"),
    ],
)
def test_make_doc_id_rejects_ambiguous_inputs(corpus_id: str, source: str) -> None:
    with pytest.raises(ValueError):
        make_doc_id(corpus_id, source)


# --- search --json payload ---------------------------------------------------


def _cli_payload(index_dir: Path, query: str, k: int, mode: str) -> dict[str, Any]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = rag_engine_main(
            ["search", str(index_dir), query, "-k", str(k), "--mode", mode, "--json"]
        )
    assert code == 0
    payload: dict[str, Any] = json.loads(out.getvalue())
    return payload


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_adapter_matches_the_rag_engine_search_json_cli(
    index_dir: Path, mode: RetrievalMode
) -> None:
    """In-process results equal `rag-engine search --json` on the same index."""
    payload = _cli_payload(index_dir, "Velmora Bridge engineer", 4, mode)
    assert payload["schema_version"] == SUPPORTED_SEARCH_SCHEMA_VERSION
    assert payload["mode"] == mode
    backend = RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode=mode)
    expected = parse_search_payload(payload)
    assert len(expected) == 4
    assert backend.search("Velmora Bridge engineer", 4) == expected


def _hit(**overrides: Any) -> dict[str, Any]:
    text = "Velmora"
    hit: dict[str, Any] = {
        "rank": 1,
        "score": 2.5,
        "retriever": "lexical",
        "components": {"lexical": 2.5},
        "ranks": {"lexical": 1},
        "doc_id": VELMORA_DOC_ID,
        "chunk_id": "c1",
        "source": VELMORA,
        "content_hash": "a" * 64,
        "heading_path": ["Velmora Bridge"],
        "page": None,
        "page_end": None,
        "start_char": 2,
        "end_char": 2 + len(text),
        "text": text,
    }
    hit.update(overrides)
    return hit


def test_payload_with_supported_version_parses_hits_and_ignores_extra_keys() -> None:
    passages = parse_search_payload(
        {"schema_version": 1, "query": "q", "mode": "lexical", "rerank": False, "hits": [_hit()]}
    )
    assert passages == [
        Passage(
            rank=1,
            score=2.5,
            doc_id=VELMORA_DOC_ID,
            chunk_id="c1",
            source=VELMORA,
            content_hash="a" * 64,
            heading_path=("Velmora Bridge",),
            start_char=2,
            end_char=9,
            text="Velmora",
        )
    ]
    assert parse_search_payload({"schema_version": 1, "hits": []}) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 2, "hits": [_hit()]}, "unsupported search payload schema_version 2"),
        ({"schema_version": 0, "hits": []}, "schema_version 0"),
        ({"schema_version": "1", "hits": []}, "schema_version '1'"),
        ({"schema_version": True, "hits": []}, "schema_version True"),
        ({"schema_version": 1.0, "hits": []}, "schema_version 1.0"),
        ({"hits": [_hit()]}, "no schema_version"),
        ([_hit()], "must be a JSON object"),
        ({"schema_version": 1}, "'hits' must be a list"),
        ({"schema_version": 1, "hits": "x"}, "'hits' must be a list"),
        ({"schema_version": 1, "hits": {"a": 1}}, "'hits' must be a list"),
        ({"schema_version": 1, "hits": [1]}, "hit 0 must be an object"),
        (
            {
                "schema_version": 1,
                "hits": [_hit(), {k: v for k, v in _hit().items() if k != "content_hash"}],
            },
            "hit 1 is missing content_hash",
        ),
        ({"schema_version": 1, "hits": [_hit(content_hash=None)]}, "hit 0 is invalid"),
        ({"schema_version": 1, "hits": [_hit(end_char=3)]}, "hit 0 is invalid"),
    ],
)
def test_payload_problems_are_clear_errors_not_empty_results(payload: object, message: str) -> None:
    with pytest.raises(BackendError, match=message):
        parse_search_payload(payload)


# --- models ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"score": math.nan}, "score must be finite"),
        ({"score": math.inf}, "score must be finite"),
        ({"start_char": 9}, "end_char must be > start_char"),
        ({"text": "Velmor"}, "text length"),
        ({"page": 1}, "both be set"),
        ({"page_end": 1}, "both be set"),
        ({"page": 3, "page_end": 2}, "page_end must be >= page"),
        ({"doc_id": "XYZ"}, "doc_id"),
        ({"content_hash": "abc"}, "content_hash"),
    ],
)
def test_passage_rejects_inconsistent_fields(overrides: dict[str, Any], message: str) -> None:
    fields = {k: v for k, v in _hit(**overrides).items() if k in Passage.model_fields}
    with pytest.raises(ValidationError, match=message):
        Passage.model_validate(fields)


def test_passage_accepts_a_page_range() -> None:
    fields = {k: v for k, v in _hit(page=2, page_end=3).items() if k in Passage.model_fields}
    assert Passage.model_validate(fields).page_end == 3


def test_stored_document_checks_its_hash() -> None:
    text = "Velmora"
    StoredDocument(doc_id=VELMORA_DOC_ID, source=VELMORA, content_hash=sha256_hex(text), text=text)
    with pytest.raises(ValidationError, match="content_hash does not match"):
        StoredDocument(
            doc_id=VELMORA_DOC_ID, source=VELMORA, content_hash=sha256_hex("x"), text=text
        )


# --- RagEngineBackend: index checks ------------------------------------------


def test_rag_engine_backend_defaults_to_lexical(index_dir: Path) -> None:
    assert RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID).mode == "lexical"
    hybrid = RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="hybrid")
    assert hybrid.mode == "hybrid"


def test_rag_engine_backend_rejects_an_unknown_mode(index_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown retrieval mode 'dense'"):
        RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID, mode="dense")  # type: ignore[arg-type]


def test_rag_engine_backend_refuses_a_directory_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="not a CitationNeeded index directory"):
        RagEngineBackend(tmp_path, corpus_id=FIXTURE_CORPUS_ID)


def _with_manifest(index_dir: Path, tmp_path: Path, edit: Callable[[str], str]) -> Path:
    copy = _copy_index(index_dir, tmp_path / "copy")
    manifest = copy / "index.json"
    manifest.write_text(edit(manifest.read_text(encoding="utf-8")), encoding="utf-8")
    return copy


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda text: text.replace('"format_version": 2', '"format_version": 1'),
            "has format_version 1; RabbitHole needs format 2",
        ),
        (
            lambda text: text.replace('"format_version": 2', '"format_version": 3'),
            "has format_version 3",
        ),
        (
            lambda text: text.replace('"format_version": 2', '"format_version": true'),
            "has format_version True",
        ),
        (lambda text: text.replace('"format_version": 2,', ""), "has format_version None"),
        (lambda _text: "[2]", "has format_version None"),
        (lambda _text: "{not json", "cannot read"),
    ],
)
def test_rag_engine_backend_requires_index_format_2(
    index_dir: Path, tmp_path: Path, edit: Callable[[str], str], message: str
) -> None:
    assert REQUIRED_INDEX_FORMAT == 2
    copy = _with_manifest(index_dir, tmp_path, edit)
    with pytest.raises(BackendError, match=message):
        RagEngineBackend(copy, corpus_id=FIXTURE_CORPUS_ID)


def test_rag_engine_backend_reports_an_unreadable_manifest(tmp_path: Path) -> None:
    (tmp_path / "index.json").mkdir()
    with pytest.raises(BackendError, match="cannot read"):
        RagEngineBackend(tmp_path, corpus_id=FIXTURE_CORPUS_ID)


def test_rag_engine_backend_refuses_another_corpus(tmp_path: Path) -> None:
    other = build_index(FIXTURE_CORPUS, tmp_path / "other", "other-corpus")
    with pytest.raises(BackendError, match="from corpus other-corpus, expected only"):
        RagEngineBackend(other, corpus_id=FIXTURE_CORPUS_ID)


def _tampered_copy(index_dir: Path, tmp_path: Path, *, rehash: bool) -> Path:
    """Change one stored character of the Velmora document (optionally fixing its hash)."""
    copy = _copy_index(index_dir, tmp_path / "copy")
    path = copy / "documents.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        record = json.loads(line)
        if record["source"] == VELMORA:
            assert "412 metres" in record["text"]
            record["text"] = record["text"].replace("412 metres", "413 metres")
            if rehash:
                record["content_hash"] = sha256_hex(record["text"])
            lines[i] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return copy


@pytest.mark.parametrize("rehash", [False, True], ids=["hash-mismatch", "span-mismatch"])
def test_rag_engine_backend_refuses_a_tampered_index(
    index_dir: Path, tmp_path: Path, rehash: bool
) -> None:
    copy = _tampered_copy(index_dir, tmp_path, rehash=rehash)
    with pytest.raises(BackendError, match="cannot load index"):
        RagEngineBackend(copy, corpus_id=FIXTURE_CORPUS_ID)


def test_rag_engine_backend_reports_missing_index_parts(index_dir: Path, tmp_path: Path) -> None:
    copy = _copy_index(index_dir, tmp_path / "copy")
    (copy / "documents.jsonl").unlink()
    with pytest.raises(BackendError, match="cannot load index"):
        RagEngineBackend(copy, corpus_id=FIXTURE_CORPUS_ID)


# --- FixtureSearchBackend specifics ------------------------------------------


def test_fixture_paragraphs_are_trimmed_runs_of_non_blank_lines() -> None:
    text = "  alpha beta\ngamma  \n\n \t\n\ndelta alpha\n"
    backend = FixtureSearchBackend("unit", {"a.txt": text})
    passages = backend.search("alpha", 5)
    assert [p.text for p in passages] == ["alpha beta\ngamma", "delta alpha"]
    assert [(p.start_char, p.end_char) for p in passages] == [(2, 18), (26, 37)]
    assert all(p.score == 1.0 for p in passages)


def test_fixture_scores_count_distinct_query_terms_and_ties_are_deterministic() -> None:
    backend = FixtureSearchBackend(
        "unit", {"b.txt": "osk bridge", "a.txt": "Osk only", "c.txt": "bridge osk osk"}
    )
    passages = backend.search("Osk bridge osk", 5)
    assert [(p.source, p.score) for p in passages] == [
        ("b.txt", 2.0),
        ("c.txt", 2.0),
        ("a.txt", 1.0),
    ]


def test_fixture_empty_document_has_no_passages_but_is_stored() -> None:
    backend = FixtureSearchBackend("unit", {"empty.txt": ""})
    assert backend.search("anything", 3) == []
    assert backend.document(make_doc_id("unit", "empty.txt")).text == ""
