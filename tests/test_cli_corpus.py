"""``arp run`` over a CitationNeeded index (``--index DIR --corpus-id ID``) and the
research tools in ``arp tools schema`` (no model calls, no network).

The index is built from the fixture corpus with CitationNeeded's offline
hashing embedder (`tests/rag_index.py`), so lexical and hybrid retrieval both
run without fastembed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_research.backends.base import BackendError
from agentic_research.backends.fixture import FixtureSearchBackend
from agentic_research.backends.rag_engine import RagEngineBackend
from agentic_research.cli import main
from agentic_research.tools.base import ErrorClass, ToolContext, ToolError
from agentic_research.tools.builtin import (
    NO_CORPUS_ID,
    NoCorpusBackend,
    catalog_registry,
    research_registry,
)
from agentic_research.tools.executor import ToolExecutor
from agentic_research.trace import (
    GateRefusal,
    Trace,
    TraceGateError,
    read_trace,
    require_citations_verified,
)
from agentic_research.trace.schema import StepSpan, ToolCallSpan
from rag_index import FIXTURE_CORPUS, FIXTURE_CORPUS_ID, build_index

FIXTURES = Path(__file__).parent / "fixtures"
VELMORA = FIXTURES / "trajectories" / "velmora_designer.json"
QUESTION = "Who designed the Velmora Bridge?"
VELMORA_DOC_ID = "7610eae52aac886e"


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_index(FIXTURE_CORPUS, tmp_path_factory.mktemp("rag") / "index", FIXTURE_CORPUS_ID)


def _velmora() -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = json.loads(VELMORA.read_text(encoding="utf-8"))
    return actions


def _write(tmp_path: Path, name: str, body: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def arp_oracle(index: Path, trajectory: Path, out: Path, *extra: str) -> int:
    return main(
        [
            "run", "--provider", "oracle", "--trajectory", str(trajectory),
            "--question", QUESTION, "--index", str(index), "--corpus-id", FIXTURE_CORPUS_ID,
            "--run-id", "corpus", "--trace-out", str(out), *extra,
        ]
    )  # fmt: skip


def _steps(trace: Trace) -> list[StepSpan]:
    return [s for s in trace.spans if isinstance(s, StepSpan)]


def _tools(trace: Trace) -> list[ToolCallSpan]:
    return [s for s in trace.spans if isinstance(s, ToolCallSpan)]


def _rejections(trace: Trace) -> list[str]:
    return [
        str(e.attributes.get("reason"))
        for step in _steps(trace)
        for e in step.events
        if e.name == "answer_rejected"
    ]


# --- oracle runs over the index -------------------------------------------------------


def test_the_fixture_trajectory_matches_the_index(index_dir: Path) -> None:
    # Guards the committed offsets and hash: they are the stored normalized text.
    doc = RagEngineBackend(index_dir, corpus_id=FIXTURE_CORPUS_ID).document(VELMORA_DOC_ID)
    (citation,) = _velmora()[2]["claims"][0]["citations"]
    assert doc.text[citation["start"] : citation["end"]] == citation["quote"]
    assert citation["content_hash"] == doc.content_hash


@pytest.mark.parametrize("retrieval", [None, "lexical", "hybrid"])
def test_an_oracle_run_over_the_index_is_verified_and_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], index_dir: Path, retrieval: str | None
) -> None:
    out = tmp_path / "t.jsonl"
    extra = [] if retrieval is None else ["--retrieval", retrieval]
    assert arp_oracle(index_dir, VELMORA, out, *extra) == 0
    mode = retrieval or "lexical"
    assert capsys.readouterr().out.splitlines() == [
        f"corpus: {FIXTURE_CORPUS_ID} ({mode}), verifier: citations",
        "run corpus: succeeded",
        "answer: Ilse Tarkand",
        f"trace: {out}",
    ]
    trace = read_trace(out)
    assert trace.run.provider == "none" and trace.run.model is None
    tools = _tools(trace)
    assert [t.tool for t in tools] == ["search_corpus", "read_document"]
    assert all(t.error is None for t in tools)
    config = trace.run_config
    assert config is not None and config.citations_verified
    assert config.corpus is not None
    assert (config.verifier, config.corpus.corpus_id, config.corpus.retrieval) == (
        "citations",
        FIXTURE_CORPUS_ID,
        mode,
    )
    output = tools[0].output
    assert output is not None and output["corpus_id"] == FIXTURE_CORPUS_ID
    passages = output["passages"]
    assert isinstance(passages, list)
    assert VELMORA_DOC_ID in {p["doc_id"] for p in passages if isinstance(p, dict)}
    assert _rejections(trace) == []
    assert require_citations_verified(trace) == config


def test_the_citation_verifier_is_on_by_default_with_a_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], index_dir: Path
) -> None:
    # The same answer without the search and the read cites text never shown.
    trajectory = _write(tmp_path, "unread.json", _velmora()[2:])
    out = tmp_path / "t.jsonl"
    assert arp_oracle(index_dir, trajectory, out) == 1
    assert "run corpus: failed:trajectory_exhausted" in capsys.readouterr().out
    assert _rejections(read_trace(out)) == ["ungrounded_citation"]


def test_accept_all_turns_the_citation_check_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], index_dir: Path
) -> None:
    trajectory = _write(tmp_path, "unread.json", _velmora()[2:])
    out = tmp_path / "t.jsonl"
    assert arp_oracle(index_dir, trajectory, out, "--verifier", "accept-all") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        f"corpus: {FIXTURE_CORPUS_ID} (lexical), verifier: accept-all",
        "run corpus: succeeded",
    ]
    # Chad's 2026-11-23 spot-check: an accept-all run must not look verified.
    config = read_trace(out).run_config
    assert config is not None and config.corpus is not None
    assert (config.verifier, config.corpus.corpus_id) == ("accept-all", FIXTURE_CORPUS_ID)
    assert not config.citations_verified
    with pytest.raises(TraceGateError) as excinfo:
        require_citations_verified(read_trace(out))
    assert excinfo.value.reason is GateRefusal.NOT_CITATIONS_VERIFIED


def test_a_misquoted_citation_is_rejected_as_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], index_dir: Path
) -> None:
    actions = _velmora()
    actions[2]["claims"][0]["citations"][0]["quote"] = "The bridge was designed by someone else."
    out = tmp_path / "t.jsonl"
    assert arp_oracle(index_dir, _write(tmp_path, "bad.json", actions), out) == 1
    capsys.readouterr()
    assert _rejections(read_trace(out)) == ["invalid_citation"]


def test_an_oracle_run_without_a_corpus_has_only_the_calculator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trajectory = _write(
        tmp_path,
        "calc.json",
        [
            {"kind": "tool_call", "tool": "calculator", "args": {"expression": "6*7"}},
            {"kind": "tool_call", "tool": "search_corpus", "args": {"query": "Velmora"}},
            {"kind": "final_answer", "answer": "42"},
        ],
    )
    out = tmp_path / "t.jsonl"
    code = main(
        ["run", "--provider", "oracle", "--trajectory", str(trajectory), "--question", "6*7?",
         "--run-id", "calc", "--trace-out", str(out)]
    )  # fmt: skip
    assert code == 0
    stdout = capsys.readouterr().out
    assert "corpus:" not in stdout and "run calc: succeeded" in stdout
    config = read_trace(out).run_config
    assert config is not None and (config.verifier, config.corpus) == ("accept-all", None)
    calc, search = _tools(read_trace(out))
    assert calc.error is None and calc.output == {"value": 42}
    assert search.error is not None and search.error.reason == "unknown_tool"


# --- the scripted LLM policy over the index ---------------------------------------------


def _search_then_answer(tmp_path: Path) -> Path:
    turns = [
        {
            "tool_calls": [
                {"id": "c1", "name": "search_corpus", "arguments": {"query": "Velmora", "k": 2}}
            ]
        },
        {"content": "Ilse Tarkand"},
        {"content": "Ilse Tarkand"},
    ]
    return _write(
        tmp_path, "script.json", {"schema_version": 1, "question": QUESTION, "turns": turns}
    )


@pytest.mark.parametrize(
    ("verifier", "code", "label", "rejections"),
    [
        # Until M5 the LLM policy's answers carry no claims (TODO.md, M5).
        (None, 1, "failed:script_exhausted", ["no_citation", "no_citation"]),
        ("accept-all", 0, "succeeded", []),
    ],
)
def test_a_scripted_run_searches_the_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    index_dir: Path,
    verifier: str | None,
    code: int,
    label: str,
    rejections: list[str],
) -> None:
    out = tmp_path / "t.jsonl"
    extra = [] if verifier is None else ["--verifier", verifier]
    args = [
        "run", "--provider", "scripted", "--script", str(_search_then_answer(tmp_path)),
        "--index", str(index_dir), "--corpus-id", FIXTURE_CORPUS_ID, "--run-id", "s",
        "--trace-out", str(out), *extra,
    ]  # fmt: skip
    assert main(args) == code
    assert f"run s: {label}" in capsys.readouterr().out
    trace = read_trace(out)
    (search,) = _tools(trace)
    assert search.tool == "search_corpus" and search.error is None
    assert _rejections(trace) == rejections


# --- bad input --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("corpus_id_alone", "--corpus-id needs --index"),
        ("retrieval_alone", "--retrieval needs --index"),
        ("index_alone", "--index needs --corpus-id"),
        ("citations_alone", "--verifier citations needs a corpus"),
        ("not_an_index", "not a CitationNeeded index directory"),
        ("foreign_corpus", "holds documents from corpus rabbithole-tasks-v1"),
        ("oracle_no_trajectory", "--provider oracle needs --trajectory"),
        ("oracle_no_question", "--provider oracle needs --question"),
        ("oracle_empty_question", "--provider oracle needs --question"),
        ("oracle_with_script", "--script is for --provider scripted"),
        ("scripted_with_trajectory", "--trajectory is for --provider oracle"),
        ("scripted_no_script", "--provider scripted needs --script"),
        ("trajectory_missing", "cannot read trajectory"),
        ("trajectory_not_json", "invalid trajectory"),
        ("trajectory_empty", "an oracle needs at least one action"),
        ("trajectory_bad_action", "invalid trajectory"),
    ],
)
def test_bad_corpus_and_oracle_input_exits_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    index_dir: Path,
    case: str,
    message: str,
) -> None:
    out = tmp_path / "t.jsonl"
    corpus = ["--index", str(index_dir), "--corpus-id", FIXTURE_CORPUS_ID]
    oracle = ["--provider", "oracle", "--trajectory", str(VELMORA), "--question", QUESTION]
    script = str(FIXTURES / "scripts" / "two_step.json")
    args: list[str] = {
        "corpus_id_alone": [*oracle, "--corpus-id", FIXTURE_CORPUS_ID],
        "retrieval_alone": [*oracle, "--retrieval", "hybrid"],
        "index_alone": [*oracle, "--index", str(index_dir)],
        "citations_alone": [*oracle, "--verifier", "citations"],
        "not_an_index": [*oracle, "--index", str(tmp_path), "--corpus-id", FIXTURE_CORPUS_ID],
        "foreign_corpus": [*oracle, "--index", str(index_dir), "--corpus-id", "other-corpus"],
        "oracle_no_trajectory": ["--provider", "oracle", "--question", QUESTION, *corpus],
        "oracle_no_question": ["--provider", "oracle", "--trajectory", str(VELMORA), *corpus],
        "oracle_empty_question": [*oracle, "--question", "", *corpus],
        "oracle_with_script": [*oracle, "--script", script, *corpus],
        "scripted_with_trajectory": [
            "--provider", "scripted", "--script", script, "--trajectory", str(VELMORA),
        ],
        "scripted_no_script": ["--provider", "scripted"],
        "trajectory_missing": [
            "--provider", "oracle", "--trajectory", str(tmp_path / "nope.json"),
            "--question", QUESTION,
        ],
        "trajectory_not_json": [
            "--provider", "oracle", "--trajectory", str(tmp_path / "x.json"),
            "--question", QUESTION,
        ],
        "trajectory_empty": [
            "--provider", "oracle", "--trajectory", str(_write(tmp_path, "e.json", [])),
            "--question", QUESTION,
        ],
        "trajectory_bad_action": [
            "--provider", "oracle", "--trajectory",
            str(_write(tmp_path, "b.json", [{"kind": "shrug"}])), "--question", QUESTION,
        ],
    }[case]  # fmt: skip
    if case == "trajectory_not_json":
        (tmp_path / "x.json").write_text("not json", encoding="utf-8")
    assert main(["run", *args, "--trace-out", str(out)]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


# --- arp tools schema ------------------------------------------------------------------


def test_tools_schema_lists_the_research_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools", "schema"]) == 0
    schemas = json.loads(capsys.readouterr().out)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["calculator", "search_corpus", "read_document"]
    # The schemas do not depend on the corpus behind the tools.
    fixture = FixtureSearchBackend(FIXTURE_CORPUS_ID, {"a.md": "Alpha."})
    assert schemas == research_registry(fixture).openai_schemas()
    assert main(["tools", "schema", "--tool", "read_document"]) == 0
    (read,) = json.loads(capsys.readouterr().out)
    assert set(read["function"]["parameters"]["required"]) == {"doc_id", "start", "end"}


async def test_the_catalog_research_tools_have_no_corpus() -> None:
    backend = NoCorpusBackend()
    assert backend.corpus_id == NO_CORPUS_ID
    with pytest.raises(BackendError, match="no corpus is loaded"):
        backend.search("q", 1)
    with pytest.raises(BackendError, match="no corpus is loaded"):
        backend.document(VELMORA_DOC_ID)
    executor = ToolExecutor(catalog_registry())
    calls: list[tuple[str, dict[str, Any]]] = [
        ("search_corpus", {"query": "Velmora"}),
        ("read_document", {"doc_id": VELMORA_DOC_ID, "start": 0, "end": 10}),
    ]
    for name, args in calls:
        result = (await executor.execute(name, args, ToolContext(run_id="r", step=0))).result
        error = result.error
        assert result.output is None and isinstance(error, ToolError), name
        assert (error.error_class, error.reason) == (ErrorClass.PERMANENT, "backend_error")
