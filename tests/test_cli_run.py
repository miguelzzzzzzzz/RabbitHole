"""``arp run --provider scripted`` and ``arp trace show`` (no model calls).

The golden test runs the two-step fixture script through the real CLI and
compares the written trace, with timings and the wall-clock start
normalized, to ``tests/fixtures/traces/two_step.golden.jsonl``. To refresh
the golden file after an intended trace change, run the test with
``ARP_UPDATE_GOLDEN=1`` and review the diff.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agentic_research.cli import main
from agentic_research.trace import read_trace

FIXTURES = Path(__file__).parent / "fixtures"
TWO_STEP = FIXTURES / "scripts" / "two_step.json"
GOLDEN = FIXTURES / "traces" / "two_step.golden.jsonl"
TIME_KEYS = {"start_s", "end_s", "at_s", "time_s"}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: 0.0 if k in TIME_KEYS and v is not None else _normalize(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def normalized_lines(path: Path) -> list[dict[str, Any]]:
    """Trace records with timings zeroed and the header's start and generator fixed."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0] |= {"started_at": "<started_at>", "generator": "<generator>"}
    return [_normalize(r) for r in records]


def write_script(tmp_path: Path, turns: list[dict[str, Any]], question: str | None = "Q?") -> Path:
    body: dict[str, Any] = {"schema_version": 1, "turns": turns}
    if question is not None:
        body["question"] = question
    path = tmp_path / "script.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def arp_run(*args: str | Path) -> int:
    return main(["run", "--provider", "scripted", *map(str, args)])


def test_two_step_run_matches_the_golden_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "trace.jsonl"
    assert arp_run("--script", TWO_STEP, "--run-id", "golden", "--trace-out", out) == 0
    assert capsys.readouterr().out.splitlines() == [
        "run golden: succeeded",
        "answer: 6*7 = 42.",
        f"trace: {out}",
    ]
    actual = normalized_lines(out)
    if os.environ.get("ARP_UPDATE_GOLDEN") == "1":  # pragma: no cover - maintenance only
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in actual)
        GOLDEN.write_text(text, encoding="utf-8")
    expected = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert actual == expected
    # The timings that were normalized away are still a valid, ordered trace.
    trace = read_trace(out)
    assert [s.span_id for s in trace.spans][-1] == "run"
    assert all(s.end_s <= trace.run.end_s for s in trace.spans)


def test_golden_file_pins_the_two_step_shape() -> None:
    records = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    kinds = [(r["kind"], r.get("span_id")) for r in records]
    assert kinds == [
        ("trace", None),
        ("provider_call", "step-0.provider-1"),
        ("tool_call", "step-0.tool-1"),
        ("step", "step-0"),
        ("provider_call", "step-1.provider-1"),
        ("step", "step-1"),
        ("run", "run"),
    ]
    tool = records[2]
    assert (tool["arguments"], tool["output"]) == ({"expression": "6*7"}, {"value": 42})
    assert records[-1]["budget_used"]["tokens"] == 138 + 167


def test_default_trace_path_and_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert arp_run("--script", TWO_STEP) == 0
    (written,) = (tmp_path / "traces").iterdir()
    assert written.name.startswith("run-") and written.suffix == ".jsonl"
    run_id = written.stem
    assert read_trace(written).header.run_id == run_id
    assert f"run {run_id}: succeeded" in capsys.readouterr().out


def test_unanswered_run_exits_1_and_still_writes_its_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = write_script(tmp_path, [{"error": {"kind": "unavailable", "message": "down"}}])
    out = tmp_path / "t.jsonl"
    assert arp_run("--script", script, "--trace-out", out, "--run-id", "r1") == 1
    assert "run r1: failed:local_server_unavailable" in capsys.readouterr().out
    run = read_trace(out).run
    assert run.error is not None and run.error.message == "down"


def test_question_override_tool_format_and_budgets(tmp_path: Path) -> None:
    turns = [
        {"content": '{"action":"tool_call","tool":"calculator","arguments":{"expression":"2+2"}}'},
        {"content": '{"action":"tool_call","tool":"calculator","arguments":{"expression":"3+3"}}'},
    ]
    script = write_script(tmp_path, turns, question=None)
    out = tmp_path / "t.jsonl"
    code = arp_run(
        "--script", script, "--question", "What is 2+2?", "--tool-format", "json",
        "--max-tool-calls", "1", "--max-tokens", "100", "--deadline-s", "60",
        "--max-steps", "5", "--seed", "3", "--trace-out", out,
    )  # fmt: skip
    assert code == 1
    trace = read_trace(out)
    assert trace.run.question == "What is 2+2?"
    assert trace.run.tool_format == "json"
    assert trace.run.events[0].attributes == {"kind": "tool_calls"}
    limits = trace.run.budget_limits
    assert (limits.steps, limits.tool_calls, limits.tokens, limits.time_s) == (5, 1, 100, 60.0)


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing", "cannot read script"),
        ("invalid", "invalid script"),
        ("no_question", "no question"),
        ("bad_run_id", "run_id must be"),
        ("exists", "already exists"),
    ],
)
def test_bad_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], setup: str, message: str
) -> None:
    script: Path = TWO_STEP
    out = tmp_path / "t.jsonl"
    extra: list[str] = []
    if setup == "missing":
        script = tmp_path / "nope.json"
    elif setup == "invalid":
        script = write_script(tmp_path, [])
    elif setup == "no_question":
        script = write_script(tmp_path, [{"content": "x"}], question=None)
    elif setup == "bad_run_id":
        extra = ["--run-id", "has space"]
    else:
        out.write_text("keep me", encoding="utf-8")
    assert arp_run("--script", script, "--trace-out", out, *extra) == 2
    assert message in capsys.readouterr().err
    if setup == "exists":
        assert out.read_text(encoding="utf-8") == "keep me"
    else:
        assert not out.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--max-steps", "0"],
        ["--deadline-s", "inf"],
        ["--deadline-s", "-1"],
        ["--provider", "openai"],
    ],
)
def test_argument_validation(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--provider", "scripted", "--script", str(TWO_STEP), *args])
    assert excinfo.value.code == 2
    assert "error" in capsys.readouterr().err


# --- arp trace show ------------------------------------------------------------------


def test_trace_show_prints_the_span_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "t.jsonl"
    assert arp_run("--script", TWO_STEP, "--run-id", "show", "--trace-out", out) == 0
    capsys.readouterr()
    assert main(["trace", "show", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("trace show  schema v1  started ")
    assert lines[1].startswith("run answered  provider=scripted model=scripted tool_format=native")
    assert lines[2:6] == [
        "  question: What is 6*7?",
        "  answer: 6*7 = 42.",
        lines[4],
        "  parse_failures: 0",
    ]
    assert lines[4].startswith("  budget: steps 2/8, tool_calls 1/-, tokens 305/-, time ")
    body = [line.split("  [")[0] for line in lines[6:]]
    assert body == [
        "  step 0",
        "    provider_call scripted/scripted  format=native  tokens in=120 out=18  "
        "finish=tool_calls",
        '    tool_call calculator attempt 1  args {"expression":"6*7"}  -> output {"value":42}',
        "  step 1",
        "    provider_call scripted/scripted  format=native  tokens in=160 out=7  finish=stop",
    ]


def test_trace_show_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["trace", "show", str(tmp_path / "nope.jsonl")]) == 2
    assert "cannot read trace" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "trace", "schema_version": 2}\n', encoding="utf-8")
    assert main(["trace", "show", str(bad)]) == 2
    assert "schema_version 2 is not supported" in capsys.readouterr().err


def test_trace_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["trace"]) == 0
    assert "show" in capsys.readouterr().out
