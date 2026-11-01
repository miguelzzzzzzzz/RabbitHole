"""Trace schema v1: JSONL round-trip, version checks, and invariants."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agentic_research.tools import ErrorClass, ToolContext, ToolError, ToolRegistry
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.trace import (
    SCHEMA_VERSION,
    BudgetLimits,
    BudgetUsage,
    ParseFailure,
    ProviderCallSpan,
    RunSpan,
    RunStatus,
    Span,
    SpanError,
    SpanEvent,
    StepSpan,
    ToolCallSpan,
    Trace,
    TraceError,
    TraceHeader,
    TraceSchemaVersionError,
    TraceWriter,
    dumps_trace,
    loads_trace,
    read_trace,
    write_trace,
)

HEADER = TraceHeader(
    run_id="run-0001",
    started_at=datetime(2026, 11, 1, 1, 2, 3, 456789, tzinfo=UTC),
    generator="agentic-research-platform test",
)


def _calculator_output(expression: str) -> dict[str, Any]:
    """Run the real calculator through the registry and return its JSON output."""
    registry = ToolRegistry([CalculatorTool()])
    args = registry.validate_args("calculator", {"expression": expression})
    assert not isinstance(args, ToolError)
    tool = registry.get("calculator")
    out = asyncio.run(tool.run(args, ToolContext(run_id=HEADER.run_id, step=1)))
    validated = registry.validate_output("calculator", out)
    assert not isinstance(validated, ToolError)
    return validated.model_dump(mode="json")


def _spans() -> list[Span]:
    """A three-step run in span-end order (children before parents, run last).

    Step 0: the model reply fails to parse. Step 1: a calculator call times
    out, then succeeds on retry. Step 2: the model answers.
    """
    timeout = ToolError(error_class=ErrorClass.TIMEOUT, reason="timeout", message="took > 5s")
    return [
        ProviderCallSpan(
            span_id="p0",
            parent_id="s0",
            start_s=0.001,
            end_s=1.25,
            provider="local",
            model="qwen3:1.7b-q4_K_M",
            tool_format="json",
            input_tokens=412,
            output_tokens=37,
            finish_reason="stop",
            parse_failure=ParseFailure(reason="invalid_json", detail="Expecting ',' at 1:18"),
        ),
        StepSpan(span_id="s0", parent_id="r", start_s=0.0, end_s=1.3, index=0),
        ProviderCallSpan(
            span_id="p1",
            parent_id="s1",
            start_s=1.3,
            end_s=2.5,
            provider="local",
            model="qwen3:1.7b-q4_K_M",
            tool_format="json",
            input_tokens=470,
            output_tokens=22,
            finish_reason="stop",
        ),
        ToolCallSpan(
            span_id="t1a",
            parent_id="s1",
            start_s=2.5,
            end_s=7.5,
            tool="calculator",
            arguments={"expression": "2**10 / 4"},
            error=SpanError.from_tool_error(timeout),
            events=[SpanEvent(name="retry_scheduled", at_s=7.5, attributes={"backoff_s": 0.4})],
        ),
        ToolCallSpan(
            span_id="t1b",
            parent_id="s1",
            start_s=7.9,
            end_s=7.901,
            tool="calculator",
            attempt=2,
            arguments={"expression": "2**10 / 4"},
            output=_calculator_output("2**10 / 4"),
        ),
        StepSpan(span_id="s1", parent_id="r", start_s=1.3, end_s=7.95, index=1),
        ProviderCallSpan(
            span_id="p2",
            parent_id="s2",
            start_s=7.95,
            end_s=9.0,
            provider="local",
            model="qwen3:1.7b-q4_K_M",
            tool_format="json",
            input_tokens=520,
            output_tokens=15,
        ),
        StepSpan(span_id="s2", parent_id="r", start_s=7.95, end_s=9.01, index=2),
        RunSpan(
            span_id="r",
            start_s=0.0,
            end_s=9.02,
            question="What is 2**10 / 4?\u2028(line separator, NEL \x85, newline \n, emoji 🐇)",
            task_id="tool-calc-001",
            provider="local",
            model="qwen3:1.7b-q4_K_M",
            tool_format="json",
            status=RunStatus.ANSWERED,
            answer="256",
            budget_used=BudgetUsage(steps=3, tool_calls=2, tokens=1476, time_s=9.02),
            budget_limits=BudgetLimits(steps=8, tool_calls=6, tokens=8000, time_s=120.0),
            parse_failures=1,
        ),
    ]


@pytest.fixture
def trace() -> Trace:
    return Trace(header=HEADER, spans=_spans())


# --- round trip -------------------------------------------------------------


def test_calculator_output_in_fixture_is_real() -> None:
    assert _calculator_output("2**10 / 4") == {"value": 256.0}


def test_write_then_read_gives_an_identical_object(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_trace(trace, path)
    loaded = read_trace(path)
    assert loaded == trace
    assert loaded.model_dump() == trace.model_dump()
    # Re-serializing gives byte-identical text, so traces diff cleanly.
    assert dumps_trace(loaded) == path.read_text(encoding="utf-8")


def test_file_is_one_json_object_per_line_with_header_first(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    write_trace(trace, path)
    lines = path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    records = [json.loads(line) for line in lines[:-1]]
    assert len(records) == 1 + len(trace.spans)
    assert records[0]["kind"] == "trace"
    assert records[0]["schema_version"] == SCHEMA_VERSION == 1
    assert [r["kind"] for r in records[1:]] == [s.kind for s in trace.spans]
    assert records[-1]["parse_failures"] == 1


def test_streamed_trace_reads_back_identical(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with TraceWriter(path, HEADER) as writer:
        for span in trace.spans:
            writer.write(span)
    assert read_trace(path) == trace


def test_crlf_file_reads(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_bytes(dumps_trace(trace).replace("\n", "\r\n").encode())
    assert read_trace(path) == trace


def test_missing_final_newline_is_accepted(trace: Trace) -> None:
    text = dumps_trace(trace)
    assert loads_trace(text.removesuffix("\n")) == trace


def test_non_utc_started_at_round_trips() -> None:
    header = HEADER.model_copy(
        update={"started_at": datetime(2026, 11, 1, 9, tzinfo=timezone(timedelta(hours=8)))}
    )
    t = Trace(header=header, spans=_spans())
    assert loads_trace(dumps_trace(t)).header.started_at.utcoffset() == timedelta(hours=8)


_text = st.text(st.characters(codec="utf-8"), min_size=1, max_size=40)
_seconds = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=150, deadline=None)
@given(question=_text, answer=_text, detail=st.text(max_size=40), a=_seconds, b=_seconds)
def test_round_trip_property(question: str, answer: str, detail: str, a: float, b: float) -> None:
    start, end = min(a, b), max(a, b)
    spans: list[Span] = [
        ProviderCallSpan(
            span_id="p",
            parent_id="s",
            start_s=start,
            end_s=end,
            provider="scripted",
            model="m",
            tool_format="native",
            parse_failure=ParseFailure(reason="prose_outside_object", detail=detail),
        ),
        ToolCallSpan(
            span_id="t",
            parent_id="s",
            start_s=start,
            end_s=end,
            tool="calculator",
            arguments={"expression": question, "nested": [a, {"b": b}, None, True]},
            output={"value": b},
        ),
        StepSpan(span_id="s", parent_id="r", start_s=start, end_s=end, index=0),
        RunSpan(
            span_id="r",
            start_s=start,
            end_s=end,
            question=question,
            provider="scripted",
            status=RunStatus.ANSWERED,
            answer=answer,
            parse_failures=1,
        ),
    ]
    t = Trace(header=HEADER, spans=spans)
    assert loads_trace(dumps_trace(t)) == t


# --- schema version ---------------------------------------------------------


def _with_header(trace: Trace, **changes: Any) -> str:
    lines = dumps_trace(trace).split("\n")
    header = json.loads(lines[0])
    for key, value in changes.items():
        if value is _DELETE:
            del header[key]
        else:
            header[key] = value
    lines[0] = json.dumps(header)
    return "\n".join(lines)


_DELETE = object()


@pytest.mark.parametrize("version", [2, 0, -1, "1", 1.0, True, None, [1]])
def test_schema_version_mismatch_raises_a_clear_error(trace: Trace, version: object) -> None:
    text = _with_header(trace, schema_version=version)
    with pytest.raises(TraceSchemaVersionError) as excinfo:
        loads_trace(text, source="old.jsonl")
    message = str(excinfo.value)
    assert message.startswith("old.jsonl: trace schema_version ")
    assert repr(version) in message
    assert "understands schema_version 1" in message
    assert excinfo.value.found == version
    assert isinstance(excinfo.value, TraceError)


def test_missing_schema_version_is_reported(trace: Trace) -> None:
    with pytest.raises(TraceError, match="no schema_version"):
        loads_trace(_with_header(trace, schema_version=_DELETE))


def test_version_is_checked_before_the_rest_of_the_header(trace: Trace) -> None:
    # A v2 header with fields v1 does not know is a version error, not a field error.
    text = _with_header(trace, schema_version=2, new_field="x", run_id=None)
    with pytest.raises(TraceSchemaVersionError):
        loads_trace(text)


def test_header_model_rejects_other_versions() -> None:
    with pytest.raises(ValidationError):
        TraceHeader.model_validate({**HEADER.model_dump(), "schema_version": 2})


# --- malformed files --------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(TraceError, match="empty trace file"):
        read_trace(path)


def test_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(TraceError, match="not valid UTF-8"):
        read_trace(path)


@pytest.mark.parametrize(
    ("first_line", "match"),
    [
        ("{not json", r"^x:1: header is not valid JSON"),
        ("[1, 2]", r"^x:1: first line must be a trace header"),
        ('{"kind": "run", "schema_version": 1}', r"^x:1: first line must be a trace header"),
    ],
)
def test_bad_first_line(trace: Trace, first_line: str, match: str) -> None:
    lines = dumps_trace(trace).split("\n")
    lines[0] = first_line
    with pytest.raises(TraceError, match=match):
        loads_trace("\n".join(lines), source="x")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"run_id": "has space"}, "run_id"),
        ({"started_at": "2026-11-01T01:02:03"}, "started_at"),  # naive datetime
        ({"surprise": 1}, "surprise"),
    ],
)
def test_invalid_header_fields(trace: Trace, change: dict[str, Any], match: str) -> None:
    with pytest.raises(TraceError, match=rf"^x:1: invalid trace header: .*{match}"):
        loads_trace(_with_header(trace, **change), source="x")


def _replace_line(trace: Trace, index: int, new: str) -> str:
    lines = dumps_trace(trace).split("\n")
    lines[index] = new
    return "\n".join(lines)


def test_invalid_span_line_reports_line_number(trace: Trace) -> None:
    with pytest.raises(TraceError, match=r"^x:3: invalid span"):
        loads_trace(_replace_line(trace, 2, "{oops"), source="x")


def test_unknown_span_kind(trace: Trace) -> None:
    with pytest.raises(TraceError, match=r"^x:2: invalid span"):
        loads_trace(_replace_line(trace, 1, '{"kind": "guardrail", "span_id": "g"}'), source="x")


def test_unknown_span_field_is_rejected(trace: Trace) -> None:
    line = json.loads(dumps_trace(trace).split("\n")[2])
    line["extra"] = 1
    with pytest.raises(TraceError, match=r"^x:3: invalid span: .*extra"):
        loads_trace(_replace_line(trace, 2, json.dumps(line)), source="x")


def test_blank_line_in_the_middle(trace: Trace) -> None:
    with pytest.raises(TraceError, match=r"^x:3: blank line"):
        loads_trace(_replace_line(trace, 2, "   "), source="x")


def test_incomplete_trace_without_run_span(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "crashed.jsonl"
    with TraceWriter(path, HEADER) as writer:
        for span in trace.spans[:-1]:  # the run span never gets written
            writer.write(span)
    with pytest.raises(TraceError, match="incomplete: no run span"):
        read_trace(path)


def test_header_only_is_incomplete() -> None:
    with pytest.raises(TraceError, match="incomplete"):
        loads_trace(HEADER.model_dump_json() + "\n")


def test_inconsistent_file_is_reported(trace: Trace) -> None:
    text = dumps_trace(trace)
    tampered = text.replace('"parse_failures":1', '"parse_failures":0')
    assert tampered != text
    with pytest.raises(TraceError, match=r"^x: inconsistent trace: .*parse_failures is 0"):
        loads_trace(tampered, source="x")


# --- cross-span invariants ---------------------------------------------------


def _without(spans: list[Span], *ids: str) -> list[Span]:
    return [s for s in spans if s.span_id not in ids]


def _replaced(spans: list[Span], span_id: str, **update: Any) -> list[Span]:
    return [s.model_copy(update=update) if s.span_id == span_id else s for s in spans]


def _invalid_trees() -> list[tuple[str, list[Span], str]]:
    spans = _spans()
    run = spans[-1]
    return [
        ("duplicate id", [*spans, spans[0]], "duplicate span_id 'p0'"),
        ("no run", spans[:-1], "exactly one run span, found 0"),
        ("two runs", [*spans, run.model_copy(update={"span_id": "r2"})], "found 2"),
        ("missing parent", _without(spans, "s0"), "parent 's0' not found"),
        ("tool under run", _replaced(spans, "t1b", parent_id="r"), "must be a step span, got run"),
        ("step under step", _replaced(spans, "s2", parent_id="s1"), "must be a run span, got step"),
        ("call under call", _replaced(spans, "p2", parent_id="p1"), "got provider_call"),
        ("index gap", _replaced(spans, "s2", index=3), r"0..n-1 without gaps, got \[0, 1, 3\]"),
        ("index repeat", _replaced(spans, "s2", index=1), r"got \[0, 1, 1\]"),
        ("parse count low", _replaced(spans, "r", parse_failures=0), "parse_failures is 0 but 1"),
        ("parse count high", _replaced(spans, "r", parse_failures=2), "parse_failures is 2 but 1"),
    ]


@pytest.mark.parametrize(
    ("spans", "match"),
    [pytest.param(s, m, id=name) for name, s, m in _invalid_trees()],
)
def test_invalid_trees_are_rejected(spans: list[Span], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Trace(header=HEADER, spans=spans)


def test_tree_helpers(trace: Trace) -> None:
    assert trace.run.span_id == "r"
    assert [s.span_id for s in trace.children("r")] == ["s0", "s1", "s2"]
    assert [s.span_id for s in trace.children("s1")] == ["p1", "t1a", "t1b"]
    assert trace.children("t1b") == []
    assert trace.spans[3].duration_s == pytest.approx(5.0)


# --- span validation ---------------------------------------------------------


def _run(**kw: Any) -> RunSpan:
    base: dict[str, Any] = {
        "span_id": "r",
        "start_s": 0.0,
        "end_s": 1.0,
        "question": "q",
        "provider": "scripted",
        "status": RunStatus.ABSTAINED,
    }
    return RunSpan(**(base | kw))


_ERR = SpanError(error_class=ErrorClass.TRANSIENT, reason="connection_refused", message="no server")


def test_run_status_rules() -> None:
    assert _run(status=RunStatus.FAILED, error=_ERR).error == _ERR
    assert _run(status=RunStatus.BUDGET_EXHAUSTED).answer is None
    with pytest.raises(ValidationError, match="failed run needs an error"):
        _run(status=RunStatus.FAILED)
    with pytest.raises(ValidationError, match="status abstained cannot carry an error"):
        _run(error=_ERR)
    with pytest.raises(ValidationError, match="answered run needs an answer"):
        _run(status=RunStatus.ANSWERED)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"start_s": 2.0, "end_s": 1.0}, "end_s < start_s"),
        ({"start_s": -0.1}, "greater than or equal to 0"),
        ({"end_s": float("nan")}, "finite number"),
        ({"end_s": float("inf")}, "finite number"),
        ({"span_id": ""}, "pattern"),
        ({"span_id": "x" * 65}, "pattern"),
        ({"parent_id": "p"}, "parent_id"),
        ({"question": ""}, "at least 1 character"),
        ({"parse_failures": -1}, "greater than or equal to 0"),
        ({"events": [SpanEvent(name="late", at_s=1.5)]}, "event late outside the span"),
        ({"budget_limits": {"steps": 0}}, "greater than or equal to 1"),
    ],
)
def test_run_span_field_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _run(**kw)


def test_spans_are_immutable() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _run().question = "changed"


def _provider(**kw: Any) -> ProviderCallSpan:
    base: dict[str, Any] = {
        "span_id": "p",
        "parent_id": "s",
        "start_s": 0.0,
        "end_s": 1.0,
        "provider": "local",
        "model": "m",
        "tool_format": "native",
    }
    return ProviderCallSpan(**(base | kw))


def test_provider_call_outcomes() -> None:
    assert _provider(error=_ERR).parse_failure is None
    assert _provider(parse_failure=ParseFailure(reason="unknown_tool")).error is None
    with pytest.raises(ValidationError, match="no reply to fail parsing"):
        _provider(error=_ERR, parse_failure=ParseFailure(reason="invalid_json"))
    with pytest.raises(ValidationError, match="tool_format"):
        _provider(tool_format="xml")
    with pytest.raises(ValidationError, match="pattern"):
        _provider(parse_failure=ParseFailure(reason="Invalid JSON"))
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _provider(attempt=0)


def _tool(**kw: Any) -> ToolCallSpan:
    base: dict[str, Any] = {
        "span_id": "t",
        "parent_id": "s",
        "start_s": 0.0,
        "end_s": 1.0,
        "tool": "calculator",
        "arguments": {"expression": "1+1"},
    }
    return ToolCallSpan(**(base | kw))


def test_tool_call_needs_exactly_one_of_output_or_error() -> None:
    assert _tool(output={"value": 2}).error is None
    assert _tool(error=_ERR).output is None
    with pytest.raises(ValidationError, match="exactly one of output or error"):
        _tool()
    with pytest.raises(ValidationError, match="exactly one of output or error"):
        _tool(output={"value": 2}, error=_ERR)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), [1.0, float("-inf")]])
def test_non_finite_json_values_are_rejected(bad: Any) -> None:
    # JSON cannot represent these; writing them would silently become null.
    with pytest.raises(ValidationError, match="without NaN or infinity"):
        _tool(arguments={"x": bad}, output={"value": 1})
    with pytest.raises(ValidationError, match="without NaN or infinity"):
        _tool(output={"value": bad})
    with pytest.raises(ValidationError, match="without NaN or infinity"):
        SpanEvent(name="e", at_s=0.0, attributes={"x": bad})


def test_non_json_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _tool(arguments={"x": (1, 2)}, output={"value": 1})
    with pytest.raises(ValidationError):
        _tool(arguments={"x": b"bytes"}, output={"value": 1})


def test_span_error_from_tool_error() -> None:
    error = ToolError(error_class=ErrorClass.INVALID_ARGS, reason="unknown_name", message="x")
    assert SpanError.from_tool_error(error).model_dump() == error.model_dump()


def test_span_error_from_tool_error_truncates_long_messages() -> None:
    error = ToolError(error_class=ErrorClass.PERMANENT, reason="upstream", message="x" * 5000)
    assert SpanError.from_tool_error(error).message == "x" * 2000
    with pytest.raises(ValidationError, match="at most 2000 characters"):
        SpanError(error_class=ErrorClass.PERMANENT, reason="upstream", message="x" * 2001)


# --- writers -----------------------------------------------------------------


def test_trace_writer_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        TraceWriter(path, HEADER)
    assert path.read_text() == "keep me"


def test_trace_writer_flushes_each_span(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = TraceWriter(path, HEADER)
    writer.write(trace.spans[0])
    # Visible on disk before close, so a crash keeps finished spans.
    assert len(path.read_text(encoding="utf-8").split("\n")) == 3
    writer.close()
    writer.close()  # idempotent


def test_trace_writer_rejects_duplicates_and_writes_after_close(
    trace: Trace, tmp_path: Path
) -> None:
    path = tmp_path / "run.jsonl"
    with TraceWriter(path, HEADER) as writer:
        writer.write(trace.spans[0])
        with pytest.raises(TraceError, match="duplicate span_id 'p0'"):
            writer.write(trace.spans[0])
    with pytest.raises(TraceError, match="writer is closed"):
        writer.write(trace.spans[1])
    assert len(path.read_text(encoding="utf-8").split("\n")) == 3


def test_write_trace_replaces_atomically(trace: Trace, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("old")
    write_trace(trace, path)
    assert read_trace(path) == trace
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.jsonl"]


def test_write_trace_failure_keeps_old_file_and_no_temp(
    trace: Trace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("old")

    def boom(_: Trace) -> str:
        raise RuntimeError("disk full")

    monkeypatch.setattr("agentic_research.trace.io.dumps_trace", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_trace(trace, path)
    assert path.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.jsonl"]
