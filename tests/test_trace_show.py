"""format_trace: every span kind, errors, events, escaping and shortening."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentic_research.tools import ErrorClass
from agentic_research.trace import (
    BudgetLimits,
    BudgetUsage,
    ParseFailure,
    ProviderCallSpan,
    RunSpan,
    RunStatus,
    SpanError,
    SpanEvent,
    StepSpan,
    ToolCallSpan,
    Trace,
    TraceHeader,
)
from agentic_research.trace.show import format_trace

HEADER = TraceHeader(
    run_id="r1", started_at=datetime(2026, 11, 14, 1, 0, tzinfo=UTC), generator="gen\x1b[31m"
)


def failed_trace() -> Trace:
    retry = SpanError(error_class=ErrorClass.TRANSIENT, reason="reset", message="conn\nreset")
    return Trace(
        header=HEADER,
        spans=[
            ToolCallSpan(
                span_id="s0.t1",
                parent_id="s0",
                start_s=0.0,
                end_s=0.5,
                tool="calc",
                arguments={"expression": "1+1"},
                error=retry,
            ),
            ToolCallSpan(
                span_id="s0.t2",
                parent_id="s0",
                start_s=1.0,
                end_s=1.5,
                tool="calc",
                attempt=2,
                arguments={"expression": "1+1"},
                output={"value": 2},
            ),
            StepSpan(
                span_id="s0",
                parent_id="run",
                start_s=0.0,
                end_s=1.5,
                index=0,
                events=[SpanEvent(name="loop_detected", at_s=1.5, attributes={"tool": "calc"})],
            ),
            ProviderCallSpan(
                span_id="s1.p1",
                parent_id="s1",
                start_s=1.5,
                end_s=2.0,
                provider="local",
                model="qwen",
                tool_format="json",
                attempt=2,
                parse_failure=ParseFailure(reason="invalid_json", detail="x" * 50),
            ),
            ProviderCallSpan(
                span_id="s1.p2",
                parent_id="s1",
                start_s=2.0,
                end_s=2.0,
                provider="local",
                model="qwen",
                tool_format="json",
                error=SpanError(error_class=ErrorClass.TRANSIENT, reason="down", message="d"),
            ),
            StepSpan(span_id="s1", parent_id="run", start_s=1.5, end_s=2.0, index=1),
            RunSpan(
                span_id="run",
                start_s=0.0,
                end_s=2.0,
                question="line one\nline two",
                task_id="t-9",
                provider="local",
                status=RunStatus.FAILED,
                error=SpanError(
                    error_class=ErrorClass.BAD_OUTPUT, reason="invalid_json", message="m"
                ),
                budget_used=BudgetUsage(steps=1, tool_calls=1, tokens=7, time_s=2.0),
                budget_limits=BudgetLimits(steps=4, tokens=100, time_s=30.0),
                parse_failures=1,
                events=[SpanEvent(name="note", at_s=2.0)],
            ),
        ],
    )


def test_format_failed_trace() -> None:
    text = format_trace(failed_trace(), max_chars=40)
    assert text.splitlines() == [
        "trace r1  schema v1  started 2026-11-14T01:00:00+00:00  generator gen\\u001b[31m",
        "run failed  provider=local  [0.000s-2.000s]",
        "  question: line one\\nline two",
        "  task_id: t-9",
        "  error bad_output/invalid_json: m",
        "  budget: steps 1/4, tool_calls 1/-, tokens 7/100, time 2.000s/30.000s",
        "  parse_failures: 1",
        "  event note @2.000s",
        "  step 0  [0.000s-1.500s]",
        '    tool_call calc attempt 1  args {"expression":"1+1"}  '
        "-> error transient/reset: conn\\nreset  [0.000s-0.500s]",
        '    tool_call calc attempt 2  args {"expression":"1+1"}  '
        '-> output {"value":2}  [1.000s-1.500s]',
        '    event loop_detected {"tool":"calc"} @1.500s',
        "  step 1  [1.500s-2.000s]",
        "    provider_call local/qwen  format=json  attempt=2  tokens in=- out=-  "
        f"parse_failure invalid_json: {'x' * 37}...  [1.500s-2.000s]",
        "    provider_call local/qwen  format=json  tokens in=- out=-  "
        "error transient/down: d  [2.000s-2.000s]",
    ]


def test_steps_print_in_index_order_and_long_json_is_shortened() -> None:
    trace = Trace(
        header=HEADER,
        spans=[
            StepSpan(span_id="b", parent_id="run", start_s=0.0, end_s=0.0, index=1),
            ToolCallSpan(
                span_id="a.t",
                parent_id="a",
                start_s=0.0,
                end_s=0.0,
                tool="t",
                arguments={"q": "y" * 100},
                output={},
            ),
            StepSpan(span_id="a", parent_id="run", start_s=0.0, end_s=0.0, index=0),
            RunSpan(
                span_id="run",
                start_s=0.0,
                end_s=0.0,
                question="q",
                provider="p",
                status=RunStatus.ABSTAINED,
            ),
        ],
    )
    lines = format_trace(trace, max_chars=10).splitlines()
    assert [line.split("  [")[0] for line in lines if line.startswith("  step")] == [
        "  step 0",
        "  step 1",
    ]
    assert '    tool_call t attempt 1  args {"q":"y...  -> output {}  [0.000s-0.000s]' in lines


def test_max_chars_floor() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        format_trace(failed_trace(), max_chars=3)
