"""Render a trace as an indented, human-readable tree (``arp trace show``).

One line per span, children under their parent: the run, then its steps in
index order, each with its provider calls and tool attempts in file order.
Events, errors and parse failures are shown on the span they belong to.
Every piece of text from the trace (questions, answers, tool output, error
messages) is shown on one line with control characters escaped, so a trace
cannot inject terminal escape sequences or fake extra lines, and long values
are shortened to ``max_chars`` with a ``...`` marker.
"""

from __future__ import annotations

import json

from agentic_research.trace.schema import (
    ProviderCallSpan,
    RunSpan,
    SpanError,
    SpanEvent,
    StepSpan,
    ToolCallSpan,
    Trace,
)

DEFAULT_MAX_CHARS = 200


def _shorten(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _one_line(text: str, max_chars: int) -> str:
    """Free text with control characters (newlines, ESC, ...) escaped JSON-style."""
    return _shorten(json.dumps(text, ensure_ascii=False)[1:-1], max_chars)


def _json(value: object, max_chars: int) -> str:
    """Compact JSON; ``json.dumps`` already escapes control characters inside strings."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _shorten(text, max_chars)


def _span_time(start_s: float, end_s: float) -> str:
    return f"[{start_s:.3f}s-{end_s:.3f}s]"


def _limit(used: float, limit: float | None, fmt: str = "{}") -> str:
    shown = "-" if limit is None else fmt.format(limit)
    return f"{fmt.format(used)}/{shown}"


def _error(error: SpanError, max_chars: int) -> str:
    message = _one_line(error.message, max_chars)
    return f"error {error.error_class.value}/{error.reason}: {message}"


def _events(events: list[SpanEvent], indent: str, max_chars: int) -> list[str]:
    lines = []
    for event in events:
        attrs = f" {_json(event.attributes, max_chars)}" if event.attributes else ""
        lines.append(f"{indent}event {event.name}{attrs} @{event.at_s:.3f}s")
    return lines


def _run_lines(run: RunSpan, max_chars: int) -> list[str]:
    describe = [f"provider={_one_line(run.provider, max_chars)}"]
    if run.model is not None:
        describe.append(f"model={_one_line(run.model, max_chars)}")
    if run.tool_format is not None:
        describe.append(f"tool_format={run.tool_format}")
    lines = [
        f"run {run.status.value}  {' '.join(describe)}  {_span_time(run.start_s, run.end_s)}",
        f"  question: {_one_line(run.question, max_chars)}",
    ]
    if run.task_id is not None:
        lines.append(f"  task_id: {_one_line(run.task_id, max_chars)}")
    if run.answer is not None:
        lines.append(f"  answer: {_one_line(run.answer, max_chars)}")
    if run.error is not None:
        lines.append(f"  {_error(run.error, max_chars)}")
    used, limits = run.budget_used, run.budget_limits
    lines.append(
        "  budget: "
        f"steps {_limit(used.steps, limits.steps)}, "
        f"tool_calls {_limit(used.tool_calls, limits.tool_calls)}, "
        f"tokens {_limit(used.tokens, limits.tokens)}, "
        f"time {_limit(used.time_s, limits.time_s, '{:.3f}s')}"
    )
    lines.append(f"  parse_failures: {run.parse_failures}")
    lines.extend(_events(run.events, "  ", max_chars))
    return lines


def _provider_line(span: ProviderCallSpan, max_chars: int) -> str:
    parts = [
        f"provider_call {_one_line(span.provider, max_chars)}/{_one_line(span.model, max_chars)}",
        f"format={span.tool_format}",
    ]
    if span.attempt != 1:
        parts.append(f"attempt={span.attempt}")
    tokens_in = "-" if span.input_tokens is None else str(span.input_tokens)
    tokens_out = "-" if span.output_tokens is None else str(span.output_tokens)
    parts.append(f"tokens in={tokens_in} out={tokens_out}")
    if span.finish_reason is not None:
        parts.append(f"finish={_one_line(span.finish_reason, max_chars)}")
    if span.parse_failure is not None:
        detail = _one_line(span.parse_failure.detail, max_chars)
        parts.append(f"parse_failure {span.parse_failure.reason}: {detail}")
    if span.error is not None:
        parts.append(_error(span.error, max_chars))
    parts.append(_span_time(span.start_s, span.end_s))
    return "  ".join(parts)


def _tool_line(span: ToolCallSpan, max_chars: int) -> str:
    if span.output is not None:
        outcome = f"-> output {_json(span.output, max_chars)}"
    else:
        assert span.error is not None
        outcome = f"-> {_error(span.error, max_chars)}"
    return (
        f"tool_call {_one_line(span.tool, max_chars)} attempt {span.attempt}  "
        f"args {_json(span.arguments, max_chars)}  {outcome}  "
        f"{_span_time(span.start_s, span.end_s)}"
    )


def format_trace(trace: Trace, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The whole trace as text (no trailing newline)."""
    if max_chars < 4:
        raise ValueError("max_chars must be >= 4")
    header = trace.header
    lines = [
        f"trace {header.run_id}  schema v{header.schema_version}  "
        f"started {header.started_at.isoformat()}  "
        f"generator {_one_line(header.generator, max_chars)}",
    ]
    run = trace.run
    lines.extend(_run_lines(run, max_chars))
    steps = sorted(
        (s for s in trace.children(run.span_id) if isinstance(s, StepSpan)),
        key=lambda s: s.index,
    )
    for step in steps:
        lines.append(f"  step {step.index}  {_span_time(step.start_s, step.end_s)}")
        for child in trace.children(step.span_id):
            if isinstance(child, ProviderCallSpan):
                lines.append(f"    {_provider_line(child, max_chars)}")
            else:  # `Trace` only allows provider and tool calls under a step
                assert isinstance(child, ToolCallSpan)
                lines.append(f"    {_tool_line(child, max_chars)}")
        lines.extend(_events(step.events, "    ", max_chars))
    return "\n".join(lines)
