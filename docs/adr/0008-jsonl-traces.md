# ADR-0008: Versioned JSONL traces instead of OpenTelemetry

Status: Accepted (2026-10-30)

## Context
Every run must be inspectable and gradable after the fact, and the failure
classifier and graders consume traces. Observability infrastructure belongs
to project 07, WhoSpentMyTokens (controller id `ai-observability-platform`;
portfolio ADR-006).

## Decision
One JSONL file per run with pydantic-validated spans (run, step,
provider_call, tool_call), parent ids, monotonic timings, token usage,
budget counters, errors with class, and guardrail events. The file carries a
`schema_version`. `arp trace show` renders a run.

## Alternatives considered
- **OpenTelemetry SDK + collector**: industry standard, but needs a collector
  to be useful, adds infrastructure, and overlaps with WhoSpentMyTokens (07).
- **Langfuse / Phoenix**: hosted or server components, and unnecessary for
  offline evaluation.

## Consequences
Traces are diffable, committed as eval artifacts where licensing allows, and
directly consumed by graders. Span names follow OTel GenAI conventions where
practical, so an exporter can be added later without changing the loop.
