# ADR-0005: Record/replay cassettes for deterministic, zero-cost CI evaluation

Status: Accepted (2026-10-30)

## Context
Live runs are not guaranteed deterministic. Local runs on the default
Qwen3-1.7B model are slow and hardware-dependent, and optional hosted calls
(if Cline is re-enabled) bill against Cline Pass limits. CI must
never call a model. Published numbers must still be
reproducible from committed artifacts.

## Decision
`RecordingProvider` stores each provider response keyed by a hash of the
normalized request (messages, tools, sampling parameters, model id).
`ReplayProvider` serves responses by key and fails loudly on a miss. Tool
outputs are not stored; they are recomputed at replay time. This also checks
tool determinism and keeps CC BY-SA corpus text out of committed cassettes,
apart from what model outputs themselves quote, which is covered by an
attribution notice. A CI job replays every committed cassette set and asserts
that the non-latency metrics equal the committed report.

## Alternatives considered
- **Re-run models in CI**: cost, time, and flakiness.
- **Commit reports only**: numbers become unverifiable.
- **Store full traces including tool outputs**: larger files, licence
  complications, and it would not catch tool non-determinism.

## Consequences
CI proves that the committed numbers come from the committed code plus the
recorded model outputs. It does not prove the model would produce the same
outputs today, and the docs say so. Any prompt or tool-schema change
invalidates cassettes (by design), so the affected configs must be re-run
locally before the change lands.
