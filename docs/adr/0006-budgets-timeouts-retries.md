# ADR-0006: Budgets, timeouts, and retries enforced by the loop with classified errors

Status: Accepted (2026-10-30)

## Context
Agents fail by looping, over-calling tools, hanging on slow tools, or
retrying errors that will never succeed. Model-side instructions ("use at
most 5 tools") are not enforcement.

## Decision
- **Hard budgets in the loop**: steps, tool calls (total and per tool),
  wall-clock deadline, tokens, and repair attempts. Exhaustion produces a
  specific terminal status.
- **Errors classified**: `transient`, `timeout`, `permanent`,
  `invalid_args`, `bad_output`. Only `transient` and `timeout` retry, with
  exponential backoff and jitter from a seeded RNG. Amendment (2026-11-11,
  from Chad's M2 review check): `timeout` may only be retried by a tool that
  declares `idempotent = True`, enforced at registration, since a timed-out
  call may have completed on the other side.
- **Per-tool timeouts** via `asyncio.timeout`, so cancellation is real.
- **Loop detection** on repeated identical calls.
- **Injectable clock and sleeper**, so all of this is unit- and
  property-tested without real waiting.

## Alternatives considered
- **Thread-pool timeouts**: threads cannot be cancelled, so work would leak.
- **Retry everything**: wastes budget on permanent errors and hides bugs.
- **Rely on the model to stop**: unenforceable.

## Consequences
Bounded cost and latency per run, with every termination explained in the
trace. Async adds some complexity to tools. CPU-bound tools (calculator) are
bounded by input limits instead of cancellation.
