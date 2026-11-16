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
- **Loop detection** on repeated identical calls. Implemented (2026-11-12)
  as `agentic_research.agent.guards.LoopGuard`: calls are identical when the
  tool and the canonicalized arguments match, counted over the whole run.
  After `max_identical_calls` runs (default 3) the next identical call is
  refused with a `loop_detected` guardrail observation; repeating it again
  ends the run as `failed:loop`. The budget re-check runs first.
- **Injectable clock and sleeper**, so all of this is unit- and
  property-tested without real waiting.
- **Budget rules** (amendment 2026-11-15, from Chad's M2 review MAJOR-1/-2
  and MINOR-4; text drafted by Bogart, reconciled with
  `agentic_research.agent.state` and `agent/loop.py`). The budget kinds are
  exactly `BudgetKind`: `steps`, `tool_calls`, `tool_calls_per_tool`,
  `deadline`, `tokens`, `repairs`; exhaustion ends the run as
  `budget_exhausted:<kind>`.
  - **Limit semantics.** A budget is exhausted once its counter reaches the
    limit (`max_steps=3` allows 3 steps). Two exceptions:
    - `max_repairs` counts repairs *allowed*: it is exhausted once
      `repairs > max_repairs`, so `max_repairs=2` permits 2 repairs and the
      3rd rejection ends the run as `budget_exhausted:repairs`. In M2 a
      repair is a verifier rejection of a `FinalAnswer`. M5's single
      malformed-call repair is a separate thing, counted in
      `parse_failures` before and after repair, not in this budget.
    - `max_tokens` is **soft**. Usage is only known after a model call
      returns, so it is charged afterwards (`usage.total_tokens`, else
      prompt + completion, else 0 and the provider-call span records what
      the server reported). Replies that fail to parse are charged too
      (`PolicyFailure.tokens`). The counter may therefore pass the limit by
      at most one policy call's usage. When it has, a `FinalAnswer` or
      `GiveUp` from that same call is honoured, since it costs nothing
      more, and a `ToolCall` is refused. The next budget check ends the run
      as `budget_exhausted:tokens`. A policy call cancelled by a timeout
      reports no usage, so it is not charged (Chad's M2 MINOR-2, open),
      unless it still returns a `Decision` or raises `PolicyFailure` after
      the cancellation; that usage is charged (see below).
  - **The deadline is hard for everything the loop awaits**: policy calls,
    tool attempts, backoff sleeps and the verifier. The policy and the
    verifier each run under `asyncio.timeout` of the step timeout or the
    time left before the deadline, whichever is shorter; tools get the
    deadline from the executor. When the deadline is the bound that fired,
    the run ends as `budget_exhausted:deadline`; when the step timeout
    fired first, as `failed:policy_timeout` or `failed:verifier_timeout`
    (error class `timeout`). An answer the verifier accepted at or after
    the deadline (on the loop's clock) is not honoured: the run ends as
    `budget_exhausted:deadline` with an `answer_after_deadline` step event.
    All of this time is charged to `elapsed_s`. Overshoot past the deadline
    is limited to cancellation latency; the property test runs on a virtual
    event-loop clock, where that latency is zero, and states its tolerance
    as the named constant `DEADLINE_TOLERANCE_S` (float rounding only).
  - **A fired time limit is final** (2026-11-16, Chad's M2 re-check N-1).
    After a policy, verifier or tool attempt's `asyncio.timeout` has fired,
    its outcome never counts: one that catches the cancellation and then
    returns, or raises anything else, is the same timeout as one that let
    the cancellation through (`budget_exhausted:deadline`,
    `failed:policy_timeout`/`failed:verifier_timeout`, or the tool's
    `tool_timeout`/deadline error). The loop checks `scope.expired()` after
    every timed call, not only on `TimeoutError`, and records a
    `late_result_ignored` step event with what came back. A call that
    swallows the cancellation and keeps running is not cancelled again, so
    the deadline overshoot is bounded only for code that lets
    `CancelledError` propagate.

## Alternatives considered
- **Thread-pool timeouts**: threads cannot be cancelled, so work would leak.
- **Retry everything**: wastes budget on permanent errors and hides bugs.
- **Rely on the model to stop**: unenforceable.

## Consequences
Bounded cost and latency per run, with every termination explained in the
trace. Async adds some complexity to tools. CPU-bound tools (calculator) are
bounded by input limits instead of cancellation.
