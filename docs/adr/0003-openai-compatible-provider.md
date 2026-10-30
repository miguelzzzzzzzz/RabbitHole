# ADR-0003: Local Qwen3-1.7B as the default provider, an optional hosted OpenAI-compatible backend, scripted and replay providers for tests and CI

Status: Accepted (2026-10-30). The draft was amended 2026-10-25 by ADR-0009
(local-first while Cline is paused). The earlier version made Cline-hosted
`cline-pass/deepseek-v4.1-flash` the default under controller ADR-016; that
default is suspended.

## Context
The owner paused Cline on 2026-10-25 ("off until the owner says otherwise").
Portfolio ADR-008 already sets local inference through OpenAI-compatible
servers. The box is CPU-only with about 3 GB of RAM to spare and no swap.
Tests and CI must run with no model at all.

What was learned about the hosted path (kept for the optional backend):
observed on 2026-10-09, the Cline endpoint
`https://api.cline.bot/api/v1/chat/completions` takes an OpenAI-style request
with a Bearer token, wraps the response as `{"data": {"choices": ...,
"usage": ...}}`, reports `cost`, `gateway_cost`, `market_cost`, and
`reasoning_tokens` (on Pass calls `cost` is a **reference cost (Pass)**, not
money spent), spent 25 of 30 output tokens on hidden reasoning, and took
about 5-9 s per tiny call. The Pass has 5-hour, weekly, and monthly limits.
Owner policy (ADR-016) allows only `cline-pass/deepseek-v4.1-flash` as a
hosted model.

## Decision
A single `ChatProvider` protocol shaped like OpenAI chat completions
(messages, `tools`, `tool_calls`, `usage`). Implementations:

- **`OpenAICompatibleProvider` (default, local)**: Qwen3-1.7B, quantized GGUF
  Q4_K_M, on CPU behind Ollama or llama.cpp `llama-server` (whichever
  installs cleanly in M1; `--jinja` for llama.cpp native tool calls).
  Temperature 0, fixed seed, non-thinking mode, pinned context length (starts
  at 4096). Model tag/digest, GGUF sha256, quantization, and server version
  are pinned in M1. A memory preflight (`arp doctor`: `MemAvailable` must
  cover GGUF size + KV cache + headroom) runs before every live run. Retries
  on 5xx/connection errors; a dead server is `local_server_unavailable`.
- **Tool-call format** is set by the M1 tool-calling probe on this model:
  native `tool_calls` if it meets the pre-registered decision rule, otherwise
  a JSON-in-text protocol with a strict parser (one JSON object, optional
  single fence, a leading `<think>` block stripped, no extraction from prose)
  and a `parse_failures` counter. Both paths are built and ablated.
  Schema-invalid or unparseable calls get one repair attempt that feeds back
  the validation error.
- **Runaway guard (all providers)**, a behaviour limit, not a spend limit: a
  run stops with `failed:runaway` after `ARP_MAX_CONSECUTIVE_FAILURES`
  (default 3) consecutive failed provider calls (including unparseable
  replies after repair) or `ARP_MAX_RETRIES_PER_RUN` (default 20) retries.
- **`HostedProvider` (optional, off by default)**: the same OpenAI-compatible
  client with a configurable base URL (`ARP_HOSTED_BASE_URL`), model
  (`ARP_HOSTED_MODEL`, checked against an allowlist that currently holds only
  `cline-pass/deepseek-v4.1-flash`), and key env var
  (`ARP_HOSTED_API_KEY_ENV`, default `CLINE_API_KEY`). Its Cline adapter:
  - unwraps `data.choices` / `data.usage`; a response without `data` is a
    typed `ProviderResponseError`
  - normalizes usage (prompt, completion, reasoning tokens; `cost`,
    `gateway_cost`, `market_cost`; missing values null, never 0)
  - `max_tokens` default and floor 1000 for that model; empty content with
    `finish_reason=length` is `truncated_by_reasoning`
  - HTTP 402 `payment_required`, non-retryable; 429
    `pass_limit_or_rate_limited`, back-off-then-stop; both count toward the
    runaway guard without burning retries
  - **Cline Pass usage logging (spend not capped)**: each call goes to the
    shared, file-locked ledger `CLINE_USAGE_LEDGER` (default
    `~/.local/state/cline-usage/ledger.jsonl`; local, never committed) with
    project id, run id, purpose, model, and an ISO timestamp with offset.
    `arp usage [--date YYYY-MM-DD]` rolls up by Asia/Manila day (midnight
    UTC+8), labelled "reference cost (Pass)". A missing cost is a warning
    (`cost_missing: true`), never zero; totals are labelled a lower bound.
  - Every hosted result has `billing_tier: cline_pass` and
    `headline_eligible: false`. While Cline is paused no hosted call is made
    and hosted rows read "not run".
- **`ScriptedProvider`**: canned completions for unit tests, including
  malformed ones, stray `<think>` blocks, and wrapped/unwrapped Cline
  payloads.
- **`ReplayProvider`**: recorded completions, see ADR-0005. Cassettes
  include provider and model id/digest and keep recorded usage (and, for
  hosted cassettes, cost) so replayed reports reproduce them. Replay never
  writes to the usage ledger.

Every result records model id, run date, provider, tool format, and
`parse_failures`. Any API key is read from the environment only and never
appears in logs, traces, cassettes, or reports, which a test asserts with a
sentinel key.

## Alternatives considered
- **Keep the hosted model as the default**: not possible while Cline is
  paused, and it would leave every headline number dependent on a paused
  service.
- **A larger local model (4B-8B)**: does not fit reliably in about 3 GB with
  no swap, and would be much slower on CPU.
- **Qwen3-0.6B as the default**: fits easily, but tool use is expected to be
  weaker; kept as an optional ablation.
- **Other hosted models or routers**: not allowed by the owner.
- **openai SDK**: does not know Cline's `data` envelope, and hides raw
  responses that replay needs.
- **LiteLLM**: broad provider coverage that this project doesn't need, and a
  large dependency.
- **In-process llama-cpp-python**: couples the agent to one runtime and puts
  model memory inside the agent process.

Spend policy: local runs are zero spend (portfolio ADR-011). If the owner
re-enables Cline, ADR-016's terms apply to the optional hosted backend only:
`cline-pass/deepseek-v4.1-flash` is the only allowed hosted id, Pass usage is
not spend-capped, per-call `cost` is logged as reference cost (Pass), and
Cline is used only for productive work (features, evals, tests, docs,
reviews).

## Consequences
- Headline results come from a 1.7B model on CPU. They may be modest,
  especially on multi-hop tasks; the project reports them honestly, sets
  thresholds from a baseline run, and publishes the failure taxonomy.
- Live evals are slow (CPU) and memory-bound; subset sizes come from a
  timing pilot, and the memory preflight prevents OOM-killed runs.
- Local results are reproducible offline with a pinned model digest; CI
  still relies on replay.
- The hosted backend stays small, mocked-tested, and ready, so hosted runs
  can be added later as a separately labelled extra without touching the
  loop.
- The whole system runs offline: oracle, heuristic, scripted, replay, and
  local modes.
