# Planning history

Changes to the plan, spec, and ADRs made while RabbitHole was staged on the
portfolio box, before this repository existed (2026-10-09 to 2026-10-29).
Kept verbatim as history, newest first. Later entries supersede earlier ones;
in particular, the 2026-10-09 entries that made Cline-hosted DeepSeek the
default model were superseded on 2026-10-25 by ADR-0009 (local-first). Changes after the repository was created are in `CHANGELOG.md`.

## 2026-10-29

- CitationNeeded `v1.0.0` is tagged at `b8015f3` (not `977f091`). The
  `977f091..v1.0.0` diff touches only docs, the `pyproject.toml` version,
  `__version__` (`0.1.0` -> `1.0.0`) and a new `tests/test_version.py`; no
  dependency changes, and upstream CI was green before the tag. The pin
  `@v1.0.0` is now final, with the interface unchanged. Removed the "gated on
  977f091" and tag-slip notes and the `__version__` mismatch caveat. The
  adapter still gates on `schema_version` and index format, not `__version__`.
  ADR-0002 addendum added, TODO item closed; spec, PLAN and ADRS updated.

## 2026-10-28

- CitationNeeded pin moved from `@v0.1.0` to `@v1.0.0`
  (`production-rag-engine @ git+https://github.com/miguelzzzzzzzz/CitationNeeded@v1.0.0`),
  gated on the `v1.0.0` tag landing at `977f091` (not tagged yet on
  2026-10-28). Interface diff `v0.1.0` (`bf0c83b`) to `977f091`: no breaking
  changes. `search --json` `schema_version` 1 and hit fields, index format 2,
  `documents.jsonl` fields and `content_hash`, `doc_id` derivation and
  `--corpus-id`, normalized-offset semantics, and the `save_index`/`load_index`
  signatures are unchanged. Upstream additions (`evaluate`/`answer` commands,
  read-only HTTP API, `compose_retriever`) are unused here.
- Adapter, `schema_version` check and citation verification unchanged; docs now
  list the hit fields and say not to gate on `rag_engine.__version__` (still
  `0.1.0` at `977f091`). ADR-0002 gains a 2026-10-28 note; spec section 5 and
  risk rows, PLAN dependency paragraph and M3, and TODO updated (diff item
  closed, tag-confirmation item added).

## 2026-10-25

- Local-first while Cline is paused (owner decision relayed by Kim: "Cline
  is off until the owner says otherwise"). New project ADR-0009; ADR-0003
  amended. The default agent/policy model is now Qwen3-1.7B (GGUF Q4_K_M) on
  CPU through an OpenAI-compatible local server (Ollama or llama.cpp
  `llama-server`, portfolio ADR-008), with a memory preflight before every
  live run (CPU-only box, about 3 GB to spare, no swap). This supersedes the
  2026-10-09 entries that made Cline-hosted `cline-pass/deepseek-v4.1-flash`
  the default.
- Tool-calling check moved from the first step of M5 (Cline) to M1 (local
  Qwen3-1.7B): `arp probe tool-calls` compares native `tool_calls` with a
  JSON-in-text protocol under a pre-registered decision rule. The JSON-in-text
  fallback has a strict parser and a `parse_failures` counter. M1 also gains
  the local runtime install, `arp doctor`, a minimal
  `OpenAICompatibleProvider`, and matching tests and acceptance criteria.
- Hosted path kept only as an optional OpenAI-compatible backend
  (`HostedProvider`: configurable base URL, model, and key env var; model
  allowlist per ADR-016). Cline Pass usage logging, the ledger, and HTTP
  402/429 handling now live only inside it. Every hosted run is "not run" in
  reports and the README, and hosted reports are `headline_eligible: false`.
- Eval plan and success criteria: every headline number comes from the local
  model; thresholds are set from a first `dev` baseline run
  (`evals/thresholds.v1.json`), not promised up front. Programmatic graders
  and eval hygiene (equal budgets and `k`, warm-up, dirty-tree flag,
  `smoke-only`) unchanged. Results table switched to local rows plus a
  separate optional-hosted table.
- The runaway guard is provider-agnostic and now covers local runs (server
  errors or crashes, timeouts, unparseable replies after repair, retry
  storms).
- Risks updated: small-model tool use and multi-hop weakness, unreliable
  native tool calls, CPU latency, RAM limits with no swap, runaway local
  runs; Cline risks folded into one optional-backend row.
- CitationNeeded pin stays `@v0.1.0`; TODO added to diff the `v1.0.0` search
  interface and bump the pin once `v1.0.0` is tagged.
- Tool-call probe reporting (Kim's approval follow-up): the 90% bar stays
  (at least 18/20). Results are reported as raw counts plus a 95% Wilson
  interval, with a note that n=20 gives a wide range (18/20 is about
  [0.70, 0.97]). The probe file (`probes/tool_calls.v1.yaml`) and the bar
  (`config/probe.toml`, with the probe file's sha256) must be committed and
  frozen before the probe runs; the probe refuses a dirty tree or a hash
  mismatch. Spec, PLAN M1 acceptance criteria and tests, ADR-0009, and TODO
  updated. Small-n rates elsewhere also use Wilson intervals.

## 2026-10-10

- Eval hygiene from Chad's CitationNeeded M4 review applied to the staged
  RabbitHole plan: equal depth / like-for-like comparisons (same search `k`,
  budgets, and scoring cut; multi-chunk-one-doc test; per-task
  `n_retrieved < k`); untimed warm-up with model/index `load_ms` excluded
  from p50/p95; reports record `git_sha` and a dirty-tree flag (refuse or
  mark dirty results for claims); no-relevant-gold queries counted as
  `n_no_relevant` instead of scored 0; tiny suites labelled `smoke-only`.

## 2026-10-09

- CitationNeeded consumer contract aligned to Chad's signed-off M3 interface
  at `cdd892e` (v0.1.0 tag still pending; pin stays
  `git+https://github.com/miguelzzzzzzzz/CitationNeeded@v0.1.0`, gated on the
  tag). Index format 2: `documents.jsonl` with `content_hash` and normalized
  text; offsets into that text (not raw files); `doc_id =
  sha256("<corpus_id>:<relpath>")[:16]`; `save_index` takes documents;
  `search --json` is `{schema_version, hits}` with per-stage `ranks`.
- `SearchBackend` / `RagEngineBackend` adapter: read `hits`, check
  `schema_version` (fail clearly if unknown), carry `doc_id`, `content_hash`,
  normalized offsets, `page`/`page_end`.
- Citation verification: slice normalized text from `documents.jsonl`; flag
  `content_hash` mismatch as `stale_source`. Removed any wording that implied
  raw-file offsets.
- Corpus setup: explicit fixed corpus ids `rabbithole-hotpotqa` and
  `rabbithole-tasks-v1` (never the folder-name fallback); format-2 build;
  pinned-`doc_id` tests.
- M3 acceptance/tests: adapter rejects unknown `schema_version`; tampered
  span caught; hash mismatch flagged stale; pinned `doc_id`.
- Risk row: upstream CitationNeeded contract drift, mitigated by the pinned
  tag, `schema_version` check, and adapter contract tests.
- ADR-0002 amended for the format-2 / searchable contract; ADR-0007 citation
  wording updated for normalized text and `content_hash`.

- Hosted model id switched to `cline-pass/deepseek-v4.1-flash` (DeepSeek V4.1
  Flash via Cline Pass). Same endpoint and key; pay-per-use credits are empty
  (HTTP 402 on the old `deepseek/` id). Bills to the owner's Cline Pass.
- Per-call `cost` on Pass calls is labelled **reference cost (Pass)** (for
  Pass limits, not money spent). Ledger and Asia/Manila daily rollups use the
  same label. `billing_tier: cline_pass`.
- Error handling: HTTP 402 is non-retryable; 429 Pass-limit/rate-limit uses
  back-off-then-stop. Both surface a clear message and count toward the
  runaway guard without burning retries. Plan milestones add a test line for
  each.
- Risk row added: Cline Pass 5-hour, weekly or monthly limit hit mid-run
  (mitigations: caching, resumable runs, local fallback, productive work only).
- Free-tier / pay-per-use-as-primary wording removed. Usage policy unchanged:
  Cline for productive work only.
- Pending "ADR-008 amendment" references replaced with ADR-016 (Cline DeepSeek
  v4.1 is the default LLM; local models are the free fallback). ADR-016 now
  names `cline-pass/deepseek-v4.1-flash` (controller commit `cd02101`).
- (Superseded on 2026-10-25 by ADR-0009; kept as history.) Provider policy:
  Cline-hosted `cline-pass/deepseek-v4.1-flash` was made the default
  agent policy model through `ClineProvider`, which unwraps the `data`
  envelope, logs cost and reasoning tokens, enforces a 1000 max_tokens floor,
  and has a spend guard. Local models are the zero-cost fallback and a
  reported secondary run. Scripted and replay providers remain for tests and
  CI. This follows ADR-016 (Cline DeepSeek v4.1 is the default LLM; local models are the free fallback).
- Every report records model id, date, provider, and reported cost. Added a
  spend/Pass-limit risk with mitigations.
- Spend guard defaults are fixed and configurable: $0.25 per run, and $1.00
  per day across all projects (a portfolio-wide cap using a shared local
  spend ledger). The call-count fallback is kept for when cost is not
  reported. The "set from a cost pilot" wording was removed.
- Spend policy: ADR-016 (Cline DeepSeek v4.1 is the default LLM; local models are the free fallback) supersedes
  the zero-spend parts of ADR-002 and ADR-011 for Cline DeepSeek only. Every
  other paid API stays at $0 unless the owner says otherwise.
- M5 now starts by checking tool-call support through Cline. The result is
  recorded in `docs/providers.md` and sets the default tool format.
- Spend caps removed (owner reversal): no per-run or daily spend limits, no
  call-count fallback, and no `*_MAX_SPEND_USD`, `CLINE_DAILY_MAX_SPEND_USD`,
  or `*_MAX_CALLS` settings or tests. This supersedes the previous fixed-cap
  entry.
- Per-call cost logging is kept. The shared ledger (`CLINE_USAGE_LEDGER`) is
  a cost and usage log only, with an Asia/Manila daily rollup (`arp usage`)
  that resets at midnight UTC+8. A missing cost is logged as a warning,
  never as zero.
- Added a runaway guard (max consecutive failures and max retries per run).
- Usage policy: Cline is used only for productive work (features, evals,
  tests, docs, reviews). Risk rows updated: spend is not capped but is logged
  and reported.
- Agent-loop step, tool-call, time, and token budgets are kept as behaviour
  limits. The `budget_exhausted:spend` status was removed, and
  `failed:runaway` was added.
