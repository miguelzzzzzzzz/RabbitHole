# TODO

Short-horizon task list. Milestones and acceptance criteria live in `PLAN.md`.

## Upstream (CitationNeeded)

- [x] CitationNeeded `v0.1.0` tag exists (`bf0c83b`, contains `cdd892e`).
- [x] Diff the `v1.0.0` interface against `v0.1.0` (2026-10-28, against
      `977f091`): no breaking changes; pin bumped to
      `git+https://github.com/miguelzzzzzzzz/CitationNeeded@v1.0.0`;
      ADR-0002 note added.
- [x] `v1.0.0` tagged at `b8015f3` (not `977f091`); re-verified 2026-10-29:
      interface unchanged from `977f091` (version string only). Pin final.
- [ ] When M3 starts: wire `RagEngineBackend` to index format 2
      (`documents.jsonl`, explicit `corpus_id`, `search --json` `hits` +
      `schema_version`, citation check on normalized text + `content_hash`).

## Model policy (2026-10-25, ADR-0009: local-first while Cline is paused)
- [ ] Do not call the Cline API until the owner re-enables it
- [ ] Default model: Qwen3-1.7B GGUF Q4_K_M on CPU via Ollama or llama.cpp `llama-server`
- [ ] Hosted backend optional only; hosted rows "not run" in reports and README
- [ ] If Cline returns: add hosted runs as a separately labelled extra (revisit ADR-0009)

## M1 - Scaffold, tool contracts, registry, calculator, trace schema, local runtime and tool-calling probe (done 2026-11-08)
- [x] Packaging (src layout, `agentic_research`, `arp` entry point), ruff, mypy --strict (src and tests), pytest, 96% branch-coverage floor, autouse network guard (2026-10-30)
- [x] CI workflow (py3.11 + py3.13; lint, format check, mypy, `pytest -m "not slow"` with coverage) (2026-10-30)
- [x] `Tool` protocol, `ToolContext`, `RetryPolicy`, `ToolResult`/`ToolError` + error-class enum (2026-10-30)
- [x] `ToolRegistry` with JSON-schema export (OpenAI `tools` format, checked against the JSON Schema meta-schema), arg/output validation to structured errors (2026-10-30)
- [x] `calculator` AST whitelist evaluator with bounds, table-driven rejection tests, and hypothesis agreement tests; `ToolFailure` for classified tool failures (2026-10-31)
- [x] Trace schema v1 (`agentic_research.trace`: header + run/step/provider_call/tool_call spans, `parse_failures`), JSONL `write_trace`/`TraceWriter`/`read_trace`, round-trip and schema-version mismatch tests (2026-11-01)
- [x] `arp tools schema` command (`--tool NAME`; built-in set from `tools.builtin.default_registry()`; output checked against the JSON Schema meta-schema) (2026-11-01)
- [x] `.env.example` (checked against the `ARP_*` names in the docs; secrets empty), README skeleton (architecture / usage / evaluation "not run"), CHANGELOG (2026-11-02)
- [x] Copy approved project ADRs (0001-0009) into `docs/adr/` with index (2026-10-30)
- [x] Install a local runtime: llama.cpp `llama-server` b12032 CPU build + Qwen3-1.7B Q4_K_M (`unsloth/Qwen3-1.7B-GGUF` @ `d7f544e`), sha256s, server version, ctx 4096, flags and reported memory buffers pinned in `docs/local-models.md`; `scripts/local_runtime.sh install|verify|serve` (2026-11-02)
- [x] `arp doctor` memory preflight (`MemAvailable` vs model + KV + compute buffers + headroom; repacked weights make this about 1.8 GiB, not GGUF + KV, see `docs/local-models.md`; swap reported but never counted) + pinned serve-flag check (`-c 4096`, `--cache-ram 0`) + `/v1/models` check + tests with fake `/proc/meminfo` (2026-11-03)
- [x] Minimal `OpenAICompatibleProvider` (chat + `tools`, timeout, usage, native `tool_calls` parsing) + MockTransport tests + one slow live Qwen test (2026-11-04)
- [x] JSON-in-text tool protocol: strict parser, `<think>` strip, reason codes, `parse_failures` counter + table-driven tests (2026-11-05)
- [x] Freeze commit: `probes/tool_calls.v1.yaml` (20 tool-expecting prompts) + `config/probe.toml` (`probe_sha256`, `min_valid = 18`, `n = 20`) before any probe run (2026-11-06)
- [x] `arp probe tool-calls` implemented (native vs json; refuses dirty tree / hash mismatch; Wilson CI; decision rule; smoke-only report) (2026-11-06)
- [x] Live `arp probe tool-calls` on the box (2026-11-06): native 20/20, json 10/20, chosen default `native`; report + docs/providers.md + ARP_TOOL_FORMAT=native committed
- [x] Chad's M1 re-check at `60e7ad9`: PASS (0 CRITICAL, 0 MAJOR); M1 may close and M2 start with the MINORs below tracked (2026-11-07)
- [x] Chad M1 FAIL remediation: observe doctor flags, `--reasoning-format none` + `reasoning_content`, MINOR-1..8 (OPT-2 deferred)
- [ ] OPT-2 (deferred, accepted by Chad): typed pydantic `ProbeReport` schema (owner: Kim)

## M1 follow-ups from Chad's re-check (non-blocking MINORs; R-8 withdrawn)
- [x] R-3 (priority, early M2; owner: Kim; done 2026-11-08): tests for a non-empty `<think>` in content in both scorers (native must fail even with valid `tool_calls`); return `think_block_unclosed` from the shared think check and test it on the native path
- [ ] R-1 (owner: Kim): `/props` only fills gaps; either compare `n_ctx`/`total_slots` against the cmdline and refuse on mismatch, or say "fills from `/props`" in docs/CHANGELOG; delete the dead `reasoning_format` branch
- [ ] R-2 (owner: Kim): doctor refuses when the observed GGUF sha256 or `llama-server --version` differs from the pins, or when hashing failed; tests for both
- [ ] R-4 (owner: Kim): one test composing `observe_runtime(proc_root=fake)` (fake `net/tcp`, `fd/` symlinks, `cmdline`) through `run_doctor` and `runtime_metadata`, asserting `observed` provenance
- [ ] R-5 (owner: Kim): render box-local paths in reports/docs as `$ARP_RUNTIME_DIR/...` or basename + sha256; keep raw cmdline only if labelled box-local
- [ ] R-6 (owner: Kim): produce the live smoke artifact from code (real timestamp, observed metadata) or relabel it as a hand-scrubbed transcription
- [ ] R-7 (owner: Kim): document the capture configuration and scrubbing of `llama_server_chat_completion_tool_reasoning.json`, or label it synthetic
- [x] R-9 (owner: Kim; done 2026-11-09): one sentence in `docs/providers.md`: JSON moved 10 to 6 between runs with different `--reasoning-format` settings; no causal claim from two runs
- [ ] OPT-1 (optional; owner: Kim): replace the always-skipping `test_live_probe_marker_exists_for_manual_runs`

## M2 - Agent loop, run state, budgets, timeouts/retries, scripted provider (in progress)
- [x] R-3 first (see above; 2026-11-08)
- [x] `RunState` + `Action` union (`ToolCall | FinalAnswer | GiveUp`) + `Policy` protocol + `Budgets` (`agentic_research.agent`; terminal statuses, JSON round-trip; 2026-11-09)
- [x] `ToolExecutor`: per-call timeout, seeded backoff retries for `transient`/`timeout` only, injectable clock/sleeper, optional deadline (`agentic_research.tools.executor`; 2026-11-10)
- [ ] Loop with budgets (steps, tool calls total/per tool, deadline, tokens, repair attempts), loop detection, observation shaping
- [ ] `ScriptedProvider`, `LLMPolicy` skeleton, `OraclePolicy`; tracer on every span; `arp run --provider scripted` and `arp trace show`

## Later (from M1)
- [ ] M5: wire `ARP_TOOL_FORMAT` from env into the agent loop (OPT-3; currently only in `.env.example`)
- [ ] Before M6 native-vs-JSON ablation: `tool_calls.v2` / `rule_version` 2 to fix contradictory json system prompt (MINOR-1)

## Eval hygiene (from CitationNeeded M4 review; wire into M4/M6)
- [ ] Equal depth / like-for-like: same search `k`, budgets, and scoring cut across compared modes; test where one doc owns several top chunks; per-task `n_retrieved < k`
- [ ] Latency: untimed warm-up per configuration; exclude model/index load from p50/p95; report `load_ms` separately
- [ ] Provenance: reports record `git_sha` + `dirty_tree`; refuse or mark dirty-tree results for claims
- [ ] Degenerate labels: no-relevant-gold queries excluded and counted as `n_no_relevant` (not scored 0)
- [ ] Tiny suites labelled `smoke-only` and kept out of README claim tables
- [ ] Headline eligibility: only clean, non-smoke `live-local` (or model-free) reports feed claim tables; hosted reports `headline_eligible: false`
- [ ] Thresholds from the first `live-local` dev baseline, written to `evals/thresholds.v1.json` before any `test` run
