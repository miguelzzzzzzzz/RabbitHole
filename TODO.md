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
- [x] When M3 starts: wire `RagEngineBackend` to index format 2
      (`documents.jsonl`, explicit `corpus_id`, `search --json` `hits` +
      `schema_version`; 2026-11-17). The citation check on normalized text +
      `content_hash` is the M3 `CitationVerifier` item below.

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
- [x] Chad's M2 spot-check nit (done 2026-11-17): a plain `pytest` with no `-m` filter passes without a local server (`test_live_server_serves_the_pinned_model` skips when unreachable); the live Qwen test accepts the empty `<think></think>` Qwen3 emits under `--reasoning-format none` via the shared think check

## M2 - Agent loop, run state, budgets, timeouts/retries, scripted provider (done 2026-11-16)
- [x] R-3 first (see above; 2026-11-08)
- [x] `RunState` + `Action` union (`ToolCall | FinalAnswer | GiveUp`) + `Policy` protocol + `Budgets` (`agentic_research.agent`; terminal statuses, JSON round-trip; 2026-11-09)
- [x] `ToolExecutor`: per-call timeout, seeded backoff retries for `transient`/`timeout` only, injectable clock/sleeper, optional deadline (`agentic_research.tools.executor`; 2026-11-10)
- [x] Chad M2 check 1: `idempotent` tool flag; registry refuses `timeout` in `retry_on` without it (2026-11-11)
- [x] Loop with budgets (steps, tool calls total/per tool, deadline, tokens, repair attempts), step timeout, verifier hook (`agentic_research.agent.loop.AgentLoop`; 2026-11-11)
- [x] Chad M2 check 2: run deadline passed into `ToolExecutor.execute`; attempts and backoff sleeps charged to `elapsed_s` + `tool_attempts`/`backoff_s`, with tests (2026-11-11)
- [x] Loop detection: identical (tool, normalized args) beyond a threshold gives a guardrail observation, then `failed:loop` (`agentic_research.agent.guards.LoopGuard`; 2026-11-12)
- [x] Observation shaping: deterministic truncation of tool outputs, eliding old observations to one-line summaries (no LLM) (`agentic_research.agent.observations`, `RunState.observations`; 2026-11-12)
- [x] `ScriptedProvider` (canned turns incl. malformed arguments/bodies and provider errors, JSON script files), `OraclePolicy`, `LLMPolicy` skeleton (native + json parsing, `parse_failures`, token charging), `PolicyFailure` -> `failed:<reason>` (2026-11-13)
- [x] Tracer on every span (run/step/provider_call/tool_call, incl. `parse_failure` on provider calls and per-attempt tool spans); `arp run --provider scripted --script FILE` and `arp trace show FILE`; golden two-step trace test (`tests/fixtures/scripts/two_step.json` -> `tests/fixtures/traces/two_step.golden.jsonl`, timings normalized) (`agentic_research.agent.tracing`, `trace.show`; 2026-11-14)
- [x] Request Chad's M2 review: done, reviewed at `f36a6c4` on 2026-11-14 (0 CRITICAL, 2 MAJOR, 7 MINOR, 7 OPTIONAL)
- [x] M2 review MAJOR-1: verifier bounded by step timeout and run deadline; `failed:verifier_timeout`; late acceptance not honoured (2026-11-15)
- [x] M2 review MAJOR-2: virtual-time budget property test asserts deadline (`DEADLINE_TOLERANCE_S`), tokens and repairs bounds; ADR-0006/SPEC/PLAN budget rules (Bogart's patch, reconciled) (2026-11-15)
- [x] M2 review MINOR-4: tokens of unparseable replies charged (`PolicyFailure.tokens`) (2026-11-15)
- [x] Chad's M2 re-check at `1200762`: PASS (0 CRITICAL, 0 MAJOR); M2 closes once new MINOR N-1 has an owner (2026-11-15)
- [x] Re-check N-1 (owner: Kim; done 2026-11-16, before M3's real verifier): `scope.expired()` checked after every timed policy, verifier and tool call, so an outcome after a fired limit (returned after swallowing the cancellation, or raised something else) is a timeout, not `succeeded`/`*_error`; `late_result_ignored` step event; README/CHANGELOG trace-overwrite wording made accurate
- [ ] M2 review MINOR-1 (before M4; owner: Kim): `arp run` publishes the trace without clobbering (`os.link` or `open(..., "x")`), exit 2 on a lost race, mid-run test (README/CHANGELOG wording already made accurate 2026-11-16; update it again when fixed)
- [ ] M2 review MINOR-2 (before M5/M6; owner: Kim): provider-call span (and tokens, if known) for a policy call cancelled by timeout
- [ ] M2 review MINOR-3 (before M4/M5 latency; owner: Kim): `started_at`/`ended_at` on `ProviderCallInfo`
- [ ] M2 review MINOR-5 (owner: Kim): `run_config` (seed, all limits, step timeout) and `budget_counters` events in the trace; `backoff_s` on failed attempt spans
- [ ] M2 review MINOR-6 (owner: Kim): schedule R-2 (and R-1) as M5 prerequisites in PLAN.md
- [ ] M2 review MINOR-7 (owner: Kim): `.env.example` budget sentence vs CLI flags
- [ ] M2 review OPT-1..OPT-7 (optional; see Chad's review)
- [x] Known limits raised with Chad (he classified them as MINOR-3, MINOR-2, MINOR-1): provider-call spans share the policy call's interval (one call per step in M2); a policy that times out reports no provider-call span; `arp run` checks for an existing trace file before the run, then replaces atomically (not an exclusive create)

## M3 - Research tools on the project-01 retriever, citations, guardrails (in progress)
- [x] Chad's hybrid nonsense-query note: `test_query_without_a_matching_term` already asserts hybrid returns k weak neighbours while lexical returns none, so abstention cannot rely on "no hits" (2026-11-17)
- [x] Search backends (2026-11-17): `Passage`/`StoredDocument`/`SearchBackend` contract, `parse_search_payload` (`schema_version` gate, `hits`), `RagEngineBackend` (format 2 only, explicit `corpus_id`, tamper refusal via `load_index`, lexical default + hybrid), `FixtureSearchBackend`; CitationNeeded `v1.0.0` git dependency (locked `b8015f3`); fictional fixture corpus `rabbithole-tasks-v1`; shared contract suite on both backends, pinned doc id, `search --json` CLI equivalence
- [x] Tools (2026-11-18): `search_corpus` (same arguments as the probe stub; the probe keeps its frozen v1 stub schema, pinned by sha256) and `read_document(doc_id, start, end)` (at most 1500 chars, end clamped), both over `SearchBackend` in a worker thread; `research_registry(backend)`; `EvidenceStore` built from the run's recorded steps (only accepted results count), keyed by `(doc_id, content_hash)`, `covers()` over merged spans
- [ ] Verifier slice: decide how a truncated observation (`Observation.truncated_chars > 0`) counts for groundedness (today the full returned span counts); replace or remove the unused M2 `RunState.evidence` / `FinalAnswer.evidence_ids` placeholder
- [ ] `arp run` over a corpus (`--index DIR --corpus-id ID`) with `research_registry`; `arp tools schema` listing the research tools
- [ ] `final_answer(answer, claims[{text, citations[...]}])` and `CitationVerifier` (validity, freshness/`stale_source`, groundedness, repair feedback) plugged into the loop's verifier hook; table-driven rejections
- [ ] `fetch_url` (fixture map by default; live only with `ARP_ALLOW_LIVE_FETCH=true` + host allowlist; socket-blocked test), HTML-to-text with offsets
- [ ] Guardrails: untrusted-data envelope, pattern-based injection detector (flags in trace + evidence store), per-task tool and URL allowlists; injection fixture pages
- [ ] Integration test: oracle trajectory (search, read, final_answer) over `RagEngineBackend` on the fixture index, accepted by the verifier
- [ ] Request Chad's M3 review

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
