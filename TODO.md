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

## M1 - Scaffold, tool contracts, registry, calculator, trace schema, local runtime and tool-calling probe (planned)
- [ ] Packaging (src layout, `agentic_research`, `arp` entry point), ruff, mypy --strict, pytest
- [ ] CI workflow (py3.11 + py3.13; lint, format check, mypy, `pytest -m "not slow"` with coverage)
- [ ] `Tool` protocol, `ToolContext`, `RetryPolicy`, `ToolResult`/`ToolError` + error-class enum
- [ ] `ToolRegistry` with JSON-schema export (OpenAI `tools` format), arg/output validation to structured errors
- [ ] `calculator` AST whitelist evaluator with bounds and table-driven rejection tests
- [ ] Trace schema v1 + JSONL writer/reader + round-trip test
- [ ] `arp tools schema` command
- [ ] `.env.example`, README skeleton (architecture / usage / evaluation "not run"), CHANGELOG
- [x] Copy approved project ADRs (0001-0009) into `docs/adr/` with index (2026-10-30)
- [ ] Install a local runtime (Ollama or llama.cpp `llama-server`, `--jinja`); pin Qwen3-1.7B Q4_K_M tag/digest, GGUF sha256, server version, ctx 4096 in `docs/local-models.md`
- [ ] `arp doctor` memory preflight (`MemAvailable` vs GGUF + KV cache + headroom; no swap) + tests with fake `/proc/meminfo`
- [ ] Minimal `OpenAICompatibleProvider` (chat + `tools`, timeout, usage) + MockTransport tests
- [ ] JSON-in-text tool protocol: strict parser, `<think>` strip, reason codes, `parse_failures` counter + table-driven tests
- [ ] Freeze commit: `probes/tool_calls.v1.yaml` (20 tool-expecting prompts) + `config/probe.toml` (`probe_sha256`, `min_valid = 18`, `n = 20`) before any probe run
- [ ] `arp probe tool-calls` (native vs json; refuses dirty tree / hash mismatch) -> raw `k/20` + 95% Wilson CI per format in `docs/providers.md` + `smoke-only` report; config test for default format

## Eval hygiene (from CitationNeeded M4 review; wire into M4/M6)
- [ ] Equal depth / like-for-like: same search `k`, budgets, and scoring cut across compared modes; test where one doc owns several top chunks; per-task `n_retrieved < k`
- [ ] Latency: untimed warm-up per configuration; exclude model/index load from p50/p95; report `load_ms` separately
- [ ] Provenance: reports record `git_sha` + `dirty_tree`; refuse or mark dirty-tree results for claims
- [ ] Degenerate labels: no-relevant-gold queries excluded and counted as `n_no_relevant` (not scored 0)
- [ ] Tiny suites labelled `smoke-only` and kept out of README claim tables
- [ ] Headline eligibility: only clean, non-smoke `live-local` (or model-free) reports feed claim tables; hosted reports `headline_eligible: false`
- [ ] Thresholds from the first `live-local` dev baseline, written to `evals/thresholds.v1.json` before any `test` run
