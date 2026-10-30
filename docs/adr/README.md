# Architecture decision records

Project-level ADRs for RabbitHole, one file per decision, in the same format
as CitationNeeded (`Status`, `Context`, `Decision`, `Alternatives considered`,
`Consequences`).

Portfolio-level decisions are already Accepted in the controller's
`DECISIONS.md` (commit `0f498a7`): ADR-006 (align catalog with the owner's
brief), ADR-007 (pinned cross-project reuse), ADR-008 (local LLMs through
OpenAI-compatible servers), and ADR-009 (programmatic graders by default).

Model policy as of 2026-10-25: **local-first while Cline is paused**
(project ADR-0009). The owner paused Cline ("off until the owner says
otherwise", relayed by Kim), so RabbitHole's default policy model is
Qwen3-1.7B (GGUF Q4_K_M) on CPU through an OpenAI-compatible local server,
under portfolio ADR-008. Controller ADR-016 (hosted
`cline-pass/deepseek-v4.1-flash` via Cline Pass) is referenced only as the
optional hosted backend; no default, headline number, or success criterion
depends on it. ADR-0003 is amended accordingly. This is a project-level
exception to ADR-016 for the duration of the pause; the controller's ADR-016
records it in a 2026-10-30 note without changing ADR-016's decision.

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-hand-rolled-agent-loop.md) | Hand-rolled agent loop instead of an agent framework | Accepted |
| [0002](0002-reuse-rag-engine-retriever.md) | Reuse CitationNeeded's retriever as the `search_corpus` backend | Accepted |
| [0003](0003-openai-compatible-provider.md) | Local Qwen3-1.7B as the default provider, an optional hosted OpenAI-compatible backend, scripted and replay providers for tests and CI | Accepted (amended 2026-10-25) |
| [0004](0004-evaluation-data-and-graders.md) | HotpotQA subset plus hand-authored tool suite, graded programmatically | Accepted |
| [0005](0005-record-replay-cassettes.md) | Record/replay cassettes for deterministic, zero-cost CI evaluation | Accepted |
| [0006](0006-budgets-timeouts-retries.md) | Budgets, timeouts, and retries enforced by the loop with classified errors | Accepted |
| [0007](0007-offline-tools-and-guardrails.md) | Offline-first tools; tool output is untrusted data; citations must be observed | Accepted |
| [0008](0008-jsonl-traces.md) | Versioned JSONL traces instead of OpenTelemetry | Accepted |
| [0009](0009-local-first-while-cline-paused.md) | Local-first while Cline is paused | Accepted (2026-10-25) |
