# Rabbit Hole

A hand-rolled, tool-using research agent built as an engineered system: an
explicit agent loop with a typed tool registry, budgets, timeouts and
retries, guardrails, citations checked against exact spans of the
[CitationNeeded](https://github.com/miguelzzzzzzzz/CitationNeeded) corpus,
versioned JSONL traces, and programmatic trajectory evals. The default policy
model is a small local model (Qwen3-1.7B, quantized, on CPU); CI never calls a
model.

**Status:** M1 in progress. Built so far: package scaffold and CI, typed
tool contracts and a validating tool registry, the `calculator` tool, trace
schema v1 (JSONL), and `arp tools schema`. Not built yet: the agent loop, the
model provider, research tools, citations, guardrails, and evals. No model
has been run, so every evaluation below is "not run".

## Architecture

The agent is a hand-rolled loop (ADR-0001), not a framework. Planned shape
(PROJECT_SPEC section 4); the "status" column says what exists today.

| Component | Role | Status |
| --- | --- | --- |
| Tool contracts and `ToolRegistry` | typed pydantic args/outputs, OpenAI `tools` schema export, argument/output validation into classified errors (`transient`, `timeout`, `permanent`, `invalid_args`, `bad_output`), retry policy | built (M1) |
| `calculator` | AST-whitelist arithmetic, bounded, no `eval` | built (M1) |
| Trace schema v1 | one JSONL file per run: header + run/step/provider_call/tool_call spans, `parse_failures` | built (M1) |
| Local runtime + `arp doctor` | Qwen3-1.7B (GGUF Q4_K_M) on CPU behind an OpenAI-compatible server; memory preflight | M1, not built |
| `OpenAICompatibleProvider` + JSON-in-text parser | chat completions with native `tools` or a strict JSON-in-text protocol; the M1 probe picks the default | M1, not built |
| Agent loop, run state, budgets, timeouts/retries | `run(task, budgets)`; failures become observations or a terminal status, never exceptions | M2 |
| Research tools, citations, guardrails | `search_corpus`/`read_document` over [CitationNeeded](https://github.com/miguelzzzzzzzz/CitationNeeded) `v1.0.0`; citations checked against exact spans; allowlists and loop detection | M3 |
| Task suites, graders, eval runner | programmatic trajectory graders; oracle and heuristic baselines | M4 |
| Local LLM policy, record/replay | live runs on the local model; cassettes replay in CI | M5 |
| Fault injection, ablations | tool errors, timeouts, malformed output, prompt injection | M6 |

## Usage

```bash
pip install -e ".[dev]"
arp --version
arp tools schema                  # all built-in tools, OpenAI `tools` format
arp tools schema --tool calculator
```

Planned commands (not built yet): `arp doctor`, `arp probe tool-calls`,
`arp run`, `arp eval`, `arp trace show`.

Configuration will come from environment variables, all listed with
comments in [`.env.example`](.env.example) (secrets empty). No code reads
them yet; each group names the milestone that wires it in.

## Evaluation

Nothing has been run. Numbers appear here only when they come from a
committed report under `evals/results/`.

| Run | Model | Status |
| --- | --- | --- |
| M1 tool-calling probe (`native` vs `json`, 20 prompts) | Qwen3-1.7B Q4_K_M, local CPU | not run |
| Oracle and heuristic baselines | none (no model) | not run |
| Local live eval, `dev` split (baseline) | Qwen3-1.7B Q4_K_M, local CPU | not run |
| Fault-injection and ablation runs | Qwen3-1.7B Q4_K_M, local CPU | not run |
| Hosted runs (optional backend) | `cline-pass/deepseek-v4.1-flash` | not run (Cline paused, ADR-0009) |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy
pytest -m "not slow" --cov
```

The fast suite blocks network access, so it never calls a model.

- [Project spec](PROJECT_SPEC.md)
- [Milestone plan](PLAN.md)
- [TODO](TODO.md)
- [Architecture decision records](docs/adr/README.md)
- [Changelog](CHANGELOG.md)

## License

MIT, see [LICENSE](LICENSE).
