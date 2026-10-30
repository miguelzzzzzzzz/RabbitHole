# Rabbit Hole

A hand-rolled, tool-using research agent built as an engineered system: an
explicit agent loop with a typed tool registry, budgets, timeouts and
retries, guardrails, citations checked against exact spans of the
[CitationNeeded](https://github.com/miguelzzzzzzzz/CitationNeeded) corpus,
versioned JSONL traces, and programmatic trajectory evals. The default policy
model is a small local model (Qwen3-1.7B, quantized, on CPU); CI never calls a
model.

**Status:** planning docs only. No code, no evaluation results yet. M1
(scaffold, tool contracts, local runtime and tool-calling probe) is next.

- [Project spec](PROJECT_SPEC.md)
- [Milestone plan](PLAN.md)
- [TODO](TODO.md)
- [Architecture decision records](docs/adr/README.md)
- [Changelog](CHANGELOG.md)

## License

MIT, see [LICENSE](LICENSE).
