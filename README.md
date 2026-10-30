# Rabbit Hole

A hand-rolled, tool-using research agent built as an engineered system: an
explicit agent loop with a typed tool registry, budgets, timeouts and
retries, guardrails, citations checked against exact spans of the
[CitationNeeded](https://github.com/miguelzzzzzzzz/CitationNeeded) corpus,
versioned JSONL traces, and programmatic trajectory evals. The default policy
model is a small local model (Qwen3-1.7B, quantized, on CPU); CI never calls a
model.

**Status:** M1 in progress: package scaffold, CI, tool contracts, and the
tool registry. No agent loop, concrete tools, or evaluation results yet;
every evaluation is "not run".

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
