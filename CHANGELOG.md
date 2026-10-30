# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project spec, milestone plan, TODO, and architecture decision records
  0001-0009 in `docs/adr/` (2026-10-30). Pre-repository planning changes are
  in `docs/planning-history.md`.
- Package scaffold (`agentic_research`, `arp --version`), ruff, strict mypy,
  pytest with a 96% branch-coverage floor, an autouse network guard for the
  fast suite, and CI on Python 3.11 and 3.13.
- Tool contracts (`agentic_research.tools`): the async `Tool` protocol,
  strict `ToolArgs`/`ToolOutput` bases, `ToolContext`, `ToolResult`, and
  `ToolError` with the error classes `transient`, `timeout`, `permanent`,
  `invalid_args`, and `bad_output`. `RetryPolicy` retries only `transient`
  and `timeout`, with capped exponential backoff and seeded jitter.
- `ToolRegistry`: validated registration (name format, unique names, strict
  argument models, positive finite timeouts, async `run`), export in the
  OpenAI `tools` format, and argument/output validation that returns
  `invalid_args`/`bad_output` errors instead of raising (unknown tool,
  invalid or duplicate-key JSON, non-object arguments, schema violations).
  Schema-violation messages omit input values; messages are capped at 1000
  characters. mypy now also checks the tests.
