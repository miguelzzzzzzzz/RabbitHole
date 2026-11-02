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
- `calculator` tool (`agentic_research.tools.calculator`): evaluates one
  arithmetic expression by walking a whitelisted AST (int/float literals,
  `+ - * / // % **`, unary `+`/`-`, constants `pi` and `e`, and the functions
  `abs round min max sqrt exp log log10 log2 sin cos tan floor ceil`); never
  calls `eval`/`exec`. Inputs are bounded instead of cancelled (ADR-0006):
  500 characters, 200 AST nodes, depth 100, integers up to 1000 bits (checked
  after every operation, and `**` refused before computing), finite real
  results only. Rejections carry reason codes and surface as `invalid_args`;
  no retries. Table-driven rejection tests and hypothesis tests checking
  agreement with Python arithmetic; `hypothesis` added to the dev extra.
- `ToolFailure`: the exception a tool's `run` raises to report a classified
  `ToolError` (the M2 executor turns it into `ToolResult.failure`).
- Trace schema v1 (`agentic_research.trace`, ADR-0008): a JSONL file per run
  whose first line is a header with `schema_version: 1`, `run_id`, and a
  timezone-aware `started_at`, followed by one span per line: `run` (question,
  provider/model, tool format, status, answer or error, budget usage and
  limits, `parse_failures`), `step`, `provider_call` (token usage, finish
  reason, optional `parse_failure` reason code, error), and `tool_call`
  (arguments plus exactly one of output or error). Timings are monotonic
  seconds since the run started; spans can carry point-in-time events.
  `Trace` checks one root run span, unique ids, parent kinds, contiguous step
  indexes, and that `parse_failures` matches the provider calls that failed
  to parse. NaN/infinity in JSON payloads is rejected instead of being
  written as `null`.
- `write_trace` (atomic), `TraceWriter` (exclusive create, one flushed line
  per finished span), and `read_trace`/`loads_trace`. The reader checks
  `schema_version` first and raises `TraceSchemaVersionError` naming the
  found and supported versions; other problems raise `TraceError` with a
  line number, and a file without a run span is reported as incomplete.
  Round-trip tests (including a hypothesis property and a span from a real
  calculator run) check that write-then-read gives an identical object and
  byte-identical text.
- `arp tools schema [--tool NAME]`: prints the built-in tools (for now
  `calculator`, from `agentic_research.tools.builtin.default_registry()`) in
  the OpenAI chat-completions `tools` format. Tests check the output against
  the JSON Schema 2020-12 meta-schema and that the printed schema accepts and
  rejects the same arguments as the registry; an unknown tool name exits 2.
- `.env.example` listing every configuration variable from PROJECT_SPEC
  section 11 with comments and empty secrets (no code reads them yet). Tests
  parse it strictly and check that it lists exactly the `ARP_*` names the
  spec, plan, ADRs, and README mention, that secret values are empty, and the
  spec's fixed defaults.
- README skeleton: status, an architecture table marking what is built,
  usage of the current CLI, and an evaluation table where every run,
  including all hosted runs, is "not run".
