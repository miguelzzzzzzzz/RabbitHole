# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Minimal `OpenAICompatibleProvider` (`agentic_research.providers`, ADR-0003): async httpx client against `{ARP_LLM_BASE_URL}/chat/completions` (default `http://127.0.0.1:8080/v1`), model from `ARP_LOCAL_MODEL` (default `Qwen3-1.7B-Q4_K_M`), optional Bearer API key that never appears in `repr` or error text, temperature 0 / seed 0 by default, and `chat_template_kwargs.enable_thinking=false` for Qwen3 non-thinking mode. Parses native `tool_calls` (arguments as a JSON string or object; malformed arguments keep `arguments_raw` + `parse_error` and are not guessed), content-only replies, and empty `tool_calls`. Typed errors: `local_server_unavailable`, HTTP errors, malformed JSON/shape. No HostedProvider, retries, runaway guard, or recording (M5). Tests use `httpx.MockTransport` under the network guard; one `@pytest.mark.slow` live test starts/stops the pinned local server. `httpx` is a runtime dependency; `pytest-asyncio` is a dev dependency.
- `arp doctor` (`agentic_research.preflight`): memory preflight that compares
  `MemAvailable` from `/proc/meminfo` to the pinned server's reported
  model + KV + compute buffers (1819.36 MiB from `docs/local-models.md`) plus
  `ARP_MIN_FREE_MB` headroom (default 1024). Swap is reported but never
  counted toward the budget. Also checks the intended serve flags match the
  pin (`-c 4096`, `--cache-ram 0`, `-np 1`, loopback, `--jinja`, `--seed 0`,
  alias `Qwen3-1.7B-Q4_K_M`) and that `GET {ARP_LLM_BASE_URL}/models` lists
  that alias. Tests use a fake meminfo file and a mocked models client; the
  CLI exits 0 on pass and 1 on refuse.
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
- Local model runtime, pinned in `docs/local-models.md` and installed on the
  dev box: llama.cpp `llama-server` release `b12032` (CPU build, asset
  sha256 checked) serving Qwen3-1.7B Q4_K_M from
  `unsloth/Qwen3-1.7B-GGUF` at a fixed revision (GGUF sha256 checked) with a
  4096-token context. `scripts/local_runtime.sh install|verify|serve`
  downloads and verifies both files and starts the server on
  `127.0.0.1:8080` with `--jinja`, one slot, and the RAM prompt cache off
  (this build defaults it to 8 GiB; the box has no swap). The doc records
  the server's reported memory buffers (about 1.8 GiB, more than GGUF + KV
  cache because the weights are repacked for the CPU) as input for
  `arp doctor`. Tests keep the script's pins, the doc, and `.env.example`
  in sync and check that `verify`/`serve` refuse missing or wrong-digest
  files; a `slow` test checks a running server serves the pinned model id.
- `.env.example`: `ARP_LOCAL_MODEL` is now the served id
  `Qwen3-1.7B-Q4_K_M`; added `ARP_RUNTIME_DIR`.
