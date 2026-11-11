# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Chad M1 re-check R-3: the shared probe think check now also returns `think_block_unclosed` for a leading `<think>` with no `</think>`, so a native reply with an unclosed think fails even when its `tool_calls` are valid (the JSON path already rejected it in the parser). Added scorer tests for a non-empty or unclosed think in `content` on both formats and both prompt kinds, plus empty-think and late-think controls. Re-scoring the `raw_content` of both committed M1 probe reports finds no leading non-empty or unclosed think, so neither report's results change; `rule_version` stays 1.
- Chad M1 review MAJOR-1: `arp doctor` observes live `llama-server` flags via `/proc/<pid>/cmdline` + `/props`; refuses if unreadable or `--cache-ram`/`-c`/`-np`/`--reasoning-format` mismatch. `runtime_metadata` fields labelled `observed` vs `pinned`; `metadata.source` repo-relative.
- Chad M1 review MAJOR-2: pin `--reasoning-format none` in `local_runtime.sh`/docs/doctor; capture `reasoning_content` on `ChatMessage` and score non-empty as `think_block_present` (native + json) with MockTransport + recorded fixtures.

### Changed
- R-9: `docs/providers.md` says JSON moved 10/20 to 6/20 between runs with different `--reasoning-format` settings and that no causal claim follows from two runs.
- M1 closed (2026-11-08) after Chad's re-check passed at `60e7ad9` (0 CRITICAL, 0 MAJOR); M2 started. His non-blocking follow-ups R-1..R-7, R-9 and OPT-1/OPT-2 are tracked in `TODO.md` with owners (R-8 withdrawn).
- Probe scores include `finish_reason`, `usage`, `reasoning_content`, `multiple_tool_calls`; `parse_failures` labelled for all 25 prompts (tool/no-tool split).
- Docs (Bogart): 20/20 means native wins on this probe set, not "100% reliable"; reliability = parse-failure counts over M2+. JSON prompt contradiction noted (no v2 at M1).
- Coverage: ~96.4% after observation/probe growth (floor 96%); CHANGELOG notes the drop. Composed `run_probe` tested without skips.

### Added
- M2 (Chad's M2 check 1): every tool now declares `idempotent: bool` (part of the `Tool` protocol). `ToolRegistry.register` refuses a tool whose `idempotent` is not a bool, and refuses `timeout` in `retry.retry_on` unless the tool is idempotent, because a call that timed out on our side may still have taken effect. `transient` retries stay allowed for any tool. `calculator` and the probe's `search_corpus` stub are marked idempotent. Note that the default `RetryPolicy()` includes `timeout`, so a non-idempotent tool must pass `retry_on={transient}` or `RetryPolicy.no_retry()`.
- M2: `agentic_research.tools.executor.ToolExecutor`: validates arguments once, then runs the tool under `asyncio.timeout` (overrunning tools are cancelled and reported as `timeout`/`tool_timeout`), retries only `transient`/`timeout` failures that the tool's `RetryPolicy` allows, with backoff jitter drawn from a seeded RNG, an injectable clock and sleeper, and an optional absolute `deadline` (timeouts shortened to it, retries skipped when the backoff would cross it). A crashing tool becomes a `permanent`/`tool_crashed` result without leaking the exception text. Returns the `ToolResult` plus per-attempt records (`AttemptRecord`) for the trace. Tests exercise cancellation, exact backoff schedules, seed reproducibility, non-retryable classes, deadline handling and a Hypothesis check of attempt/sleep counts. No loop yet.
- M2: `agentic_research.agent` with the `Action` union (`ToolCall | FinalAnswer | GiveUp`, tagged on `kind`), the `Policy` protocol, serializable `RunState` (task, plan, messages, evidence, step records, budget counters, terminal status with `budget_exhausted:<kind>` / `failed:<reason>` labels, JSON round-trip) and `Budgets` (steps, tool calls total and per tool, deadline, tokens, repairs) with a fixed-order `exhausted()` check. No loop yet; the loop and executor come next.
- Scrubbed llama-server fixtures under `tests/fixtures/` (MINOR-7); live smoke-only JSON under `evals/results/` (MINOR-6).
- Second M1 tool-call probe run (`evals/results/m1-toolcall-probe-20261107T011142Z.json`): native 20/20, json 6/20, default stays `native` under the frozen rule; first run kept. Observed launch flags include `--reasoning-format none`.
- Live M1 tool-calling probe results (`smoke-only`): native 20/20 (Wilson 95% ≈ [0.84, 1.00]), json 10/20 (≈ [0.30, 0.70]); chosen default `native` under the pre-registered rule. Report at `evals/results/m1-toolcall-probe-20261106T011244Z.json`; `docs/providers.md` updated; `ARP_TOOL_FORMAT=native`.
- `arp probe tool-calls` (`agentic_research.probe`): loads the frozen `config/probe.toml` + `probes/tool_calls.v1.yaml`, refuses a dirty tree or probe-file sha256 mismatch (no request sent), runs `native` and `json` formats against a local `OpenAICompatibleProvider` (temperature 0, seed 0, `enable_thinking=false`), scores structurally valid / correct tool / schema-valid args / no-tool decisions, applies the pre-registered 18/20 decision rule, writes a `smoke-only` report under `evals/results/m1-toolcall-probe-*.json`, and updates `docs/providers.md`. Includes a stub `search_corpus` schema for the probe, Wilson 95% helper, and offline unit tests with a scripted provider. Live run is separate.
- Frozen M1 tool-calling probe bar: `probes/tool_calls.v1.yaml` (exactly 20 tool-expecting prompts over `calculator` + stub `search_corpus`, plus 5 no-tool prompts scored separately) and `config/probe.toml` (`probe_sha256`, `min_valid = 18`, `n = 20`, `rule_version = 1`, temperature 0, seed 0, `enable_thinking = false`). CI asserts the count and that the on-disk sha256 matches the config. No probe run yet; `arp probe tool-calls` lands next.
- Strict JSON-in-text tool protocol (`agentic_research.providers.toolcall_json`, ADR-0003): `parse_toolcall_json` accepts exactly one JSON object, optionally inside a single json or bare code fence, shaped as `{"action":"tool_call","tool":...,"arguments":{...}}` or `{"action":"final_answer","answer":...}`. Leading empty/whitespace-only `<think>...</think>` is stripped as preprocessing (`empty_think_stripped=True`, not a failure); a non-empty think body fails with `think_block_present` and the remainder is not parsed; an unclosed think is `think_block_unclosed`. Rejects invalid JSON, multiple objects, leading/trailing prose, unsupported fences, unknown action/keys, unknown tools, and arguments that fail the caller-supplied pydantic allowlist — never recovers a call from surrounding prose. Every rejection yields a trace `ParseFailure` reason code and increments an optional `ParseFailureCounter`. Table-driven tests cover success and each rejection class; no network and no model.
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
