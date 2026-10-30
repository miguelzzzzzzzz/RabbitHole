# PLAN: Rabbit Hole

Milestones are ordered. Each ends with green CI on the pushed SHA, an updated
CHANGELOG, and TODO.md rolled forward. Every acceptance criterion can be
checked by running a command; "CI" means the GitHub Actions workflow on the
milestone's final SHA. Commits are coherent units of work with conventional
subjects. No progress, cosmetic, or padding commits. No numbers appear in docs
unless they come from a committed report under `evals/results/`.

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | Scaffold, tool contracts, registry, calculator, trace schema, local runtime and tool-calling probe | planned |
| M2 | Agent loop, run state, budgets, timeouts/retries, scripted provider | planned |
| M3 | Research tools on the project-01 retriever, citations, guardrails | planned |
| M4 | Task suites, graders, eval runner, oracle and heuristic baselines | planned |
| M5 | Local LLM policy (Qwen3-1.7B, default), call parsing/repair, optional hosted backend, record/replay, baseline run | planned |
| M6 | Fault injection, failure taxonomy, baselines and ablations | planned |
| M7 | Container, CI image smoke eval, docs, v0.1.0 release | planned |

Dependency: M3 needs CitationNeeded (project 01) at its v1.0.0 release tag.
The dependency spec is
`production-rag-engine @ git+https://github.com/miguelzzzzzzzz/CitationNeeded@v1.0.0`.
The `v0.1.0` tag (`bf0c83b`, contains Chad's signed-off `cdd892e`) exists; the
2026-10-28 diff of `v0.1.0` against `977f091` (the release
candidate) found **no breaking changes** to the consumer contract, only
additions (ADR-0002, 2026-10-28 note). The pin is **final**: `v1.0.0` is
tagged at `b8015f3`, re-verified 2026-10-29; the interface is unchanged from
`977f091` (version string only). The rename
kept the distribution name `production-rag-engine` and the import `rag_engine`.
The portfolio runs one active project at a time, so that tag should exist before
M1 starts.

Model policy (2026-10-25, ADR-0009): local-first while Cline is paused. The
default policy model is Qwen3-1.7B (GGUF Q4_K_M) on CPU through an
OpenAI-compatible local server (Ollama or llama.cpp `llama-server`, portfolio
ADR-008). No milestone calls the Cline API. The hosted backend is built as an
optional extra with mocked tests only, and every hosted run is "not run".

## M1 - Scaffold, tool contracts, registry, calculator, trace schema, local runtime and tool-calling probe

Scope
- Packaging (src layout, `agentic_research` package, `arp` CLI entry point),
  ruff, mypy --strict, pytest, and a CI workflow on py3.11 and py3.13 that
  mirrors 01.
- `Tool` protocol, `ToolContext`, `RetryPolicy`, `ToolResult`/`ToolError`
  with an error-class enum (`transient`, `permanent`, `invalid_args`,
  `timeout`, `bad_output`).
- `ToolRegistry`: registration, duplicate-name rejection, JSON-schema export
  in the OpenAI `tools` format, argument validation that returns a structured
  `invalid_args` observation, and output validation that returns `bad_output`.
- `calculator` tool: an AST whitelist evaluator (numbers, + - * / // % **,
  parentheses, unary minus, a small math-function allowlist) with bounds on
  exponent and result size. No `eval` or `exec`.
- Trace schema v1 (pydantic): `RunSpan`, `StepSpan`, `ProviderCallSpan`,
  `ToolCallSpan`, JSONL writer/reader, and a `schema_version` field.
- `.env.example`, `README` skeleton (architecture, usage, and evaluation
  sections with "not run" placeholders, and hosted runs listed as "not run"),
  CHANGELOG, `docs/adr/0001-0009` copied from the approved ADRs.
- **Local runtime** (moved here from M5): install one OpenAI-compatible CPU
  runtime on the box (Ollama binary, or the llama.cpp `llama-server` CPU
  build / `llama-cpp-python[server]`; llama.cpp needs `--jinja` for native
  tool calls). Pull Qwen3-1.7B as GGUF Q4_K_M and pin tag/digest, GGUF file
  sha256, quantization, server version, and context length (start at 4096)
  in `docs/local-models.md`. Optional: Qwen3-0.6B Q4_K_M for a later
  ablation.
- **Memory preflight**: `arp doctor` reads `MemAvailable` from
  `/proc/meminfo` and the swap total, and refuses a live run unless free
  memory covers the GGUF size + KV cache for the configured context + headroom
  (default 1 GB, `ARP_MIN_FREE_MB`). It also checks that the server answers
  `/v1/models`. The box is CPU-only with about 3 GB to spare and no swap.
- **Minimal `OpenAICompatibleProvider`** (httpx, async): chat completions
  with optional `tools`, timeout, and usage capture. Enough for the probe;
  M5 adds retries, the runaway guard wiring, and recording.
- **Tool-calling probe** (moved here from the first step of M5, where it was
  a Cline check): `arp probe tool-calls` sends a fixed, versioned probe set
  (`probes/tool_calls.v1.yaml`: 20 tool-expecting prompts over the
  registry's real schemas, e.g. `calculator` plus a stub `search_corpus`
  schema, plus a few prompts where the right action is to answer without a
  tool, scored separately) to Qwen3-1.7B
  through the local server, at temperature 0, non-thinking mode, in two
  formats:
  - `native`: OpenAI `tools` in the request, `tool_calls` in the reply.
  - `json`: the JSON-in-text protocol below, no `tools` field.
  For each format it records: replies with a structurally valid call, correct
  tool name, schema-valid arguments, correct no-tool decisions, parse
  failures, and latency. Results go to `docs/providers.md` and
  `evals/results/m1-toolcall-probe-*.json` (labelled `smoke-only`; never a
  headline number) with model digest, server version, date, and the memory
  preflight result.
  - Decision rule, fixed before the probe runs: `native` becomes the default
    only if it gets schema-valid calls on at least 90% of tool-expecting
    probes (at least 18/20) and its count is not lower than `json`'s;
    otherwise the default is `json`. The rule and the raw counts are written
    down whatever the outcome.
  - Reporting: each format's result is a raw count (for example `18/20`)
    plus a 95% Wilson interval (for example 18/20 gives about [0.70, 0.97];
    20/20 gives about [0.84, 1.00]). With n=20 the interval is wide, so
    passing the bar shows "probably usable", not "90% reliable". The real
    reliability measure is `parse_failures` on the full M5 runs.
  - Freeze before running: the probe file and the bar are committed in their
    own commit before the first probe run. The bar lives in config
    (`config/probe.toml`: `probe_file = "probes/tool_calls.v1.yaml"`,
    `probe_sha256 = "<sha256 of the file>"`, `min_valid = 18`, `n = 20`,
    `rule_version = 1`). `arp probe tool-calls` refuses to run if the tree is
    dirty or the probe file's sha256 does not match `probe_sha256`, and the
    probe report records the probe sha256, the config values, and the freeze
    commit SHA. Changing prompts or the bar means a new probe version
    (`tool_calls.v2.yaml`, `rule_version = 2`) and a CHANGELOG note, never an
    edit in place after results are seen.
- **JSON-in-text tool protocol** (the fallback, built either way): the model
  replies with exactly one JSON object, optionally inside a single
  ```` ```json ```` fence: `{"action": "tool_call", "tool": "<name>",
  "arguments": {...}}` or `{"action": "final_answer", ...}`. The strict
  parser strips one leading `<think>...</think>` block (Qwen3 may emit an
  empty one in non-thinking mode), then rejects: invalid JSON, more than one
  object, non-whitespace text outside the object/fence, unknown keys,
  unknown tool names, and arguments that fail the tool's pydantic model. It
  never "guesses" a call out of prose. Every rejection increments a
  `parse_failures` counter (per call, per run, per report) with a reason
  code; M5 adds one repair attempt on top.

Deliverables
- `src/agentic_research/{tools/,trace/,cli.py}`, `tests/`,
  `.github/workflows/ci.yml`, `docs/adr/`.
- `src/agentic_research/{providers/openai_compat.py,providers/toolcall_json.py,preflight.py,probe.py}`,
  `probes/tool_calls.v1.yaml`, `config/probe.toml`, `docs/local-models.md`,
  `docs/providers.md`, `evals/results/m1-toolcall-probe-*.json`.

Acceptance criteria
- `pytest -q -m "not slow"` passes locally and in CI. `ruff check .`,
  `ruff format --check .`, and `mypy` are clean.
- `arp tools schema` prints valid OpenAI-format tool schemas. A test validates
  them against the JSON Schema meta-schema.
- The calculator rejects every disallowed construct in a table-driven test
  (names, attributes, calls outside the allowlist, huge exponents, lambdas,
  comprehensions).
- A trace round-trip test passes (write, then read gives an identical object).
- The local runtime is installed and pinned in `docs/local-models.md`
  (model tag/digest, GGUF sha256, Q4_K_M, server version, context length),
  and `arp doctor` passes on the box before the probe.
- `probes/tool_calls.v1.yaml` and `config/probe.toml` (with
  `probe_sha256`, `min_valid = 18`, `n = 20`) are committed in a freeze commit
  that comes **before** the probe report in git history (checked in review).
  A CI test asserts the probe file has exactly 20 tool-expecting prompts and
  that its sha256 equals `probe_sha256`.
- `arp probe tool-calls` (manual, `slow`, on the box; never in CI) has run
  for both formats on a clean tree. `docs/providers.md` and the probe report
  record the date, model digest, server version, freeze commit SHA, probe
  sha256, each format's raw count (`k/20`) with its 95% Wilson interval, a
  note that n=20 gives a wide interval, the decision rule, and the chosen
  default. A config test checks that the default `ARP_TOOL_FORMAT` matches
  the recorded outcome.
- The JSON-in-text parser passes its full table-driven test matrix in CI,
  and `parse_failures` appears in the trace schema and the report schema.
- If no runtime can be installed or the model does not fit in memory, this
  is recorded as a blocker in `state.json`, the probe result is "not run",
  and the rest of M1 still ships.

Tests
- Registry: duplicate name, unknown tool, invalid args become `invalid_args`
  (not an exception), and output schema violations become `bad_output`.
- Calculator: correctness on hand-computed cases, plus property tests
  (hypothesis) checking agreement with Python arithmetic on generated safe
  expressions.
- Trace: schema-version mismatch raises a clear error.
- JSON-in-text parser (table-driven): valid tool call; valid final answer;
  fenced and unfenced; leading empty and non-empty `<think>` block; invalid
  JSON; two objects; prose before/after the object; unknown key; unknown tool;
  wrong arg types; missing required arg. Each case is accepted, or rejected
  with the expected reason code and `parse_failures` incremented by one.
- Native `tool_calls` parsing against recorded Ollama- and llama.cpp-shaped
  payloads (`httpx.MockTransport`): valid call, arguments as a JSON string,
  malformed arguments, empty `tool_calls`, content-only reply.
- Memory preflight with a fake `/proc/meminfo`: passes above the threshold,
  refuses below it with a clear message, reports swap 0, and records the
  values in the result.
- Probe scoring with scripted replies (no model): counts and the decision
  rule produce the expected default for "native reliable", "native
  unreliable", and "both poor" fixtures, including the boundary (18/20
  passes, 17/20 fails).
- Wilson interval helper checked against hand-computed values (18/20 gives
  about [0.699, 0.972]; 0/20 and 20/20 edge cases).
- Probe freeze guard: a dirty tree or a probe-file sha256 mismatch makes
  `arp probe tool-calls` refuse with a clear message (no request sent).
- A network guard fixture fails any test that opens a real connection, so CI
  never calls a model.

## M2 - Agent loop, run state, budgets, timeouts/retries, scripted provider

Scope
- `RunState` (serializable): task, plan, messages, evidence store, step
  records, budget counters, and status enum with terminal statuses
  `succeeded`, `abstained`, `budget_exhausted:<kind>`, and `failed:<reason>`.
- `Action` union: `ToolCall | FinalAnswer | GiveUp`. A `Policy` protocol.
- Loop: budget check, then policy (with a step timeout), then guardrails,
  then executor, then observation. `FinalAnswer` goes through a pluggable
  verifier hook (a real verifier arrives in M3).
- Budgets: max steps, max tool calls (total and per tool), wall-clock
  deadline, token budget, and max repair attempts. All are enforced by the
  loop.
- `ToolExecutor`: `asyncio.timeout` per call, retries with exponential
  backoff and jitter from a seeded RNG for `transient`/`timeout` errors only,
  and an injectable clock and sleeper.
- Loop detection: an identical (tool, normalized args) repeated beyond a
  threshold produces a guardrail observation, then `failed:loop`.
- Observation shaping: deterministic truncation of tool outputs and eliding
  of old observations to one-line summaries, all without an LLM.
- `ScriptedProvider` (canned completions, including malformed ones) and
  `LLMPolicy` skeleton. `OraclePolicy` (follows a given trajectory).
- Tracer wired into every span. `arp run --provider scripted --script FILE`
  and `arp trace show FILE`.

Deliverables
- `src/agentic_research/{agent/,providers/scripted.py,policies/}` and the
  trace viewer.

Acceptance criteria
- With a scripted policy and a fake clock, each budget kind triggers its
  exact terminal status (one test per kind).
- Hypothesis property test: for any generated scripted action sequence and
  budget config, counters never exceed limits and every run ends in a
  terminal status with no unhandled exception.
- A tool that sleeps past its timeout is cancelled. A retry happens only for
  retryable error classes, and attempt counts match the policy (fake clock,
  no real sleeping).
- `arp run --provider scripted --script tests/fixtures/scripts/two_step.json`
  exits 0 and writes a trace whose spans match a golden file.

Tests
- Unit tests for the executor (timeout, retry/backoff schedule, non-retryable
  errors), budgets, loop detection, truncation, and state serialization
  round-trip.
- Golden-trace test for a two-step scripted run (timings normalized).

## M3 - Research tools on the project-01 retriever, citations, guardrails

Scope
- `SearchBackend` protocol and the `RagEngineBackend` adapter over
  CitationNeeded (distribution `production-rag-engine`; git dependency pinned
  to `v1.0.0`, final: tagged at `b8015f3`, with
  `allow-direct-references` in hatch). Contract (index format 2) as signed off
  at `cdd892e` and re-verified unchanged at `977f091` (2026-10-28)
  and at `v1.0.0` = `b8015f3` (2026-10-29, version string only):
  - Each index stores `documents.jsonl` (doc id, source, `content_hash`,
    normalized text). `provenance()` includes `content_hash` and `page_end`.
    Citations are re-checked on ingest, build, and load; `load_index` refuses
    a tampered index. Old (format-1) indexes must be rebuilt.
  - Offsets (`start_char`/`end_char`) are into the **normalized** document
    text, not the raw file.
  - `doc_id = sha256("<corpus_id>:<relpath>")[:16]`. RabbitHole **always**
    passes an explicit `--corpus-id` / `corpus_id` (never the folder-name
    fallback; Chad's re-check note).
  - `save_index` takes the documents too.
  - `search --json` returns an object with `schema_version` and results under
    `hits` (no longer a bare list). Rerank keeps per-stage `ranks`. Payload
    keys: `schema_version` (1), `query`, `mode`, `rerank`, `hits`; each hit
    carries `rank`, `score`, `retriever`, optional `components`/`ranks`,
    `doc_id`, `chunk_id`, `source`, `content_hash`, `heading_path`, `page`,
    `page_end`, `start_char`, `end_char`, `text`.
  - Not used by RabbitHole: CitationNeeded's v1.0.0 additions (`evaluate` and
    `answer` commands, the read-only FastAPI service, `compose_retriever`).
    The adapter stays on the in-process API / `search --json`.
- Adapter behaviour: read `hits`, check `schema_version` (fail clearly if
  unknown), and carry `doc_id`, `content_hash`, normalized offsets, `page` /
  `page_end` into `Passage`. BM25 mode by default, hybrid as a config option
  (fastembed, `slow` tests only).
- Corpus setup: fixture corpus with `corpus_id=rabbithole-tasks-v1`; HotpotQA
  corpus (M4) with `corpus_id=rabbithole-hotpotqa`. Both built as index
  format 2. A test pins a known `doc_id` for a fixed relative path under each
  corpus id.
- `FixtureSearchBackend`: a tiny keyword backend over in-memory fixtures for
  unit tests, so the default test suite has no dependency on 01's index
  build. It exposes the same Passage fields (`content_hash`, normalized
  offsets, `page`/`page_end`).
- Tools:
  - `search_corpus(query, k, filters)` returns passages with
    `doc_id`, `content_hash`, `start_char`, `end_char`, `page`, `page_end`,
    `text`, `score`.
  - `read_document(doc_id, start, end)` returns bounded text sliced from the
    normalized stored document (with offsets).
  - `fetch_url(url)` serves the fixture map by default and does live HTTP
    only when `ARP_ALLOW_LIVE_FETCH=true` and the host is allowlisted.
    Output is HTML-to-text with offsets.
  - `final_answer(answer, claims[{text, citations[{source_id, start, end, quote, content_hash?}]}])`.
- `CitationVerifier`:
  - Validity: quote equals `documents.jsonl` normalized text at
    `[start_char:end_char]` exactly (not raw-file bytes).
  - Freshness: cited `content_hash` matches the stored document; mismatch =>
    `stale_source`.
  - Groundedness: the span lies inside text observed in this run's evidence
    store.
  - On an invalid or stale citation it returns feedback if the repair budget
    remains, otherwise the run fails.
- Guardrails:
  - Tool outputs are wrapped in an untrusted-data envelope in prompts.
  - An injected-instruction detector (pattern-based, conservative) flags
    content in traces and the evidence store.
  - Per-task tool allowlist and URL allowlist.

Deliverables
- `src/agentic_research/{tools/search.py,tools/read.py,tools/fetch.py,tools/answer.py,citations.py,guardrails.py,backends/}`
- Fixture corpus and fixture pages under `tests/fixtures/`.

Acceptance criteria
- An integration test (no network) builds a `RagEngineBackend` over the
  fixture corpus with 01's BM25 index (format 2, explicit
  `corpus_id=rabbithole-tasks-v1`), then runs an oracle trajectory
  (search, read, final_answer). The verifier accepts it.
- The verifier rejects, in table-driven tests: off-by-one spans, a quote that
  doesn't match the normalized stored text, a tampered span (text at offsets
  altered in a copy of `documents.jsonl`), a `content_hash` mismatch flagged
  `stale_source`, a span from an unobserved document, and an unknown doc_id.
- Adapter rejects an unknown `schema_version` with a clear error (not a
  silent empty hit list). Adapter accepts `schema_version` 1 with results
  under `hits` (the value at both `v0.1.0` and `v1.0.0`). The adapter does
  not gate on `rag_engine.__version__`; it gates on `schema_version` and
  index `format_version` 2.
- A pinned-`doc_id` test: for a fixed fixture relative path under
  `rabbithole-tasks-v1`, `doc_id == sha256("rabbithole-tasks-v1:<relpath>")[:16]`.
- `fetch_url` with live fetch disabled never opens a socket (asserted with a
  socket-blocking fixture). A non-allowlisted host is refused even when live
  fetch is enabled.
- Injection fixture pages are flagged in the trace.
- CI green. 01's dependency is installed from the pinned `v1.0.0` ref.

Tests
- Adapter contract tests: the same suite runs against `FixtureSearchBackend`
  and `RagEngineBackend` (interface conformance), including unknown
  `schema_version`, `hits` parsing, and Passage field presence
  (`content_hash`, normalized offsets, `page`/`page_end`).
- Offset invariant: `read_document` text equals the normalized
  `source.text[start:end]` from `documents.jsonl` (and equals the passage
  text returned by search for that span).
- Span / hash tests: tampered span caught; hash mismatch flagged stale.
- Pinned doc_id under the fixed corpus id.
- Guardrail unit tests: allowlists, envelope formatting, detector positives
  and negatives.

## M4 - Task suites, graders, eval runner, oracle and heuristic baselines

Scope
- Task schema (pydantic) and hand-authored suite `tasks/v1/`, with
  categories `lookup`, `multi_hop`, `compute`, `fetch`, `unanswerable`,
  `injection`, `distractor`, and `fault` (fault specs are wired up in M6).
  Dev/test split fixed in files.
- HotpotQA loader:
  - Download script with a pinned revision and sha256. The CMU URL is a
    fallback.
  - Seeded stratified sampler with committed ID lists `dev`/`test`.
  - Corpus builder: one document per paragraph, deduplicated, with gold
    supporting sentences mapped to char spans in the *normalized* text.
    Indexed as CitationNeeded format 2 with explicit
    `corpus_id=rabbithole-hotpotqa` (pinned-`doc_id` test).
  - Data is never committed.
- Graders:
  - Answer: normalized EM/F1 following the official HotpotQA normalization,
    numeric tolerance, set match, and abstention.
  - Citation validity, groundedness, and support precision/recall against
    gold spans.
  - Tool-selection: required recall, unnecessary rate, forbidden violations,
    first-tool, hallucinated tool names.
  - Call counts and efficiency ratio, redundant calls.
  - Latency breakdown, tokens.
  - Composite task completion.
- Runner: `arp eval --suite ... --mode {oracle,heuristic,...} --split ...`
  writes a JSON report with config, suite version and hash, dataset checksum,
  git SHA, dirty-tree flag, hardware, timestamp, per-task records (including
  `n_retrieved` vs scoring `k` when retrieval is involved), aggregates with n
  and bootstrap CIs, and per-category breakdown. Equal depth: every mode in a
  comparison uses the same search `k`, budgets, and scoring cut before
  aggregation. Untimed warm-up once per configuration; model/index `load_ms`
  reported separately and excluded from p50/p95. Tasks/queries with no
  relevant gold evidence are excluded from completion/citation aggregates and
  counted as `n_no_relevant`. Suites too small for claims are labelled
  `smoke-only`.
- `HeuristicPolicy` (rule-based router plus extractive answer) as the no-LLM
  baseline.
- Report-to-Markdown table generator for the README (skips `smoke-only` and
  dirty-tree rows for claim tables).

Deliverables
- `tasks/v1/`, `src/agentic_research/eval/`, `scripts/download_hotpotqa.py`,
  `evals/data/hotpotqa_ids.v1.{dev,test}.txt`.
- Committed reports `evals/results/v1-oracle-*.json` and
  `evals/results/v1-heuristic-*.json` from executed runs.

Acceptance criteria
- `pytest` schema test: every task validates, and every gold evidence span
  resolves to its expected quote.
- `arp eval --suite tasks/v1 --mode oracle` scores 100% on every grader
  (asserted in CI). On the HotpotQA subset, oracle reaches 100% when the data
  is available (CI job with cached download).
- `arp eval --mode heuristic` runs end to end and writes a report. Whatever it
  scores is committed as-is.
- Grader unit tests compare against hand-computed examples, including
  HotpotQA EM/F1 normalization parity with the official script on a fixture
  of predictions.
- Equal-depth: a comparison fixture where one document owns several of the
  top chunks scores every mode at the same post-dedup `k`, and a mode that
  cannot fill `k` is reported per-task (`n_retrieved < k`) rather than
  silently under-filled.
- Latency: warm-up is untimed; a fixture that sleeps during load shows
  `load_ms` outside p50/p95.
- Provenance: report schema requires `git_sha` and `dirty_tree`; committing a
  dirty-tree report fails the honesty check (or the report is marked and
  excluded from README claims).
- Degenerate labels: a HotpotQA-format fixture with no relevant gold spans is
  counted in `n_no_relevant` and absent from completion aggregates (not
  scored 0).
- A 3-task smoke suite is labelled `smoke-only` in its report and rejected
  from README claim generation.

Tests
- Each grader: positive, negative, and edge cases (empty answer, multiple
  citations, overlapping spans, alias match, numeric tolerance boundary).
- Sampler determinism: same seed gives the same IDs. Corpus builder offset
  invariant on a small embedded HotpotQA-format fixture written for the test,
  so the test needs no download.
- Equal-depth / multi-chunk-one-doc fixture as above.
- Dirty-tree flag detection (mocked `git status`).
- No-relevant-label exclusion counter.

## M5 - Local LLM policy (Qwen3-1.7B, default), call parsing/repair, optional hosted backend, record/replay, baseline run

Scope
- `OpenAICompatibleProvider` (local, **default**), hardened from M1: retries
  on 5xx/connection errors with backoff, timeouts sized for CPU, usage
  capture (prompt/completion tokens; estimated and labelled if the server
  omits them), tokens/s, and `local_server_unavailable` classification for
  connection refused or a server that died (e.g. OOM-killed). The memory
  preflight (`arp doctor`) runs before every live run and is recorded in the
  report.
- **Runaway guard (all providers, including local)**:
  `ARP_MAX_CONSECUTIVE_FAILURES=3` consecutive failed provider calls
  (transport errors, timeouts, 5xx, unparseable replies left after repair) or
  `ARP_MAX_RETRIES_PER_RUN=20` retries in total end the run with
  `failed:runaway`.
- `LLMPolicy`: versioned prompt templates (`prompts/v1/*.md`) written for a
  small model (short system prompt, few tools per task, tight schemas),
  tool-schema injection, native `tool_calls` parsing and the M1 JSON-in-text
  parser, with the default format taken from the M1 probe. On schema-invalid,
  unparseable, or hallucinated calls it makes one repair attempt that feeds
  the validation error back (configurable). `parse_failures` is counted
  before and after repair. Optional plan step.
- `HostedProvider` (**optional**, off by default; no live calls while Cline
  is paused): the same OpenAI-compatible client with configurable
  `ARP_HOSTED_BASE_URL`, `ARP_HOSTED_MODEL`, and `ARP_HOSTED_API_KEY_ENV`
  (the key is read from that env var only). A model-id allowlist enforces
  owner policy (portfolio ADR-016: only `cline-pass/deepseek-v4.1-flash`).
  Its Cline adapter keeps:
  - unwrap of the `{"data": {...}}` envelope
  - usage normalization (prompt, completion, and reasoning tokens; `cost`,
    `gateway_cost`, `market_cost`)
  - `max_tokens` default and floor of 1000 for that model
  - HTTP 402 classified `payment_required`, non-retryable, clear message,
    counts toward the runaway guard without burning retries
  - 429 (Pass limit or rate limit) classified `pass_limit_or_rate_limited`,
    back-off-then-stop, clear message, counts toward the runaway guard without
    burning retries
  - per-call cost and usage logging to the shared ledger
    (`CLINE_USAGE_LEDGER`) with an Asia/Manila daily rollup (`arp usage`);
    reported `cost` is labelled **reference cost (Pass)**, missing cost is a
    warning, never zero, and spend is not capped
  - per-run hosted cost log in the run result and report (`billing_tier:
    cline_pass`)
  Hosted results are marked `headline_eligible: false` and rendered only in
  the separate "optional hosted extra" table, which reads "not run" while
  Cline is paused.
- Record/replay:
  - `RecordingProvider` writes cassettes keyed by a hash of the normalized
    request (messages, tools, sampling params, model id/digest), storing
    provider responses only.
  - `ReplayProvider` serves them and fails loudly on a cache miss.
  - Tool outputs are recomputed at replay time.
- Timing pilot on the `dev` split (a handful of tasks) with the local model to
  pick subset sizes that fit CPU wall-clock. Recorded in the report config.
- **Baseline run**: the first `live-local` agent and single-shot runs on the
  `dev` split are the baseline. Any regression thresholds are derived from it
  and written to `evals/thresholds.v1.json` (with the baseline report path
  and git SHA) before any `test` split run. No threshold is promised up front.
- First `live-local` runs (agent and single-shot baseline, same model, same
  budgets and `k`) on the `test` split. Reports and cassettes committed.

Deliverables
- `src/agentic_research/providers/{openai_compat.py,hosted.py,recording.py,replay.py}`,
  `src/agentic_research/policies/llm.py`, `prompts/v1/`,
  `evals/cassettes/<run-id>/`, updated `docs/providers.md` and
  `docs/local-models.md`, `evals/thresholds.v1.json`.
- `evals/results/v1-live-local-qwen3-1.7b-q4_k_m-*.json` (dev baseline and
  test runs). No `live-hosted` report while Cline is paused.

Acceptance criteria
- `OpenAICompatibleProvider` tests (CI, `httpx.MockTransport`): happy path,
  native `tool_calls` parsing, JSON-in-text replies, usage capture (and
  labelled estimate when absent), 5xx/connection retry, timeout, malformed
  JSON, connection refused classified `local_server_unavailable`.
- Runaway guard (CI, scripted local provider): consecutive failures, a retry
  storm, and repeated unparseable replies after repair each end the run with
  `failed:runaway` at the configured count.
- Parser/repair matrix with scripted malformed completions (bad JSON, unknown
  tool, wrong arg types, missing required args, calls embedded in prose,
  stray `<think>` blocks): each is repaired or classified as expected, and
  `parse_failures` before/after repair is correct in the report.
- `HostedProvider` tests (CI only, `httpx.MockTransport` with recorded
  Cline-shaped payloads; no live call):
  - configurable base URL, model, and key env var; a non-allowlisted model id
    is rejected; it is never selected unless `--provider hosted` is given
  - unwrap: a `{"data": ...}` payload parses, and a payload missing `data`
    raises `ProviderResponseError`
  - cost and reasoning-token logging land in the span, run result, and
    report; missing fields are null, not 0
  - max_tokens floor: a default request sends 1000, and config below 1000 is
    rejected for that model
  - usage ledger: each call appends an entry with project, run, purpose, and
    reference cost (Pass); missing cost is logged with `cost_missing: true`
    and never counted as $0; the daily rollup splits at midnight Asia/Manila
    (timestamps at 15:59:59Z and 16:00:00Z)
  - HTTP 402: non-retryable, clear message, increments consecutive failures
    without burning retries
  - 429 Pass-limit/rate-limit: back-off-then-stop, clear message, counts
    toward the runaway guard without burning retries
  - a sentinel hosted API key is absent from logs, traces, cassettes, and
    reports
- Report schema test (CI): every live report has model id, date, provider,
  `headline_eligible`, tool format, `parse_failures`, and the memory
  preflight; hosted reports also have reported-cost fields. The README claim
  generator rejects any report with `headline_eligible: false`.
- A guard fixture fails any test that opens a real network connection, so CI
  never calls a model.
- CI `eval-replay` job: `arp eval --mode replay --cassettes evals/cassettes/<run-id>`
  reproduces each committed `live-local` report's non-latency metrics
  exactly.
- `evals/thresholds.v1.json` exists, cites the dev baseline report, and was
  committed before the first `test` split report (checked by git history in
  the review).
- `pytest -m slow` (manual, on the box) runs a 3-task `smoke-only` run
  against the running local server after `arp doctor` passes. Never part of
  default CI. No hosted smoke runs while Cline is paused.
- If no local runtime works, the model does not fit in memory, or the local
  runs cannot finish, this is recorded as a blocker in `state.json`. The
  affected results stay "not run", and every other M5 deliverable still
  ships.

Tests
- Cassette key stability: reordering dict keys gives the same hash. A
  sampling-param or model-digest change gives a different hash.
- Replay miss gives a clear error naming the request hash.
- Prompt-template versioning: the report records the prompt version and
  hash.

## M6 - Fault injection, failure taxonomy, baselines and ablations

Scope
- `FaultInjector` tool wrapper with seeded faults: transient error (recovers
  after k attempts), permanent error, delay beyond the timeout,
  schema-invalid output, empty results, and injected-instruction content.
  Wired to the `fault` tasks and to a robustness sweep over a base task set.
- Failure classifier over traces: deterministic rules producing primary and
  secondary labels from the taxonomy in PROJECT_SPEC section 7, each with a
  unit test.
- Comparison runner: single-shot baseline, heuristic, and agent. Ablations:
  planning on/off, repair on/off, native vs JSON tool format, BM25 vs hybrid
  backend (local only), and optionally Qwen3-1.7B vs Qwen3-0.6B (local).
  All headline comparisons use the local model. A hosted-vs-local comparison
  is an optional labelled extra, added later only if Cline is re-enabled, in
  a separate table; while Cline is paused it is "not run". Every compared
  configuration shares the
  same search `k`, step/tool/token budgets, task IDs, and scoring cut
  (like-for-like); per-task `n_retrieved < k` is reported. Paired tests
  (McNemar for completion, paired bootstrap otherwise).
- Robustness and comparison reports, and a generated failure-taxonomy table.

Deliverables
- `src/agentic_research/eval/{faults.py,taxonomy.py,compare.py}`.
- `evals/results/v1-robustness-*.json`, `evals/results/v1-compare-*.json`,
  and cassettes for every committed `live-local` config (and any later
  `live-hosted` extra).

Acceptance criteria
- `arp eval --suite tasks/v1 --faults all --mode oracle` (CI): 100% of
  faulted runs end in a classified terminal status with zero unhandled
  exceptions. The oracle recovers wherever recovery is possible by design.
- Every taxonomy label has a test that triggers it from a synthetic trace,
  and an "unclassified" count is reported, never hidden.
- Comparison reports exist for every configuration that was run, including
  losing ones. CI replay reproduces them. Compared rows share depth/budget/`k`;
  a drift check fails if two rows in one report used different scoring cuts.
- The README comparison and failure tables are generated from reports by
  script, and a CI check fails if they drift.

Tests
- Fault determinism: the same seed gives the same fault schedule.
- Taxonomy precedence rules are tested (e.g., timeout followed by budget
  exhaustion gives primary `timeout_unrecovered`).
- Statistical helpers are checked against hand-computed McNemar and
  bootstrap examples.

## M7 - Container, CI image smoke eval, docs, v0.1.0 release

Scope
- Dockerfile (multi-stage, non-root, slim). Entrypoint is the `arp` CLI. An
  example `compose.yaml` pairs it with an Ollama service, marked unvalidated
  on the dev box.
- CI job: build the image, then run `arp eval --mode oracle` and
  `--mode replay` inside the container.
- Hardening: structured logging, clear error messages for config/provider
  failures, and a dependency audit (`pip-audit` in CI, non-blocking at
  first).
- Docs:
  - README: architecture, quickstart, local runtime setup (Qwen3-1.7B
    Q4_K_M, memory check), results tables generated from `live-local` and
    model-free reports (each row with model id, date, provider), a separate
    "optional hosted extra" table showing "not run" while Cline is paused,
    and limitations (small CPU model, CPU latency, RAM limits).
  - `docs/architecture.md`, `docs/evaluation.md` (methodology, threats to
    validity), and `docs/adr/` index.
- Tag v0.1.0 and a CHANGELOG release section.

Deliverables
- `Dockerfile`, `compose.yaml`, an updated `.github/workflows/ci.yml`, docs,
  and the release tag.

Acceptance criteria
- The CI image-build job is green and the in-container oracle and replay
  evals pass.
- `python scripts/review_project.py <checkout> --run-checks` from the
  controller reports no CRITICAL/MAJOR findings (required files, secrets
  scan, tests, CI, committed eval reports, README sections).
- Every number in the README appears in a committed report referenced by
  path. A test or script check verifies this.

Tests
- Container smoke (CI).
- Docs link check and README table drift check (CI).
