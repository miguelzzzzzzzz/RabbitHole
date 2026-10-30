# PROJECT_SPEC: RabbitHole

A tool-using research agent built as an engineered system rather than a demo:
an explicit, hand-rolled agent loop with planning, typed tool selection and
routing, multi-step tool use, serializable run state, hard budgets,
retries/timeouts, guardrails, and per-step tracing. Every answer carries
citations to exact character spans in the *normalized* document text
(CitationNeeded index format 2), and the citations are checked
programmatically against the stored text and ``content_hash``. Evaluation uses a versioned task suite and programmatic
graders, with no LLM judge. The default agent policy model is **Qwen3-1.7B
running locally on CPU** (quantized GGUF, Q4_K_M) behind an OpenAI-compatible
local server (Ollama or llama.cpp `llama-server`, per portfolio ADR-008). Every
headline number comes from that local model. A hosted OpenAI-compatible
backend exists only as an optional, separately labelled extra; while Cline is
paused, all hosted runs are reported as "not run" (project ADR-0009). CI never
calls a model: it runs scripted and replayed providers at zero cost.

Status: in progress (started 2026-10-30, after CitationNeeded v1.0.0).
Last revised: 2026-10-30 (repository created; local-first while Cline is
paused, ADR-0009).
Repository: RabbitHole, https://github.com/miguelzzzzzzzz/RabbitHole. The Python package
(`agentic_research`) and CLI (`arp`) names are unchanged.
Catalog: project 2 in the controller's `portfolio.yaml`, which follows the
owner's brief. At controller commit `0f498a7`, catalog ids still use the
pre-rename slugs, so this project's id is `agentic-research-platform`. Names
used in this document map to controller ids as follows:

| # | Repo name | Controller id (at `0f498a7`) |
| --- | --- | --- |
| 01 | CitationNeeded (renamed from production-rag-engine; distribution `production-rag-engine`, import `rag_engine` unchanged) | `production-rag-engine` |
| 02 | RabbitHole | `agentic-research-platform` |
| 03 | TrustIssues | `llm-evaluation-lab` |
| 07 | WhoSpentMyTokens | `ai-observability-platform` |

Relevant portfolio
decisions in `DECISIONS.md`, all Accepted (2026-10-09):
- ADR-006: Align project catalog with the owner's brief
- ADR-007: Cross-project reuse through pinned releases behind
  consumer-side interfaces
- ADR-008: Local LLM inference through OpenAI-compatible servers; CI never
  calls a model. **This is the operative model decision for RabbitHole while
  Cline is paused** (project ADR-0009, owner decision relayed by Kim on
  2026-10-25: "Cline is off until the owner says otherwise").
- ADR-016 (hosted `cline-pass/deepseek-v4.1-flash` via Cline Pass) is kept
  only as the definition of the **optional hosted backend**: it is the only
  allowed hosted model id, Pass usage is logged as **reference cost (Pass)**
  (not money spent, rolled up per Asia/Manila day), and Cline is used only for
  productive work. While Cline is paused, no Cline API call is made, and no
  RabbitHole default, headline number, or success criterion depends on it.
- ADR-009: Programmatic graders by default; LLM-as-judge only with measured
  agreement

## 1. Problem statement

Most "research agent" repositories wrap a framework's ReAct agent around a
web-search API and a frontier model, then show a few transcripts. They cannot
answer the questions an engineering team asks before it ships an agent:

- Does the agent pick the right tool, and how many calls does it waste?
- When a tool fails, times out, or returns garbage, does the run recover,
  stop cleanly, or loop until it runs out of budget?
- Are the cited sources real, and does the cited text support the answer?
- What does a run cost in steps, tokens, and latency, and can a failed run be
  replayed exactly to debug it?

CitationNeeded (project 01) answers "is retrieval good?" for a single
query-to-answer pipeline. This project answers a different question. Given a
retriever and other tools, can an agent decompose a question, route among
tools, gather evidence over several steps under budgets, and produce a
grounded, verifiable answer? And how does it fail when it doesn't?

## 2. Goals and non-goals

Goals
- A small, readable agent loop (policy, then guardrails, then executor, then
  observation) with an explicit run-state model. The loop is a pure
  orchestration layer, and every LLM call sits behind an interface.
- Typed tool registry: pydantic argument/output models, JSON-schema export in
  the OpenAI `tools` format, argument validation, and structured error
  observations returned to the policy instead of exceptions.
- Planning: an optional explicit plan step (sub-questions stored in state),
  ablatable on/off.
- Multi-step tool use over offline tools: `search_corpus`, `read_document`,
  `fetch_url` (local fixture pages by default), `calculator` (safe AST
  evaluator), and a terminal `final_answer` with structured citations.
- Budgets enforced by the loop rather than the model: max steps, max tool calls
  (total and per tool), wall-clock deadline, token budget (from provider usage
  or an estimate), and repair attempts.
- Per-tool timeouts, retries with backoff for transient errors only (errors
  are classified), loop detection (repeated identical calls), and graceful
  terminal statuses (`succeeded`, `abstained`, `budget_exhausted:<kind>`,
  `failed:<reason>`).
- Guardrails: tool outputs treated as untrusted data and delimited in
  prompts; injected-instruction detection in fetched content; URL allowlist;
  per-task tool allowlist; citations must reference text the agent actually
  observed in this run.
- Tracing: a versioned JSONL trace per run (run, step, provider-call, and
  tool-call spans with timings, token usage, errors, and budget counters),
  plus a CLI trace viewer.
- An OpenAI-compatible provider interface with four implementations:
  - `OpenAICompatibleProvider` (local, **default**): Qwen3-1.7B (GGUF
    Q4_K_M) on CPU behind Ollama or llama.cpp `llama-server`, temperature 0,
    non-thinking mode, with a memory preflight before every live run.
  - `HostedProvider` (**optional**, off by default): the same
    OpenAI-compatible client pointed at a configurable base URL, model id, and
    API-key env var. Its Cline adapter keeps the Cline-specific handling:
    `{"data": ...}` unwrap, reported cost and reasoning tokens, a max_tokens
    floor, per-call **reference cost (Pass)** in the shared usage ledger,
    HTTP 402 non-retryable, and 429 Pass-limit/rate-limit back-off-then-stop.
    Every hosted run is labelled as an extra and is "not run" while Cline is
    paused.
  - scripted: canned completions, for tests.
  - replay: recorded completions, for CI.
- A provider-agnostic runaway guard in the loop that stops a run on repeated
  provider failures (local server errors, timeouts, crashes, unparseable
  replies after repair) or too many retries, for local and hosted runs alike.
- An evaluation harness covering task completion, tool-selection accuracy,
  citation correctness, tool-call counts, latency, and failure modes, with
  baselines, ablations, and fault injection.

Non-goals
- Building a better retriever. Retrieval comes from project 01 (ADR-0002);
  retrieval-quality metrics belong to 01.
- Open-ended live web research as the default. A live HTTP fetch tool exists
  but is opt-in, allowlisted, and never used in tests or CI.
- Arbitrary code execution or a Python sandbox tool (security scope too large
  for this project).
- Multi-agent orchestration, parallel tool calls, long-term memory, or
  resumable runs across processes. State is serializable for tracing and
  replay only.
- A general LLM evaluation framework (project 03, TrustIssues), or
  OpenTelemetry pipelines, dashboards, and cost tracking (project 07,
  WhoSpentMyTokens). Traces use a documented JSONL schema that
  could later be exported.
- Fine-tuning models for tool use.
- A web UI.

## 3. Users and use cases

- An engineer evaluating how well a small (1.7B) model running locally on a
  CPU-only box can drive a tool-using agent for internal research questions,
  who needs measured tool-selection accuracy, parse-failure rates, latency, and
  failure modes rather than anecdotes.
- An engineer debugging an agent regression, who needs to replay a recorded
  run exactly, inspect each step's trace, and see which budget or guardrail
  ended it.
- An engineer adding a new tool, who needs a typed contract, schema export,
  timeout/retry policy, and a fault-injection test pattern to copy.
- Interview reviewers, who should be able to clone the repo, run
  `arp eval --provider replay` offline in minutes, and get the same numbers
  as the committed report.

## 4. Architecture

```
                 +-------------------------- run(task, budgets) --------------------------+
                 |                                                                        |
 task/question ->|  RunState (pydantic, serializable)                                     |
                 |   plan, messages, evidence store, step records, budget counters, status|
                 |                                                                        |
                 |   loop:                                                                |
                 |     budgets.check(state) ---- exhausted --> finish(budget_exhausted)    |
                 |     action = Policy.next_action(state)      (timeout, retry transient) |
                 |        |   ToolCall | FinalAnswer | GiveUp                             |
                 |        v                                                               |
                 |     Guardrails.check(action, state)  allowlist, per-tool caps,         |
                 |        |                             loop detection, arg validation    |
                 |        v                                                               |
                 |     ToolExecutor.run(call)  timeout -> retry(backoff, transient only)  |
                 |        |                    -> ToolResult | ToolError (classified)     |
                 |        v                                                               |
                 |     observation (untrusted-data envelope, truncated) -> state          |
                 |     FinalAnswer -> CitationVerifier (span == normalized stored text;   |
                 |                    content_hash mismatch => stale)                      |
                 |        invalid & repair budget left -> feedback observation, continue  |
                 +--------------------------------+---------------------------------------+
                                                  |
                         Tracer (JSONL spans) <---+---> RunResult (answer, citations, status)

 Policies:  LLMPolicy(ChatProvider, PromptBuilder, ActionParser+Repair)
            HeuristicPolicy (rule-based router, no LLM; baseline)
            OraclePolicy (gold trajectory; validates graders and harness)
 Providers: OpenAICompatibleProvider(httpx; local Qwen3-1.7B, default live)
            | HostedProvider(httpx; optional extra, Cline adapter; off while Cline paused)
            | ScriptedProvider (tests) | ReplayProvider(cassette; CI)
 Tool-call formats: native tools/tool_calls | JSON-in-text (strict parser,
            parse-failure counter); default chosen by the M1 probe
 Tools:     search_corpus -> SearchBackend -> rag_engine Retriever (pinned dep)
            read_document (doc_id, char range) | fetch_url (fixtures; live opt-in)
            calculator (AST, no eval) | final_answer (structured, terminal)
 Eval:      TaskSuite(v1, versioned) -> Runner(mode) -> Graders -> JSON report
            FaultInjector wraps tools: error / timeout / malformed / empty / injection
```

Key interfaces (Python `Protocol`s; names are indicative):

```python
class Tool(Protocol[ArgsT, OutT]):
    name: str
    description: str
    args_model: type[ArgsT]          # pydantic; exported as JSON schema
    output_model: type[OutT]
    timeout_s: float
    retry: RetryPolicy               # attempts, backoff, which error classes retry
    async def run(self, args: ArgsT, ctx: ToolContext) -> OutT: ...

class ChatProvider(Protocol):        # OpenAI chat-completions shape
    name: str
    async def complete(self, request: ChatRequest) -> ChatResponse: ...
    # ChatResponse: message (content | tool_calls), usage, model, latency_ms, raw

class Policy(Protocol):
    async def next_action(self, state: RunState) -> Action: ...
    # Action = ToolCall | FinalAnswer | GiveUp

class SearchBackend(Protocol):       # consumer-side seam over project 01
    def search(self, query: str, k: int, filters: Mapping[str, Any] | None) -> list[Passage]: ...
    def get_document(self, doc_id: str) -> SourceDocument: ...
    # Passage: doc_id, content_hash, start_char, end_char (into normalized text),
    #          page, page_end, text, score. Adapter reads search --json `hits`,
    #          checks schema_version == 1 (fail clearly if unknown; 1 at both
    #          v0.1.0 and v1.0.0), never assumes a bare list.
    #          SourceDocument.text is the normalized text from
    #          documents.jsonl; content_hash is sha256 of that text.

class Grader(Protocol):
    name: str
    def grade(self, task: Task, result: RunResult, trace: Trace) -> GradeRecord: ...
```

Design rules: the loop never raises on tool/provider failure (every failure
becomes a classified observation or terminal status). Time comes from an
injectable clock, so budget and timeout logic is unit-tested without
sleeping. Tool execution is sequential, so traces are deterministic. The
CLI, Docker image, and eval runner all compose the same `run()`.

## 5. Technology choices

| Concern | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | same toolchain as 01; `asyncio.timeout`, `StrEnum` |
| Agent loop | hand-rolled, small (target: a few hundred lines of orchestration) | the loop is the skill being shown; frameworks hide budgets, retries, and traces (ADR-0001) |
| Concurrency | `asyncio` with `asyncio.timeout` | coroutine cancellation gives real per-tool timeouts; sync threads cannot be cancelled |
| Data models | pydantic v2 | tool arg validation, JSON-schema export, trace/report schemas |
| LLM access | `httpx` against OpenAI-style chat completions. `OpenAICompatibleProvider` talks to the local server (`ARP_LLM_BASE_URL`). The optional `HostedProvider` reuses the same client with a configurable base URL, model id, and API-key env var; its Cline adapter unwraps the `data` envelope | one small client, thin per-backend adapters; no SDK lock-in (ADR-0003) |
| Default policy model | **Qwen3-1.7B** (Apache-2.0), quantized **GGUF Q4_K_M**, on CPU via Ollama or llama.cpp `llama-server` (whichever installs cleanly on the box in M1; both are OpenAI-compatible per portfolio ADR-008). Temperature 0, fixed seed, non-thinking mode, context length pinned (starts at 4096; raised only if the memory check passes). The box is CPU-only with about 3 GB of RAM to spare and **no swap**, so a memory preflight runs before every live run (see section 7) | local-first while Cline is paused (ADR-0009, under portfolio ADR-008). Model name, tag/digest, GGUF file sha256, quantization, server and version recorded in every report |
| Optional smaller local model | Qwen3-0.6B (GGUF Q4_K_M), same runtime | optional "model A vs model B" ablation only; never a headline row |
| Optional hosted backend | OpenAI-compatible, configurable `ARP_HOSTED_BASE_URL`, `ARP_HOSTED_MODEL`, `ARP_HOSTED_API_KEY_ENV`. Owner policy (portfolio ADR-016) currently allows only `cline-pass/deepseek-v4.1-flash` via Cline Pass, enforced by an allowlist; `max_tokens` floor 1000 for that model because hidden reasoning tokens use most of the output budget | **not the default**; off while Cline is paused, all hosted runs "not run"; if Cline returns, hosted runs are added as a separately labelled extra (ADR-0009) |
| Tool-call formats | native `tools`/`tool_calls`, or a JSON-in-text protocol with a strict parser and a parse-failure counter. The default is set by the M1 tool-calling probe on the local model: a frozen, versioned 20-prompt probe file and a 90% bar (at least 18/20) in config, both committed before the probe runs; results reported as raw counts plus 95% Wilson intervals | small local models often emit malformed native calls; both paths are built and measured |
| Retrieval | CitationNeeded (distribution `production-rag-engine`, from `github.com/miguelzzzzzzzz/CitationNeeded`) pinned to `v1.0.0` (pin final: tagged at `b8015f3`, re-verified 2026-10-29, interface unchanged from `977f091` (version string only); contract unchanged from `v0.1.0`), behind `SearchBackend`. Index format 2: `documents.jsonl` with `content_hash` and normalized text; offsets into that text; `doc_id = sha256("<corpus_id>:<relpath>")[:16]`; `search --json` is `{schema_version, hits}` | reuses a tested, verifiable-offset retriever; keeps 02 about orchestration (ADR-0002, portfolio ADR-007) |
| CI retrieval mode | BM25-only (no model download); hybrid with fastembed in local runs | deterministic, fast CI |
| Calculator | stdlib `ast` whitelist evaluator | no `eval`; bounded exponent/size |
| HTML fetch/parse | fixture map + stdlib `html.parser`; `httpx` for opt-in live fetch | offline by default (ADR-0007) |
| Tracing | JSONL with versioned schema; `rich` for the CLI viewer (optional) | diffable, replayable, no collector infra (ADR-0008) |
| Determinism | scripted + replay providers, recorded cassettes | zero-cost reproducible CI evals (ADR-0005) |
| Eval data | HotpotQA distractor (dev) subset + hand-authored tool suite | public labeled multi-hop data with gold supporting sentences, plus tool-specific tasks (ADR-0004) |
| Statistics | stdlib + numpy bootstrap; McNemar for paired completion | CIs and paired comparisons with no heavy deps |
| Quality | ruff, mypy --strict, pytest (+ hypothesis for budget invariants) | matches the portfolio quality gate |
| Packaging | Docker (built and smoke-tested in GitHub Actions only) | no local Docker (portfolio ADR-003) |

Not used, deliberately: LangChain/LangGraph, LlamaIndex agents, CrewAI,
AutoGen, smolagents, PydanticAI, and the OpenAI Agents SDK (ADR-0001), as
well as OpenTelemetry and any vector DB server.

## 6. Data (sources, licenses, labeling)

1. **HotpotQA, distractor setting, dev split** (multi-hop Wikipedia QA with
   sentence-level supporting facts). License: CC BY-SA 4.0
   (<https://hotpotqa.github.io/>). It has 7,405 dev questions, and each
   comes with 10 paragraphs (2 gold plus 8 distractors) and
   `supporting_facts` given as `[title, sent_id]`.
   - Acquisition: a download script fetches a pinned revision (Hugging Face
     `hotpotqa/hotpot_qa`, `distractor/validation` parquet; the CMU JSON URL
     is a fallback) and verifies a sha256. The data is never committed.
     (The CMU HTTP host timed out from the dev box on 2026-10-09, while the
     HF mirror responded.)
   - Subset: a seeded, stratified sample (by `type` bridge/comparison and
     `level`). The sample IDs are committed as `evals/data/hotpotqa_ids.v1.txt`
     so the subset is fixed. Size is set by a CPU timing pilot with the
     local Qwen3-1.7B in M5 and recorded, so it is not assumed here.
   - Corpus: the union of all context paragraphs of the sampled questions,
     deduplicated by (title, text). Pooling makes `search_corpus` a real
     retrieval problem rather than a 10-paragraph lookup. Each paragraph is
     one document whose text is the concatenation of its sentences, so gold
     supporting sentences map to exact `(doc_id, start_char, end_char)`
     spans by construction (offsets into the *normalized* document text that
     CitationNeeded stores in `documents.jsonl`). Built as index format 2
     with an explicit fixed `corpus_id` of `rabbithole-hotpotqa` (never the
     folder-name fallback; renaming the ingest root must not change doc ids).
     A unit test pins at least one known `(relpath -> doc_id)` pair via
     `sha256("rabbithole-hotpotqa:<relpath>")[:16]`.
   - Labels: the gold answer (official HotpotQA normalization), gold evidence
     spans (supporting sentences), and gold tools `{search_corpus, read_document}`
     with a reference minimum call count derived from the gold paragraphs.
   - Splits: `dev` (prompt and policy iteration) and `test` (held out,
     reported) are disjoint ID lists, fixed before any model run.
   - CC BY-SA note: cassettes and traces can contain model output that quotes
     HotpotQA text. Committed cassettes therefore store provider responses
     only. Tool outputs are recomputed at replay time from the downloaded
     corpus, and `evals/cassettes/` carries a CC BY-SA 4.0 attribution notice.

2. **Hand-authored tool suite `tasks/v1/`** (written for this repo; MIT
   for code, CC BY 4.0 for fixture text). This is a small corpus of
   project-written documents (clearly labelled synthetic fixtures about a
   fictional domain, never presented as real data), plus fixture HTML pages
   for `fetch_url`. The fixture corpus is indexed with explicit
   `corpus_id=rabbithole-tasks-v1` (index format 2). Task categories:
   - `lookup` (search, then read)
   - `multi_hop` (two documents)
   - `compute` (search, then calculator)
   - `fetch` (fixture page)
   - `unanswerable` (gold = abstain)
   - `injection` (a fixture page contains instructions the agent must ignore)
   - `distractor` (a tempting wrong tool or document)
   - `fault` (paired with a fault-injection spec)
   - Each task YAML has: `id`, `version`, `category`, `split`, `question`,
     `gold.answer` (+ aliases, numeric tolerance, or list semantics),
     `gold.evidence` (doc_id + char span + expected quote), `gold.tools`
     (`required`, `acceptable_sets`, `forbidden`, `expected_first`),
     `gold.min_calls`, optional `budgets`, and optional `faults`.
   - Labeling: authored by hand. A test asserts that every gold span resolves
     to its expected quote and that every task validates against the schema.
     Suite version and sha256 go into every report. The final task count is
     whatever the committed suite contains. Target scale is a few dozen
     tasks, enough for at least 3 tasks per category.

3. **Unit-test fixtures**: tiny documents and canned provider completions,
   including deliberately malformed tool calls.

## 7. Evaluation methodology

Principle: every metric is computed by a deterministic program from the run
result, the trace, and gold labels. No LLM-as-judge (portfolio ADR-009). The
honest consequence: free-form answer quality beyond short factual answers is
not measured. Tasks are designed with short, gradable answers (entity, number,
date, yes/no, list, or abstain).

Run modes, always reported separately and labelled in every report:

| Mode | Policy / provider | Cost | Where | Purpose |
| --- | --- | --- | --- | --- |
| `oracle` | gold trajectory | free | CI | grader and harness validation (should score 100%; anything less is a bug) |
| `heuristic` | rule-based router, extractive answer | free | CI | no-LLM baseline for routing and completion |
| `replay` | recorded completions of a `live-local` (or, later, `live-hosted`) run | free (re-reports the recorded usage without calling a model) | CI | proves committed numbers reproduce from committed artifacts |
| `live-local` | **Qwen3-1.7B (GGUF Q4_K_M) via the local OpenAI-compatible server (default)** | zero spend, slow on CPU | dev box, manual | **the primary results; every headline number comes from here** |
| `live-hosted` | optional `HostedProvider` (Cline adapter: `cline-pass/deepseek-v4.1-flash`) | provider-reported `cost`, logged as **reference cost (Pass)** (not money spent) | dev box, manual, only if the owner re-enables Cline | optional labelled extra. **"not run" while Cline is paused**; never in headline numbers or claim tables |

Every live report records:
- model id, run date, provider (`local` or `hosted`), and
  `headline_eligible` (true only for `live-local` runs on a clean tree that
  are not `smoke-only`)
- prompt, completion, and (if any) reasoning tokens, the call count, and
  `max_tokens`
- tool-call format, `parse_failures` (count and rate per provider call, before
  and after repair), and runaway-guard trips
- the memory preflight result (`MemAvailable` before the run, the configured
  threshold, and peak server RSS when measurable)
- hosted runs only: the reported cost (sums of `usage.cost`, `gateway_cost`,
  and `market_cost`, null if not reported, never 0), labelled **reference
  cost (Pass)**, plus `billing_tier: cline_pass`

Local reports also record model name, tag, and digest, GGUF file sha256,
quantization, server name and version, context length, temperature, seed,
thinking mode, prompt version, tokens/s, hardware (CPU model, cores, RAM, swap, OS),
task-suite version and hash, dataset checksum, git SHA, a dirty-tree flag
(`git status --porcelain` non-empty), and timestamp. Results from a dirty
tree are refused for committed reports, or clearly marked `dirty: true` and
excluded from README claims. CPU inference at temperature 0 with a fixed
seed is close to deterministic but not guaranteed, so repeats are
configurable and the number of repeats `n` is reported.

Memory preflight (before every `live-local` run and the M1 probe): read
`MemAvailable` from `/proc/meminfo` and refuse to start unless it covers the
GGUF file size plus the KV cache for the configured context plus a fixed
headroom (default 1 GB, `ARP_MIN_FREE_MB` overrides). There is no swap, so an
over-commit means the kernel OOM-kills the server, not a slow run. The check
result goes into the report. A server that dies mid-run surfaces as provider
errors that trip the runaway guard (`failed:runaway`), never a hang.

Thresholds are set from a baseline, not promised up front. A 1.7B model on
CPU may score low on multi-hop tasks, and that is a result, not a failure of
the plan. The first `live-local` run on the `dev` split is the **baseline**.
Any regression thresholds (for example "completion must not drop by more than
the baseline CI half-width") are written to `evals/thresholds.v1.json` with
the baseline report path and git SHA before any `test` split run, and are
never tuned on `test`.

Eval hygiene (like-for-like comparisons):
- **Equal depth / budget.** Every compared configuration (agent policy,
  single-shot, heuristic, ablation, search backend) uses the same candidate
  depth, step/tool/token budgets, and scoring `k` before aggregation. A
  configuration that returns fewer than `k` items for a task is reported
  per-task (`n_retrieved < k`) and never silently scored as if it filled
  `k`. Dedup to unique docs must not let one baseline score on fewer items
  than another.
- **Latency.** An untimed warm-up call runs once per configuration before
  timed tasks. Model load and index load are excluded from p50/p95 and
  reported separately as `load_ms`.
- **Degenerate labels.** Tasks or HotpotQA queries with no relevant gold
  evidence spans are excluded from completion/citation aggregates and
  counted separately (`n_no_relevant`), never silently scored 0. Intentional
  `unanswerable` tasks (gold = abstain) stay in the suite and are graded
  for abstention.
- **Tiny sets.** Any suite or subset too small to support claims (for
  example a 3-task provider smoke) is labelled `smoke-only` in the report
  and must not appear in README claim tables.
- **Small-n rates.** Any rate over a small count (the M1 tool-call probe,
  per-category completion, fault recovery per type) is shown as a raw count
  `k/n` plus a 95% Wilson interval, never as a bare percentage.

Metrics (per task, then aggregated overall and per category, with sample
size and bootstrap 95% CIs):

- **Task completion**. A task passes only if all hold: correct answer (normalized
  exact match, numeric tolerance, or set equality; abstention for
  `unanswerable`), at least one valid citation overlapping a gold evidence
  span, a terminal status of `succeeded` or `abstained` as expected, and no
  budget violation. Answer EM/F1 (HotpotQA normalization) are also reported
  on their own.
- **Tool-selection accuracy**. Required-tool recall, unnecessary-tool rate
  (calls outside every acceptable set), forbidden-tool violations,
  first-tool accuracy, hallucinated tool names, and schema-invalid call rate
  (before and after repair). Set-based comparison, so any valid order passes.
- **Citation correctness**. Several separate checks:
  - Validity: the quoted text equals the *normalized* stored document text
    at `[start_char:end_char]` exactly (from the index's `documents.jsonl`,
    not the raw file on disk). Offsets are into that normalized text.
  - Freshness: the citation's `content_hash` matches the stored document's
    `content_hash`; a mismatch is flagged `stale_source` (source changed or
    wrong document).
  - Groundedness: the span lies inside text the agent actually observed in
    this run.
  - Support precision/recall: character-overlap of cited spans with gold
    evidence spans, with the threshold stated in the report. For HotpotQA
    this is computed at sentence level, comparable to the official
    supporting-fact metric.
- **Tool-call counts**. Total and per tool, efficiency ratio (calls / gold
  `min_calls`), redundant calls (identical tool+args repeated), and steps
  per run.
- **Latency**. Wall-clock per run and per step, split into provider time,
  tool time, and harness overhead, with p50/p95 after an untimed warm-up per
  configuration. Model/index load is excluded from those percentiles and
  reported separately as `load_ms`. In `replay`, provider time is the
  recorded value and labelled as recorded. Harness and tool time are
  measured live.
- **Tokens and cost**. Prompt, completion, and reasoning tokens per run,
  from provider `usage` when available, otherwise estimated (and labelled).
  Local runs report zero spend plus measured wall-clock and tokens/s
  (portfolio ADR-011). For the optional `live-hosted` extra only, the
  provider-reported `cost`, `gateway_cost`, and `market_cost` per run and per
  eval, labelled "provider-reported, reference cost (Pass)".
- **Failure modes**. A deterministic classifier over traces assigns each
  non-passing run one primary and any secondary labels:
  - Policy: `wrong_tool`, `hallucinated_tool`, `malformed_call_unrepaired`
    (includes JSON-in-text parse failures left after repair),
    `premature_finish`, `no_citation`, `invalid_citation`,
    `ungrounded_citation`, `stale_source`, `wrong_answer_valid_citations`,
    `abstained_when_answerable`, `answered_when_unanswerable`,
    `followed_injection`, `loop`.
  - Tools: `tool_error_unrecovered`, `timeout_unrecovered`.
  - Budgets: `budget_exhausted:{steps,tool_calls,time,tokens}`. These are
    behaviour limits of the agent loop, not spend limits.
  - Runaway: `failed:runaway` (repeated provider failures or retries,
    local or hosted).
  - Provider: `provider_error`, `local_server_unavailable` (connection
    refused or server died, e.g. OOM-killed).
  Each label has a unit-tested detector, and the counts make up the failure
  taxonomy table.
- **Fault injection / robustness**. A `FaultInjector` wraps tools with
  seeded faults: transient error (recovers after k attempts), permanent
  error, delay beyond the timeout, schema-invalid output, empty results, and
  injected-instruction content. Reported as recovery rate per fault type
  (the task still passes, or the run ends with the correct clean terminal
  status) and the share of faulted runs that end in a classified terminal
  status. That share is an engineering invariant: unhandled exceptions are
  bugs.

Baselines and comparisons (all reported, including ones that lose):
1. Single-shot retrieve-then-answer: one `search_corpus` call, then one model
   call, with no loop. This is 01's pattern with the same local model, so it
   isolates the value of multi-step agency, especially on multi-hop tasks.
2. Heuristic router (no LLM) for tool selection.
3. Oracle trajectory (upper bound and harness check).
4. Ablations on the same model:
   - planning step on/off
   - malformed-call repair on/off
   - native `tools` vs JSON-in-content
   - BM25 vs hybrid search backend (local only)
   - model A vs model B: Qwen3-1.7B vs Qwen3-0.6B (optional, local)
   - optional hosted extra (later, only if Cline is re-enabled): the same
     suite on the hosted backend, reported in a separate, labelled table
     and never mixed into headline rows
Every compared row uses the same search `k`, step/tool/token budgets, and
task IDs (like-for-like). Paired comparisons on identical task IDs use
McNemar's test for completion and a paired bootstrap for continuous
metrics. No tuning on the `test` split.

Regression gate in CI: `replay` over the committed cassettes must reproduce
the committed report's non-latency metrics exactly, and `oracle` must score
100% on every grader.

Results (placeholder, to be filled only from committed reports under `evals/results/`):

| Mode / config | Suite | n | Completion | Tool-sel. recall | Citation validity | Calls / task | p95 latency | Reported cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | v1 | not run | not run | not run | not run | not run | not run | n/a |
| heuristic | v1 | not run | not run | not run | not run | not run | not run | n/a |
| single-shot (Qwen3-1.7B Q4_K_M, local CPU) | v1 | not run | not run | not run | not run | not run | not run | zero spend |
| agent (Qwen3-1.7B Q4_K_M, local CPU) | v1 | not run | not run | not run | not run | not run | not run | zero spend |

Optional hosted extra (separate table; never a headline number):

| Mode / config | Suite | n | Completion | Tool-sel. recall | Citation validity | Calls / task | p95 latency | Reported cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent (hosted, `cline-pass/deepseek-v4.1-flash`) | v1 | not run (Cline paused) | not run | not run | not run | not run | not run | not run |

Each live row also shows model id, run date, and provider.

## 8. Success criteria

All criteria are verifiable by running something, and none presuppose a result.
No numeric quality threshold is promised for the 1.7B local model; thresholds
come from the first baseline run (section 7).

- Harness correctness: `oracle` mode scores 100% on every grader across both
  suites (CI). Every grader has unit tests with positive and negative cases.
  Every failure-mode detector has a test that triggers it.
- Budgets: a property test (hypothesis) shows that for any scripted policy, a
  run never exceeds max steps, max tool calls (total and per tool), or the
  deadline (with an injected clock).
- Robustness: across the fault-injection suite, 100% of runs end in a
  classified terminal status with no unhandled exception (CI). Recovery rates
  per fault type are reported, whatever they are.
- Citations: no answer is reported as `succeeded` with a citation that fails
  the validity, freshness (`content_hash`), or groundedness checks (enforced
  by the verifier, tested). Validity slices normalized text from
  `documents.jsonl`; a hash mismatch is `stale_source`.
- Tool calling (M1): the probe set (`probes/tool_calls.v1.yaml`, 20
  tool-expecting prompts) and the bar (`config/probe.toml`: at least 18/20,
  i.e. 90%, schema-valid calls) were committed and frozen before the probe
  ran. The result is reported per format as a raw count (for example
  `18/20`) with a 95% Wilson interval, noting that n=20 gives a wide range
  (18/20 is about [0.70, 0.97]). It is recorded in `docs/providers.md`, and
  the configured default tool format matches it (config test). Parse failures
  are counted in every live report.
- Agency vs single-shot: on the HotpotQA `test` subset, the agent loop is
  compared with the single-shot baseline on task completion using the same
  local model (Qwen3-1.7B), with McNemar's test. Whichever way it comes out is
  reported; if the agent does not win, the README explains why with the
  failure taxonomy as evidence.
- Headline provenance: every headline/README claim number comes from a
  `live-local` (or model-free) report with `headline_eligible: true`. A CI
  check fails if a hosted, `smoke-only`, or dirty-tree report feeds a claim
  table. While Cline is paused, the README lists hosted runs as "not run".
- Reproducibility: CI `replay` reproduces each committed `live-local` (and any
  later `live-hosted`) report's non-latency metrics exactly. Every README
  number traces to a committed report with git SHA, dirty-tree flag (clean),
  and config.
- Guardrails: on `injection` tasks, `followed_injection` is detected and
  reported. Live fetch is off by default, and its allowlist is tested.
- Engineering: CI green (ruff, mypy --strict, pytest). The Docker image
  builds in CI and runs `arp eval --mode oracle` and `--mode replay` inside
  the container. `.env.example` is present, and the portfolio review gate
  passes with no CRITICAL/MAJOR findings.

## 9. Milestones

See `PLAN.md` for scope, deliverables, acceptance criteria, and tests:
M1 tool contracts, trace schema, local runtime, and the local tool-calling
probe (Qwen3-1.7B); M2 agent loop, budgets, retries/timeouts;
M3 research tools on the project-01 retriever, citations, and guardrails;
M4 task suites, graders, eval runner, oracle/heuristic baselines;
M5 local LLM policy (default), parsing/repair, optional hosted backend,
record/replay, baseline and first local runs;
M6 fault injection, failure taxonomy, comparisons and ablations;
M7 container, docs, and v0.1.0 release.

## 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A 1.7B local model is weaker at tool use and multi-hop reasoning than hosted models, so headline numbers may be low | Treated as a measured result, not a blocker. Tight tool schemas (few tools, few required args, enums where possible); per-task tool allowlists; hard step/tool budgets; non-thinking mode; one repair attempt; the planning-step ablation. Honest reporting: thresholds come from a baseline run, the failure taxonomy is published, and losing comparisons are kept. An optional hosted extra can be added later as a separately labelled table if Cline returns |
| Qwen3-1.7B native tool calls through the local server are unreliable (template quirks, malformed `tool_calls`, stray `<think>` blocks) | Checked in M1 by the tool-calling probe before anything builds on it; result recorded in `docs/providers.md` and sets the default format. Fallback: a JSON-in-text protocol with a strict parser and a parse-failure counter. Both paths are built and measured. llama.cpp `llama-server` must run with `--jinja` for native tool calls |
| CPU latency (small model, no GPU) makes evals slow | Timing pilot sets subset sizes; tokens/s and `load_ms` reported; short prompts and truncated observations; manual `slow` runs only, never in default CI; cassettes so nothing is re-run unnecessarily |
| RAM limits: about 3 GB to spare on a 16 GB box with **no swap**; over-commit gets the server OOM-killed | Q4_K_M GGUF (Qwen3-1.7B, about 1.1-1.4 GB depending on packaging; measured and recorded in M1), pinned context length, memory preflight before every live run (refuse below the threshold), one model loaded at a time, no concurrent evals. A server crash mid-run becomes provider errors that trip the runaway guard |
| Runaway local runs (server down, repeated timeouts, endless parse failures, retry storms) | The runaway guard is provider-agnostic: `failed:runaway` after `ARP_MAX_CONSECUTIVE_FAILURES` consecutive failed provider calls (including unparseable replies after repair) or `ARP_MAX_RETRIES_PER_RUN` retries, for local and hosted runs alike |
| No Ollama or llama.cpp installed on the box | M1 installs one runtime (Ollama binary, or the llama.cpp `llama-server` CPU build / `llama-cpp-python[server]`) and documents it. If install is blocked, M1 records a blocker in `state.json`; model-free work (M1-M4, oracle/heuristic/scripted/replay) continues, and live results stay "not run" until a runtime works |
| Optional hosted backend: Cline paused; when re-enabled, Pass usage is not spend-capped and has 5-hour, weekly, and monthly limits | Off by default; no Cline API call while paused; hosted runs "not run". If re-enabled: per-call `cost` logged as **reference cost (Pass)** in the shared usage ledger with an Asia/Manila daily rollup (missing cost is a warning, never zero); HTTP 402 non-retryable; 429 Pass-limit/rate-limit back-off-then-stop (both count toward the runaway guard without burning retries); cassettes; productive work only. No headline number depends on it |
| Hosted model drift or non-determinism (optional extra only) | Run date recorded; hosted results kept in a separate labelled table; replay cassettes make CI exact regardless |
| Local inference non-determinism | Temperature 0, fixed seed, pinned model digest, repeats with variance. Replay cassettes make CI exact regardless |
| API key leakage (optional hosted backend) | The hosted key (env var named by `ARP_HOSTED_API_KEY_ENV`, default `CLINE_API_KEY`) is read from the environment only. It is never logged or written to traces, cassettes, or reports, and tests assert this with a sentinel key. `.env.example` lists the name with an empty value |
| Project 01 API changes or release slips | Consumer-side `SearchBackend` protocol and a single adapter module. Pin `git+https://github.com/miguelzzzzzzzz/CitationNeeded@v1.0.0` is final: tagged at `b8015f3`, re-verified 2026-10-29, interface unchanged from `977f091` (version string only). The 2026-10-28 diff against `v0.1.0` (`search --json` shape, `schema_version`, index format, offsets, `doc_id`, `save_index`/`load_index`) found no breaking changes (ADR-0002 note). 02 starts only after 01 is complete (`max_active_projects: 1`) |
| Upstream CitationNeeded contract drift (search JSON shape, index format, offset semantics, doc_id scheme) after the pin | Mitigated by the pinned `v1.0.0` tag (contract re-verified against `v0.1.0` on 2026-10-28); adapter checks `schema_version` and fails clearly on unknown versions; adapter contract tests (unknown schema_version rejected, tampered span caught, hash mismatch flagged stale, pinned `doc_id` for `rabbithole-hotpotqa` / `rabbithole-tasks-v1`); index format 2 rebuild required for old indexes; always pass explicit `--corpus-id` / `corpus_id` (folder-name fallback must never be used) |
| Grader bugs inflate scores | Oracle must score 100%. Adversarial unit tests (wrong span, off-by-one, ungrounded quote). Graders reviewed against HotpotQA's official eval script |
| Gold tool sets are too strict (multiple valid trajectories) | `acceptable_sets` plus set-based metrics. Order is only checked for `expected_first` |
| HotpotQA licensing (CC BY-SA) | Not committed. Download script with checksum. Cassettes store model responses only, with attribution notice |
| Hand-authored suite overfits to the author's agent | Dev/test split fixed before model runs. Prompt iteration only on dev. The suite is versioned and changes are logged in CHANGELOG |
| Prompt injection via fetched pages | Untrusted-data envelopes, detection, URL allowlist, no side-effecting tools, live fetch opt-in |
| Overlap with TrustIssues (03) / WhoSpentMyTokens (07) | Graders here are agent-trajectory specific. Tracing is JSONL only, with no OTel or dashboards. Boundaries are stated in Non-goals and portfolio ADR-006 |

## 11. Deployment

- CLI: `arp run "question" --provider {local,hosted,scripted,replay} --budget-steps N ...`
  (default `local`), `arp doctor` (memory preflight and local server check),
  `arp probe tool-calls` (the M1 probe),
  `arp eval --suite tasks/v1 --mode {oracle,heuristic,replay,live-hosted,live-local}`,
  `arp trace show <run.jsonl>`.
- Container: multi-stage, non-root image with the CLI and the hand-authored
  suite. The default entrypoint runs a replay/oracle eval. The local model
  server is reached through `ARP_LLM_BASE_URL`; it is not bundled. An
  example `compose.yaml` pairs the image with an Ollama container for local
  use. That example is documented as not validated on the dev box. The
  optional hosted backend needs its key passed at runtime, never baked into
  the image.
- Configuration, all documented in `.env.example` with empty secrets:
  - `ARP_PROVIDER=local` (default)
  - `ARP_LLM_BASE_URL` (the local server's OpenAI-compatible URL, e.g.
    Ollama `http://127.0.0.1:11434/v1` or llama.cpp `llama-server`
    `http://127.0.0.1:8080/v1`), `ARP_LOCAL_MODEL` (Qwen3-1.7B Q4_K_M tag or
    GGUF name), `ARP_LOCAL_CTX=4096`, `ARP_TOOL_FORMAT` (`native` or `json`,
    default from the M1 probe), `ARP_MIN_FREE_MB` (memory preflight headroom)
  - `ARP_MAX_CONSECUTIVE_FAILURES=3`, `ARP_MAX_RETRIES_PER_RUN=20` (runaway
    guard, all providers)
  - Optional hosted backend (unused while Cline is paused):
    `ARP_HOSTED_BASE_URL` (Cline chat-completions endpoint when using the
    Cline adapter), `ARP_HOSTED_MODEL` (allowlist: `cline-pass/deepseek-v4.1-flash`),
    `ARP_HOSTED_API_KEY_ENV` (name of the key env var, default
    `CLINE_API_KEY`; the key itself is a secret with an empty value in
    `.env.example`), `ARP_HOSTED_MAX_TOKENS=1000` (floor for that model),
    `CLINE_USAGE_LEDGER` (shared reference-cost (Pass) and usage log; not a
    limit)
  - `ARP_ALLOW_LIVE_FETCH=false`, `ARP_FETCH_ALLOWLIST`, and budget defaults
- Docker is not available on the development box, so the image build and
  in-container smoke eval are validated in GitHub Actions only (portfolio
  ADR-003).
