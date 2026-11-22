# ADR-0004: HotpotQA subset plus hand-authored tool suite, graded programmatically

Status: Accepted (2026-10-30)

## Context
The evaluation must cover task completion, tool-selection accuracy, citation
correctness, tool-call counts, latency, and failure modes, without an LLM
judge (portfolio ADR-009). Data must be licensed and small enough for CPU
runs.

## Decision
Use two suites:

- **HotpotQA distractor dev** (CC BY-SA 4.0): a seeded, stratified subset
  with committed ID lists and fixed dev/test splits. The corpus pools all
  context paragraphs of the sampled questions. Gold supporting sentences
  become exact char spans. Data is downloaded with a checksum and never
  committed.
- **Hand-authored suite `tasks/v1`** (synthetic fixture documents written for
  this repo and labelled as such): covers tool behaviors HotpotQA cannot,
  namely calculator, fetch, unanswerable, prompt injection, distractor tools,
  and faults. Each task carries gold answer/aliases, gold evidence spans,
  acceptable tool sets, forbidden tools, and a reference minimum call count.

Graders are deterministic: normalized EM/F1 (official HotpotQA
normalization), numeric/set match, abstention, citation validity,
groundedness, and gold overlap, tool-set metrics, counts, and latency
breakdown.

## Alternatives considered
- **LLM-generated tasks**: risk evaluating the generator, and need a model.
- **GAIA / WebArena-style benchmarks**: need live web, strong models, or
  heavy environments. Some have restrictive access terms.
- **SciFact (01's corpus)**: not multi-hop, and has no tool diversity.
- **MuSiQue / 2WikiMultiHopQA**: viable multi-hop alternatives, kept as
  possible secondary sets.

## Consequences
Every metric is reproducible and grader correctness is testable (the oracle
must score 100%). Free-form answer quality is not measured and is stated as
such. A hand-authored suite can reflect its author's biases. Mitigations are
a fixed dev/test split before model runs, versioning, and reporting the suite
hash.

## Amendment (2026-11-19): abstention and truncated evidence
Drafted by Bogart against `b852949` for the M3 citation-verifier slice.

**Abstention.**
- The agent abstains only through the policy's `GiveUp`, which ends the run
  `abstained`. "No hits" is never the abstain signal: hybrid search always
  returns `k` neighbours, even for a nonsense query.
- The verifier backs it up: a `FinalAnswer` is rejected unless every claim
  is cited and every citation is valid, fresh and grounded (one repair,
  ADR-0006; wording tightened 2026-11-22, see the verifier contract below). A run that runs out
  of repairs ends `budget_exhausted:repairs`, which is graded as *not*
  abstaining; the agent never earns abstention credit by failing.
- An optional retrieval score floor, if one is ever added (for example a
  guardrail observation "no passage scored above the floor" that the policy
  can act on), is set per search mode (`lexical`, `dense`, `hybrid`,
  `rerank`, since RRF scores are not comparable with BM25 or cosine scores),
  tuned on `dev` only and frozen in `evals/thresholds.v1.json` (value per
  mode, dev report path, git SHA) before any `test` run. It is advice to the
  policy, never an automatic abstain in the loop.
- `answered_when_unanswerable` and `abstained_when_answerable` are always
  reported side by side (rate, count, Wilson interval) with the unanswerable
  task count, plus an abstain-everything baseline row, so a floor cannot win
  one rate by quietly losing the other.

**Truncated observations and groundedness.** A span grounds a citation only
if the policy was shown that span's text in full, untruncated, in the
observation of the step that returned it. A partial view counts as not seen
(no partial-span credit: mapping a cut in canonical JSON back to character
offsets is error-prone because of escaping).
- `search_corpus` observations are cut at passage boundaries: trailing
  passages that do not fit are dropped whole and announced
  (`...[N passages omitted]`), and the observation records
  `shown_passages`. No passage is cut mid-text.
  - Note (2026-11-20, from Chad's spot-check of `d670ed4`): the marker also
    lists the omitted passages' `(doc_id, start_char, end_char)`, best first
    (no text, so not evidence), so a policy shown no passage at all can
    recover with `read_document`. The marker counts against the limit;
    locators that do not fit are counted as `+M more`, and the marker shrinks
    to the bare count before another passage is dropped.
- `read_document` keeps its slice cap (1500) below the shaper's default
  limit (2000) for plain prose; a slice that is truncated anyway (heavy JSON
  escaping) adds no evidence.
- The evidence store reads each step's observation and adds only fully
  shown spans; steps without an observation add nothing. Dropped spans are
  counted per step: the loop puts an `evidence_dropped_truncated` step
  event (`tool`, `shown`, `dropped`, `reason`) on every step that dropped
  spans, including a result that got no observation (added 2026-11-21;
  `reason` is `truncated` or `no_observation`, added 2026-11-22).
- Evidence is read from a result by its tool's name, so the loop refuses
  to start with a registry where `search_corpus` or `read_document` has
  another output model (added 2026-11-22, from Chad's spot-check of
  `4444938`). A run whose evidence cannot be read must not end `abstained`
  because the policy gave up. A tool swapped in after the loop started
  ends the run `failed:evidence_unreadable` at its first successful result.
- Later elision does not remove evidence: once shown in full, a span stays
  observed after its observation is elided to a summary. Elision is context
  management; the model did see the text.

## Amendment (2026-11-22): what the verifier checks
From Chad's spot-check of `4444938`, with Bogart's wording. Chad checks at
the M3 review that these docs match the code.

**Every claim cited, every citation valid.** The `CitationVerifier` accepts
a `FinalAnswer` only if it has at least one claim, every claim has at least
one citation, and every citation passes, in this order: the `doc_id` is
known (`invalid_citation`), the cited `content_hash`, if given, matches the
stored document (`stale_source`), the quote is exactly the normalized stored
text at `[start, end)` (`invalid_citation`), and the span lies inside text
this run was shown in full (`ungrounded_citation`). No claims, or a claim
without a citation, is `no_citation`.

**Not an entailment check.** The verifier checks citation validity,
freshness and shown-grounding only. It does not check that a quote supports
its claim, that the claims support the answer, or that the answer is right.
An answer can cite real, shown text and still be wrong. So "verified" does
not mean "correct", and a `succeeded` run is not a correct one. Correctness
is graded in M4 against gold answers and evidence, and the failure-mode
classifier (PROJECT_SPEC) labels `answered_when_unanswerable` (a verified
answer on a task with no answer) and `wrong_answer_valid_citations` (valid
citations, wrong answer) separately from citation failures. Reports must
not present the verifier's acceptance rate as accuracy.
