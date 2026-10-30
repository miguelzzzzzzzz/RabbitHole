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
