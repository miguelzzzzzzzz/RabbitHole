# ADR-0002: Reuse CitationNeeded's retriever as the `search_corpus` backend

Status: Accepted (2026-10-30). The draft was amended 2026-10-09 for CitationNeeded
index format 2 / searchable contract at `cdd892e`; amended 2026-10-28 to pin
`v1.0.0` (no contract change)

## Context
Research tasks need corpus search. Project 01 provides typed chunks with the
invariant `document.text[start_char:end_char] == chunk.text` over *normalized*
document text, BM25 plus dense hybrid retrieval, and its own retrieval evals.
Chad's M3 interface review at `84bce05` found two MAJORs (no stored document
text / `content_hash` for citation re-check; `doc_id` dependent on ingest
root). Both are fixed at `cdd892e` (Chad signed off; tagged `v0.1.0` at `bf0c83b`).
Citation verification in this project depends on that contract. Retrieval
quality is not the skill this project demonstrates. Portfolio ADR-007 sets
the reuse rules.

## Decision
Depend on CitationNeeded (project 01) at its v1.0.0 tag, pinned by tag and
commit SHA as a git dependency:
`production-rag-engine @ git+https://github.com/miguelzzzzzzzz/CitationNeeded@v1.0.0`.
The pin is final: `v1.0.0` is tagged at `b8015f3`, re-verified 2026-10-29,
interface unchanged from `977f091` (version string only; see the note below). The
repository was renamed, but the distribution name and the `rag_engine` import
were not.

Wrap it in a consumer-side `SearchBackend` protocol
(`search(query, k, filters) -> list[Passage]`, `get_document(doc_id)`) with a
single adapter module, `backends/rag_engine.py`. The adapter:

1. Consumes `search --json` as an object with `schema_version` and results
   under `hits` (not a bare list). Unknown `schema_version` fails clearly.
2. Maps each hit to a `Passage` carrying `doc_id`, `content_hash`, normalized
   `start_char`/`end_char`, `page`/`page_end`, `text`, and `score`.
3. Resolves documents from the index's `documents.jsonl` (normalized text +
   `content_hash`). Offsets are into that text, never into raw-file bytes.
4. Treats a `content_hash` mismatch as a stale source for citation checks.
5. Builds and loads indexes as format 2 via `save_index(..., documents)` /
   `load_index` (which re-verifies chunks and refuses tampered indexes).
6. **Always** passes an explicit `corpus_id` (`rabbithole-hotpotqa` for the
   HotpotQA corpus, `rabbithole-tasks-v1` for the hand-authored fixture
   corpus). Never rely on the folder-name fallback
   (`doc_id = sha256("<corpus_id>:<relpath>")[:16]`).

CI and default evals use BM25-only mode, which needs no model downloads.
Hybrid (fastembed bge-small) is a local configuration and an ablation. Unit
tests use `FixtureSearchBackend`, a tiny keyword backend behind the same
protocol, and a shared contract test suite runs against both backends
(unknown `schema_version`, tampered span, hash-mismatch stale, pinned
`doc_id`).

## Alternatives considered
- **Re-implement BM25 here**: duplicates 01, adds no new skill, and loses the
  verifiable-offset chunk model (`documents.jsonl` + `content_hash`).
- **Call 01 over HTTP (its FastAPI service)**: realistic for microservices,
  but adds a running service to every test and eval, and latency noise to
  agent measurements.
- **Copy 01's retrieval code in**: drift and duplicated maintenance.
- **A vector DB server**: infrastructure without an evaluation need at this
  scale.
- **Pin an untagged SHA**: not needed; `v1.0.0` is tagged at `b8015f3`.
- **Stay on `v0.1.0`**: same consumer contract, but misses upstream M4-M7
  fixes (locked fastembed first loads, thread-safe reranker load).

## Consequences
02 shows composition with a versioned upstream component, and its metrics
isolate orchestration from retrieval, since the backend is held fixed and
recorded in reports. If 01's public API changes, only the adapter changes
(plus the `schema_version` gate and contract tests). Format-1 indexes are
unsupported and must be rebuilt. Retrieval failures show up in this project's
failure taxonomy as `wrong_answer`/`no_citation` with a retrieval-miss
secondary label (gold doc not in any search result), and hash mismatches as
`stale_source`, so they are visible but not conflated with policy errors.

### Note (2026-10-28): `v0.1.0` -> `v1.0.0`, no contract change
Diffed `v0.1.0` (`bf0c83b`) against `977f091`, the commit about to be tagged
`v1.0.0` (no `v1.0.0` tag existed on 2026-10-28; upstream's draft release
notes say it waits on Chad's sign-off of fixes from his review at `900575f`).
Result: **no breaking changes** to anything this ADR relies on.
`models.py`, `documents.py`, ingestion, chunking and `retrieval/index_store.py`
are byte-identical between the two refs. Checked:
- `search --json`: `schema_version` still 1; keys `schema_version`, `query`,
  `mode`, `rerank`, `hits`; hit fields unchanged (`rank`, `score`,
  `retriever`, optional `components`/`ranks`, `doc_id`, `chunk_id`, `source`,
  `content_hash`, `heading_path`, `page`, `page_end`, `start_char`,
  `end_char`, `text`). Same tiny corpus indexed and searched (lexical and
  hybrid) at both refs gave identical JSON.
- Index format version still 2 (`index.json`; `load_index` rejects others);
  `documents.jsonl` fields unchanged (`doc_id`, `corpus_id`, `source`,
  `title`, `format`, `metadata`, `sections`, `text`, `content_hash` =
  sha256 of normalized text); built files byte-identical across refs.
- `doc_id = sha256("<corpus_id>:<relpath>")[:16]` unchanged; `index`/`ingest`
  `--corpus-id` flag unchanged (help text identical).
- Offsets still index the normalized document text.
- `save_index(directory, dense, lexical, documents)` and
  `load_index(directory, config=None, *, tokenizer=tokenize)` signatures
  unchanged.
Additive only: `evaluate` and `answer` CLI commands; a read-only FastAPI
service (`GET /health`, `POST /search` with its own schema_version 1 whose hits
match `search --json`, `POST /answer`; optional `api` extra); retrieval
exports `RETRIEVAL_MODES`, `RetrievalMode`, `compose_retriever`; new
`Settings.llm`/`Settings.service`. RabbitHole uses none of them, and the
HTTP alternative above stays rejected. Adapter changes: none beyond the pin.
The `schema_version == 1` gate, the citation check (slice normalized text from
`documents.jsonl`, `content_hash` mismatch -> `stale_source`) and the
pinned-`doc_id` tests stay as written. The adapter gates on `schema_version`
and index format, never on `rag_engine.__version__`.
Addendum (2026-10-29): `v1.0.0` tagged at `b8015f3`; `977f091..b8015f3` touches only docs, the version string (`0.1.0` -> `1.0.0`) and `tests/test_version.py`, so the pin `@v1.0.0` is final.
