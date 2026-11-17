"""Build a CitationNeeded format-2 index over a fixture corpus (test helper).

Uses CitationNeeded's own ingest, chunking and ``save_index`` with the offline
hashing embedder, so no model and no network are needed.
"""

from __future__ import annotations

from pathlib import Path

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, IngestionConfig
from rag_engine.ingestion import ingest_path
from rag_engine.retrieval import DenseRetriever, HashingEmbedder, LexicalRetriever, save_index

FIXTURE_CORPUS = Path(__file__).resolve().parent / "fixtures" / "corpus" / "rabbithole-tasks-v1"
FIXTURE_CORPUS_ID = "rabbithole-tasks-v1"


def build_index(corpus: Path, out: Path, corpus_id: str) -> Path:
    report = ingest_path(corpus, IngestionConfig(corpus_id=corpus_id))
    if report.skipped:
        raise AssertionError(f"fixture files skipped: {report.skipped}")
    # Small chunks so each fixture document yields several passages.
    chunks = chunk_documents(
        report.documents, ChunkingConfig(chunk_size=32, chunk_overlap=0, min_chunk_tokens=0)
    )
    dense = DenseRetriever(HashingEmbedder())
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)
    save_index(out, dense, lexical, report.documents)
    return out
