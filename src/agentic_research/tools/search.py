"""`search_corpus`: top-k passages from the run's `SearchBackend`.

The tool returns each passage with everything a citation needs (``doc_id``,
``content_hash``, offsets into the normalized stored text, ``page`` /
``page_end``, ``text``, ``score``). The backend's search is synchronous and may
be CPU-bound (BM25, an embedding for hybrid search), so it runs in a worker
thread and never blocks the event loop. A timed-out call cannot stop that
thread; it finishes in the background and its result is dropped, because the
executor has already classified the attempt as a timeout.

Failures: a whitespace-only query is ``invalid_args/blank_query``; a backend
that cannot serve the request (`BackendError`) is ``permanent/backend_error``.
Anything else a backend raises is a bug and surfaces as the executor's
``permanent/tool_crashed``.

The arguments accept exactly what M1's schema-only probe stub accepted (same
fields, defaults and bounds; ``tests/test_research_tools.py`` checks this).
The probe keeps its own frozen copy of the stub schema so that
``probes/tool_calls.v1`` stays reproducible.
"""

from __future__ import annotations

import asyncio
from typing import Final

from pydantic import Field

from agentic_research.backends.base import (
    BackendError,
    Passage,
    SearchBackend,
    validate_search_args,
)
from agentic_research.tools.base import (
    ErrorClass,
    RetryPolicy,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolOutput,
)

__all__ = [
    "MAX_SEARCH_K",
    "SEARCH_CORPUS",
    "SearchCorpusArgs",
    "SearchCorpusOutput",
    "SearchCorpusTool",
]

SEARCH_CORPUS: Final = "search_corpus"
#: Upper bound on ``k`` (the probe stub's bound).
MAX_SEARCH_K: Final = 50


class SearchCorpusArgs(ToolArgs):
    """Arguments of ``search_corpus``."""

    query: str = Field(min_length=1, description="Natural-language search query.")
    k: int = Field(default=5, ge=1, le=MAX_SEARCH_K, description="Number of passages to return.")
    filters: dict[str, str] | None = Field(
        default=None, description="Optional metadata filters (string keys and values)."
    )


class SearchCorpusOutput(ToolOutput):
    """The passages found, best first (ranks 1..n), from corpus ``corpus_id``."""

    corpus_id: str = Field(min_length=1)
    passages: list[Passage]


class SearchCorpusTool:
    """The ``search_corpus`` tool over one `SearchBackend`."""

    name = SEARCH_CORPUS
    description = (
        "Search the research corpus for relevant passages. Returns up to k passages, "
        "best first, each with doc_id, content_hash, start_char/end_char offsets into "
        "the stored document text, and the passage text. Filters are exact matches on "
        "doc_id or source."
    )
    args_model: type[ToolArgs] = SearchCorpusArgs
    output_model: type[ToolOutput] = SearchCorpusOutput
    retry = RetryPolicy.no_retry()  # a local index fails the same way twice
    idempotent = True  # read-only

    def __init__(self, backend: SearchBackend, *, timeout_s: float = 30.0) -> None:
        self._backend = backend
        self.timeout_s = timeout_s

    async def run(self, args: SearchCorpusArgs, ctx: ToolContext) -> SearchCorpusOutput:
        try:
            validate_search_args(args.query, args.k)
        except ValueError as exc:
            raise ToolFailure(
                ToolError(
                    error_class=ErrorClass.INVALID_ARGS, reason="blank_query", message=str(exc)
                )
            ) from None
        try:
            passages = await asyncio.to_thread(
                self._backend.search, args.query, args.k, args.filters
            )
        except BackendError as exc:
            raise ToolFailure(
                ToolError(
                    error_class=ErrorClass.PERMANENT, reason="backend_error", message=str(exc)
                )
            ) from None
        return SearchCorpusOutput(corpus_id=self._backend.corpus_id, passages=passages)
