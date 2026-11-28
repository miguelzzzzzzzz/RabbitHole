"""`read_document`: a bounded slice of a stored document's normalized text.

``read_document(doc_id, start, end)`` returns ``text[start:end]`` of the
document the backend stores, the same text every citation offset indexes, so a
quote taken from the result can be cited as ``[start_char, end_char)`` as is.

* At most `MAX_READ_CHARS` characters per call (``end - start``); a longer
  range is ``invalid_args`` before the tool runs.
* ``end`` past the end of the document is clamped; the output reports the
  actual ``end_char`` and the document's length (``doc_length``), so the
  model can page through a document.
* ``start`` at or past the end of the document is
  ``invalid_args/start_out_of_range``; an unknown ``doc_id`` is
  ``invalid_args/unknown_doc_id``; a backend failure is
  ``permanent/backend_error``.

The lookup runs in a worker thread, like ``search_corpus``.
"""

from __future__ import annotations

import asyncio
from typing import Final, Self

from pydantic import Field, model_validator

from agentic_research.backends.base import BackendError, SearchBackend, UnknownDocumentError
from agentic_research.tools.base import (
    ErrorClass,
    Offset,
    RetryPolicy,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolOutput,
)

__all__ = [
    "MAX_READ_CHARS",
    "READ_DOCUMENT",
    "ReadDocumentArgs",
    "ReadDocumentOutput",
    "ReadDocumentTool",
]

READ_DOCUMENT: Final = "read_document"
#: Most characters one call returns. Kept below the observation shaper's
#: default 2000-character limit so that, for typical text, the whole slice
#: reaches the policy (JSON escaping and the other fields add some overhead).
MAX_READ_CHARS: Final = 1500

_DOC_ID_PATTERN: Final = r"^[0-9a-f]{16}$"


class ReadDocumentArgs(ToolArgs):
    """Arguments of ``read_document``: the half-open range ``[start, end)``."""

    doc_id: str = Field(
        pattern=_DOC_ID_PATTERN, description="doc_id of a passage returned by search_corpus."
    )
    start: Offset = Field(ge=0, description="First character offset (inclusive).")
    end: Offset = Field(
        ge=1,
        description=(
            f"End character offset (exclusive); at most {MAX_READ_CHARS} characters "
            "after start. Clamped to the document length."
        ),
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be > start")
        if self.end - self.start > MAX_READ_CHARS:
            raise ValueError(f"at most {MAX_READ_CHARS} characters per call (end - start)")
        return self


class ReadDocumentOutput(ToolOutput):
    """``text == document.text[start_char:end_char]`` of the stored document."""

    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    source: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)
    doc_length: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def _check_slice(self) -> Self:
        if not self.start_char < self.end_char <= self.doc_length:
            raise ValueError("need start_char < end_char <= doc_length")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("text length must equal end_char - start_char")
        return self


class ReadDocumentTool:
    """The ``read_document`` tool over one `SearchBackend`."""

    name = READ_DOCUMENT
    description = (
        "Read part of a stored document by doc_id: returns text[start:end] (at most "
        f"{MAX_READ_CHARS} characters) with its offsets, content_hash and the document "
        "length. Offsets are the ones citations use."
    )
    args_model: type[ToolArgs] = ReadDocumentArgs
    output_model: type[ToolOutput] = ReadDocumentOutput
    retry = RetryPolicy.no_retry()
    idempotent = True  # read-only

    def __init__(self, backend: SearchBackend, *, timeout_s: float = 10.0) -> None:
        self._backend = backend
        self.timeout_s = timeout_s

    async def run(self, args: ReadDocumentArgs, ctx: ToolContext) -> ReadDocumentOutput:
        try:
            document = await asyncio.to_thread(self._backend.document, args.doc_id)
        except UnknownDocumentError:
            raise _failure(
                ErrorClass.INVALID_ARGS,
                "unknown_doc_id",
                f"no document with doc_id {args.doc_id!r} in corpus {self._backend.corpus_id!r}",
            ) from None
        except BackendError as exc:
            raise _failure(ErrorClass.PERMANENT, "backend_error", str(exc)) from None
        length = len(document.text)
        if args.start >= length:
            raise _failure(
                ErrorClass.INVALID_ARGS,
                "start_out_of_range",
                f"start {args.start} is past the end of the document ({length} characters)",
            )
        end = min(args.end, length)
        return ReadDocumentOutput(
            doc_id=document.doc_id,
            source=document.source,
            content_hash=document.content_hash,
            start_char=args.start,
            end_char=end,
            doc_length=length,
            text=document.text[args.start : end],
        )


def _failure(error_class: ErrorClass, reason: str, message: str) -> ToolFailure:
    return ToolFailure(ToolError(error_class=error_class, reason=reason, message=message))
