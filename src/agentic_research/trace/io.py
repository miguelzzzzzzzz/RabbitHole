"""Write and read JSONL trace files.

`write_trace` saves a complete `Trace` atomically. `TraceWriter` streams
one span per line as a run progresses, so a crash leaves every finished span
on disk; reading such a file fails with a clear "incomplete" error rather
than returning a partial trace. `read_trace`/`loads_trace` check the header's
``schema_version`` before anything else, so a file from another schema
version is reported as such instead of as a pile of validation errors.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from pydantic import TypeAdapter, ValidationError

from agentic_research.trace.schema import (
    SCHEMA_VERSION,
    RunSpan,
    Span,
    Trace,
    TraceHeader,
)

_SPAN_ADAPTER: TypeAdapter[Span] = TypeAdapter(Span)


class TraceError(ValueError):
    """A trace file is malformed, inconsistent, or incomplete."""


class TraceSchemaVersionError(TraceError):
    """A trace file declares a schema version this reader does not support."""

    def __init__(self, source: str, found: object) -> None:
        super().__init__(
            f"{source}: trace schema_version {found!r} is not supported "
            f"(this reader understands schema_version {SCHEMA_VERSION})"
        )
        self.found = found


def _first_error(exc: ValidationError) -> str:
    err = exc.errors(include_url=False, include_input=False)[0]
    loc = ".".join(str(part) for part in err["loc"])
    return f"{loc}: {err['msg']}" if loc else str(err["msg"])


def _dump_line(record: TraceHeader | Span) -> str:
    # Pydantic escapes control characters, so one record is always one line.
    return record.model_dump_json() + "\n"


def dumps_trace(trace: Trace) -> str:
    """Serialize ``trace`` to JSONL text (header line, then one span per line)."""
    return _dump_line(trace.header) + "".join(_dump_line(span) for span in trace.spans)


def _check_version(first_line: str, source: str) -> None:
    try:
        raw = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise TraceError(f"{source}:1: header is not valid JSON ({exc.msg})") from exc
    if not isinstance(raw, dict) or raw.get("kind") != "trace":
        raise TraceError(f"{source}:1: first line must be a trace header (kind 'trace')")
    if "schema_version" not in raw:
        raise TraceError(f"{source}:1: trace header has no schema_version")
    version = raw["schema_version"]
    # bool is an int subclass; True must not pass as version 1.
    if type(version) is not int or version != SCHEMA_VERSION:
        raise TraceSchemaVersionError(source, version)


def loads_trace(text: str, source: str = "<string>") -> Trace:
    """Parse JSONL trace text; raises `TraceError` with a line number on failure."""
    # Split on "\n" only: str.splitlines() would also split inside JSON strings
    # that contain U+2028, U+0085 and similar characters.
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines:
        raise TraceError(f"{source}: empty trace file")

    _check_version(lines[0], source)
    try:
        header = TraceHeader.model_validate_json(lines[0])
    except ValidationError as exc:
        raise TraceError(f"{source}:1: invalid trace header: {_first_error(exc)}") from exc

    spans: list[Span] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            raise TraceError(f"{source}:{lineno}: blank line")
        try:
            spans.append(_SPAN_ADAPTER.validate_json(line))
        except ValidationError as exc:
            raise TraceError(f"{source}:{lineno}: invalid span: {_first_error(exc)}") from exc

    if not any(isinstance(span, RunSpan) for span in spans):
        raise TraceError(
            f"{source}: trace is incomplete: no run span (the run did not finish writing)"
        )
    try:
        return Trace(header=header, spans=spans)
    except ValidationError as exc:
        raise TraceError(f"{source}: inconsistent trace: {_first_error(exc)}") from exc


def read_trace(path: str | os.PathLike[str]) -> Trace:
    """Read and validate a JSONL trace file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TraceError(f"{p}: trace file is not valid UTF-8") from exc
    return loads_trace(text, source=str(p))


def write_trace(trace: Trace, path: str | os.PathLike[str]) -> None:
    """Write ``trace`` to ``path`` atomically (temp file in the same directory, then rename)."""
    p = Path(path)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps_trace(trace))
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class TraceWriter:
    """Append spans to a new trace file as they finish.

    The file is created exclusively (an existing trace is never overwritten)
    and each line is flushed as it is written. Span ids must be unique; the
    full tree is validated when the file is read back.
    """

    def __init__(self, path: str | os.PathLike[str], header: TraceHeader) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = self.path.open("x", encoding="utf-8", newline="\n")
        self._ids: set[str] = set()
        self._write(_dump_line(header))

    def _write(self, line: str) -> None:
        if self._fh is None:
            raise TraceError(f"{self.path}: trace writer is closed")
        self._fh.write(line)
        self._fh.flush()

    def write(self, span: Span) -> None:
        if span.span_id in self._ids:
            raise TraceError(f"{self.path}: duplicate span_id {span.span_id!r}")
        self._write(_dump_line(span))
        self._ids.add(span.span_id)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
