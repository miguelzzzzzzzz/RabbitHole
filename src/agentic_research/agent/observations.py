"""Deterministic observation shaping (no LLM).

The loop turns each step into at most one `Observation`, which is what a
policy is shown; the full `ToolResult` stays in the step record for traces
and verifiers.

- A successful tool result is rendered as canonical JSON (sorted keys,
  compact separators), an error as ``<class>/<reason>: <message>``, a refused
  call as its guardrail message and a rejected answer as the verifier's
  feedback. The text is cut to ``max_chars`` characters with an explicit
  ``...[truncated N chars]`` marker; ``truncated_chars`` records N.
- A successful ``search_corpus`` result is cut at passage boundaries
  instead: the longest prefix of its passages that fits in ``max_chars``
  together with the omission marker is rendered, the rest are dropped whole,
  and ``shown_passages`` records how many were rendered. No passage is ever
  cut mid-text, so the evidence store can count exactly the passages the
  policy saw (ADR-0004). The marker names the omitted passages by location
  only, best first, so the policy can fetch them with ``read_document``
  even when no passage was shown::

      ...[2 passages omitted; read_document(doc_id, start, end):
      (0123456789abcdef, 0, 2400), (0123456789abcdef, 2401, 2600)]

  (one line in the observation).

  A locator carries no passage text, so it is not evidence. Locators that
  do not fit are counted as ``+M more`` (the marker shrinks to the bare
  ``...[N passages omitted]`` if no locator fits). A passage-cut observation
  is never longer than ``max_chars``. If not even the result with no
  passages plus the bare marker fits (an absurdly long ``corpus_id`` or a
  tiny ``max_chars``), the generic character cut applies with
  ``shown_passages = 0``.
- Only the ``keep_recent`` newest observations keep their text. Older ones
  are elided to a one-line summary (step, tool, outcome, size) that contains
  no tool output.
- Calls refused by a budget, accepted answers and give-ups get no
  observation (the first ends the run, the others have nothing to show). A
  guardrail step always gets one, including the one that ends the run as
  ``failed:loop``, so the record shows what the policy was told.

Same steps in, same observations out: no clock, randomness or model is
involved. Wrapping observations as untrusted data in prompts is M3
guardrail work.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from agentic_research.agent.actions import ToolCall
from agentic_research.agent.state import Observation, StepRecord
from agentic_research.tools.search import SEARCH_CORPUS

TRUNCATION_MARKER = "...[truncated {n} chars]"
PASSAGES_MARKER = "...[{n} {noun} omitted]"
#: How the omission marker introduces the locators of omitted passages.
LOCATOR_HINT = "read_document(doc_id, start, end)"
_NAME_LIMIT = 64


def one_line(text: str, limit: int) -> str:
    """``text`` with all whitespace runs collapsed to one space, cut to ``limit``."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def truncate(text: str, max_chars: int) -> tuple[str, int]:
    """``(text cut to max_chars plus a marker, characters dropped)``."""
    if len(text) <= max_chars:
        return text, 0
    dropped = len(text) - max_chars
    return text[:max_chars] + TRUNCATION_MARKER.format(n=dropped), dropped


def passages_marker(
    omitted: int,
    locators: Sequence[tuple[str, int, int]] = (),
    room: int | None = None,
) -> str | None:
    """The marker announcing ``omitted`` (>= 1) dropped passages.

    ``locators`` are ``(doc_id, start_char, end_char)`` of (some of) the
    omitted passages, best first. With ``room``, the marker lists as many of
    them as fit in ``room`` characters (the rest as ``+M more``), falls back
    to the bare marker, and is ``None`` if not even that fits. Without
    ``room`` every locator is listed.
    """
    noun = "passage" if omitted == 1 else "passages"
    bare = PASSAGES_MARKER.format(n=omitted, noun=noun)
    for listed in range(len(locators), 0, -1):
        items = [f"({d}, {s}, {e})" for d, s, e in locators[:listed]]
        if listed < omitted:
            items.append(f"+{omitted - listed} more")
        marker = f"...[{omitted} {noun} omitted; {LOCATOR_HINT}: {', '.join(items)}]"
        if room is None or len(marker) <= room:
            return marker
    return bare if room is None or len(bare) <= room else None


def _locator(passage: Any) -> tuple[str, int, int] | None:
    """``(doc_id, start_char, end_char)`` of a rendered passage, if it has them."""
    if not isinstance(passage, dict):
        return None
    doc_id, start, end = passage.get("doc_id"), passage.get("start_char"), passage.get("end_char")
    if not isinstance(doc_id, str) or not doc_id:
        return None
    if type(start) is not int or type(end) is not int:
        return None
    return doc_id, start, end


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class ObservationShaper:
    """Builds and ages the policy-facing observations of a run."""

    def __init__(self, *, max_chars: int = 2000, keep_recent: int = 3) -> None:
        for name, value, low in (("max_chars", max_chars, 16), ("keep_recent", keep_recent, 1)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < low:
                raise ValueError(f"{name} must be >= {low}")
        self.max_chars = max_chars
        self.keep_recent = keep_recent

    def observe(self, step: StepRecord) -> Observation | None:
        """The observation for one step, or ``None`` if the policy needs none."""
        action = step.action
        if step.guardrail is not None and isinstance(action, ToolCall):
            g = step.guardrail
            return self._make(
                step,
                "guardrail",
                f"guardrail {g.code}: {g.message}",
                f"guardrail {g.code} on {one_line(action.tool, _NAME_LIMIT)}",
            )
        if step.result is not None:
            result = step.result
            tool = one_line(result.tool, _NAME_LIMIT)
            if result.error is None:
                output = result.output or {}
                passages = output.get("passages")
                if result.tool == SEARCH_CORPUS and isinstance(passages, list):
                    return self._search(step, output, passages)
                body = _canonical(result.output)
                return self._make(
                    step,
                    "tool",
                    f"{result.tool} ok: {body}",
                    f"{tool} ok, {len(body)} chars elided",
                )
            err = result.error
            code = f"{err.error_class.value}/{err.reason}"
            return self._make(
                step, "tool", f"{result.tool} error {code}: {err.message}", f"{tool} error {code}"
            )
        if step.verdict is not None and not step.verdict.accepted:
            v = step.verdict
            return self._make(
                step,
                "verifier",
                f"answer rejected {v.reason}: {v.message}",
                f"answer rejected {v.reason}",
            )
        return None

    def add(self, observations: list[Observation], new: Observation) -> None:
        """Append ``new`` and elide everything older than the ``keep_recent`` newest."""
        observations.append(new)
        full = [i for i, o in enumerate(observations) if not o.elided]
        for i in full[: -self.keep_recent]:
            observations[i] = observations[i].elide()

    def _search(self, step: StepRecord, output: dict[str, Any], passages: list[Any]) -> Observation:
        """A ``search_corpus`` observation cut at passage boundaries."""
        total = len(passages)
        full_body = _canonical(output)
        full_text = f"{SEARCH_CORPUS} ok: {full_body}"
        located = [_locator(p) for p in passages]
        for shown in range(total, -1, -1):
            body = (
                full_body
                if shown == total
                else _canonical({**output, "passages": passages[:shown]})
            )
            text = f"{SEARCH_CORPUS} ok: {body}"
            if shown == total:
                if len(text) > self.max_chars:
                    continue
                marker = ""
            else:
                # Listed locators must be a leading run of the omitted passages,
                # so "+M more" always means "the M after the listed ones".
                locators: list[tuple[str, int, int]] = []
                for loc in located[shown:]:
                    if loc is None:
                        break
                    locators.append(loc)
                fitted = passages_marker(total - shown, locators, self.max_chars - len(text))
                if fitted is None:
                    continue
                marker = fitted
            dropped = len(full_text) - len(text)
            summary = f"{SEARCH_CORPUS} ok, {len(body)} chars elided"
            if shown < total:
                summary += f", {shown} of {total} passages shown"
            return Observation(
                step=step.index,
                source="tool",
                text=text + marker,
                summary=f"step {step.index}: {summary}",
                truncated_chars=dropped,
                shown_passages=shown,
            )
        # Not even the passage-free result and the bare marker fit: character
        # cut, no passage shown.
        cut = self._make(
            step, "tool", full_text, f"{SEARCH_CORPUS} ok, {len(full_body)} chars elided"
        )
        return cut.model_copy(update={"shown_passages": 0})

    def _make(
        self,
        step: StepRecord,
        source: Literal["tool", "guardrail", "verifier"],
        text: str,
        summary: str,
    ) -> Observation:
        shown, dropped = truncate(text, self.max_chars)
        return Observation(
            step=step.index,
            source=source,
            text=shown,
            summary=f"step {step.index}: {summary}",
            truncated_chars=dropped,
        )
