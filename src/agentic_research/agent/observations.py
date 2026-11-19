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
  instead: the longest prefix of its passages whose JSON fits in
  ``max_chars`` is rendered, the rest are dropped whole and announced with
  ``...[N passages omitted]``, and ``shown_passages`` records how many were
  rendered. No passage is ever cut mid-text, so the evidence store can count
  exactly the passages the policy saw (ADR-0004). If not even the result
  with no passages fits (an absurdly long ``corpus_id``), the generic
  character cut applies with ``shown_passages = 0``.
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
from typing import Any, Literal

from agentic_research.agent.actions import ToolCall
from agentic_research.agent.state import Observation, StepRecord
from agentic_research.tools.search import SEARCH_CORPUS

TRUNCATION_MARKER = "...[truncated {n} chars]"
PASSAGES_MARKER = "...[{n} {noun} omitted]"
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


def passages_marker(omitted: int) -> str:
    """The marker announcing ``omitted`` (>= 1) dropped passages."""
    return PASSAGES_MARKER.format(n=omitted, noun="passage" if omitted == 1 else "passages")


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
        for shown in range(total, -1, -1):
            body = (
                full_body
                if shown == total
                else _canonical({**output, "passages": passages[:shown]})
            )
            text = f"{SEARCH_CORPUS} ok: {body}"
            if len(text) > self.max_chars:
                continue
            dropped = len(full_text) - len(text)
            summary = f"{SEARCH_CORPUS} ok, {len(body)} chars elided"
            if shown < total:
                summary += f", {shown} of {total} passages shown"
                text += passages_marker(total - shown)
            return Observation(
                step=step.index,
                source="tool",
                text=text,
                summary=f"step {step.index}: {summary}",
                truncated_chars=dropped,
                shown_passages=shown,
            )
        # Not even the passage-free result fits: character cut, no passage shown.
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
