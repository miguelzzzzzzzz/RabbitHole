"""Loop detection: refuse a tool call the run has already made too often.

Two calls are *identical* when they name the same tool and their arguments
normalize to the same canonical JSON (`canonical_args`). Counting is over the
whole run, not just consecutive steps, so an ``A, B, A, B`` cycle is caught
as well as ``A, A, A``. Only calls that actually ran count; a call refused
by a budget or a guardrail does not.

With ``max_identical_calls=n`` an identical call may run ``n`` times. The
next one is not run: the loop records a ``loop_detected`` guardrail
observation so the policy can change course. If the policy repeats that call
again anyway, the run ends as ``failed:loop``. The check reads only the
recorded steps, so it holds no per-run state and survives a JSON round-trip
of the `RunState`.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentic_research.agent.actions import ToolCall
from agentic_research.agent.observations import one_line
from agentic_research.agent.state import StepRecord

LOOP_DETECTED = "loop_detected"
"""Guardrail code recorded on a refused identical call."""

LOOP = "loop"
"""Failure reason once the policy ignores the guardrail (``failed:loop``)."""


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)  # 2.0 and 2 are the same argument; -0.0 becomes 0
        return value
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _normalize(v) for k, v in value.items()}
        # Non-string keys (only possible for args built in code): tag every key
        # with its type so 1 and "1" stay distinct and sorting cannot fail.
        return {f"{type(k).__name__}:{k!r}": _normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    return value


def _opaque(value: object) -> str:
    return f"<{type(value).__name__}:{value!r}>"


def canonical_args(args: dict[str, Any]) -> str:
    """Arguments as canonical JSON, for comparing calls.

    Keys are sorted and kept exactly (non-string keys are tagged with their
    type); string values are NFC-normalized and stripped of surrounding
    whitespace; integral floats become ints; list order is kept. Values JSON
    cannot encode are compared by type and ``repr``. This is a comparison key
    only, never what the tool receives.
    """
    return json.dumps(
        _normalize(args),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_opaque,
    )


def call_key(call: ToolCall) -> tuple[str, str]:
    """``(tool, canonical args)``: equal keys mean identical calls."""
    return call.tool, canonical_args(call.args)


class LoopCheck(StrEnum):
    """What the loop should do with a tool call."""

    OK = "ok"  # run it
    WARN = "warn"  # refuse it with a guardrail observation
    STOP = "stop"  # the guardrail was ignored: end the run as failed:loop


@dataclass(frozen=True)
class LoopGuard:
    """Allows an identical call ``max_identical_calls`` times per run."""

    max_identical_calls: int = 3

    def __post_init__(self) -> None:
        # bool is an int subclass; True is not a sensible threshold.
        if isinstance(self.max_identical_calls, bool) or not isinstance(
            self.max_identical_calls, int
        ):
            raise TypeError("max_identical_calls must be an int")
        if self.max_identical_calls < 1:
            raise ValueError("max_identical_calls must be >= 1")

    def check(self, steps: Sequence[StepRecord], call: ToolCall) -> LoopCheck:
        key = call_key(call)
        ran = 0
        warned = False
        for step in steps:
            action = step.action
            if not isinstance(action, ToolCall) or call_key(action) != key:
                continue
            if step.result is not None:
                ran += 1
            elif step.guardrail is not None and step.guardrail.code == LOOP_DETECTED:
                warned = True
        if ran < self.max_identical_calls:
            return LoopCheck.OK
        return LoopCheck.STOP if warned else LoopCheck.WARN

    def message(self, call: ToolCall) -> str:
        """The guardrail text shown to the policy (fixed wording, no tool output)."""
        return (
            f"not run: this exact call to {one_line(call.tool, 64)} already ran "
            f"{self.max_identical_calls} time(s) in this run; change the arguments, "
            "use another tool, or answer. Repeating it again ends the run."
        )
