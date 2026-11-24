"""The pluggable `FinalAnswer` verifier hook and how a run's trace names it."""

from __future__ import annotations

from typing import ClassVar, Final, Protocol

from agentic_research.agent.actions import FinalAnswer
from agentic_research.agent.state import RunState, Verdict

#: Longest verifier label a trace stores (`RunConfig.verifier`).
MAX_LABEL_CHARS: Final = 200


class Verifier(Protocol):
    """Accepts or rejects a final answer.

    A rejection becomes a repair: the loop records the verdict on the step,
    charges one repair, and asks the policy again. Like a policy, a verifier
    gets a copy of the run state. A verifier class may declare a ``label``
    class attribute; the trace's ``run_config`` event records it (see
    `verifier_label`).
    """

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict: ...


class AcceptAll:
    """The default verifier: every answer is accepted."""

    label: ClassVar[str] = "accept-all"

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        return Verdict.accept()


def verifier_label(verifier: Verifier) -> str:
    """The name a trace records for ``verifier``.

    The ``label`` its class declares (``accept-all``, ``citations``), or
    ``custom:<class qualname>`` when the class declares none (or a non-string
    or empty one), cut to `MAX_LABEL_CHARS`. Read from the class, not the
    instance, so an instance attribute cannot change what the trace says.
    """
    label = getattr(type(verifier), "label", None)
    if not isinstance(label, str) or not label:
        label = f"custom:{type(verifier).__qualname__}"
    return label[:MAX_LABEL_CHARS]
