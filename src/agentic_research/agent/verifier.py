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
    gets a copy of the run state. The trace's ``run_config`` event names the
    verifier (see `verifier_label`); only the built-in verifiers get a
    reserved label, so any other class is recorded as ``custom:...``.
    """

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict: ...


class AcceptAll:
    """The default verifier: every answer is accepted."""

    label: ClassVar[str] = "accept-all"

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        return Verdict.accept()


def verifier_label(verifier: Verifier) -> str:
    """The name a trace records for ``verifier``.

    The labels ``accept-all`` and ``citations`` are reserved: a verifier gets
    one only when its exact class (``type(verifier) is ...``, so not a
    subclass, and not an object whose ``__class__`` claims to be one) is
    `AcceptAll` or `agentic_research.citations.CitationVerifier`. Every other
    verifier is ``custom:<module>.<qualname>`` of its class, whatever
    ``label`` it declares, cut to `MAX_LABEL_CHARS`. A subclass of
    `CitationVerifier` can override ``verify``, so it is not the citation
    verifier as far as a grader is concerned (Chad's 2026-11-25 spot-check).
    The label still records configuration, not an attestation: a trace file
    can be edited.
    """
    # Imported here: `agentic_research.citations` imports `agentic_research.agent`.
    from agentic_research.citations import CitationVerifier

    cls = type(verifier)
    for reserved in (AcceptAll, CitationVerifier):
        if cls is reserved:
            return reserved.label
    return f"custom:{cls.__module__}.{cls.__qualname__}"[:MAX_LABEL_CHARS]
