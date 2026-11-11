"""The pluggable `FinalAnswer` verifier hook (the citation verifier arrives in M3)."""

from __future__ import annotations

from typing import Protocol

from agentic_research.agent.actions import FinalAnswer
from agentic_research.agent.state import RunState, Verdict


class Verifier(Protocol):
    """Accepts or rejects a final answer.

    A rejection becomes a repair: the loop records the verdict on the step,
    charges one repair, and asks the policy again. Like a policy, a verifier
    gets a copy of the run state.
    """

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict: ...


class AcceptAll:
    """The default verifier: every answer is accepted."""

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        return Verdict.accept()
