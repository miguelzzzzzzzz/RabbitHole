"""The `Policy` protocol: given the run state, choose the next action."""

from __future__ import annotations

from typing import Protocol

from agentic_research.agent.actions import FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.state import RunState


class Policy(Protocol):
    """Chooses the next action from the current `RunState`.

    A policy must not mutate the state: the loop owns every transition
    (counters, step records, status). It is async so the loop can bound it
    with ``asyncio.timeout``.
    """

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp: ...
