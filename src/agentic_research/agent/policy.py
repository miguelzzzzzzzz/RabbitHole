"""The `Policy` protocol: given the run state, choose the next action."""

from __future__ import annotations

from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentic_research.agent.actions import Action, FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.state import RunState


class Decision(BaseModel):
    """An action plus what choosing it cost.

    ``tokens`` is the model usage of the call that produced the action (0 for
    policies that call no model); the loop charges it to the token budget.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    tokens: Annotated[int, Field(ge=0)] = 0


class Policy(Protocol):
    """Chooses the next action from the current `RunState`.

    The loop passes a copy of its state and owns every transition (counters,
    step records, status), so a policy cannot change the run by mutating what
    it is given. It is async so the loop can bound it with ``asyncio.timeout``.
    A policy that calls a model returns a `Decision` so its token usage is
    charged; a bare action costs nothing.
    """

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp | Decision: ...
