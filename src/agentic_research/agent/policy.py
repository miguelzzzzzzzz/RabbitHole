"""The `Policy` protocol: given the run state, choose the next action."""

from __future__ import annotations

import re
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentic_research.agent.actions import Action, FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.state import RunState

_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_reason_code(value: str) -> bool:
    """Whether ``value`` can be a run's failure reason (``[a-z][a-z0-9_]*``, <= 64 chars)."""
    return len(value) <= 64 and _CODE_RE.fullmatch(value) is not None


class PolicyFailure(Exception):  # noqa: N818 - a classified outcome, like ToolFailure
    """Raised by a policy that cannot produce an action, with a reason code.

    The loop ends the run as ``failed:<reason>`` (any other exception from a
    policy is ``failed:policy_error``). Model-backed policies use it for
    provider errors and unparseable replies; repair of malformed replies is
    M5 work. ``reason`` must be a short lowercase code (``[a-z][a-z0-9_]*``,
    at most 64 characters) so it fits `RunState.detail`.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        if not is_reason_code(reason):
            raise ValueError(f"invalid policy failure reason {reason!r}")
        self.reason = reason
        self.message = message
        super().__init__(f"{reason}: {message}" if message else reason)


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
    charged; a bare action costs nothing. To end the run with a specific
    reason, raise `PolicyFailure`.
    """

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp | Decision: ...
