"""`OraclePolicy`: follows a given trajectory of actions.

The oracle baseline (M4) replays a gold trajectory; in M2 it drives the loop
with known actions. The action for a call is chosen by how many steps the
run has recorded (``len(state.steps)``), not by an internal cursor, so the
policy holds no run state and the same trajectory gives the same run every
time. Every loop step records exactly one step (a verifier rejection or a
loop-guard refusal included), so action ``i`` answers step ``i``.

A run that needs more actions than the trajectory has ends as
``failed:trajectory_exhausted``. The oracle calls no model and costs no
tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from agentic_research.agent.actions import Action, FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.policy import PolicyFailure
from agentic_research.agent.state import RunState

TRAJECTORY_EXHAUSTED: Final = "trajectory_exhausted"
_TRAJECTORY: Final = TypeAdapter(list[Action])


class OraclePolicy:
    """Returns ``trajectory[len(state.steps)]``."""

    def __init__(self, trajectory: Sequence[ToolCall | FinalAnswer | GiveUp]) -> None:
        if not trajectory:
            raise ValueError("an oracle needs at least one action")
        for action in trajectory:
            if not isinstance(action, ToolCall | FinalAnswer | GiveUp):
                raise TypeError(f"not an action: {type(action).__name__}")
        self.trajectory: tuple[ToolCall | FinalAnswer | GiveUp, ...] = tuple(trajectory)

    @classmethod
    def from_json(cls, data: str | bytes) -> OraclePolicy:
        """From a JSON list of actions (each tagged with ``kind``)."""
        return cls(_TRAJECTORY.validate_json(data))

    @classmethod
    def load(cls, path: str | Path) -> OraclePolicy:
        return cls.from_json(Path(path).read_bytes())

    async def next_action(self, state: RunState) -> ToolCall | FinalAnswer | GiveUp:
        index = len(state.steps)
        if index >= len(self.trajectory):
            raise PolicyFailure(
                TRAJECTORY_EXHAUSTED,
                f"step {index} has no action ({len(self.trajectory)} in the trajectory)",
            )
        return self.trajectory[index]
