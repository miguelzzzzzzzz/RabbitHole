"""The `Policy` protocol: given the run state, choose the next action."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_research.agent.actions import Action, FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.state import RunState
from agentic_research.trace.schema import ParseFailure, SpanError

_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_reason_code(value: str) -> bool:
    """Whether ``value`` can be a run's failure reason (``[a-z][a-z0-9_]*``, <= 64 chars)."""
    return len(value) <= 64 and _CODE_RE.fullmatch(value) is not None


class ProviderCallInfo(BaseModel):
    """One model call a policy made, reported so the loop can trace it.

    The loop turns each into a ``provider_call`` span under the current step.
    Timing is not part of it: the span covers the policy call (M2 policies
    make at most one provider call per step). A reply that could not become
    an action carries ``parse_failure``; a call that failed carries
    ``error``; never both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tool_format: Literal["native", "json"]
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    finish_reason: str | None = None
    parse_failure: ParseFailure | None = None
    error: SpanError | None = None

    @model_validator(mode="after")
    def _not_both(self) -> Self:
        if self.error is not None and self.parse_failure is not None:
            raise ValueError("a failed provider call has no reply to fail parsing")
        return self


class PolicyFailure(Exception):  # noqa: N818 - a classified outcome, like ToolFailure
    """Raised by a policy that cannot produce an action, with a reason code.

    The loop ends the run as ``failed:<reason>`` (any other exception from a
    policy is ``failed:policy_error``). Model-backed policies use it for
    provider errors and unparseable replies; repair of malformed replies is
    M5 work. ``reason`` must be a short lowercase code (``[a-z][a-z0-9_]*``,
    at most 64 characters) so it fits `RunState.detail`. ``calls`` are the
    provider calls made before failing, so they still reach the trace.
    """

    def __init__(
        self, reason: str, message: str = "", *, calls: Sequence[ProviderCallInfo] = ()
    ) -> None:
        if not is_reason_code(reason):
            raise ValueError(f"invalid policy failure reason {reason!r}")
        self.reason = reason
        self.message = message
        self.calls: tuple[ProviderCallInfo, ...] = tuple(calls)
        super().__init__(f"{reason}: {message}" if message else reason)


class Decision(BaseModel):
    """An action plus what choosing it cost.

    ``tokens`` is the model usage of the call that produced the action (0 for
    policies that call no model); the loop charges it to the token budget.
    ``calls`` describes the provider calls behind it, for the trace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    tokens: Annotated[int, Field(ge=0)] = 0
    calls: tuple[ProviderCallInfo, ...] = ()


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
