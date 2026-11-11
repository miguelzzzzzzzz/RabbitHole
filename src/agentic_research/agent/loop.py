"""The budgeted agent loop (ADR-0001, ADR-0006).

One iteration is: budget check, policy (bounded by the step timeout and the
run deadline), then either the verifier hook (`FinalAnswer`), abstention
(`GiveUp`) or the tool executor (`ToolCall`). The loop owns every state
transition and never raises for a policy, verifier or tool failure: each run
ends in a terminal `RunState`.

Budget rules:

- Before each policy call, `Budgets.exhausted` is checked; the first
  exhausted budget ends the run as ``budget_exhausted:<kind>``.
- A `ToolCall` is checked again (now including the per-tool limit and any
  tokens or time the policy just used) before it runs. A refused call is
  recorded as a step without a result.
- ``max_tokens`` is soft: usage is charged after the policy returns, so the
  counter can pass the limit. A `FinalAnswer` or `GiveUp` from that call is
  still honoured (it costs nothing more); a `ToolCall` is refused.
- The run deadline is passed to the executor as an absolute time on the
  executor's clock, so tool timeouts are shortened to it and a retry whose
  backoff would cross it is skipped. Attempt durations and backoff sleeps
  are charged to ``elapsed_s`` (plus ``tool_attempts``/``backoff_s``).
- A rejected answer costs one repair; the run ends as
  ``budget_exhausted:repairs`` once repairs exceed ``max_repairs``.

Time: budgets and tool deadlines use the executor's clock, while policy and
tool timeouts run on the event loop's clock via ``asyncio.timeout``. In
production both are monotonic seconds; tests inject a fake executor clock.

Loop detection, observation shaping and tracing are not wired in yet.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from agentic_research.agent.actions import FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.policy import Decision, Policy
from agentic_research.agent.state import (
    BudgetKind,
    Budgets,
    RunState,
    RunStatus,
    Task,
    Verdict,
)
from agentic_research.agent.verifier import AcceptAll, Verifier
from agentic_research.tools.base import ToolContext, ToolResult
from agentic_research.tools.executor import ToolExecutor

POLICY_TIMEOUT = "policy_timeout"
POLICY_ERROR = "policy_error"
INVALID_ACTION = "invalid_action"
VERIFIER_ERROR = "verifier_error"


@dataclass(frozen=True)
class _Stop:
    status: RunStatus
    detail: str | BudgetKind


class AgentLoop:
    """Runs one task to a terminal state under `Budgets`.

    ``max_steps`` is required so that every run terminates whatever the
    policy does. ``step_timeout_s`` bounds each policy call; the deadline
    bounds it too when it is closer.
    """

    def __init__(
        self,
        policy: Policy,
        executor: ToolExecutor,
        budgets: Budgets,
        *,
        verifier: Verifier | None = None,
        step_timeout_s: float | None = None,
        run_id: str = "run",
    ) -> None:
        if budgets.max_steps is None:
            raise ValueError("AgentLoop needs budgets.max_steps so every run terminates")
        valid_timeout = step_timeout_s is None or (
            math.isfinite(step_timeout_s) and step_timeout_s > 0
        )
        if not valid_timeout:
            raise ValueError("step_timeout_s must be a finite number > 0")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self._policy = policy
        self._executor = executor
        self._budgets = budgets
        self._verifier: Verifier = verifier if verifier is not None else AcceptAll()
        self._step_timeout_s = step_timeout_s
        self._run_id = run_id

    async def run(self, task: Task) -> RunState:
        state = RunState(task=task)
        clock = self._executor.clock
        start = clock()
        deadline = None if self._budgets.deadline_s is None else start + self._budgets.deadline_s

        def tick() -> None:
            state.counters.elapsed_s = max(0.0, clock() - start)

        while not state.terminal:
            tick()
            kind = self._budgets.exhausted(state.counters)
            if kind is not None:
                state.finish(RunStatus.BUDGET_EXHAUSTED, detail=kind)
                break
            decision = await self._decide(state, deadline)
            tick()
            if isinstance(decision, _Stop):
                state.finish(decision.status, detail=decision.detail)
                break
            state.counters.tokens += decision.tokens
            action = decision.action
            if isinstance(action, GiveUp):
                self._record(state, action)
                state.finish(RunStatus.ABSTAINED, detail=action.reason)
            elif isinstance(action, FinalAnswer):
                await self._answer(state, action)
            else:
                await self._call_tool(state, action, deadline)
            tick()
        return state

    async def _decide(self, state: RunState, deadline: float | None) -> Decision | _Stop:
        limit = self._step_timeout_s
        deadline_binds = False
        if deadline is not None:
            remaining = deadline - self._executor.clock()
            if remaining <= 0:
                return _Stop(RunStatus.BUDGET_EXHAUSTED, BudgetKind.DEADLINE)
            if limit is None or remaining < limit:
                limit, deadline_binds = remaining, True
        try:
            async with asyncio.timeout(limit) as scope:
                raw = await self._policy.next_action(state.model_copy(deep=True))
        except TimeoutError:
            if not scope.expired():  # the policy raised its own TimeoutError
                return _Stop(RunStatus.FAILED, POLICY_ERROR)
            if deadline_binds:
                return _Stop(RunStatus.BUDGET_EXHAUSTED, BudgetKind.DEADLINE)
            return _Stop(RunStatus.FAILED, POLICY_TIMEOUT)
        except Exception:  # a broken policy must not crash the loop
            return _Stop(RunStatus.FAILED, POLICY_ERROR)
        if isinstance(raw, Decision):
            return raw
        if isinstance(raw, ToolCall | FinalAnswer | GiveUp):
            return Decision(action=raw)
        return _Stop(RunStatus.FAILED, INVALID_ACTION)

    async def _answer(self, state: RunState, action: FinalAnswer) -> None:
        try:
            verdict = await self._verifier.verify(action, state.model_copy(deep=True))
        except Exception:
            self._record(state, action)
            state.finish(RunStatus.FAILED, detail=VERIFIER_ERROR)
            return
        if not isinstance(verdict, Verdict):
            self._record(state, action)
            state.finish(RunStatus.FAILED, detail=VERIFIER_ERROR)
            return
        self._record(state, action, verdict=verdict)
        if verdict.accepted:
            state.finish(RunStatus.SUCCEEDED, answer=action.answer)
        else:
            state.counters.repairs += 1

    async def _call_tool(self, state: RunState, action: ToolCall, deadline: float | None) -> None:
        kind = self._budgets.exhausted(state.counters, tool=action.tool)
        if kind is not None:
            self._record(state, action)
            state.finish(RunStatus.BUDGET_EXHAUSTED, detail=kind)
            return
        ctx = ToolContext(run_id=self._run_id, step=len(state.steps))
        execution = await self._executor.execute(action.tool, action.args, ctx, deadline=deadline)
        counters = state.counters
        counters.count_tool_call(action.tool)
        counters.tool_attempts += len(execution.attempts)
        counters.backoff_s += execution.total_backoff_s
        self._record(state, action, result=execution.result)

    @staticmethod
    def _record(
        state: RunState,
        action: ToolCall | FinalAnswer | GiveUp,
        *,
        result: ToolResult | None = None,
        verdict: Verdict | None = None,
    ) -> None:
        state.add_step(action, result=result, verdict=verdict)
        state.counters.steps += 1
