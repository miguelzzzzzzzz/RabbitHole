"""Which traces may feed answer and citation scores (the M4 verifier gate).

A trace's ``run_config`` event records how the run was set up; it is a
configuration record, not an attestation. `require_citations_verified` is
the check the M4 runner applies before it scores a trace's answer or
citations. It accepts a trace only when:

- the run span has exactly one ``run_config`` event, and it is the span's
  first event (zero is a trace written before 2026-11-24; two is an
  ambiguous record);
- that event is a valid `RunConfig` whose ``verifier`` is ``citations``
  (``accept-all`` and ``custom:*`` runs never had their citations checked);
- the step events agree with the run's status: an ``answered`` run has
  exactly one ``answer_accepted`` event, on its last step (traces written
  before 2026-11-25 have none, so they are refused); any other run has none,
  except a ``budget_exhausted`` run whose last step accepted an answer too
  late (``answer_after_deadline``).

`Trace` validation already enforces the first two shapes for traces it
builds; the gate checks them again so it holds for a trace assembled
without validation (``Trace.model_construct``). Refusals carry a
`GateRefusal` code so a runner can count them or report them in flagged
rows that never feed claim tables.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import ValidationError

from agentic_research.trace.schema import (
    ANSWER_ACCEPTED_EVENT,
    ANSWER_AFTER_DEADLINE_EVENT,
    RUN_CONFIG_EVENT,
    RunConfig,
    RunStatus,
    StepSpan,
    Trace,
)


class GateRefusal(StrEnum):
    """Why a trace may not feed answer or citation scores."""

    NO_RUN_CONFIG = "no_run_config"
    DUPLICATE_RUN_CONFIG = "duplicate_run_config"
    RUN_CONFIG_NOT_FIRST = "run_config_not_first"
    INVALID_RUN_CONFIG = "invalid_run_config"
    NOT_CITATIONS_VERIFIED = "not_citations_verified"
    ANSWER_NOT_ACCEPTED = "answer_not_accepted"
    ACCEPTANCE_MISMATCH = "acceptance_mismatch"


class TraceGateError(ValueError):
    """A trace the verifier gate refuses; ``reason`` says why."""

    def __init__(self, reason: GateRefusal, message: str) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason


def require_citations_verified(trace: Trace) -> RunConfig:
    """The trace's `RunConfig` if its answer and citations may be scored.

    Raises `TraceGateError` otherwise (see the module docstring).
    """
    config = _run_config(trace)
    if not config.citations_verified:
        raise TraceGateError(
            GateRefusal.NOT_CITATIONS_VERIFIED,
            f"the run's verifier is {config.verifier!r}, not 'citations'",
        )
    _check_acceptance(trace)
    return config


def _run_config(trace: Trace) -> RunConfig:
    events = trace.run.events
    positions = [i for i, event in enumerate(events) if event.name == RUN_CONFIG_EVENT]
    if not positions:
        raise TraceGateError(
            GateRefusal.NO_RUN_CONFIG, f"the run span has no {RUN_CONFIG_EVENT} event"
        )
    if len(positions) > 1:
        raise TraceGateError(
            GateRefusal.DUPLICATE_RUN_CONFIG,
            f"the run span has {len(positions)} {RUN_CONFIG_EVENT} events",
        )
    if positions[0] != 0:
        raise TraceGateError(
            GateRefusal.RUN_CONFIG_NOT_FIRST,
            f"{RUN_CONFIG_EVENT} is event {positions[0]} on the run span, not the first",
        )
    try:
        return RunConfig.model_validate(events[0].attributes)
    except ValidationError as exc:
        raise TraceGateError(
            GateRefusal.INVALID_RUN_CONFIG,
            f"{exc.error_count()} validation error(s) in the {RUN_CONFIG_EVENT} event",
        ) from None


def _check_acceptance(trace: Trace) -> None:
    steps = sorted((s for s in trace.spans if isinstance(s, StepSpan)), key=lambda s: s.index)
    accepted = [s for s in steps for e in s.events if e.name == ANSWER_ACCEPTED_EVENT]
    status = trace.run.status
    if not accepted:
        if status is RunStatus.ANSWERED:
            raise TraceGateError(
                GateRefusal.ANSWER_NOT_ACCEPTED,
                f"the run is answered but no step has an {ANSWER_ACCEPTED_EVENT} event",
            )
        return
    if len(accepted) > 1 or accepted[0].span_id != steps[-1].span_id:
        where = ", ".join(f"step {s.index}" for s in accepted)
        raise TraceGateError(
            GateRefusal.ACCEPTANCE_MISMATCH,
            f"{ANSWER_ACCEPTED_EVENT} must be one event on the last step, found on {where}",
        )
    late = any(e.name == ANSWER_AFTER_DEADLINE_EVENT for e in accepted[0].events)
    if late and status is not RunStatus.BUDGET_EXHAUSTED:
        raise TraceGateError(
            GateRefusal.ACCEPTANCE_MISMATCH,
            f"the run is {status.value} but its answer came after the deadline "
            f"({ANSWER_AFTER_DEADLINE_EVENT}); that run ends budget_exhausted",
        )
    if not late and status is not RunStatus.ANSWERED:
        raise TraceGateError(
            GateRefusal.ACCEPTANCE_MISMATCH,
            f"the run is {status.value} but its last step accepted an answer in time",
        )
