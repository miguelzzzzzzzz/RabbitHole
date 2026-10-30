from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from agentic_research.tools import (
    RETRYABLE,
    ErrorClass,
    RetryPolicy,
    ToolContext,
    ToolError,
    ToolOutput,
    ToolResult,
)


class Echo(ToolOutput):
    text: str


def _error(error_class: ErrorClass) -> ToolError:
    return ToolError(error_class=error_class, reason="test_reason", message="boom")


def test_only_transient_and_timeout_are_retryable() -> None:
    assert frozenset({ErrorClass.TRANSIENT, ErrorClass.TIMEOUT}) == RETRYABLE
    for error_class in ErrorClass:
        assert _error(error_class).retryable is (error_class in RETRYABLE)


@pytest.mark.parametrize("reason", ["", "Has Spaces", "1starts_with_digit", "dash-ed"])
def test_tool_error_reason_must_be_a_snake_case_code(reason: str) -> None:
    with pytest.raises(ValidationError):
        ToolError(error_class=ErrorClass.PERMANENT, reason=reason, message="x")


def test_tool_result_success_serializes_output_to_json_data() -> None:
    result = ToolResult.success("echo", Echo(text="hi"))
    assert result.ok
    assert result.output == {"text": "hi"}
    assert result.error is None


def test_tool_result_failure_carries_the_error() -> None:
    error = _error(ErrorClass.TIMEOUT)
    result = ToolResult.failure("echo", error)
    assert not result.ok
    assert result.error == error
    assert result.output is None


def test_tool_result_rejects_both_or_neither() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ToolResult(tool="echo")
    with pytest.raises(ValidationError, match="exactly one"):
        ToolResult(tool="echo", output={"text": "hi"}, error=_error(ErrorClass.PERMANENT))


def test_tool_result_round_trips_through_json() -> None:
    result = ToolResult.failure("echo", _error(ErrorClass.BAD_OUTPUT))
    assert ToolResult.model_validate_json(result.model_dump_json()) == result


def test_default_retry_policy_retries_only_retryable_classes() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.should_retry(_error(ErrorClass.TRANSIENT), attempt=1)
    assert policy.should_retry(_error(ErrorClass.TIMEOUT), attempt=2)
    assert not policy.should_retry(_error(ErrorClass.TRANSIENT), attempt=3)
    for error_class in (ErrorClass.PERMANENT, ErrorClass.INVALID_ARGS, ErrorClass.BAD_OUTPUT):
        assert not policy.should_retry(_error(error_class), attempt=1)


def test_retry_on_can_narrow_to_one_class() -> None:
    policy = RetryPolicy(retry_on=frozenset({ErrorClass.TRANSIENT}))
    assert policy.should_retry(_error(ErrorClass.TRANSIENT), attempt=1)
    assert not policy.should_retry(_error(ErrorClass.TIMEOUT), attempt=1)


@pytest.mark.parametrize(
    "error_class", [ErrorClass.PERMANENT, ErrorClass.INVALID_ARGS, ErrorClass.BAD_OUTPUT]
)
def test_retry_on_rejects_non_retryable_classes(error_class: ErrorClass) -> None:
    with pytest.raises(ValidationError, match=error_class.value):
        RetryPolicy(retry_on=frozenset({ErrorClass.TRANSIENT, error_class}))


def test_no_retry_policy_never_retries() -> None:
    policy = RetryPolicy.no_retry()
    assert policy.max_attempts == 1
    assert not policy.should_retry(_error(ErrorClass.TRANSIENT), attempt=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"initial_backoff_s": -0.1},
        {"multiplier": 0.5},
        {"jitter": 1.5},
        {"initial_backoff_s": float("nan")},
        {"max_backoff_s": float("inf")},
        {"initial_backoff_s": 2.0, "max_backoff_s": 1.0},
    ],
)
def test_retry_policy_rejects_invalid_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_attempt_numbers_are_one_based() -> None:
    policy = RetryPolicy()
    with pytest.raises(ValueError, match="1-based"):
        policy.should_retry(_error(ErrorClass.TRANSIENT), attempt=0)
    with pytest.raises(ValueError, match="1-based"):
        policy.backoff_s(0, random.Random(0))


class _FixedRandom(random.Random):
    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


def test_backoff_without_jitter_is_exponential_and_capped() -> None:
    policy = RetryPolicy(
        max_attempts=10, initial_backoff_s=0.5, multiplier=2.0, max_backoff_s=3.0, jitter=0.0
    )
    rng = random.Random(0)
    assert [policy.backoff_s(n, rng) for n in range(1, 7)] == [0.5, 1.0, 2.0, 3.0, 3.0, 3.0]


def test_backoff_jitter_bounds_use_the_supplied_rng() -> None:
    policy = RetryPolicy(initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=8.0, jitter=0.5)
    # rng.random() == 0 keeps the full delay; values near 1 remove up to `jitter` of it.
    assert policy.backoff_s(2, _FixedRandom(0.0)) == 2.0
    assert policy.backoff_s(2, _FixedRandom(0.5)) == 1.5
    assert policy.backoff_s(2, _FixedRandom(0.999)) == pytest.approx(1.001)


def test_backoff_is_reproducible_from_a_seed() -> None:
    policy = RetryPolicy()
    first = [policy.backoff_s(n, random.Random(42)) for n in (1, 2, 3)]
    second = [policy.backoff_s(n, random.Random(42)) for n in (1, 2, 3)]
    assert first == second
    for n, delay in zip((1, 2, 3), first, strict=True):
        base = min(0.5 * 2.0 ** (n - 1), 8.0)
        assert base * 0.5 <= delay <= base


def test_backoff_does_not_overflow_for_large_attempt_numbers() -> None:
    policy = RetryPolicy(multiplier=1e300, max_backoff_s=5.0, jitter=0.0)
    assert policy.backoff_s(1000, random.Random(0)) == 5.0


def test_zero_backoff_stays_zero() -> None:
    policy = RetryPolicy(initial_backoff_s=0.0, max_backoff_s=0.0)
    assert policy.backoff_s(5, random.Random(0)) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": "", "step": 0},
        {"run_id": "r", "step": -1},
        {"run_id": "r", "step": 0, "attempt": 0},
    ],
)
def test_tool_context_validates_fields(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToolContext(**kwargs)  # type: ignore[arg-type]


def test_tool_context_defaults_to_first_attempt() -> None:
    assert ToolContext(run_id="run-1", step=0).attempt == 1
