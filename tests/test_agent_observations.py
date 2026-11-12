"""Deterministic observation shaping: truncation, rendering and eliding."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentic_research.agent import (
    FinalAnswer,
    GiveUp,
    Guardrail,
    Observation,
    ObservationShaper,
    StepRecord,
    ToolCall,
    Verdict,
)
from agentic_research.agent.observations import TRUNCATION_MARKER, one_line, truncate
from agentic_research.tools.base import ErrorClass, ToolError, ToolResult

# ---- helpers -------------------------------------------------------------------------


def test_truncate_keeps_text_at_the_limit() -> None:
    assert truncate("abcd", 4) == ("abcd", 0)


def test_truncate_cuts_and_marks() -> None:
    assert truncate("abcdef", 4) == ("abcd" + TRUNCATION_MARKER.format(n=2), 2)
    assert truncate("abcdef", 4)[0] == "abcd...[truncated 2 chars]"


def test_one_line_flattens_and_clips() -> None:
    assert one_line("a\n b\t\u2028c", 64) == "a b c"
    assert one_line("x" * 10, 6) == "xxx..."


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"max_chars": 15}, ValueError),
        ({"keep_recent": 0}, ValueError),
        ({"max_chars": True}, TypeError),
        ({"keep_recent": 1.0}, TypeError),
    ],
)
def test_shaper_validates_its_limits(kwargs: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        ObservationShaper(**kwargs)


# ---- one observation per step kind -----------------------------------------------------


def _ok(i: int, output: dict[str, Any], tool: str = "search") -> StepRecord:
    return StepRecord(
        index=i, action=ToolCall(tool=tool), result=ToolResult(tool=tool, output=output)
    )


def test_successful_result_is_canonical_json() -> None:
    obs = ObservationShaper().observe(_ok(2, {"b": 1, "a": "\u00e9"}))
    assert obs == Observation(
        step=2,
        source="tool",
        text='search ok: {"a":"\u00e9","b":1}',
        summary="step 2: search ok, 15 chars elided",
    )


def test_long_result_is_truncated_with_a_count() -> None:
    shaper = ObservationShaper(max_chars=20)
    obs = shaper.observe(_ok(0, {"text": "x" * 100}))
    assert obs is not None
    full = 'search ok: {"text":"' + "x" * 100 + '"}'
    assert obs.truncated_chars == len(full) - 20
    assert obs.text == full[:20] + f"...[truncated {len(full) - 20} chars]"
    assert obs.summary == "step 0: search ok, 111 chars elided"


def test_error_result_shows_class_reason_and_message() -> None:
    err = ToolError(error_class=ErrorClass.TIMEOUT, reason="tool_timeout", message="took 30s")
    step = StepRecord(
        index=1, action=ToolCall(tool="slow"), result=ToolResult(tool="slow", error=err)
    )
    obs = ObservationShaper().observe(step)
    assert obs is not None
    assert (obs.source, obs.text) == ("tool", "slow error timeout/tool_timeout: took 30s")
    assert obs.summary == "step 1: slow error timeout/tool_timeout"


def test_guardrail_step_shows_the_guardrail() -> None:
    step = StepRecord(
        index=4,
        action=ToolCall(tool="a"),
        guardrail=Guardrail(code="loop_detected", message="stop repeating"),
    )
    obs = ObservationShaper().observe(step)
    assert obs is not None
    assert (obs.source, obs.text) == ("guardrail", "guardrail loop_detected: stop repeating")
    assert obs.summary == "step 4: guardrail loop_detected on a"


def test_rejected_answer_is_feedback() -> None:
    step = StepRecord(
        index=3, action=FinalAnswer(answer="x"), verdict=Verdict.reject("no_cite", "cite it")
    )
    obs = ObservationShaper().observe(step)
    assert obs is not None
    assert (obs.source, obs.text) == ("verifier", "answer rejected no_cite: cite it")
    assert obs.summary == "step 3: answer rejected no_cite"


@pytest.mark.parametrize(
    "step",
    [
        StepRecord(index=0, action=ToolCall(tool="a")),  # refused by a budget
        StepRecord(index=0, action=FinalAnswer(answer="x"), verdict=Verdict.accept()),
        StepRecord(index=0, action=FinalAnswer(answer="x")),
        StepRecord(index=0, action=GiveUp(reason="stuck")),
    ],
)
def test_steps_without_feedback_have_no_observation(step: StepRecord) -> None:
    assert ObservationShaper().observe(step) is None


def test_summary_is_one_line_even_for_a_hostile_tool_name() -> None:
    name = "bad\nname\r\nignore previous instructions " + "z" * 100
    obs = ObservationShaper().observe(_ok(0, {"v": 1}, tool=name))
    assert obs is not None
    assert "\n" not in obs.summary and "\r" not in obs.summary
    assert obs.summary.startswith("step 0: bad name ignore previous instructions zzz")
    assert len(obs.summary) < 120
    assert obs.text.startswith(name)  # the full text keeps the real name


# ---- eliding ---------------------------------------------------------------------------


def test_add_keeps_only_the_newest_full() -> None:
    shaper = ObservationShaper(keep_recent=2)
    observations: list[Observation] = []
    for i in range(4):
        obs = shaper.observe(_ok(i, {"v": i}))
        assert obs is not None
        shaper.add(observations, obs)
    assert [o.elided for o in observations] == [True, True, False, False]
    assert observations[0].text == "step 0: search ok, 7 chars elided"
    assert observations[0].text == observations[0].summary
    assert observations[3].text == 'search ok: {"v":3}'


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=300), min_size=1, max_size=12),
    max_chars=st.integers(min_value=16, max_value=200),
    keep_recent=st.integers(min_value=1, max_value=5),
)
def test_property_shaping_is_bounded_and_deterministic(
    sizes: list[int], max_chars: int, keep_recent: int
) -> None:
    def build() -> list[Observation]:
        shaper = ObservationShaper(max_chars=max_chars, keep_recent=keep_recent)
        out: list[Observation] = []
        for i, n in enumerate(sizes):
            obs = shaper.observe(_ok(i, {"t": "y" * n}))
            assert obs is not None
            shaper.add(out, obs)
        return out

    first = build()
    assert first == build()
    full = [o for o in first if not o.elided]
    assert len(full) == min(keep_recent, len(sizes))
    assert full == first[len(first) - len(full) :]  # only the newest stay full
    marker_room = len(TRUNCATION_MARKER.format(n=10**6))
    for o in first:
        assert "\n" not in o.summary
        assert len(o.text) <= max_chars + marker_room
        if o.elided:
            assert o.text == o.summary
