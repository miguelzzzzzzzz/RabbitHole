"""OraclePolicy and LLMPolicy, alone and driving the AgentLoop (scripted provider, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_research.agent import (
    AgentLoop,
    Budgets,
    Decision,
    FinalAnswer,
    GiveUp,
    Guardrail,
    LoopGuard,
    ObservationShaper,
    PolicyFailure,
    RunState,
    Task,
    ToolCall,
    Verdict,
)
from agentic_research.agent.policy import ProviderCallInfo, is_reason_code
from agentic_research.policies import LLMPolicy, OraclePolicy, charged_tokens
from agentic_research.policies.llm import JSON_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from agentic_research.providers import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FunctionCall,
    ProviderError,
    ScriptedError,
    ScriptedFunctionCall,
    ScriptedProvider,
    ScriptedTurn,
    TokenUsage,
    load_script,
)
from agentic_research.providers import ToolCall as WireToolCall
from agentic_research.tools import ToolExecutor, ToolRegistry, ToolResult
from agentic_research.tools.builtin import default_registry
from agentic_research.tools.calculator import CalculatorOutput

FIXTURE = Path(__file__).parent / "fixtures" / "scripts" / "two_step.json"
TASK = Task(question="What is 6*7?")
CALC = ToolCall(tool="calculator", args={"expression": "6*7"})


class RejectOnce:
    """Rejects the first answer, accepts the rest."""

    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, answer: FinalAnswer, state: RunState) -> Verdict:
        self.calls += 1
        if self.calls == 1:
            return Verdict.reject("no_citation", "cite a source")
        return Verdict.accept()


def make_loop(policy: object, *, max_steps: int = 6, **kwargs: object) -> AgentLoop:
    return AgentLoop(
        policy,  # type: ignore[arg-type]
        ToolExecutor(default_registry()),
        Budgets(max_steps=max_steps),
        **kwargs,  # type: ignore[arg-type]
    )


def tool_turn(name: str, arguments: dict[str, object] | str | None) -> ScriptedTurn:
    return ScriptedTurn(tool_calls=(ScriptedFunctionCall(name=name, arguments=arguments),))


def reply(**message: object) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", **message),  # type: ignore[arg-type]
        model="m",
        latency_ms=0.0,
    )


# --- PolicyFailure ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "ok"),
    [("script_exhausted", True), ("a1_b", True), ("", False), ("Bad", False), ("x" * 65, False)],
)
def test_is_reason_code(value: str, ok: bool) -> None:
    assert is_reason_code(value) is ok


def test_policy_failure_validates_its_reason() -> None:
    assert str(PolicyFailure("nope")) == "nope"
    assert str(PolicyFailure("nope", "why")) == "nope: why"
    with pytest.raises(ValueError, match="invalid policy failure reason"):
        PolicyFailure("Not A Code")


async def test_loop_ends_with_the_policy_failure_reason() -> None:
    class Fails:
        async def next_action(self, state: RunState) -> FinalAnswer:
            raise PolicyFailure("my_reason", "detail")

    state = await make_loop(Fails()).run(TASK)
    assert state.terminal_label == "failed:my_reason"
    assert state.steps == []


# --- OraclePolicy -------------------------------------------------------------


async def test_oracle_follows_the_trajectory() -> None:
    oracle = OraclePolicy([CALC, FinalAnswer(answer="42")])
    state = await make_loop(oracle).run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.answer == "42"
    assert [s.action for s in state.steps] == [CALC, FinalAnswer(answer="42")]
    assert state.steps[0].result is not None
    assert state.steps[0].result.output == {"value": 42}
    assert state.counters.tokens == 0


async def test_oracle_action_index_is_the_step_count() -> None:
    oracle = OraclePolicy([CALC, FinalAnswer(answer="42")])
    state = RunState(task=TASK)
    assert await oracle.next_action(state) == CALC
    assert await oracle.next_action(state) == CALC  # no hidden cursor
    state.add_step(CALC, result=ToolResult.success("calculator", CalculatorOutput(value=42)))
    assert await oracle.next_action(state) == FinalAnswer(answer="42")


async def test_oracle_past_its_trajectory_fails_the_run() -> None:
    state = await make_loop(OraclePolicy([CALC])).run(TASK)
    assert state.terminal_label == "failed:trajectory_exhausted"
    assert len(state.steps) == 1


async def test_a_rejected_answer_consumes_the_next_oracle_action() -> None:
    trajectory = [FinalAnswer(answer="42"), FinalAnswer(answer="42 [doc1]")]
    state = await make_loop(OraclePolicy(trajectory), verifier=RejectOnce()).run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.answer == "42 [doc1]"
    assert state.counters.repairs == 1


async def test_oracle_give_up_abstains() -> None:
    state = await make_loop(OraclePolicy([GiveUp(reason="no_evidence")])).run(TASK)
    assert state.terminal_label == "abstained:no_evidence"


async def test_oracle_from_json_and_load(tmp_path: Path) -> None:
    data = json.dumps(
        [
            {"kind": "tool_call", "tool": "calculator", "args": {"expression": "6*7"}},
            {"kind": "final_answer", "answer": "42"},
        ]
    )
    assert OraclePolicy.from_json(data).trajectory == (CALC, FinalAnswer(answer="42"))
    path = tmp_path / "trajectory.json"
    path.write_text(data)
    assert OraclePolicy.load(path).trajectory == (CALC, FinalAnswer(answer="42"))


def test_oracle_rejects_empty_or_non_actions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        OraclePolicy([])
    with pytest.raises(TypeError, match="not an action"):
        OraclePolicy([{"kind": "final_answer", "answer": "x"}])  # type: ignore[list-item]


# --- LLMPolicy: end to end on the scripted provider ---------------------------


async def test_two_step_script_runs_to_success_in_native_format() -> None:
    script = load_script(FIXTURE)
    assert script.question is not None
    provider = ScriptedProvider.from_script(script)
    policy = LLMPolicy(provider, default_registry())
    state = await make_loop(policy).run(Task(question=script.question))
    assert state.terminal_label == "succeeded"
    assert state.answer == "6*7 = 42."
    assert state.counters.tokens == 138 + 167
    assert state.counters.tool_calls == 1
    assert policy.parse_failures.total == 0
    first, second = provider.requests
    assert first.tools == tuple(default_registry().openai_schemas())
    assert [m.role for m in first.messages] == ["system", "user"]
    assert first.messages[0].content == NATIVE_SYSTEM_PROMPT
    assert first.messages[1].content == "What is 6*7?"
    assert [m.role for m in second.messages] == ["system", "user", "assistant", "tool"]
    assistant, tool = second.messages[2:]
    assert assistant.content is None
    (call,) = assistant.tool_calls or ()
    assert (call.id, call.function.name) == ("call_0", "calculator")
    assert call.function.arguments == {"expression": "6*7"}
    assert call.function.arguments_raw == '{"expression":"6*7"}'
    assert tool.tool_call_id == "call_0"
    assert tool.content == 'calculator ok: {"value":42}'


async def test_json_format_end_to_end() -> None:
    provider = ScriptedProvider(
        [
            ScriptedTurn(
                content='```json\n{"action":"tool_call","tool":"calculator",'
                '"arguments":{"expression":"6*7"}}\n```'
            ),
            ScriptedTurn(content='<think>\n</think>{"action":"final_answer","answer":" 42 "}'),
        ]
    )
    policy = LLMPolicy(provider, default_registry(), tool_format="json")
    state = await make_loop(policy).run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.answer == "42"
    first, second = provider.requests
    assert first.tools is None
    system = first.messages[0].content or ""
    assert system.startswith(JSON_SYSTEM_PROMPT)
    assert json.loads(system[len(JSON_SYSTEM_PROMPT) :]) == default_registry().openai_schemas()
    assert [m.role for m in second.messages] == ["system", "user", "assistant", "user"]
    assert json.loads(second.messages[2].content or "") == {
        "action": "tool_call",
        "arguments": {"expression": "6*7"},
        "tool": "calculator",
    }
    assert second.messages[3].content == 'Observation: calculator ok: {"value":42}'


async def test_unknown_tool_in_native_format_becomes_a_tool_error_not_a_dead_run() -> None:
    provider = ScriptedProvider([tool_turn("search_web", {"q": "x"}), ScriptedTurn(content="42")])
    policy = LLMPolicy(provider, default_registry())
    state = await make_loop(policy).run(TASK)
    assert state.terminal_label == "succeeded"
    result = state.steps[0].result
    assert result is not None and result.error is not None
    tool_message = provider.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert "search_web error" in (tool_message.content or "")
    assert policy.parse_failures.total == 0


@pytest.mark.parametrize(
    ("turn", "label"),
    [
        (tool_turn("calculator", '{"expression": '), "failed:invalid_json"),
        (tool_turn("calculator", None), "failed:missing_arguments"),
        (tool_turn("calculator", "[]"), "failed:arguments_not_object"),
        (
            ScriptedTurn(
                tool_calls=(
                    ScriptedFunctionCall(name="calculator", arguments={"expression": "1"}),
                    ScriptedFunctionCall(name="calculator", arguments={"expression": "2"}),
                )
            ),
            "failed:multiple_tool_calls",
        ),
        (ScriptedTurn(content="   "), "failed:empty_reply"),
        (ScriptedTurn(content="<think></think>"), "failed:empty_reply"),
        (ScriptedTurn(), "failed:empty_reply"),
        (ScriptedTurn(content="<think>hmm</think>42"), "failed:think_block_present"),
        (ScriptedTurn(content="<think>42"), "failed:think_block_unclosed"),
        (
            ScriptedTurn(
                reasoning_content="let me think",
                tool_calls=tool_turn("calculator", {"expression": "1"}).tool_calls,
            ),
            "failed:think_block_present",
        ),
    ],
)
async def test_native_unparseable_replies_fail_the_run_and_are_counted(
    turn: ScriptedTurn, label: str
) -> None:
    policy = LLMPolicy(ScriptedProvider([turn]), default_registry())
    state = await make_loop(policy).run(TASK)
    assert state.terminal_label == label
    assert state.steps == []
    reason = label.removeprefix("failed:")
    assert policy.parse_failures.by_reason == {reason: 1}


@pytest.mark.parametrize(
    ("turn", "label"),
    [
        (ScriptedTurn(content="The answer is 42"), "failed:leading_content"),
        (ScriptedTurn(content='{"action": '), "failed:invalid_json"),
        (
            ScriptedTurn(content='{"action":"tool_call","tool":"nope","arguments":{}}'),
            "failed:unknown_tool",
        ),
        (
            ScriptedTurn(content='{"action":"tool_call","tool":"calculator","arguments":{}}'),
            "failed:invalid_arguments",
        ),
        (ScriptedTurn(content='{"action":"final_answer","answer":"  "}'), "failed:empty_answer"),
        (tool_turn("calculator", {"expression": "1"}), "failed:unexpected_tool_calls"),
        (
            ScriptedTurn(reasoning_content="x", content='{"action":"final_answer","answer":"1"}'),
            "failed:think_block_present",
        ),
    ],
)
async def test_json_unparseable_replies_fail_the_run_and_are_counted(
    turn: ScriptedTurn, label: str
) -> None:
    policy = LLMPolicy(ScriptedProvider([turn]), default_registry(), tool_format="json")
    state = await make_loop(policy).run(TASK)
    assert state.terminal_label == label
    assert policy.parse_failures.by_reason == {label.removeprefix("failed:"): 1}


@pytest.mark.parametrize(
    ("turns", "label"),
    [
        (
            [ScriptedTurn(error=ScriptedError(kind="unavailable"))],
            "failed:local_server_unavailable",
        ),
        ([ScriptedTurn(error=ScriptedError(kind="http", status_code=500))], "failed:http_error"),
        ([ScriptedTurn(raw={"choices": []})], "failed:malformed_response"),
        ([tool_turn("calculator", {"expression": "1"})], "failed:script_exhausted"),
    ],
)
async def test_provider_errors_end_the_run_with_their_reason(
    turns: list[ScriptedTurn], label: str
) -> None:
    policy = LLMPolicy(ScriptedProvider(turns), default_registry())
    state = await make_loop(policy).run(TASK)
    assert state.terminal_label == label
    assert policy.parse_failures.total == 0


async def test_provider_reason_that_is_not_a_code_maps_to_provider_error() -> None:
    class Odd:
        name = "odd"

        async def complete(self, request: ChatRequest) -> ChatResponse:
            raise ProviderError("weird", reason="Not-A-Code")

    state = await make_loop(LLMPolicy(Odd(), default_registry())).run(TASK)
    assert state.terminal_label == "failed:provider_error"


async def test_next_action_returns_a_decision_with_charged_tokens() -> None:
    turn = ScriptedTurn(content="42", usage=TokenUsage(prompt_tokens=10, completion_tokens=2))
    policy = LLMPolicy(ScriptedProvider([turn]), default_registry(), model="qwen", max_tokens=64)
    decision = await policy.next_action(RunState(task=TASK))
    info = ProviderCallInfo(
        provider="scripted",
        model="qwen",
        tool_format="native",
        input_tokens=10,
        output_tokens=2,
        finish_reason="stop",
    )
    assert decision == Decision(action=FinalAnswer(answer="42"), tokens=12, calls=(info,))
    request = policy.provider.requests[0]  # type: ignore[attr-defined]
    assert (request.model, request.max_tokens) == ("qwen", 64)


@pytest.mark.parametrize(
    ("usage", "tokens"),
    [
        (None, 0),
        (TokenUsage(), 0),
        (TokenUsage(prompt_tokens=5), 0),
        (TokenUsage(prompt_tokens=5, completion_tokens=3), 8),
        (TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=9), 9),
        (TokenUsage(total_tokens=-4), 0),
        (TokenUsage(prompt_tokens=-9, completion_tokens=3), 0),
    ],
)
def test_charged_tokens(usage: TokenUsage | None, tokens: int) -> None:
    assert charged_tokens(usage) == tokens


async def test_token_budget_counts_scripted_usage() -> None:
    turns = [
        ScriptedTurn(
            tool_calls=(ScriptedFunctionCall(name="calculator", arguments={"expression": "1"}),),
            usage=TokenUsage(total_tokens=50),
        ),
        ScriptedTurn(content="never asked"),
    ]
    provider = ScriptedProvider(turns)
    loop = AgentLoop(
        LLMPolicy(provider, default_registry()),
        ToolExecutor(default_registry()),
        Budgets(max_steps=5, max_tokens=40),
    )
    state = await loop.run(TASK)
    assert state.terminal_label == "budget_exhausted:tokens"
    assert state.counters.tokens == 50
    assert len(provider.requests) == 1


# --- LLMPolicy: message building ----------------------------------------------


@pytest.mark.parametrize(
    ("tool_format", "first", "second", "shown"),
    [
        ("native", "42", "42 [doc1]", "42"),
        (
            "json",
            '{"action":"final_answer","answer":"42"}',
            '{"action":"final_answer","answer":"42 [doc1]"}',
            '{"action":"final_answer","answer":"42"}',
        ),
    ],
)
async def test_verifier_feedback_is_shown_after_the_rejected_answer(
    tool_format: str, first: str, second: str, shown: str
) -> None:
    provider = ScriptedProvider([ScriptedTurn(content=first), ScriptedTurn(content=second)])
    policy = LLMPolicy(provider, default_registry(), tool_format=tool_format)  # type: ignore[arg-type]
    state = await make_loop(policy, verifier=RejectOnce()).run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.answer == "42 [doc1]"
    assistant, feedback = provider.requests[1].messages[2:]
    assert assistant == ChatMessage(role="assistant", content=shown)
    assert feedback == ChatMessage(
        role="user", content="Observation: answer rejected no_citation: cite a source"
    )


async def test_loop_guardrail_is_shown_as_the_tool_result() -> None:
    same = tool_turn("calculator", {"expression": "6*7"})
    provider = ScriptedProvider([same, same, ScriptedTurn(content="42")])
    policy = LLMPolicy(provider, default_registry())
    loop = make_loop(policy, loop_guard=LoopGuard(max_identical_calls=1))
    state = await loop.run(TASK)
    assert state.terminal_label == "succeeded"
    assert state.steps[1].guardrail is not None
    guard_message = provider.requests[2].messages[-1]
    assert guard_message.role == "tool"
    assert guard_message.tool_call_id == "call_1"
    assert (guard_message.content or "").startswith("guardrail loop_detected:")


async def test_old_observations_are_sent_as_their_summaries() -> None:
    turns = [tool_turn("calculator", {"expression": f"{i}+1"}) for i in range(3)]
    provider = ScriptedProvider([*turns, ScriptedTurn(content="done")])
    policy = LLMPolicy(provider, default_registry())
    state = await make_loop(policy, shaper=ObservationShaper(keep_recent=1)).run(TASK)
    assert state.terminal_label == "succeeded"
    tool_texts = [m.content for m in provider.requests[3].messages if m.role == "tool"]
    assert tool_texts == [
        state.observations[0].summary,
        state.observations[1].summary,
        'calculator ok: {"value":3}',
    ]
    assert state.observations[0].elided and state.observations[1].elided


def test_steps_without_observations_and_give_up_steps() -> None:
    state = RunState(task=TASK)
    state.add_step(CALC)  # e.g. refused by a budget: no result, no observation
    state.add_step(FinalAnswer(answer="42"), verdict=Verdict.accept())
    state.add_step(GiveUp(reason="stuck"))
    native = LLMPolicy(ScriptedProvider([]), default_registry()).build_messages(state)
    assert [m.role for m in native] == ["system", "user", "assistant", "tool", "assistant"]
    assert native[3].content == "(no result)"
    assert native[4].content == "42"
    json_policy = LLMPolicy(ScriptedProvider([]), default_registry(), tool_format="json")
    as_json = json_policy.build_messages(state)
    assert [m.role for m in as_json] == ["system", "user", "assistant", "assistant"]


def test_guardrail_step_in_json_format() -> None:
    state = RunState(task=TASK)
    state.add_step(CALC, guardrail=Guardrail(code="loop_detected", message="stop"))
    shaper = ObservationShaper()
    observation = shaper.observe(state.steps[0])
    assert observation is not None
    state.observations.append(observation)
    policy = LLMPolicy(ScriptedProvider([]), default_registry(), tool_format="json")
    messages = policy.build_messages(state)
    assert messages[-1] == ChatMessage(
        role="user", content="Observation: guardrail loop_detected: stop"
    )


def test_empty_registry_sends_no_tools_and_custom_prompt_is_used() -> None:
    policy = LLMPolicy(ScriptedProvider([]), ToolRegistry(), system_prompt="Be brief.")
    request = policy.build_request(RunState(task=TASK))
    assert request.tools is None
    assert request.messages[0].content == "Be brief."
    assert (request.temperature, request.seed, request.enable_thinking) == (0.0, 0, False)


def test_llm_policy_argument_validation() -> None:
    with pytest.raises(ValueError, match="tool format"):
        LLMPolicy(ScriptedProvider([]), default_registry(), tool_format="xml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_tokens"):
        LLMPolicy(ScriptedProvider([]), default_registry(), max_tokens=0)


def test_parse_reply_detail_is_capped() -> None:
    policy = LLMPolicy(ScriptedProvider([]), default_registry())
    long_name = "x" * 3000
    function = FunctionCall(name=long_name, parse_error="invalid_json: x")
    call = WireToolCall(id="c", function=function)
    with pytest.raises(PolicyFailure) as info:
        policy.parse_reply(reply(tool_calls=(call,)))
    assert info.value.reason == "invalid_json"
    assert len(info.value.message) == 2000


def test_argument_parse_error_that_is_not_a_code_maps_to_invalid_arguments() -> None:
    policy = LLMPolicy(ScriptedProvider([]), default_registry())
    function = FunctionCall(name="calculator", parse_error="Weird Error")
    with pytest.raises(PolicyFailure) as info:
        policy.parse_reply(reply(tool_calls=(WireToolCall(id="c", function=function),)))
    assert info.value.reason == "invalid_arguments"
    assert policy.parse_failures.by_reason == {"invalid_arguments": 1}
