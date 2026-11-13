"""`LLMPolicy` skeleton: ask a `ChatProvider` for the next action.

Each call builds the conversation from the run state, makes one provider
call and turns the reply into one action:

- Messages: the system prompt, the task question, then for every recorded
  step the action as an assistant message followed by its shaped
  observation (`RunState.observations`; old ones are already one-line
  summaries). In ``native`` format a tool step is an assistant ``tool_calls``
  message plus a ``tool`` message with id ``call_<step>``; in ``json`` format
  the action is the JSON envelope and the observation a user message.
  Guardrail and verifier feedback are sent the same way.
- ``native`` replies: a leaked think block (non-empty ``reasoning_content``
  or a non-empty/unclosed leading ``<think>``) is rejected; exactly one tool
  call becomes a `ToolCall` (its arguments must be a JSON object; the tool
  name and argument schema are left to the registry, so the model sees a
  structured tool error rather than a dead run); no tool call makes the
  content, after one empty leading think block, the final answer.
- ``json`` replies go through `parse_toolcall_json` with the registry's
  argument models; native ``tool_calls`` in this mode are rejected.
- Token usage is ``usage.total_tokens``, else prompt + completion tokens,
  else 0 when the server reported none (the soft token budget then
  undercounts; the provider-call trace span records what was reported).

A reply that cannot become an action increments `parse_failures` and raises
`PolicyFailure` with the parse reason code, so the run ends as
``failed:<reason>``; provider errors end it as ``failed:<provider reason>``.
Repairing malformed replies, recording/replay, and ``GiveUp`` from a model
are M5 work. No prompt here has been tuned against a model.
"""

from __future__ import annotations

import json
from typing import Final, Literal, NoReturn

from pydantic import BaseModel

from agentic_research.agent.actions import FinalAnswer, GiveUp, ToolCall
from agentic_research.agent.policy import Decision, PolicyFailure, is_reason_code
from agentic_research.agent.state import Observation, RunState
from agentic_research.providers.errors import ProviderError
from agentic_research.providers.toolcall_json import (
    REASON_EMPTY_REPLY,
    ParsedToolCall,
    ParseFailureCounter,
    leaked_think_reason,
    parse_toolcall_json,
    strip_empty_think,
)
from agentic_research.providers.types import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    FunctionCall,
    TokenUsage,
)
from agentic_research.providers.types import ToolCall as WireToolCall
from agentic_research.tools.registry import ToolRegistry
from agentic_research.trace.schema import MAX_MESSAGE_CHARS, ParseFailure

ToolFormat = Literal["native", "json"]

MULTIPLE_TOOL_CALLS: Final = "multiple_tool_calls"
UNEXPECTED_TOOL_CALLS: Final = "unexpected_tool_calls"
EMPTY_ANSWER: Final = "empty_answer"
PROVIDER_ERROR: Final = "provider_error"
INVALID_ARGUMENTS: Final = "invalid_arguments"

NATIVE_SYSTEM_PROMPT: Final = (
    "You are a research agent. Answer the user's question. Use the provided "
    "tools when they help, one tool call at a time, and read each tool result "
    "before deciding what to do next. When you can answer, reply with the "
    "answer as plain text and no tool call."
)

JSON_SYSTEM_PROMPT: Final = (
    "You are a research agent. Answer the user's question. Use the tools "
    "below when they help, one tool call at a time, and read each "
    "observation before deciding what to do next.\n"
    "Reply with exactly one JSON object, optionally inside a single ```json "
    "fence, and nothing else. Either\n"
    '{"action":"tool_call","tool":"<name>","arguments":{...}}\n'
    "or, when you can answer,\n"
    '{"action":"final_answer","answer":"..."}\n'
    "Tools (OpenAI function schemas):\n"
)

OBSERVATION_PREFIX: Final = "Observation: "


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def charged_tokens(usage: TokenUsage | None) -> int:
    """Tokens to charge for one reply (0 when the server reported none)."""
    if usage is None:
        return 0
    if usage.total_tokens is not None:
        return max(0, usage.total_tokens)
    if usage.prompt_tokens is not None and usage.completion_tokens is not None:
        return max(0, usage.prompt_tokens + usage.completion_tokens)
    return 0


class LLMPolicy:
    """A `Policy` backed by a chat provider (one call per step)."""

    def __init__(
        self,
        provider: ChatProvider,
        registry: ToolRegistry,
        *,
        tool_format: ToolFormat = "native",
        model: str | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if tool_format not in ("native", "json"):
            raise ValueError(f"unknown tool format {tool_format!r}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.provider = provider
        self.registry = registry
        self.tool_format: ToolFormat = tool_format
        self.model = model
        self.max_tokens = max_tokens
        self._schemas = tuple(registry.openai_schemas())
        if system_prompt is None:
            system_prompt = (
                NATIVE_SYSTEM_PROMPT
                if tool_format == "native"
                else JSON_SYSTEM_PROMPT + _compact(list(self._schemas))
            )
        self.system_prompt = system_prompt
        self.parse_failures = ParseFailureCounter()

    def build_request(self, state: RunState) -> ChatRequest:
        """The chat request for the next action (pure; no provider call)."""
        return ChatRequest(
            messages=self.build_messages(state),
            tools=self._schemas if self.tool_format == "native" and self._schemas else None,
            model=self.model,
            max_tokens=self.max_tokens,
        )

    def build_messages(self, state: RunState) -> tuple[ChatMessage, ...]:
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=state.task.question),
        ]
        observed = {o.step: o for o in state.observations}
        for step in state.steps:
            messages.extend(self._step_messages(step.index, step.action, observed.get(step.index)))
        return tuple(messages)

    def _step_messages(
        self, index: int, action: ToolCall | FinalAnswer | GiveUp, observation: Observation | None
    ) -> list[ChatMessage]:
        if isinstance(action, GiveUp):  # ends the run; never followed by another call
            return []
        if self.tool_format == "native" and isinstance(action, ToolCall):
            call_id = f"call_{index}"
            call = WireToolCall(
                id=call_id,
                function=FunctionCall(
                    name=action.tool, arguments=action.args, arguments_raw=_compact(action.args)
                ),
            )
            text = observation.text if observation is not None else "(no result)"
            return [
                ChatMessage(role="assistant", tool_calls=(call,)),
                ChatMessage(role="tool", tool_call_id=call_id, content=text),
            ]
        if self.tool_format == "native":
            assert isinstance(action, FinalAnswer)
            content = action.answer
        elif isinstance(action, ToolCall):
            envelope = {"action": "tool_call", "tool": action.tool, "arguments": action.args}
            content = _compact(envelope)
        else:
            content = _compact({"action": "final_answer", "answer": action.answer})
        out = [ChatMessage(role="assistant", content=content)]
        if observation is not None:
            out.append(ChatMessage(role="user", content=OBSERVATION_PREFIX + observation.text))
        return out

    async def next_action(self, state: RunState) -> Decision:
        request = self.build_request(state)
        try:
            response = await self.provider.complete(request)
        except ProviderError as exc:
            reason = exc.reason if is_reason_code(exc.reason) else PROVIDER_ERROR
            raise PolicyFailure(reason, exc.message) from exc
        action = self.parse_reply(response)
        return Decision(action=action, tokens=charged_tokens(response.usage))

    def parse_reply(self, response: ChatResponse) -> ToolCall | FinalAnswer:
        """One action from a reply, or `PolicyFailure` (counted in `parse_failures`)."""
        message = response.message
        think = leaked_think_reason(message)
        if think is not None:
            self._fail(think, "reasoning leaked into the reply")
        if self.tool_format == "json":
            return self._parse_json(message)
        calls = message.tool_calls or ()
        if len(calls) > 1:
            self._fail(MULTIPLE_TOOL_CALLS, f"{len(calls)} tool calls; one per step")
        if calls:
            fn = calls[0].function
            if fn.arguments is None or fn.parse_error is not None:
                detail = fn.parse_error or "missing_arguments"
                reason = detail.split(":", 1)[0]
                if not is_reason_code(reason):
                    reason = INVALID_ARGUMENTS
                self._fail(reason, f"{fn.name}: {detail}")
            assert fn.arguments is not None
            return ToolCall(tool=fn.name, args=fn.arguments)
        answer = strip_empty_think(message.content or "").strip()
        if not answer:
            self._fail(REASON_EMPTY_REPLY, "no tool call and no content")
        return FinalAnswer(answer=answer)

    def _parse_json(self, message: ChatMessage) -> ToolCall | FinalAnswer:
        if message.tool_calls:
            self._fail(UNEXPECTED_TOOL_CALLS, "native tool_calls in json tool format")
        models: dict[str, type[BaseModel]] = {t.name: t.args_model for t in self.registry}
        result = parse_toolcall_json(message.content or "", models, failures=self.parse_failures)
        if result.failure is not None:
            raise PolicyFailure(result.failure.reason, result.failure.detail)
        parsed = result.action
        if isinstance(parsed, ParsedToolCall):
            return ToolCall(tool=parsed.tool, args=parsed.arguments)
        assert parsed is not None
        if not parsed.answer.strip():
            self._fail(EMPTY_ANSWER, "final_answer with an empty answer")
        return FinalAnswer(answer=parsed.answer.strip())

    def _fail(self, reason: str, detail: str) -> NoReturn:
        failure = ParseFailure(reason=reason, detail=detail[:MAX_MESSAGE_CHARS])
        self.parse_failures.record(failure)
        raise PolicyFailure(failure.reason, failure.detail)
