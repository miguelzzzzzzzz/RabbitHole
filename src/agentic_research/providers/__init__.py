"""Chat providers (OpenAI-compatible local server first; ADR-0003) and the scripted provider."""

from agentic_research.providers.errors import (
    LocalServerUnavailableError,
    ProviderError,
    ProviderHTTPError,
    ProviderResponseError,
)
from agentic_research.providers.openai_compat import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    OpenAICompatibleProvider,
    base_url_from_environ,
    model_from_environ,
    parse_chat_completion,
    parse_function_arguments,
    parse_tool_calls,
    parse_usage,
)
from agentic_research.providers.scripted import (
    Script,
    ScriptedError,
    ScriptedFunctionCall,
    ScriptedProvider,
    ScriptedTurn,
    load_script,
)
from agentic_research.providers.toolcall_json import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ParseFailureCounter,
    ToolCallJsonResult,
    parse_toolcall_json,
)
from agentic_research.providers.types import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    FunctionCall,
    TokenUsage,
    ToolCall,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "ChatMessage",
    "ChatProvider",
    "ChatRequest",
    "ChatResponse",
    "FunctionCall",
    "LocalServerUnavailableError",
    "OpenAICompatibleProvider",
    "ParseFailureCounter",
    "ParsedFinalAnswer",
    "ParsedToolCall",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderResponseError",
    "Script",
    "ScriptedError",
    "ScriptedFunctionCall",
    "ScriptedProvider",
    "ScriptedTurn",
    "TokenUsage",
    "ToolCall",
    "ToolCallJsonResult",
    "base_url_from_environ",
    "load_script",
    "model_from_environ",
    "parse_chat_completion",
    "parse_function_arguments",
    "parse_tool_calls",
    "parse_toolcall_json",
    "parse_usage",
]
