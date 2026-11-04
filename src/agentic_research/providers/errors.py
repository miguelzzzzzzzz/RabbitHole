"""Typed provider failures.

Provider calls raise these so the agent loop (M2) and the probe can classify
outcomes without inspecting exception strings. Retries and the runaway guard
arrive in M5; this module only names the failure modes the M1 provider can hit.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for chat-provider failures.

    ``reason`` is a short machine-readable code (for example
    ``local_server_unavailable``). ``message`` is human-readable detail that is
    safe to show the model and to write into a trace.
    """

    reason: str = "provider_error"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        self.message = message
        if reason is not None:
            self.reason = reason
        super().__init__(f"{self.reason}: {message}")


class LocalServerUnavailableError(ProviderError):
    """The local OpenAI-compatible server could not be reached."""

    reason = "local_server_unavailable"


class ProviderHTTPError(ProviderError):
    """The server answered with a non-success HTTP status."""

    reason = "http_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        reason: str | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, reason=reason or self.reason)


class ProviderResponseError(ProviderError):
    """The response body was not a usable OpenAI chat-completions payload."""

    reason = "malformed_response"
