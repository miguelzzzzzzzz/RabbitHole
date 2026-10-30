"""Shared fixtures.

The autouse network guard makes any test that opens an internet socket fail,
so the fast suite (and CI) can never call a model or a remote service. Tests
that need the network must be marked both ``slow`` and ``allow_network``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to open a network connection."""


def _is_unix_socket(sock: socket.socket) -> bool:
    return getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX


@pytest.fixture(autouse=True)
def network_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("allow_network") is not None:
        if request.node.get_closest_marker("slow") is None:
            raise pytest.UsageError("allow_network tests must also be marked slow")
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if _is_unix_socket(self):
            real_connect(self, address)
            return
        raise NetworkBlockedError(f"network access blocked in tests: connect to {address!r}")

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if _is_unix_socket(self):
            return real_connect_ex(self, address)
        raise NetworkBlockedError(f"network access blocked in tests: connect to {address!r}")

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError(f"network access blocked in tests: resolve {host!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield
