from __future__ import annotations

import re
import socket

import pytest

from conftest import NetworkBlockedError


def test_tcp_connect_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkBlockedError, match=re.escape("127.0.0.1")),
    ):
        sock.connect(("127.0.0.1", 9))


def test_connect_ex_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkBlockedError),
    ):
        sock.connect_ex(("127.0.0.1", 9))


def test_name_resolution_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError, match=re.escape("example.com")):
        socket.getaddrinfo("example.com", 443)


def test_create_connection_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 443), timeout=1)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs Unix sockets")
def test_unix_sockets_still_work() -> None:
    left, right = socket.socketpair(socket.AF_UNIX)
    with left, right:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"
