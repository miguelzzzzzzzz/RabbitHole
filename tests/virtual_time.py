"""A virtual-time asyncio event loop for deterministic timing tests.

`VirtualTimeLoop.time()` is a counter that moves only when the loop would
otherwise block waiting for its next timer: instead of sleeping until the
timer is due, the loop jumps straight to it. ``asyncio.sleep``,
``asyncio.timeout`` and ``call_later`` therefore run their real code paths
(a timeout really cancels the awaited coroutine), but a 3600 s sleep takes
no wall time and every run is deterministic. I/O readiness is still polled
(with a zero timeout), so the loop's self-pipe keeps working.

Not for code that waits on threads or real I/O while timers are pending: the
clock jumps to the next timer as soon as nothing is ready, without waiting
for the thread. With no timer at all, the loop waits up to `IDLE_WAIT_S` real
seconds for I/O, then reports a deadlock.

Use `run_virtual` (or ``asyncio.Runner(loop_factory=VirtualTimeLoop)``), and
`loop_clock` as the injected clock so budgets and timeouts share one clock.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

#: Real seconds to wait for I/O when no timer is scheduled before calling it a deadlock.
IDLE_WAIT_S = 5.0


class VirtualTimeDeadlockError(RuntimeError):
    """The loop has no timer to jump to and no I/O arrived: nothing can make progress."""


class _JumpingSelector(selectors.DefaultSelector):
    def __init__(self, loop_time: list[float]) -> None:
        super().__init__()
        self._time = loop_time

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        if timeout is None:  # no timer at all: only I/O (e.g. a thread wake-up) can help
            events = super().select(IDLE_WAIT_S)
            if not events:
                raise VirtualTimeDeadlockError("no timer scheduled and no I/O ready")
            return events
        events = super().select(0)
        if not events and timeout > 0:
            self._time[0] += timeout  # jump to the next timer instead of waiting for it
        return events


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock starts at 0 and jumps to each due timer."""

    def __init__(self) -> None:
        self._virtual_now = [0.0]
        super().__init__(_JumpingSelector(self._virtual_now))

    def time(self) -> float:
        return self._virtual_now[0]


def run_virtual(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on a fresh `VirtualTimeLoop`."""
    with asyncio.Runner(loop_factory=VirtualTimeLoop) as runner:
        return runner.run(coro)


def loop_clock() -> float:
    """The running loop's clock (virtual under `VirtualTimeLoop`)."""
    return asyncio.get_running_loop().time()
