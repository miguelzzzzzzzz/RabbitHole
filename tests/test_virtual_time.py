"""The virtual-time test loop jumps to timers instead of waiting for them."""

from __future__ import annotations

import asyncio
import time

import pytest

from virtual_time import (
    VirtualTimeDeadlockError,
    VirtualTimeLoop,
    loop_clock,
    run_virtual,
)


def test_sleep_advances_virtual_time_without_waiting() -> None:
    async def main() -> float:
        start = loop_clock()
        await asyncio.sleep(3600)
        await asyncio.sleep(0.25)
        return loop_clock() - start

    wall = time.monotonic()
    assert run_virtual(main()) == pytest.approx(3600.25)
    assert time.monotonic() - wall < 5


def test_timeout_cancels_at_the_virtual_deadline() -> None:
    cancelled: list[float] = []

    async def sleeper() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(loop_clock())
            raise

    async def main() -> None:
        async with asyncio.timeout(1.5):
            await sleeper()

    with pytest.raises(TimeoutError):
        run_virtual(main())
    assert cancelled == [pytest.approx(1.5)]


def test_timers_fire_in_order_at_their_virtual_times() -> None:
    async def main() -> list[tuple[str, float]]:
        order: list[tuple[str, float]] = []

        async def after(delay: float, label: str) -> None:
            await asyncio.sleep(delay)
            order.append((label, loop_clock()))

        await asyncio.gather(after(2, "b"), after(1, "a"), after(3, "c"))
        return order

    assert run_virtual(main()) == [("a", 1.0), ("b", 2.0), ("c", 3.0)]


def test_waiting_on_nothing_is_reported_as_a_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("virtual_time.IDLE_WAIT_S", 0.01)

    async def main() -> None:
        await asyncio.get_running_loop().create_future()  # never resolved

    with pytest.raises(VirtualTimeDeadlockError):
        run_virtual(main())


def test_clock_starts_at_zero() -> None:
    loop = VirtualTimeLoop()
    try:
        assert loop.time() == 0.0
    finally:
        loop.close()
