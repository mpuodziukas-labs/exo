"""Behavioral tests for exo.master.request_hedging.

HedgingController manages speculative duplicate inference to improve tail
latency.  Key behaviors:
  - should_hedge() gates on EXO_HEDGING_ENABLED env var
  - update_p95() updates the internal hedge delay reference
  - maybe_hedge() returns 0.0 when disabled, effective delay when enabled
  - run_with_hedge() races primary vs hedge; first wins, loser is cancelled
  - stats() tracks wins and cancellations
"""

from __future__ import annotations

import asyncio
import math

import pytest

from exo.master.request_hedging import HedgingController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enabled() -> HedgingController:
    ctrl = HedgingController()
    ctrl.enabled = True
    return ctrl


def _disabled() -> HedgingController:
    ctrl = HedgingController()
    ctrl.enabled = False
    return ctrl


# ---------------------------------------------------------------------------
# should_hedge() / maybe_hedge()
# ---------------------------------------------------------------------------


def test_should_hedge_false_when_disabled() -> None:
    """should_hedge returns False when hedging is disabled."""
    assert _disabled().should_hedge("t1") is False


def test_should_hedge_true_when_enabled() -> None:
    """should_hedge returns True when hedging is enabled."""
    assert _enabled().should_hedge("t2") is True


def test_maybe_hedge_returns_zero_when_disabled() -> None:
    """maybe_hedge returns 0.0 (skip hedge) when controller is disabled."""
    ctrl = _disabled()
    assert ctrl.maybe_hedge("t3", delay_ms=500.0) == 0.0


def test_maybe_hedge_returns_passed_delay_when_enabled() -> None:
    """maybe_hedge returns the caller's delay_ms when enabled."""
    ctrl = _enabled()
    assert ctrl.maybe_hedge("t4", delay_ms=300.0) == 300.0


def test_maybe_hedge_uses_p95_when_delay_is_zero() -> None:
    """maybe_hedge falls back to the stored p95 when delay_ms=0."""
    ctrl = _enabled()
    ctrl.update_p95(450.0)
    assert math.isclose(
        ctrl.maybe_hedge("t5", delay_ms=0.0), 450.0, rel_tol=1e-6, abs_tol=1e-12
    )


# ---------------------------------------------------------------------------
# update_p95()
# ---------------------------------------------------------------------------


def test_update_p95_ignores_non_positive_values() -> None:
    """update_p95 must not update the delay for zero or negative values."""
    ctrl = _enabled()
    original = ctrl.p95_latency_ms
    ctrl.update_p95(0.0)
    assert ctrl.p95_latency_ms == original
    ctrl.update_p95(-50.0)
    assert ctrl.p95_latency_ms == original


def test_update_p95_stores_positive_value() -> None:
    """update_p95 stores a valid positive p95."""
    ctrl = _enabled()
    ctrl.update_p95(800.0)
    assert math.isclose(ctrl.p95_latency_ms, 800.0, rel_tol=1e-6, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# run_with_hedge() — asyncio race tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_wins_when_fastest() -> None:
    """Primary wins when it finishes before the hedge delay fires.

    delay_ms=5000 ensures the hedge wrapper's asyncio.sleep hasn't fired
    by the time the primary coroutine returns.  asyncio.wait(FIRST_COMPLETED)
    picks the primary task.

    BUG NOTE: when primary wins before delay_ms fires, run_with_hedge cancels
    the _fire_hedge wrapper task before it reaches `await hedge_coro`.  The
    hedge_coro coroutine object is therefore never awaited, producing a
    RuntimeWarning from the Python runtime.  This is an implementation bug in
    request_hedging.py — the coroutine should be closed explicitly.
    """
    ctrl = _enabled()

    async def fast_primary() -> str:
        return "primary-result"

    async def hedge_never_reached() -> str:
        # Body never reached because the _fire_hedge wrapper is cancelled
        # before it awaits this coroutine.
        return "hedge-result"

    result, winner = await ctrl.run_with_hedge(
        fast_primary(), hedge_never_reached(), delay_ms=5000.0
    )
    assert winner == "primary"
    assert result == "primary-result"
    assert ctrl.stats()["primary_wins"] == 1


@pytest.mark.asyncio
async def test_hedge_wins_when_primary_is_slow() -> None:
    """Hedge wins when primary sleeps beyond the hedge delay."""
    ctrl = _enabled()

    async def slow_primary() -> str:
        await asyncio.sleep(10.0)
        return "primary-result"

    async def fast_hedge() -> str:
        return "hedge-result"

    # delay_ms=0 → hedge fires immediately and wins
    result, winner = await ctrl.run_with_hedge(
        slow_primary(), fast_hedge(), delay_ms=0.0
    )
    assert winner == "hedge"
    assert result == "hedge-result"
    assert ctrl.stats()["hedge_wins"] == 1


@pytest.mark.asyncio
async def test_losing_task_is_cancelled() -> None:
    """The losing task is cancelled and stats.cancelled increments.

    NOTE: asyncio.Task.cancel() is asynchronous — it schedules CancelledError
    injection but does not immediately execute the coroutine cleanup.
    The test verifies the stats counter is updated (synchronous), not that the
    CancelledError has been propagated (which requires an event-loop tick after
    run_with_hedge returns).
    """
    ctrl = _enabled()

    async def slow_primary() -> str:
        await asyncio.sleep(5.0)
        return "primary"

    async def fast_hedge() -> str:
        return "hedge"

    # delay=0: hedge fires immediately and wins; primary is the loser
    _result, winner = await ctrl.run_with_hedge(
        slow_primary(), fast_hedge(), delay_ms=0.0
    )
    assert winner == "hedge"
    # stats.cancelled incremented synchronously by run_with_hedge
    assert ctrl.stats()["cancelled"] == 1


@pytest.mark.asyncio
async def test_total_hedged_increments_per_call() -> None:
    """total_hedged is incremented for each run_with_hedge call."""
    ctrl = _enabled()

    async def instant() -> int:
        return 1

    for _ in range(3):
        await ctrl.run_with_hedge(instant(), instant(), delay_ms=0.0)

    assert ctrl.stats()["total_hedged"] == 3
