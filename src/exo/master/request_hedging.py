"""Request hedging for distributed ML inference.

Fire a duplicate inference to a backup trace_id if primary exceeds P95 latency.
First response wins; the losing task is cancelled.

Usage:
    # Env var to enable:
    EXO_HEDGING_ENABLED=1

    # In the API handler (non-streaming only):
    if HEDGING.should_hedge(trace_id):
        result, winner = await HEDGING.run_with_hedge(
            primary_coro=_do_collect(command),
            hedge_coro=_do_collect(hedge_command),
            delay_ms=HEDGING._p95_latency_ms,
        )
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class HedgeConfig:
    enabled: bool
    hedge_delay_ms: float = 200.0
    max_hedge_attempts: int = 1


@dataclass
class HedgeStats:
    total_hedged: int = 0
    hedge_wins: int = 0  # hedge coroutine responded first
    primary_wins: int = 0  # primary coroutine responded first
    cancelled: int = 0  # losing tasks cancelled


class HedgingController:
    """
    Manages request hedging.

    After `delay_ms` the hedge coroutine is dispatched via asyncio.ensure_future.
    asyncio.wait(FIRST_COMPLETED) picks the winner; the loser is cancelled.

    Thread-safety: stats counters are mutated only from the event-loop thread
    (all callers are async), so no lock is needed.
    """

    def __init__(self) -> None:
        self._stats: HedgeStats = HedgeStats()
        self._p95_latency_ms: float = 200.0
        self._enabled: bool = os.getenv("EXO_HEDGING_ENABLED", "0") == "1"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_hedge(self, trace_id: str) -> bool:
        """Return True when hedging is enabled. Streaming requests must NOT hedge."""
        return self._enabled

    def update_p95(self, p95_ms: float) -> None:
        """Update the hedge delay to match the cluster P95 total latency."""
        if p95_ms > 0:
            self._p95_latency_ms = p95_ms
            logger.debug(f"[hedging] p95 updated to {p95_ms:.1f}ms")

    async def run_with_hedge(
        self,
        primary_coro: Any,
        hedge_coro: Any,
        delay_ms: float,
    ) -> tuple[Any, str]:
        """
        Race primary_coro against hedge_coro (fired after delay_ms).

        Returns (result, winner) where winner is "primary" or "hedge".
        Cancels the losing task immediately after the winner resolves.

        Raises whatever the winning coroutine raises.
        """
        self._stats.total_hedged += 1

        primary_task: asyncio.Task[Any] = asyncio.ensure_future(primary_coro)

        async def _fire_hedge() -> Any:
            await asyncio.sleep(delay_ms / 1000.0)
            return await hedge_coro

        hedge_wrapper: asyncio.Task[Any] = asyncio.ensure_future(_fire_hedge())

        try:
            done, pending = await asyncio.wait(
                {primary_task, hedge_wrapper},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # Caller cancelled the whole request; propagate cleanly.
            for t in (primary_task, hedge_wrapper):
                t.cancel()
            raise

        # Cancel the loser and reap it so cancellation completes.
        for t in pending:
            t.cancel()
            self._stats.cancelled += 1
        for t in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if hedge_wrapper in pending:
            # Cancellation can land during the delay sleep, in which case
            # hedge_coro was never awaited — close it to avoid a
            # "coroutine was never awaited" RuntimeWarning. close() is a
            # no-op for coroutines that already started or finished.
            hedge_coro.close()

        winner_task = next(iter(done))

        if winner_task is primary_task:
            self._stats.primary_wins += 1
            who = "primary"
            logger.debug(f"[hedging] primary won (p95={delay_ms:.0f}ms)")
            # If hedge actually fired (delay expired) but primary still won,
            # we still need to await the hedge_wrapper cancel.
        else:
            # hedge_wrapper won — that means the delay fired AND the hedge
            # coroutine finished before primary.
            self._stats.hedge_wins += 1
            who = "hedge"
            logger.info(
                f"[hedging] hedge won — primary was slower than {delay_ms:.0f}ms"
            )

        return winner_task.result(), who

    def maybe_hedge(self, trace_id: str, delay_ms: float = 1000.0) -> float:
        """Return the hedge delay in ms to use for this request.

        Callers that want the full race loop should use run_with_hedge directly.
        This method exists so callers can obtain the configured delay and decide
        whether to proceed with hedging.  Returns 0.0 when hedging is disabled.
        """
        if not self._enabled:
            return 0.0
        effective_delay = delay_ms if delay_ms > 0 else self._p95_latency_ms
        logger.debug(
            f"[hedging] maybe_hedge trace_id={trace_id!r} delay_ms={effective_delay:.0f}"
        )
        return effective_delay

    def cancel_hedge(self, trace_id: str) -> None:
        """No-op cancellation hook for callers that dispatch the hedge manually.

        run_with_hedge cancels the losing task internally via asyncio.Task.cancel().
        This method is provided so external callers that manage hedge tasks
        themselves have a consistent interface to signal cancellation intent.
        The actual cancellation must be performed by the caller on the asyncio.Task.
        """
        logger.debug(f"[hedging] cancel_hedge signalled trace_id={trace_id!r}")

    def stats(self) -> dict[str, object]:
        s = self._stats
        total = max(s.total_hedged, 1)
        return {
            "enabled": self._enabled,
            "p95_latency_ms": round(self._p95_latency_ms, 2),
            "total_hedged": s.total_hedged,
            "hedge_wins": s.hedge_wins,
            "primary_wins": s.primary_wins,
            "cancelled": s.cancelled,
            "hedge_win_rate": round(s.hedge_wins / total, 4),
        }


HEDGING = HedgingController()
