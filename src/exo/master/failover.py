"""
failover.py — Node failover coordinator for distributed ML inference.

Detects node failure via circuit breaker (OPEN state) or heartbeat eviction,
selects the healthiest surviving node, and records each failover lifecycle from
trigger through completion.

Usage::

    from exo.master.failover import FAILOVER

    if FAILOVER.should_failover(node_id):
        event = FAILOVER.trigger(task_id, node_id, reason="heartbeat_evicted")
        target = FAILOVER.select_failover_node(node_id)
        # ... reschedule task on target ...
        FAILOVER.complete(task_id, target, success=target is not None)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger

from exo.master.circuit_breaker import CIRCUIT_BREAKERS, CircuitState
from exo.master.event_stream import emit as emit_cluster_event
from exo.master.heartbeat_monitor import HEARTBEAT_MONITOR

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

EVENT_NODE_FAILED: str = "node_failed"
EVENT_FAILOVER_COMPLETE: str = "failover_complete"


@dataclass
class FailoverEvent:
    task_id: str
    failed_node_id: str
    failover_node_id: str | None
    reason: str
    triggered_at: float
    completed_at: float | None
    success: bool


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class FailoverCoordinator:
    """
    Lifecycle manager for per-task node failovers.

    Typical call sequence
    ---------------------
    1. ``should_failover(node_id)``  — gate check before any action
    2. ``trigger(task_id, node_id, reason)``  — open the failover
    3. ``select_failover_node(node_id)``  — pick the replacement
    4. ``complete(task_id, target, success)``  — close the failover
    """

    def __init__(self, history_maxlen: int = 200) -> None:
        self._events: deque[FailoverEvent] = deque(maxlen=history_maxlen)
        self._active_failovers: dict[str, FailoverEvent] = {}
        self._total_failovers: int = 0
        self._successful_failovers: int = 0

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def should_failover(self, node_id: str) -> bool:
        """Return True when the node's circuit breaker is OPEN *or* the
        heartbeat monitor has evicted it."""
        cb_open = CIRCUIT_BREAKERS.get(node_id).state == CircuitState.OPEN
        hb_evicted = node_id in HEARTBEAT_MONITOR.evicted_nodes()
        if cb_open or hb_evicted:
            logger.debug(
                f"[failover] should_failover node_id={node_id!r} "
                f"cb_open={cb_open} hb_evicted={hb_evicted}"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def trigger(self, task_id: str, failed_node_id: str, reason: str) -> FailoverEvent:
        """Open a failover for *task_id*.  Idempotent — returns the existing
        event if a failover is already active for this task."""
        if task_id in self._active_failovers:
            logger.warning(
                f"[failover] trigger called but failover already active "
                f"task_id={task_id!r} — returning existing event"
            )
            return self._active_failovers[task_id]

        event = FailoverEvent(
            task_id=task_id,
            failed_node_id=failed_node_id,
            failover_node_id=None,
            reason=reason,
            triggered_at=time.time(),
            completed_at=None,
            success=False,
        )
        self._active_failovers[task_id] = event
        self._total_failovers += 1

        logger.warning(
            f"[failover] triggered task_id={task_id!r} "
            f"failed_node={failed_node_id!r} reason={reason!r}"
        )
        emit_cluster_event(
            EVENT_NODE_FAILED,
            {
                "task_id": task_id,
                "failed_node_id": failed_node_id,
                "reason": reason,
                "triggered_at": event.triggered_at,
            },
            node_id=failed_node_id,
        )
        return event

    def complete(
        self, task_id: str, failover_node_id: str | None, success: bool
    ) -> None:
        """Finalise an active failover; move it to history."""
        event = self._active_failovers.pop(task_id, None)
        if event is None:
            logger.warning(
                f"[failover] complete called for unknown task_id={task_id!r} — ignoring"
            )
            return

        event.failover_node_id = failover_node_id
        event.completed_at = time.time()
        event.success = success

        if success:
            self._successful_failovers += 1

        self._events.append(event)
        latency_ms = round((event.completed_at - event.triggered_at) * 1000, 1)
        logger.info(
            f"[failover] complete task_id={task_id!r} "
            f"target={failover_node_id!r} success={success} "
            f"latency_ms={latency_ms}"
        )
        emit_cluster_event(
            EVENT_FAILOVER_COMPLETE,
            {
                "task_id": task_id,
                "failed_node_id": event.failed_node_id,
                "failover_node_id": failover_node_id,
                "success": success,
                "latency_ms": latency_ms,
            },
            node_id=failover_node_id or event.failed_node_id,
        )

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def select_failover_node(self, failed_node_id: str) -> str | None:
        """Pick the healthiest surviving node.

        Strategy: from HEARTBEAT_MONITOR.alive_nodes(), exclude the failed
        node, then return the one with the lowest circuit-breaker error_rate.
        Returns None when no healthy candidates exist.
        """
        candidates: list[str] = [
            n for n in HEARTBEAT_MONITOR.alive_nodes() if n != failed_node_id
        ]
        if not candidates:
            logger.warning(
                f"[failover] select_failover_node: no alive nodes excluding "
                f"failed_node={failed_node_id!r}"
            )
            return None

        # Filter out nodes whose circuit breaker is OPEN
        healthy = [
            n for n in candidates if CIRCUIT_BREAKERS.get(n).state != CircuitState.OPEN
        ]
        pool = healthy if healthy else candidates  # fall back to all alive if all open

        best = min(pool, key=lambda n: CIRCUIT_BREAKERS.get(n).error_rate)
        logger.info(
            f"[failover] selected node={best!r} "
            f"error_rate={CIRCUIT_BREAKERS.get(best).error_rate:.4f} "
            f"from {len(pool)} candidate(s)"
        )
        return best

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def history(self, n: int = 20) -> list[FailoverEvent]:
        """Return the last *n* completed failover events (newest last)."""
        events = list(self._events)
        return events[-n:]

    def stats(self) -> dict[str, Any]:
        success_rate = (
            self._successful_failovers / self._total_failovers
            if self._total_failovers > 0
            else 0.0
        )
        return {
            "total": self._total_failovers,
            "successful": self._successful_failovers,
            "active": len(self._active_failovers),
            "success_rate": round(success_rate, 4),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

FAILOVER = FailoverCoordinator()
