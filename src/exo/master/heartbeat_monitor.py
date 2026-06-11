"""
heartbeat_monitor.py — Worker heartbeat tracking and auto-eviction.

Master evicts dead workers after EXO_EVICTION_THRESHOLD (default 3) missed
heartbeats.  On eviction the circuit breaker for that node is tripped
immediately so in-flight routing stops using the dead node.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass

import anyio
from loguru import logger

from exo.master.circuit_breaker import CIRCUIT_BREAKERS
from exo.master.drift_detector import DRIFT_DETECTOR
from exo.master.event_stream import emit as emit_cluster_event
from exo.master.runbook_executor import RUNBOOK_EXECUTOR
from exo.master.webhook_notifier import WEBHOOK_NOTIFIER
from exo.master.worker_reconnect import RECONNECT_TRACKER

_HEARTBEAT_INTERVAL: float = float(
    os.getenv("EXO_HEARTBEAT_INTERVAL_SECONDS", "10.0")
)
_EVICTION_THRESHOLD: int = int(os.getenv("EXO_EVICTION_THRESHOLD", "3"))


@dataclass
class HeartbeatRecord:
    node_id: str
    last_seen: float
    missed: int
    evicted: bool
    first_seen: float


class HeartbeatMonitor:
    """
    Tracks per-worker liveness via periodic heartbeats.

    Workers call beat() when they are alive.  The master calls
    check_all() on a fixed interval; nodes that exceed the missed-beat
    threshold are evicted and their circuit breaker is tripped.
    """

    def __init__(
        self,
        heartbeat_interval: float = _HEARTBEAT_INTERVAL,
        eviction_threshold: int = _EVICTION_THRESHOLD,
    ) -> None:
        self._records: dict[str, HeartbeatRecord] = {}
        self._heartbeat_interval: float = heartbeat_interval
        self._eviction_threshold: int = eviction_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, node_id: str) -> None:
        """Ensure a record exists for node_id (idempotent)."""
        if node_id not in self._records:
            now = time.monotonic()
            self._records[node_id] = HeartbeatRecord(
                node_id=node_id,
                last_seen=now,
                missed=0,
                evicted=False,
                first_seen=now,
            )
            logger.debug(f"[heartbeat] registered node {node_id!r}")

    def beat(self, node_id: str) -> HeartbeatRecord:
        """Record a heartbeat from node_id.  Creates the record if absent."""
        now = time.monotonic()
        if node_id not in self._records:
            self._records[node_id] = HeartbeatRecord(
                node_id=node_id,
                last_seen=now,
                missed=0,
                evicted=False,
                first_seen=now,
            )
            logger.info(f"[heartbeat] new node {node_id!r} registered via first beat")
        else:
            rec = self._records[node_id]
            was_evicted = rec.evicted
            rec.last_seen = now
            rec.missed = 0
            rec.evicted = False
            if was_evicted:
                logger.info(f"[heartbeat] node {node_id!r} recovered after eviction")
                RECONNECT_TRACKER.on_reconnect(node_id)
        # Piggyback a drift snapshot on every live heartbeat.
        try:
            DRIFT_DETECTOR.register_snapshot(DRIFT_DETECTOR.local_snapshot(node_id))
        except Exception as exc:  # never let drift capture crash the heartbeat path
            logger.warning(f"[heartbeat] drift snapshot failed for {node_id!r}: {exc}")
        return self._records[node_id]

    def check_all(self) -> list[str]:
        """
        Advance the missed-beat counter for every stale node.

        Returns a list of node_ids that were *newly* evicted this cycle
        (missed >= threshold and not already marked evicted).
        Side-effect: records a circuit-breaker failure for each eviction.
        """
        now = time.monotonic()
        stale_cutoff = self._heartbeat_interval  # one interval = one missed beat
        newly_evicted: list[str] = []

        for node_id, rec in self._records.items():
            if rec.evicted:
                continue  # already handled

            age = now - rec.last_seen
            if age >= stale_cutoff:
                rec.missed += 1
                logger.debug(
                    f"[heartbeat] node {node_id!r} missed beat "
                    f"#{rec.missed}/{self._eviction_threshold} (age={age:.1f}s)"
                )

            if rec.missed >= self._eviction_threshold:
                rec.evicted = True
                newly_evicted.append(node_id)
                logger.warning(
                    f"[heartbeat] evicting node {node_id!r} "
                    f"after {rec.missed} missed heartbeats — tripping circuit breaker"
                )
                CIRCUIT_BREAKERS.get(node_id).record_failure()
                RECONNECT_TRACKER.on_disconnect(node_id)
                emit_cluster_event(
                    "worker_evicted",
                    {
                        "missed_heartbeats": rec.missed,
                        "eviction_threshold": self._eviction_threshold,
                        "last_seen": rec.last_seen,
                        "uptime_s": round(time.monotonic() - rec.first_seen, 1),
                    },
                    node_id=node_id,
                )
                WEBHOOK_NOTIFIER.notify("worker_down", {"node_id": node_id})
                # No running event loop -> skip async runbook dispatch.
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().create_task(
                        RUNBOOK_EXECUTOR.trigger("worker_down", {"node_id": node_id})
                    )

        return newly_evicted

    def is_alive(self, node_id: str) -> bool:
        """True if node is registered, not evicted, and last_seen within 3× interval."""
        rec = self._records.get(node_id)
        if rec is None or rec.evicted:
            return False
        age = time.monotonic() - rec.last_seen
        return age <= self._heartbeat_interval * 3

    def evicted_nodes(self) -> list[str]:
        return [r.node_id for r in self._records.values() if r.evicted]

    def alive_nodes(self) -> list[str]:
        return [
            r.node_id
            for r in self._records.values()
            if not r.evicted
        ]

    def stats(self) -> dict[str, object]:
        now = time.monotonic()
        records_out: list[dict[str, object]] = []
        for rec in self._records.values():
            records_out.append(
                {
                    "node_id": rec.node_id,
                    "last_seen_ago_s": round(now - rec.last_seen, 2),
                    "missed": rec.missed,
                    "evicted": rec.evicted,
                    "uptime_s": round(now - rec.first_seen, 1),
                }
            )
        return {
            "heartbeat_interval_s": self._heartbeat_interval,
            "eviction_threshold": self._eviction_threshold,
            "total_nodes": len(self._records),
            "alive": len(self.alive_nodes()),
            "evicted": len(self.evicted_nodes()),
            "nodes": records_out,
        }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run_check_loop(self) -> None:
        """Runs check_all() every heartbeat_interval seconds indefinitely."""
        logger.info(
            f"[heartbeat] monitor started — interval={self._heartbeat_interval}s "
            f"eviction_threshold={self._eviction_threshold}"
        )
        while True:
            await anyio.sleep(self._heartbeat_interval)
            newly_evicted = self.check_all()
            if newly_evicted:
                logger.warning(
                    f"[heartbeat] eviction cycle — evicted nodes: {newly_evicted}"
                )


# Module-level singleton
HEARTBEAT_MONITOR = HeartbeatMonitor()
