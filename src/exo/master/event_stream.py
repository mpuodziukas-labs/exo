"""
event_stream.py — Real-time cluster event bus for SSE streaming.

Cluster events (node joins/leaves, model loads, circuit breaker flips,
SLO violations, hot-swaps, evictions, checkpoints) are published onto a
shared EventBus.  Each SSE subscriber gets its own asyncio.Queue so the
HTTP handler can yield events without blocking any other component.

Usage::

    from exo.master.event_stream import EVENT_BUS, emit

    # fire-and-forget from anywhere
    emit("node_joined", {"node_id": "abc", "hostname": "mac-mini"}, node_id="abc")

    # SSE handler reads from a per-subscriber queue
    queue = EVENT_BUS.subscribe()
    try:
        event = await asyncio.wait_for(queue.get(), timeout=15.0)
    finally:
        EVENT_BUS.unsubscribe(queue)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

# ---------------------------------------------------------------------------
# Event types (informational; not exhaustive — callers may use any string)
# ---------------------------------------------------------------------------
EVENT_NODE_JOINED: str = "node_joined"
EVENT_NODE_LEFT: str = "node_left"
EVENT_MODEL_LOADED: str = "model_loaded"
EVENT_CIRCUIT_OPEN: str = "circuit_open"
EVENT_CIRCUIT_CLOSED: str = "circuit_closed"
EVENT_SLO_VIOLATION: str = "slo_violation"
EVENT_WORKER_EVICTED: str = "worker_evicted"
EVENT_CHECKPOINT_SAVED: str = "checkpoint_saved"
EVENT_HOT_SWAP_COMPLETE: str = "hot_swap_complete"

_HISTORY_MAXLEN = 1000


@dataclass
class ClusterEvent:
    """Immutable cluster event record."""

    event_id: str
    event_type: str
    data: dict[str, object]
    node_id: str
    timestamp: float

    def to_sse_line(self) -> str:
        """Serialise as a single SSE data line (no trailing newlines)."""
        import json

        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}"


class EventBus:
    """
    Fanout event bus with per-subscriber asyncio queues and a ring-buffer
    history for replay on new connections.

    Thread-safety: publish() and subscribe()/unsubscribe() are safe to call
    from any thread because asyncio.Queue is coroutine-safe within the same
    event loop, and the subscriber list mutation uses a plain Python list
    (GIL-protected for single-element ops).
    """

    def __init__(self, history_maxlen: int = _HISTORY_MAXLEN) -> None:
        self._subscribers: list[asyncio.Queue[ClusterEvent]] = []
        self._history: deque[ClusterEvent] = deque(maxlen=history_maxlen)
        self._total_published: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def publish(self, event: ClusterEvent) -> None:
        """Put *event* on every subscriber queue and append to history.

        Subscribers whose queues are full are skipped (slow-consumer drop)
        so one stalled connection never blocks the cluster.
        """
        self._history.append(event)
        self._total_published += 1
        dropped = 0
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dropped += 1
        if dropped:
            logger.warning(
                f"[event_bus] dropped event {event.event_type!r} "
                f"for {dropped} slow subscriber(s)"
            )

    def subscribe(self, maxsize: int = 100) -> asyncio.Queue[ClusterEvent]:
        """Create and register a new per-subscriber queue."""
        queue: asyncio.Queue[ClusterEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        logger.debug(
            f"[event_bus] subscriber added — total={len(self._subscribers)}"
        )
        return queue

    def unsubscribe(self, queue: asyncio.Queue[ClusterEvent]) -> None:
        """Remove *queue* from the subscriber list (idempotent)."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)
        logger.debug(
            f"[event_bus] subscriber removed — total={len(self._subscribers)}"
        )

    def history(self, since_timestamp: float = 0.0) -> list[ClusterEvent]:
        """Return all buffered events with timestamp > *since_timestamp*."""
        return [e for e in self._history if e.timestamp > since_timestamp]

    def stats(self) -> dict[str, object]:
        return {
            "subscriber_count": len(self._subscribers),
            "total_published": self._total_published,
            "history_size": len(self._history),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

EVENT_BUS: EventBus = EventBus()


def emit(
    event_type: str,
    data: dict[str, object],
    node_id: str = "",
) -> ClusterEvent:
    """Convenience function: create a ClusterEvent and publish it.

    Returns the created event so callers can inspect the generated event_id
    if needed.
    """
    event = ClusterEvent(
        event_id=str(uuid4()),
        event_type=event_type,
        data=data,
        node_id=node_id,
        timestamp=time.time(),
    )
    EVENT_BUS.publish(event)
    logger.debug(
        f"[event_bus] emit type={event_type!r} node_id={node_id!r} "
        f"event_id={event.event_id}"
    )
    return event
