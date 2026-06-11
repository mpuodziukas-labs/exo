from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generic, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass(order=True)
class PriorityItem(Generic[T]):
    # Negative priority so higher number = higher priority in min-heap
    neg_priority: int
    sequence: int  # tiebreaker — FIFO within same priority
    timestamp: float = field(compare=False)
    item: T = field(compare=False)


class PriorityRequestQueue:
    """
    Min-heap priority queue for inference requests.

    Priority levels:
    - 10 = CRITICAL (health checks, internal monitoring)
    - 7  = HIGH (premium tier / realtime use cases)
    - 5  = NORMAL (default)
    - 2  = LOW (batch processing, background tasks)
    - 0  = BACKGROUND (best-effort, preemptible)

    Preemption: when a HIGH request arrives and the queue is full,
    it displaces the lowest-priority pending request (if lower than HIGH).
    """

    PRIORITY_CRITICAL = 10
    PRIORITY_HIGH = 7
    PRIORITY_NORMAL = 5
    PRIORITY_LOW = 2
    PRIORITY_BACKGROUND = 0

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._heap: list[PriorityItem[Any]] = []
        self._seq = 0
        self._lock = Lock()
        self._enqueued_total: int = 0
        self._preempted_total: int = 0
        self._dequeued_total: int = 0

    def enqueue(self, item: Any, priority: int = 5) -> bool:
        """
        Add item to queue. Returns True if enqueued, False if rejected.
        If queue is full and item has higher priority than minimum, preempts lowest.
        """
        with self._lock:
            if len(self._heap) >= self.max_size:
                # Try preemption: find lowest priority item (min-heap root has most-negative neg_priority = lowest actual priority)
                if self._heap and (-self._heap[0].neg_priority) < priority:
                    # Preempt the lowest priority item
                    evicted = heapq.heappop(self._heap)
                    self._preempted_total += 1
                    logger.warning(
                        f"Priority queue: preempted priority={-evicted.neg_priority} "
                        f"to admit priority={priority}"
                    )
                else:
                    logger.warning(
                        f"Priority queue full (max={self.max_size}) — rejecting priority={priority}"
                    )
                    return False

            self._seq += 1
            heapq.heappush(
                self._heap,
                PriorityItem(
                    neg_priority=-priority,
                    sequence=self._seq,
                    timestamp=time.monotonic(),
                    item=item,
                ),
            )
            self._enqueued_total += 1
            return True

    def dequeue(self) -> tuple[Any, int] | None:
        """Return (item, priority) or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            entry = heapq.heappop(self._heap)
            self._dequeued_total += 1
            return entry.item, -entry.neg_priority

    def peek_priority(self) -> int | None:
        """Return the highest priority of queued items, or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            return -self._heap[0].neg_priority

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._heap),
                "max_size": self.max_size,
                "enqueued_total": self._enqueued_total,
                "dequeued_total": self._dequeued_total,
                "preempted_total": self._preempted_total,
                "highest_pending_priority": (-self._heap[0].neg_priority) if self._heap else None,
            }

    def prometheus_metrics(self) -> str:
        s = self.stats()
        return (
            "# HELP exo_priority_queue_size Current priority queue depth\n"
            "# TYPE exo_priority_queue_size gauge\n"
            f"exo_priority_queue_size {s['size']}\n"
            "# HELP exo_priority_queue_preempted_total Total preempted requests\n"
            "# TYPE exo_priority_queue_preempted_total counter\n"
            f"exo_priority_queue_preempted_total {s['preempted_total']}\n"
        )


PRIORITY_QUEUE = PriorityRequestQueue(
    max_size=int(os.getenv("EXO_PRIORITY_QUEUE_MAX_SIZE", "128"))
)
