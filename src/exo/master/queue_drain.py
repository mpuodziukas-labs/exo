"""
Inference queue drain on shutdown: when SIGTERM is received, waits for all
in-flight inference requests to complete before stopping. Tracks active
request lifecycle (started/finished) so the drain loop knows when to exit.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_DRAIN_TIMEOUT = 30.0  # max seconds to wait for drain


@dataclass
class InFlightRequest:
    request_id: str
    model_id: str
    started_at: float = field(default_factory=time.time)

    def age_s(self) -> float:
        return time.time() - self.started_at


class QueueDrainManager:
    """
    register(request_id, model_id): marks a request as in-flight.
    complete(request_id): removes from in-flight set.
    drain(timeout): waits until all in-flight complete or timeout.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, InFlightRequest] = {}
        self._draining = False

    def register(self, request_id: str, model_id: str) -> None:
        self._inflight[request_id] = InFlightRequest(
            request_id=request_id, model_id=model_id
        )

    def complete(self, request_id: str) -> None:
        self._inflight.pop(request_id, None)

    @property
    def active_count(self) -> int:
        return len(self._inflight)

    @property
    def is_draining(self) -> bool:
        return self._draining

    async def drain(self, timeout: float = _DRAIN_TIMEOUT) -> bool:
        """
        Wait for all in-flight requests to complete.
        Returns True if drained cleanly, False if timed out.
        """
        self._draining = True
        deadline = time.monotonic() + timeout
        logger.info(
            f"QueueDrain: waiting for {len(self._inflight)} in-flight requests (max {timeout}s)"
        )

        while self._inflight and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

        if self._inflight:
            remaining = list(self._inflight.keys())
            logger.warning(
                f"QueueDrain: timed out with {len(remaining)} requests still in-flight: {remaining[:5]}"
            )
            self._draining = False
            return False

        logger.info("QueueDrain: all requests drained cleanly")
        self._draining = False
        return True

    def reject_new(self) -> bool:
        """Returns True if new requests should be rejected (draining in progress)."""
        return self._draining

    def get_status(self) -> dict[str, Any]:
        return {
            "in_flight": self.active_count,
            "draining": self._draining,
            "requests": [
                {"id": r.request_id, "model": r.model_id, "age_s": round(r.age_s(), 2)}
                for r in self._inflight.values()
            ],
        }


QUEUE_DRAIN = QueueDrainManager()
