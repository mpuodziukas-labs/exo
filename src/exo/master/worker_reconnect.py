"""
Worker reconnect backoff: when a worker disconnects, tracks reconnect attempts
with exponential backoff. Prevents thundering-herd reconnects after a master restart.
Each worker gets an independent backoff state; resets on successful reconnect.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

_BASE_DELAY = 1.0     # seconds
_MAX_DELAY = 60.0     # seconds
_MULTIPLIER = 2.0


@dataclass
class ReconnectState:
    node_id: str
    attempt: int = 0
    next_allowed_at: float = 0.0
    last_disconnect_at: float = 0.0
    last_reconnect_at: float = 0.0

    def backoff_delay(self) -> float:
        return min(_BASE_DELAY * (_MULTIPLIER ** self.attempt), _MAX_DELAY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "attempt": self.attempt,
            "next_allowed_at": self.next_allowed_at,
            "backoff_delay_s": round(self.backoff_delay(), 2),
            "last_disconnect_at": self.last_disconnect_at,
            "last_reconnect_at": self.last_reconnect_at,
        }


class WorkerReconnectTracker:
    """
    on_disconnect(node_id): marks disconnect, computes next allowed reconnect time.
    can_reconnect(node_id): True if backoff window has passed.
    on_reconnect(node_id): resets backoff on successful reconnect.
    """

    def __init__(self) -> None:
        self._states: dict[str, ReconnectState] = {}

    def _get_or_create(self, node_id: str) -> ReconnectState:
        if node_id not in self._states:
            self._states[node_id] = ReconnectState(node_id=node_id)
        return self._states[node_id]

    def on_disconnect(self, node_id: str) -> float:
        """Returns delay seconds before reconnect is allowed."""
        state = self._get_or_create(node_id)
        delay = state.backoff_delay()
        state.last_disconnect_at = time.time()
        state.next_allowed_at = time.time() + delay
        state.attempt += 1
        logger.info(
            f"Worker disconnect: node={node_id} attempt={state.attempt} "
            f"backoff={delay:.1f}s"
        )
        return delay

    def can_reconnect(self, node_id: str) -> bool:
        state = self._states.get(node_id)
        if state is None:
            return True
        return time.time() >= state.next_allowed_at

    def on_reconnect(self, node_id: str) -> None:
        if node_id in self._states:
            state = self._states[node_id]
            state.attempt = 0
            state.next_allowed_at = 0.0
            state.last_reconnect_at = time.time()
            logger.info(f"Worker reconnect: node={node_id} backoff reset")

    def get_all(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._states.values()]

    def get_stats(self) -> dict[str, Any]:
        backing_off = [s for s in self._states.values() if time.time() < s.next_allowed_at]
        return {
            "tracked_nodes": len(self._states),
            "backing_off": len(backing_off),
            "nodes": self.get_all(),
        }


RECONNECT_TRACKER = WorkerReconnectTracker()
