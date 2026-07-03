"""
Per-client SLO violation throttle.

Rule:
  If a client (identified by API key or IP address) accumulates > 5 SLO
  violations within a 60-second sliding window, throttle them to 1 request
  per 10 seconds for the next 120 seconds.

Classes:
  - ``ClientSLOThrottle`` — core state machine
  - ``CLIENT_SLO_THROTTLE`` — module-level singleton

Integration with ``SLOBudgetTracker``:
  After every ``SLOBudgetTracker.record()`` call, pass the violation result
  to ``ClientSLOThrottle.observe(client_id, violated)`` to keep both
  components in sync.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_VIOLATION_WINDOW_S: float = 60.0  # sliding window for counting violations
_VIOLATION_THRESHOLD: int = 5  # violations in window → throttle
_THROTTLE_MIN_GAP_S: float = 10.0  # throttled client: min seconds between requests
_THROTTLE_DURATION_S: float = 120.0  # how long the throttle lasts


@dataclass
class _ClientRecord:
    client_id: str
    violation_timestamps: deque[float] = field(default_factory=lambda: deque())
    throttle_until: float = 0.0  # epoch; 0 = not throttled
    last_allowed_at: float = (
        0.0  # last time a request was let through (for rate-limiting)
    )
    total_violations: int = 0
    total_throttled_requests: int = 0

    def _prune_window(self) -> None:
        cutoff = time.time() - _VIOLATION_WINDOW_S
        while self.violation_timestamps and self.violation_timestamps[0] < cutoff:
            self.violation_timestamps.popleft()

    def recent_violations(self) -> int:
        self._prune_window()
        return len(self.violation_timestamps)

    def is_throttled(self) -> bool:
        return time.time() < self.throttle_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "recent_violations": self.recent_violations(),
            "total_violations": self.total_violations,
            "throttled": self.is_throttled(),
            "throttle_until": self.throttle_until if self.is_throttled() else None,
            "throttle_remaining_s": max(
                0.0, round(self.throttle_until - time.time(), 2)
            )
            if self.is_throttled()
            else 0.0,
            "total_throttled_requests": self.total_throttled_requests,
        }


class ClientSLOThrottle:
    """
    Tracks per-client SLO violations and enforces rate-limiting when a client
    violates the SLO too frequently.

    Observe each request outcome::

        throttle = ClientSLOThrottle()
        throttle.observe("api-key-abc123", violated=True)

    Before admitting a new request::

        allowed, reason = throttle.check_allowed("api-key-abc123")
        if not allowed:
            return 429, reason
    """

    def __init__(self) -> None:
        self._clients: dict[str, _ClientRecord] = {}

    def _get_or_create(self, client_id: str) -> _ClientRecord:
        if client_id not in self._clients:
            self._clients[client_id] = _ClientRecord(client_id=client_id)
        return self._clients[client_id]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, client_id: str, *, violated: bool) -> None:
        """
        Record the outcome of a completed request for *client_id*.

        If *violated* is True, the violation is appended to the sliding
        window.  If the threshold is crossed, a throttle is applied
        immediately.
        """
        record = self._get_or_create(client_id)
        if not violated:
            return

        now = time.time()
        record.violation_timestamps.append(now)
        record.total_violations += 1

        # recent_violations() prunes the window internally before counting.
        recent = record.recent_violations()
        if recent > _VIOLATION_THRESHOLD and not record.is_throttled():
            record.throttle_until = now + _THROTTLE_DURATION_S
            logger.warning(
                "ClientSLOThrottle: throttling client={} (violations={} in {}s) for {}s",
                client_id,
                recent,
                _VIOLATION_WINDOW_S,
                _THROTTLE_DURATION_S,
            )

    def check_allowed(self, client_id: str) -> tuple[bool, str]:
        """
        Return (True, "ok") if the client may proceed, or
        (False, reason) if the request should be rejected / rate-limited.

        Throttled clients may send at most 1 request every
        ``_THROTTLE_MIN_GAP_S`` seconds.
        """
        if client_id not in self._clients:
            return True, "ok"

        record = self._clients[client_id]

        if not record.is_throttled():
            return True, "ok"

        now = time.time()
        gap = now - record.last_allowed_at
        if gap >= _THROTTLE_MIN_GAP_S:
            record.last_allowed_at = now
            logger.debug(
                "ClientSLOThrottle: throttled client={} allowed (gap={:.1f}s)",
                client_id,
                gap,
            )
            return True, "throttled_but_allowed"

        record.total_throttled_requests += 1
        remaining = round(_THROTTLE_MIN_GAP_S - gap, 1)
        reason = (
            f"SLO throttle active: retry in {remaining}s "
            f"(throttle expires in {record.to_dict()['throttle_remaining_s']}s)"
        )
        logger.debug(
            "ClientSLOThrottle: throttled client={} BLOCKED retry_in={}s",
            client_id,
            remaining,
        )
        return False, reason

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        record = self._clients.get(client_id)
        return record.to_dict() if record else None

    def all_clients(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._clients.values()]

    def throttled_clients(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._clients.values() if r.is_throttled()]

    def stats(self) -> dict[str, Any]:
        all_c = self.all_clients()
        return {
            "tracked_clients": len(all_c),
            "throttled_clients": sum(1 for c in all_c if c["throttled"]),
            "total_violations": sum(c["total_violations"] for c in all_c),
            "total_throttled_requests": sum(
                c["total_throttled_requests"] for c in all_c
            ),
            "violation_window_s": _VIOLATION_WINDOW_S,
            "violation_threshold": _VIOLATION_THRESHOLD,
            "throttle_duration_s": _THROTTLE_DURATION_S,
            "throttle_min_gap_s": _THROTTLE_MIN_GAP_S,
        }

    def reset(self, client_id: str) -> bool:
        """Clear throttle and violation history for a client (admin use)."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info("ClientSLOThrottle: reset client={}", client_id)
            return True
        return False


# Module-level singleton
CLIENT_SLO_THROTTLE = ClientSLOThrottle()
