"""
Cluster quorum check: enforces a minimum quorum of healthy workers before
accepting inference requests. Default quorum = 1 (any single healthy worker).
Configurable via EXO_QUORUM_MIN env var. If quorum not met, returns 503.
Also tracks quorum history for observability.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger

_DEFAULT_QUORUM = int(os.getenv("EXO_QUORUM_MIN", "1"))
_HISTORY_SIZE = 60  # last 60 samples


@dataclass
class QuorumSample:
    timestamp: float
    healthy_workers: int
    quorum_met: bool


class QuorumChecker:
    """
    update(healthy_workers): call from health check loop to update quorum state.
    quorum_met: True if current healthy workers >= required quorum.
    Logs transitions on quorum gain/loss.
    """

    def __init__(self, min_quorum: int = _DEFAULT_QUORUM) -> None:
        self._min_quorum = min_quorum
        self._healthy = 0
        self._met = False
        self._history: deque[QuorumSample] = deque(maxlen=_HISTORY_SIZE)
        self._lost_at: float | None = None
        logger.info(f"Quorum checker: min_quorum={min_quorum}")

    def update(self, healthy_workers: int) -> None:
        met = healthy_workers >= self._min_quorum
        if met != self._met:
            if met:
                lost_for = time.time() - self._lost_at if self._lost_at else 0
                logger.info(f"Quorum RESTORED: healthy={healthy_workers} (lost for {lost_for:.1f}s)")
                self._lost_at = None
            else:
                logger.warning(
                    f"Quorum LOST: healthy={healthy_workers} < required={self._min_quorum}"
                )
                self._lost_at = time.time()
        self._healthy = healthy_workers
        self._met = met
        self._history.append(QuorumSample(
            timestamp=time.time(),
            healthy_workers=healthy_workers,
            quorum_met=met,
        ))

    @property
    def quorum_met(self) -> bool:
        # quorum_min == 0 means local-only / bypass mode — always pass.
        return self._min_quorum == 0 or self._met

    @property
    def healthy_workers(self) -> int:
        return self._healthy

    def get_status(self) -> dict[str, Any]:
        recent = list(self._history)[-10:]
        return {
            "quorum_met": self._met,
            "healthy_workers": self._healthy,
            "min_quorum": self._min_quorum,
            "lost_at": self._lost_at,
            "recent_samples": [
                {"ts": s.timestamp, "healthy": s.healthy_workers, "met": s.quorum_met}
                for s in recent
            ],
        }

    @property
    def quorum_threshold(self) -> float:
        """The fractional threshold (default 0.5 for majority)."""
        return _DEFAULT_QUORUM_THRESHOLD


# Default fractional quorum threshold for the static helper.
_DEFAULT_QUORUM_THRESHOLD: float = 0.5


def quorum_met(total: int, alive: int, threshold: float = _DEFAULT_QUORUM_THRESHOLD) -> bool:
    """Return True if the fraction of alive nodes meets or exceeds threshold.

    Edge cases:
    - total == 0 → False (empty cluster, no quorum possible)
    - threshold == 0.0 → True for any alive >= 0 (but total > 0 required)
    """
    if total <= 0:
        return False
    fraction = alive / total
    return fraction >= threshold


QUORUM_CHECKER = QuorumChecker()
