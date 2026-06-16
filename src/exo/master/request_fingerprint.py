"""
Request fingerprinting for abuse detection: generates a fingerprint for each
request based on prompt content hash + client IP + model. Detects repeated
identical requests (scraping/abuse) and applies backpressure.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_WINDOW = 300.0  # 5-minute rolling window
_REPEAT_THRESHOLD = 10  # same fingerprint N times = suspicious


@dataclass
class FingerprintRecord:
    fingerprint: str
    client_ip: str
    model_id: str
    count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    flagged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint[:16],
            "client_ip": self.client_ip[:20],
            "model_id": self.model_id,
            "count": self.count,
            "flagged": self.flagged,
            "age_s": round(time.time() - self.first_seen, 1),
        }


class RequestFingerprintDetector:
    """
    fingerprint(prompt, client_ip, model_id): returns fingerprint string.
    record(fingerprint, client_ip, model_id): tracks and detects abuse.
    Returns True if this fingerprint is flagged (too many repeats).
    """

    def __init__(self) -> None:
        self._records: dict[str, FingerprintRecord] = {}
        self._timestamps: dict[str, deque[float]] = defaultdict(lambda: deque())
        self._flagged_count = 0

    @staticmethod
    def fingerprint(prompt: str, client_ip: str, model_id: str) -> str:
        content = f"{prompt[:200]}|{client_ip}|{model_id}"
        return hashlib.sha256(content.encode()).hexdigest()

    def record(self, fp: str, client_ip: str, model_id: str) -> bool:
        now = time.time()
        cutoff = now - _WINDOW
        ts = self._timestamps[fp]
        while ts and ts[0] < cutoff:
            ts.popleft()
        ts.append(now)

        if fp not in self._records:
            self._records[fp] = FingerprintRecord(
                fingerprint=fp, client_ip=client_ip, model_id=model_id
            )
        rec = self._records[fp]
        rec.count = len(ts)
        rec.last_seen = now

        if rec.count >= _REPEAT_THRESHOLD and not rec.flagged:
            rec.flagged = True
            self._flagged_count += 1
            logger.warning(
                f"FingerprintDetector: abuse detected ip={client_ip} "
                f"model={model_id} count={rec.count} fp={fp[:16]}"
            )
        return rec.flagged

    def get_flagged(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records.values() if r.flagged]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_fingerprints": len(self._records),
            "flagged_count": self._flagged_count,
            "flagged": self.get_flagged(),
        }


FINGERPRINT_DETECTOR = RequestFingerprintDetector()
