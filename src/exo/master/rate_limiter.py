from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter for a single client.

    capacity: max burst size (tokens)
    refill_rate: tokens added per second
    tokens: current token count
    """

    capacity: float
    refill_rate: float
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, cost: float = 1.0) -> bool:
        """Attempt to consume `cost` tokens. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    @property
    def wait_time_seconds(self) -> float:
        """Seconds until one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / max(self.refill_rate, 1e-10)


class RateLimiter:
    """
    Per-client token bucket rate limiter.

    Clients are identified by:
    1. X-API-Key header (if present)
    2. Client IP address
    3. "anonymous" bucket (shared for unauthenticated clients)

    Configuration via env vars:
    - EXO_RATE_LIMIT_RPM: requests per minute per client (default 60)
    - EXO_RATE_LIMIT_BURST: burst capacity multiplier (default 1.5x)
    - EXO_RATE_LIMIT_ENABLED: "1" to enable (default "1")
    - EXO_RATE_LIMIT_ANONYMOUS_RPM: RPM for anonymous clients (default 10)
    """

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = defaultdict(self._new_bucket)
        self._lock = Lock()
        self._rejected_total: int = 0
        self._allowed_total: int = 0

    @property
    def enabled(self) -> bool:
        return os.getenv("EXO_RATE_LIMIT_ENABLED", "1") == "1"

    def _rpm_for_client(self, client_id: str) -> float:
        if client_id == "anonymous":
            return float(os.getenv("EXO_RATE_LIMIT_ANONYMOUS_RPM", "10"))
        return float(os.getenv("EXO_RATE_LIMIT_RPM", "60"))

    def _new_bucket(self) -> TokenBucket:
        rpm = float(os.getenv("EXO_RATE_LIMIT_RPM", "60"))
        burst = float(os.getenv("EXO_RATE_LIMIT_BURST", "1.5"))
        refill_rate = rpm / 60.0  # tokens per second
        capacity = rpm * burst / 60.0
        return TokenBucket(capacity=capacity, refill_rate=refill_rate, tokens=capacity)

    def _get_or_create_bucket(self, client_id: str) -> TokenBucket:
        if client_id not in self._buckets:
            rpm = self._rpm_for_client(client_id)
            burst = float(os.getenv("EXO_RATE_LIMIT_BURST", "1.5"))
            refill_rate = rpm / 60.0
            capacity = rpm * burst / 60.0
            self._buckets[client_id] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate,
                tokens=capacity,
            )
        return self._buckets[client_id]

    def check(self, client_id: str, cost: float = 1.0) -> tuple[bool, float]:
        """
        Check if client_id is within rate limit.
        Returns (allowed, retry_after_seconds).
        """
        if not self.enabled:
            return True, 0.0

        with self._lock:
            bucket = self._get_or_create_bucket(client_id)
            allowed = bucket.consume(cost)
            wait = 0.0 if allowed else bucket.wait_time_seconds

            if allowed:
                self._allowed_total += 1
            else:
                self._rejected_total += 1
                logger.warning(
                    f"Rate limit exceeded: client={client_id} "
                    f"retry_after={wait:.1f}s"
                )

        return allowed, wait

    def client_id_from_request(self, api_key: str | None, client_ip: str | None) -> str:
        """Determine the client identifier for rate limiting."""
        if api_key:
            return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:8]
        if client_ip:
            return f"ip:{client_ip}"
        return "anonymous"

    def reset_client(self, client_id: str) -> None:
        with self._lock:
            self._buckets.pop(client_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._allowed_total + self._rejected_total
            return {
                "enabled": self.enabled,
                "active_clients": len(self._buckets),
                "allowed_total": self._allowed_total,
                "rejected_total": self._rejected_total,
                "rejection_rate": round(self._rejected_total / max(total, 1), 4),
                "rpm_per_client": float(os.getenv("EXO_RATE_LIMIT_RPM", "60")),
                "anonymous_rpm": float(os.getenv("EXO_RATE_LIMIT_ANONYMOUS_RPM", "10")),
            }

    def prometheus_metrics(self) -> str:
        s = self.stats()
        return (
            "# HELP exo_rate_limit_rejected_total Requests rejected by rate limiter\n"
            "# TYPE exo_rate_limit_rejected_total counter\n"
            f"exo_rate_limit_rejected_total {s['rejected_total']}\n"
            "# HELP exo_rate_limit_active_clients Active rate-limited clients\n"
            "# TYPE exo_rate_limit_active_clients gauge\n"
            f"exo_rate_limit_active_clients {s['active_clients']}\n"
        )


RATE_LIMITER = RateLimiter()
