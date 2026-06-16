"""bandwidth_throttle.py — token-bucket rate limiter for inter-node transfers.

Prevents TB4 link saturation during model weight transfers from starving
inference traffic.  Default limit: 30 Gbps (leaving headroom on 40 Gbps TB4).

Environment:
    EXO_BW_LIMIT_MBPS  — per-node default limit in Mbps (default: 30000.0)
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from loguru import logger

from exo.master.connection_pool import CONNECTION_POOL

_DEFAULT_LIMIT_MBPS: float = float(os.getenv("EXO_BW_LIMIT_MBPS", "30000.0"))
_MAX_WAIT_SECONDS: float = 5.0


@dataclass
class BandwidthBucket:
    node_id: str
    capacity_mbps: float  # bucket ceiling == refill rate
    current_tokens: float  # bytes available right now
    last_refill: float  # monotonic timestamp of last refill
    bytes_throttled: int = field(default=0)  # cumulative bytes that had to wait

    # capacity in bytes
    @property
    def _capacity_bytes(self) -> float:
        return self.capacity_mbps * 1e6 / 8


class BandwidthThrottler:
    """
    Token-bucket throttler, one bucket per remote node.

    Refill is continuous: tokens accumulate proportionally to elapsed time
    since last_refill, capped at bucket capacity.

    Usage:
        wait = BANDWIDTH_THROTTLER.consume(node_id, num_bytes)
        if wait > 0:
            await asyncio.sleep(wait)   # caller decides whether to actually sleep
    """

    def __init__(self, default_limit_mbps: float = _DEFAULT_LIMIT_MBPS) -> None:
        self._buckets: dict[str, BandwidthBucket] = {}
        self._default_limit_mbps: float = default_limit_mbps
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, node_id: str, limit_mbps: float | None = None) -> None:
        """Create a bucket for *node_id*.  Idempotent; existing bucket unchanged."""
        with self._lock:
            if node_id in self._buckets:
                return
            limit = limit_mbps if limit_mbps is not None else self._default_limit_mbps
            cap_bytes = limit * 1e6 / 8
            self._buckets[node_id] = BandwidthBucket(
                node_id=node_id,
                capacity_mbps=limit,
                current_tokens=cap_bytes,  # start full
                last_refill=time.monotonic(),
            )
            logger.info(
                f"[bandwidth_throttle] registered node={node_id} "
                f"limit={limit:.0f} Mbps ({cap_bytes / 1e6:.1f} MB/s)"
            )

    # ------------------------------------------------------------------ #
    # Core token-bucket logic                                              #
    # ------------------------------------------------------------------ #

    def consume(self, node_id: str, bytes_to_send: int) -> float:
        """
        Attempt to consume *bytes_to_send* from the bucket.

        Returns:
            0.0   — tokens available; caller may send immediately.
            >0.0  — seconds the caller should wait before sending.
                    Capped at _MAX_WAIT_SECONDS.

        Auto-registers unknown nodes with the default limit.
        """
        with self._lock:
            if node_id not in self._buckets:
                # Auto-register so callers don't have to pre-register.
                limit = self._default_limit_mbps
                cap_bytes = limit * 1e6 / 8
                self._buckets[node_id] = BandwidthBucket(
                    node_id=node_id,
                    capacity_mbps=limit,
                    current_tokens=cap_bytes,
                    last_refill=time.monotonic(),
                )

            bucket = self._buckets[node_id]
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            cap_bytes = bucket._capacity_bytes

            # Refill proportionally to elapsed time.
            bucket.current_tokens = min(
                cap_bytes,
                bucket.current_tokens + elapsed * cap_bytes,  # tokens/sec == capacity
            )
            bucket.last_refill = now

            if bucket.current_tokens >= bytes_to_send:
                bucket.current_tokens -= bytes_to_send
                return 0.0

            # Not enough tokens — compute wait.
            deficit = bytes_to_send - bucket.current_tokens
            wait = deficit / cap_bytes  # seconds
            wait = min(wait, _MAX_WAIT_SECONDS)
            bucket.bytes_throttled += bytes_to_send
            logger.debug(
                f"[bandwidth_throttle] throttling node={node_id} "
                f"bytes={bytes_to_send} wait={wait:.3f}s"
            )
            return wait

    # ------------------------------------------------------------------ #
    # Record transfer (wires into CONNECTION_POOL stats)                  #
    # ------------------------------------------------------------------ #

    def record_transfer(self, node_id: str, bytes_transferred: int) -> None:
        """
        Notify CONNECTION_POOL of bytes sent to *node_id*.

        Picks the active (last-used) connection for that node and calls
        record_send so pool bytes_sent counters stay accurate.
        """
        conn = CONNECTION_POOL.get_connection(node_id)
        if conn is not None:
            CONNECTION_POOL.record_send(conn.conn_id, bytes_transferred)

    # ------------------------------------------------------------------ #
    # Observability                                                        #
    # ------------------------------------------------------------------ #

    def utilization(self, node_id: str) -> float:
        """Return 0.0–1.0: fraction of capacity currently consumed (token depletion)."""
        with self._lock:
            bucket = self._buckets.get(node_id)
            if bucket is None:
                return 0.0
            cap = bucket._capacity_bytes
            if cap == 0:
                return 0.0
            return max(0.0, min(1.0, 1.0 - bucket.current_tokens / cap))

    def stats(self) -> dict[str, object]:
        """Per-node utilization + cumulative throttled bytes."""
        with self._lock:
            result: dict[str, object] = {}
            for node_id, bucket in self._buckets.items():
                cap = bucket._capacity_bytes
                util = (
                    max(0.0, min(1.0, 1.0 - bucket.current_tokens / cap))
                    if cap
                    else 0.0
                )
                result[node_id] = {
                    "limit_mbps": bucket.capacity_mbps,
                    "utilization": round(util, 4),
                    "current_tokens_bytes": int(bucket.current_tokens),
                    "bytes_throttled": bucket.bytes_throttled,
                }
            return result

    def to_prometheus(self) -> str:
        """Emit exo_bw_utilization gauge per node (0.0–1.0)."""
        lines = [
            "# HELP exo_bw_utilization Bandwidth utilization per node (0=idle, 1=saturated)",
            "# TYPE exo_bw_utilization gauge",
        ]
        with self._lock:
            for node_id, bucket in self._buckets.items():
                cap = bucket._capacity_bytes
                util = (
                    max(0.0, min(1.0, 1.0 - bucket.current_tokens / cap))
                    if cap
                    else 0.0
                )
                lines.append(f'exo_bw_utilization{{node="{node_id}"}} {util:.6f}')
        return "\n".join(lines) + "\n"


BANDWIDTH_THROTTLER = BandwidthThrottler()
