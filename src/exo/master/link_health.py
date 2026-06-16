from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class LinkSample:
    timestamp: float
    latency_ms: float
    bytes_sent: int
    bytes_recv: int
    packet_loss_pct: float = 0.0


@dataclass
class NodeLinkStats:
    node_id: str
    samples: list[LinkSample] = field(default_factory=list)
    _max_samples: int = 120  # 2 minutes at 1Hz

    def add_sample(self, sample: LinkSample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self._max_samples:
            self.samples = self.samples[-self._max_samples :]

    @property
    def p50_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        lats = sorted(s.latency_ms for s in self.samples)
        return lats[len(lats) // 2]

    @property
    def p99_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        lats = sorted(s.latency_ms for s in self.samples)
        return lats[min(int(len(lats) * 0.99), len(lats) - 1)]

    @property
    def avg_throughput_mbps(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        total_bytes = sum(s.bytes_sent + s.bytes_recv for s in self.samples[-10:])
        elapsed = (
            self.samples[-1].timestamp - self.samples[-10].timestamp
            if len(self.samples) >= 10
            else 1.0
        )
        return (total_bytes / max(elapsed, 0.001)) / (1024 * 1024) * 8  # Mbps

    @property
    def health_status(self) -> str:
        if not self.samples:
            return "unknown"
        recent = self.samples[-3:]
        avg_lat = sum(s.latency_ms for s in recent) / len(recent)
        avg_loss = sum(s.packet_loss_pct for s in recent) / len(recent)
        if avg_loss > 5.0 or avg_lat > 50.0:
            return "degraded"
        if avg_loss > 1.0 or avg_lat > 20.0:
            return "warning"
        return "healthy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.health_status,
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "avg_throughput_mbps": round(self.avg_throughput_mbps, 2),
            "sample_count": len(self.samples),
            "last_sample_ts": self.samples[-1].timestamp if self.samples else 0.0,
        }


class LinkHealthMonitor:
    """
    Monitors TB4 link health by tracking round-trip latency and bytes
    transferred per node. Runs as a background task.
    """

    def __init__(self, probe_interval_seconds: float = 5.0) -> None:
        self._probe_interval = probe_interval_seconds
        self._stats: dict[str, NodeLinkStats] = {}
        self._running = False
        self._bytes_counters: dict[str, tuple[int, int]] = {}  # node_id -> (sent, recv)

    def register_node(self, node_id: str) -> None:
        if node_id not in self._stats:
            self._stats[node_id] = NodeLinkStats(node_id=node_id)
            self._bytes_counters[node_id] = (0, 0)

    def record_message(self, node_id: str, bytes_sent: int, bytes_recv: int) -> None:
        if node_id not in self._stats:
            self.register_node(node_id)
        sent, recv = self._bytes_counters[node_id]
        self._bytes_counters[node_id] = (sent + bytes_sent, recv + bytes_recv)

    async def probe_node(self, node_id: str, host: str) -> float:
        """Measure round-trip latency to a node via TCP connect."""
        start = time.monotonic()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, 52415), timeout=2.0
            )
            latency_ms = (time.monotonic() - start) * 1000
            writer.close()
            await writer.wait_closed()
            return latency_ms
        except Exception as exc:
            logger.debug(f"Link probe failed for {node_id}: {exc}")
            return 999.0

    async def run(self, node_hosts: dict[str, str]) -> None:
        """
        Background probe loop.
        node_hosts: {node_id: ip_or_hostname}
        """
        self._running = True
        for node_id in node_hosts:
            self.register_node(node_id)

        while self._running:
            for node_id, host in node_hosts.items():
                latency_ms = await self.probe_node(node_id, host)
                sent, recv = self._bytes_counters.get(node_id, (0, 0))
                sample = LinkSample(
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    bytes_sent=sent,
                    bytes_recv=recv,
                )
                if node_id not in self._stats:
                    self.register_node(node_id)
                self._stats[node_id].add_sample(sample)
                self._bytes_counters[node_id] = (0, 0)  # reset interval counters

                if latency_ms > 50.0:
                    logger.warning(
                        f"Link health: node={node_id} latency={latency_ms:.1f}ms (degraded)"
                    )

            await asyncio.sleep(self._probe_interval)

    def stop(self) -> None:
        self._running = False

    def get_stats(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._stats.values()]

    def get_node_stats(self, node_id: str) -> dict[str, Any] | None:
        if node_id in self._stats:
            return self._stats[node_id].to_dict()
        return None

    def prometheus_metrics(self) -> str:
        lines = [
            "# HELP exo_link_latency_p50_ms P50 round-trip latency per node (ms)",
            "# TYPE exo_link_latency_p50_ms gauge",
        ]
        for node_id, stats in self._stats.items():
            lines.append(
                f'exo_link_latency_p50_ms{{node="{node_id}"}} {stats.p50_latency_ms:.3f}'
            )
        lines += [
            "# HELP exo_link_latency_p99_ms P99 round-trip latency per node (ms)",
            "# TYPE exo_link_latency_p99_ms gauge",
        ]
        for node_id, stats in self._stats.items():
            lines.append(
                f'exo_link_latency_p99_ms{{node="{node_id}"}} {stats.p99_latency_ms:.3f}'
            )
        lines += [
            "# HELP exo_link_throughput_mbps Average throughput per node (Mbps)",
            "# TYPE exo_link_throughput_mbps gauge",
        ]
        for node_id, stats in self._stats.items():
            lines.append(
                f'exo_link_throughput_mbps{{node="{node_id}"}} {stats.avg_throughput_mbps:.2f}'
            )
        return "\n".join(lines) + "\n"


LINK_MONITOR = LinkHealthMonitor(probe_interval_seconds=5.0)
