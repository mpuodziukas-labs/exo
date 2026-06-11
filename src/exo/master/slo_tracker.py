from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger

_TTFT_SLO_MS: float = float(os.getenv("EXO_TTFT_SLO_MS", "500"))
_WINDOW = 1000


@dataclass
class SloSample:
    ttft_ms: float
    total_ms: float
    tokens: int
    timestamp: float = field(default_factory=time.time)


class ClientSloStats:
    def __init__(self, client_key: str) -> None:
        self.client_key = client_key
        self.samples: deque[SloSample] = deque(maxlen=_WINDOW)
        self.violations: int = 0
        self._lock = Lock()

    def add_sample(self, ttft_ms: float, total_ms: float, tokens: int) -> None:
        with self._lock:
            self.samples.append(SloSample(ttft_ms=ttft_ms, total_ms=total_ms, tokens=tokens))

    def _sorted_ttft(self) -> list[float]:
        return sorted(s.ttft_ms for s in self.samples)

    def _sorted_total(self) -> list[float]:
        return sorted(s.total_ms for s in self.samples)

    @property
    def p50_ttft_ms(self) -> float:
        vals = self._sorted_ttft()
        if not vals:
            return 0.0
        return vals[int(len(vals) * 0.50)]

    @property
    def p99_ttft_ms(self) -> float:
        vals = self._sorted_ttft()
        if not vals:
            return 0.0
        return vals[min(int(len(vals) * 0.99), len(vals) - 1)]

    @property
    def p50_total_ms(self) -> float:
        vals = self._sorted_total()
        if not vals:
            return 0.0
        return vals[int(len(vals) * 0.50)]

    @property
    def p99_total_ms(self) -> float:
        vals = self._sorted_total()
        if not vals:
            return 0.0
        return vals[min(int(len(vals) * 0.99), len(vals) - 1)]

    @property
    def avg_tokens_per_request(self) -> float:
        with self._lock:
            if not self.samples:
                return 0.0
            return sum(s.tokens for s in self.samples) / len(self.samples)

    def check_slo(self, ttft_slo_ms: float = _TTFT_SLO_MS) -> bool:
        p99 = self.p99_ttft_ms
        if p99 > ttft_slo_ms:
            with self._lock:
                self.violations += 1
            logger.warning(
                f"SLO violation client={self.client_key} p99_ttft={p99:.1f}ms slo={ttft_slo_ms:.1f}ms"
            )
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_key": self.client_key,
            "sample_count": len(self.samples),
            "p50_ttft_ms": round(self.p50_ttft_ms, 2),
            "p99_ttft_ms": round(self.p99_ttft_ms, 2),
            "p50_total_ms": round(self.p50_total_ms, 2),
            "p99_total_ms": round(self.p99_total_ms, 2),
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 2),
            "violations": self.violations,
        }


class SloTracker:
    def __init__(self) -> None:
        self._clients: dict[str, ClientSloStats] = {}
        self._lock = Lock()

    def record(self, client_key: str, ttft_ms: float, total_ms: float, tokens: int) -> None:
        with self._lock:
            if client_key not in self._clients:
                self._clients[client_key] = ClientSloStats(client_key)
            stats = self._clients[client_key]
        stats.add_sample(ttft_ms, total_ms, tokens)

    def get(self, client_key: str) -> ClientSloStats | None:
        return self._clients.get(client_key)

    def all_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            clients = list(self._clients.values())
        return [c.to_dict() for c in clients]

    def violating_clients(self, ttft_slo_ms: float = _TTFT_SLO_MS) -> list[str]:
        with self._lock:
            clients = list(self._clients.values())
        return [c.client_key for c in clients if not c.check_slo(ttft_slo_ms)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            clients = list(self._clients.values())
        if not clients:
            return {
                "client_count": 0,
                "global_p50_ttft_ms": 0.0,
                "global_p99_ttft_ms": 0.0,
                "global_p50_total_ms": 0.0,
                "global_p99_total_ms": 0.0,
                "total_violations": 0,
                "slo_ms": _TTFT_SLO_MS,
            }
        all_ttft = sorted(s.ttft_ms for c in clients for s in c.samples)
        all_total = sorted(s.total_ms for c in clients for s in c.samples)
        n_ttft = len(all_ttft)
        n_total = len(all_total)
        return {
            "client_count": len(clients),
            "global_p50_ttft_ms": round(all_ttft[int(n_ttft * 0.50)] if n_ttft else 0.0, 2),
            "global_p99_ttft_ms": round(all_ttft[min(int(n_ttft * 0.99), n_ttft - 1)] if n_ttft else 0.0, 2),
            "global_p50_total_ms": round(all_total[int(n_total * 0.50)] if n_total else 0.0, 2),
            "global_p99_total_ms": round(all_total[min(int(n_total * 0.99), n_total - 1)] if n_total else 0.0, 2),
            "total_violations": sum(c.violations for c in clients),
            "slo_ms": _TTFT_SLO_MS,
        }


SLO_TRACKER = SloTracker()
