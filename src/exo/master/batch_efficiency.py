"""
Continuous batching efficiency tracker: measures how well the batch processor
fills each batch (fill_ratio = actual_tokens / max_batch_tokens).
Tracks padding waste, batch sizes, and throughput efficiency.
Surfaces these as Prometheus metrics and API endpoint.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class BatchSample:
    batch_size: int          # number of sequences in batch
    total_tokens: int        # actual tokens processed
    padded_tokens: int       # padding added
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def fill_ratio(self) -> float:
        total = self.total_tokens + self.padded_tokens
        return self.total_tokens / max(total, 1)

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / max(self.duration_ms / 1000, 0.001)


class BatchEfficiencyTracker:
    """
    record(batch_size, total_tokens, padded_tokens, duration_ms): tracks a batch.
    get_stats(): returns efficiency metrics.
    prometheus_metrics(): returns Prometheus text format.
    """

    def __init__(self) -> None:
        self._samples: deque[BatchSample] = deque(maxlen=500)
        self._total_batches = 0
        self._total_tokens = 0
        self._total_padded = 0

    def record(
        self,
        batch_size: int,
        total_tokens: int,
        padded_tokens: int = 0,
        duration_ms: float = 1.0,
    ) -> None:
        sample = BatchSample(
            batch_size=batch_size,
            total_tokens=total_tokens,
            padded_tokens=padded_tokens,
            duration_ms=duration_ms,
        )
        self._samples.append(sample)
        self._total_batches += 1
        self._total_tokens += total_tokens
        self._total_padded += padded_tokens

        if self._total_batches % 100 == 0:
            avg_fill = self._avg_fill_ratio()
            logger.debug(f"BatchEfficiency: avg_fill={avg_fill:.2%} total_batches={self._total_batches}")

    def _avg_fill_ratio(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.fill_ratio for s in self._samples) / len(self._samples)

    def _avg_batch_size(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.batch_size for s in self._samples) / len(self._samples)

    def _avg_tps(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.tokens_per_sec for s in self._samples) / len(self._samples)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_batches": self._total_batches,
            "total_tokens_processed": self._total_tokens,
            "total_padding_waste": self._total_padded,
            "avg_fill_ratio_pct": round(self._avg_fill_ratio() * 100, 2),
            "avg_batch_size": round(self._avg_batch_size(), 2),
            "avg_tokens_per_sec": round(self._avg_tps(), 2),
            "padding_waste_pct": round(
                100 * self._total_padded / max(self._total_tokens + self._total_padded, 1), 2
            ),
        }

    def prometheus_metrics(self) -> str:
        stats = self.get_stats()
        lines = [
            "# HELP exo_batch_fill_ratio_pct Average batch fill ratio",
            "# TYPE exo_batch_fill_ratio_pct gauge",
            f'exo_batch_fill_ratio_pct {stats["avg_fill_ratio_pct"]}',
            "# HELP exo_batch_padding_waste_pct Percentage of tokens that are padding",
            "# TYPE exo_batch_padding_waste_pct gauge",
            f'exo_batch_padding_waste_pct {stats["padding_waste_pct"]}',
            "# HELP exo_batch_avg_size Average sequences per batch",
            "# TYPE exo_batch_avg_size gauge",
            f'exo_batch_avg_size {stats["avg_batch_size"]}',
        ]
        return "\n".join(lines) + "\n"


BATCH_EFFICIENCY = BatchEfficiencyTracker()
