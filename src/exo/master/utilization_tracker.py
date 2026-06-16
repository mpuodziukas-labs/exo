"""Per-node CPU, memory, and GPU/Metal utilization tracker.

Samples local hardware every 5 s, retains a 2-minute rolling window (120 samples),
and exposes stats via Prometheus text format and JSON REST responses.

Apple Silicon GPU: reads the AGXAccelerator IORegistry entry for
"Device Utilization %" from the "PerformanceStatistics" dictionary.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False
    logger.debug("psutil not available — CPU/memory utilization reporting limited")


@dataclass
class UtilizationSample:
    node_id: str
    cpu_pct: float
    memory_pct: float
    gpu_pct: float | None  # None when GPU metrics are unavailable
    gpu_memory_gb: float | None  # None when GPU metrics are unavailable
    timestamp: float = field(default_factory=time.time)


class NodeUtilization:
    """Rolling window of utilization samples for a single node."""

    _WINDOW_SIZE: int = 120  # 2 min at 1 Hz (sampled every 5 s → ~24 non-None readings)
    _AVG_SAMPLES: int = 10  # number of most-recent samples used for averages

    def __init__(self, node_id: str) -> None:
        self.node_id: str = node_id
        self._samples: deque[UtilizationSample] = deque(maxlen=self._WINDOW_SIZE)

    def add_sample(self, sample: UtilizationSample) -> None:
        self._samples.append(sample)

    # ------------------------------------------------------------------ #
    #  Computed properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def _recent(self) -> list[UtilizationSample]:
        return list(self._samples)[-self._AVG_SAMPLES :]

    @property
    def avg_cpu_pct(self) -> float:
        recent = self._recent
        if not recent:
            return 0.0
        return sum(s.cpu_pct for s in recent) / len(recent)

    @property
    def avg_memory_pct(self) -> float:
        recent = self._recent
        if not recent:
            return 0.0
        return sum(s.memory_pct for s in recent) / len(recent)

    @property
    def avg_gpu_pct(self) -> float | None:
        recent = [s.gpu_pct for s in self._recent if s.gpu_pct is not None]
        if not recent:
            return None
        return sum(recent) / len(recent)

    @property
    def peak_cpu_pct(self) -> float:
        samples = list(self._samples)
        if not samples:
            return 0.0
        return max(s.cpu_pct for s in samples)

    def to_dict(self) -> dict[str, Any]:
        latest = self._samples[-1] if self._samples else None
        return {
            "node_id": self.node_id,
            "sample_count": len(self._samples),
            "avg_cpu_pct": round(self.avg_cpu_pct, 2),
            "avg_memory_pct": round(self.avg_memory_pct, 2),
            "avg_gpu_pct": round(self.avg_gpu_pct, 2)
            if self.avg_gpu_pct is not None
            else None,
            "peak_cpu_pct": round(self.peak_cpu_pct, 2),
            "latest": {
                "cpu_pct": round(latest.cpu_pct, 2),
                "memory_pct": round(latest.memory_pct, 2),
                "gpu_pct": round(latest.gpu_pct, 2)
                if latest.gpu_pct is not None
                else None,
                "gpu_memory_gb": round(latest.gpu_memory_gb, 2)
                if latest.gpu_memory_gb is not None
                else None,
                "timestamp": latest.timestamp,
            }
            if latest
            else None,
        }


class UtilizationTracker:
    """Cluster-wide utilization tracker.

    Records samples for each node ID.  Local sampling uses psutil for CPU/memory
    and an ``ioreg`` subprocess for Apple Silicon GPU metrics.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeUtilization] = {}

    # ------------------------------------------------------------------ #
    #  Sampling                                                            #
    # ------------------------------------------------------------------ #

    def sample_local(self, node_id: str) -> UtilizationSample:
        """Collect a single hardware sample for *node_id* on the current machine."""
        cpu_pct = self._read_cpu()
        memory_pct = self._read_memory()
        gpu_pct, gpu_memory_gb = self._read_gpu_metal()

        return UtilizationSample(
            node_id=node_id,
            cpu_pct=cpu_pct,
            memory_pct=memory_pct,
            gpu_pct=gpu_pct,
            gpu_memory_gb=gpu_memory_gb,
        )

    @staticmethod
    def _read_cpu() -> float:
        if not _HAS_PSUTIL:
            return 0.0
        try:
            return _psutil.cpu_percent(interval=0.1)
        except Exception as exc:
            logger.warning(f"[utilization] CPU sample failed: {exc}")
            return 0.0

    @staticmethod
    def _read_memory() -> float:
        if not _HAS_PSUTIL:
            return 0.0
        try:
            return _psutil.virtual_memory().percent
        except Exception as exc:
            logger.warning(f"[utilization] memory sample failed: {exc}")
            return 0.0

    @staticmethod
    def _read_gpu_metal() -> tuple[float | None, float | None]:
        """Parse Apple Silicon GPU utilization via ioreg AGXAccelerator.

        Returns (gpu_utilization_pct, gpu_allocated_memory_gb) or (None, None)
        when the data is unavailable (non-macOS, missing AGXAccelerator, timeout).
        """
        if sys.platform != "darwin":
            return None, None

        try:
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug(f"[utilization] ioreg unavailable: {exc}")
            return None, None

        if result.returncode != 0 or not result.stdout:
            return None, None

        # The output contains a line like:
        #   "PerformanceStatistics" = {"Device Utilization %"=42,"In use system memory"=1234567}
        # We find the PerformanceStatistics value blob and extract fields from it.
        perf_match = re.search(
            r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', result.stdout
        )
        if not perf_match:
            return None, None

        stats_blob = perf_match.group(1)

        gpu_pct: float | None = None
        gpu_memory_gb: float | None = None

        util_match = re.search(
            r'"Device Utilization %"\s*=\s*([0-9]+(?:\.[0-9]+)?)', stats_blob
        )
        if util_match:
            gpu_pct = float(util_match.group(1))

        # "In use system memory" is reported in bytes (unified memory used by GPU)
        mem_match = re.search(r'"In use system memory"\s*=\s*([0-9]+)', stats_blob)
        if mem_match:
            gpu_memory_gb = int(mem_match.group(1)) / 1e9

        return gpu_pct, gpu_memory_gb

    # ------------------------------------------------------------------ #
    #  Recording / retrieval                                               #
    # ------------------------------------------------------------------ #

    def record(self, sample: UtilizationSample) -> None:
        node = self._nodes.setdefault(sample.node_id, NodeUtilization(sample.node_id))
        node.add_sample(sample)

    def get_node(self, node_id: str) -> NodeUtilization | None:
        return self._nodes.get(node_id)

    def all_stats(self) -> list[dict[str, Any]]:
        return [node.to_dict() for node in self._nodes.values()]

    # ------------------------------------------------------------------ #
    #  Prometheus export                                                   #
    # ------------------------------------------------------------------ #

    def to_prometheus(self) -> str:
        if not self._nodes:
            return ""

        lines: list[str] = [
            "# HELP exo_node_cpu_pct Average CPU utilisation over last 10 samples (percent)",
            "# TYPE exo_node_cpu_pct gauge",
        ]
        for node in self._nodes.values():
            lines.append(
                f'exo_node_cpu_pct{{node_id="{node.node_id}"}} {node.avg_cpu_pct:.2f}'
            )

        lines += [
            "# HELP exo_node_memory_pct Average memory utilisation over last 10 samples (percent)",
            "# TYPE exo_node_memory_pct gauge",
        ]
        for node in self._nodes.values():
            lines.append(
                f'exo_node_memory_pct{{node_id="{node.node_id}"}} {node.avg_memory_pct:.2f}'
            )

        lines += [
            "# HELP exo_node_peak_cpu_pct Peak CPU utilisation in rolling window (percent)",
            "# TYPE exo_node_peak_cpu_pct gauge",
        ]
        for node in self._nodes.values():
            lines.append(
                f'exo_node_peak_cpu_pct{{node_id="{node.node_id}"}} {node.peak_cpu_pct:.2f}'
            )

        # GPU gauge — only emit when at least one sample has a value
        gpu_lines: list[str] = []
        for node in self._nodes.values():
            avg_gpu = node.avg_gpu_pct
            if avg_gpu is not None:
                gpu_lines.append(
                    f'exo_node_gpu_pct{{node_id="{node.node_id}"}} {avg_gpu:.2f}'
                )
        if gpu_lines:
            lines += [
                "# HELP exo_node_gpu_pct Average GPU utilisation over last 10 samples (percent)",
                "# TYPE exo_node_gpu_pct gauge",
            ]
            lines.extend(gpu_lines)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    #  Background loop                                                     #
    # ------------------------------------------------------------------ #

    async def run_sample_loop(self, node_id: str) -> None:
        """Sample local hardware every 5 s and record the result."""
        logger.info(f"[utilization] starting sample loop for node_id={node_id!r}")
        while True:
            try:
                sample = self.sample_local(node_id)
                self.record(sample)
                logger.debug(
                    f"[utilization] node={node_id!r} "
                    f"cpu={sample.cpu_pct:.1f}% "
                    f"mem={sample.memory_pct:.1f}% "
                    f"gpu={sample.gpu_pct}"
                )
            except Exception as exc:
                logger.warning(f"[utilization] sample loop error: {exc}")
            await asyncio.sleep(5)


UTILIZATION_TRACKER = UtilizationTracker()
