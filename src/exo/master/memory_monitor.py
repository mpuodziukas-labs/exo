from __future__ import annotations

import os
import resource
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

from loguru import logger


@dataclass
class MemorySnapshot:
    timestamp: float
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_used_gb: float
    ram_pressure: float  # 0.0-1.0
    swap_pressure: float  # 0.0-1.0


class MemoryPressureMonitor:
    """
    Tracks system RAM and swap usage. Emits warnings at thresholds and
    signals the admission controller to tighten limits under pressure.

    On macOS (Apple Silicon), also reads memory-pressure via vm_stat.
    Thresholds:
    - WARNING:  ram_pressure > 0.85
    - CRITICAL: ram_pressure > 0.92 -> triggers KV cache eviction requests
    - FATAL:    ram_pressure > 0.97 -> emergency shutdown signal
    """

    WARNING_THRESHOLD = float(os.getenv("EXO_MEM_WARNING_THRESHOLD", "0.85"))
    CRITICAL_THRESHOLD = float(os.getenv("EXO_MEM_CRITICAL_THRESHOLD", "0.92"))
    FATAL_THRESHOLD = float(os.getenv("EXO_MEM_FATAL_THRESHOLD", "0.97"))

    # Rolling window capacity for pressure history
    _WINDOW_SIZE = 120  # ~20 minutes at 10s sample interval

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: MemorySnapshot | None = None
        self._warning_count: int = 0
        self._critical_count: int = 0
        self._pressure_window: deque[float] = deque(maxlen=self._WINDOW_SIZE)
        # Throttle: only emit >0.75 WARNING once per 30s
        self._last_warning_logged_at: float = 0.0

    def sample(self) -> MemorySnapshot:
        try:
            import psutil
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            snap = MemorySnapshot(
                timestamp=time.time(),
                ram_total_gb=vm.total / 1e9,
                ram_used_gb=vm.used / 1e9,
                ram_available_gb=vm.available / 1e9,
                swap_total_gb=sw.total / 1e9,
                swap_used_gb=sw.used / 1e9,
                ram_pressure=vm.percent / 100.0,
                swap_pressure=(sw.used / max(sw.total, 1)),
            )
        except ImportError:
            # psutil not available — use macOS sysctl fallback
            snap = self._sample_macos()

        with self._lock:
            self._latest = snap
            self._pressure_window.append(snap.ram_pressure)
            if snap.ram_pressure > self.FATAL_THRESHOLD:
                logger.critical(
                    f"Memory FATAL: {snap.ram_pressure:.1%} RAM used "
                    f"({snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB)"
                )
            elif snap.ram_pressure > self.CRITICAL_THRESHOLD:
                self._critical_count += 1
                logger.warning(
                    f"Memory CRITICAL: {snap.ram_pressure:.1%} RAM used — "
                    f"KV cache eviction recommended"
                )
            elif snap.ram_pressure > self.WARNING_THRESHOLD:
                self._warning_count += 1
                logger.warning(
                    f"Memory WARNING: {snap.ram_pressure:.1%} RAM used"
                )
        return snap

    def _sample_macos(self) -> MemorySnapshot:
        """macOS fallback using sysctl without psutil."""
        import subprocess
        total_gb = 0.0
        used_gb = 0.0
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2
            )
            total_gb = int(result.stdout.strip()) / 1e9
            # vm_stat for page counts
            vm_result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2
            )
            page_size = resource.getpagesize()  # 16 KB on Apple Silicon, 4 KB on x86
            free_pages = 0
            for line in vm_result.stdout.splitlines():
                if "Pages free" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                    break
            available_gb = (free_pages * page_size) / 1e9
            used_gb = max(0.0, total_gb - available_gb)
        except Exception as exc:
            logger.warning(f"Memory sample failed: {exc}")

        pressure = used_gb / max(total_gb, 1.0)
        return MemorySnapshot(
            timestamp=time.time(),
            ram_total_gb=total_gb,
            ram_used_gb=used_gb,
            ram_available_gb=max(0.0, total_gb - used_gb),
            swap_total_gb=0.0,
            swap_used_gb=0.0,
            ram_pressure=pressure,
            swap_pressure=0.0,
        )

    @property
    def current_pressure(self) -> float:
        with self._lock:
            return self._latest.ram_pressure if self._latest else 0.0

    def is_critical(self) -> bool:
        return self.current_pressure > self.CRITICAL_THRESHOLD

    def is_fatal(self) -> bool:
        return self.current_pressure > self.FATAL_THRESHOLD

    def record_pressure(self, value: float) -> None:
        """Manually record a pressure sample into the rolling window (useful for testing)."""
        with self._lock:
            self._pressure_window.append(float(value))

    def pressure_level(self) -> str:
        """Return human-readable level: 'normal' | 'warning' | 'critical' | 'oom'."""
        p = self.current_pressure
        if p > self.FATAL_THRESHOLD:
            return "oom"
        if p > self.CRITICAL_THRESHOLD:
            return "critical"
        if p > self.WARNING_THRESHOLD:
            return "warning"
        return "normal"

    def p99_pressure(self) -> float:
        """99th-percentile pressure over the rolling window. Returns 0.0 if window empty."""
        with self._lock:
            if not self._pressure_window:
                return 0.0
            sorted_window = sorted(self._pressure_window)
            idx = max(0, int(len(sorted_window) * 0.99) - 1)
            return sorted_window[idx]

    def above_threshold(self, threshold: float) -> bool:
        """Return True when current_pressure exceeds threshold."""
        return self.current_pressure > threshold

    def pressure_trend(self) -> str:
        """
        Compute trend over the last 10 samples in the rolling window.
        Returns 'increasing' | 'stable' | 'decreasing'.
        A change of >=2pp is required to call a trend directional.
        """
        with self._lock:
            window = list(self._pressure_window)
        if len(window) < 2:
            return "stable"
        half = max(1, len(window) // 2)
        first_half_avg = sum(window[:half]) / half
        second_half_avg = sum(window[half:]) / max(1, len(window) - half)
        delta = second_half_avg - first_half_avg
        if delta > 0.02:
            return "increasing"
        if delta < -0.02:
            return "decreasing"
        return "stable"

    def check_oom_pressure_warning(self) -> bool:
        """
        Emit a WARNING log if pressure > 0.75, throttled to once per 30s.
        Returns True if the warning was emitted this call.
        """
        p = self.current_pressure
        if p > 0.75:
            now = time.time()
            with self._lock:
                if (now - self._last_warning_logged_at) >= 30.0:
                    self._last_warning_logged_at = now
                    logger.warning(
                        f"[OOM guard] Memory pressure WARNING: {p:.1%} > 75% — "
                        "admitting request but approaching limit"
                    )
                    return True
        return False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            if self._latest is None:
                return {"status": "not_sampled"}
            snap = self._latest
            level = "ok"
            if snap.ram_pressure > self.FATAL_THRESHOLD:
                level = "fatal"
            elif snap.ram_pressure > self.CRITICAL_THRESHOLD:
                level = "critical"
            elif snap.ram_pressure > self.WARNING_THRESHOLD:
                level = "warning"
            return {
                "level": level,
                "ram_pressure": round(snap.ram_pressure, 4),
                "ram_used_gb": round(snap.ram_used_gb, 2),
                "ram_total_gb": round(snap.ram_total_gb, 2),
                "ram_available_gb": round(snap.ram_available_gb, 2),
                "swap_pressure": round(snap.swap_pressure, 4),
                "swap_used_gb": round(snap.swap_used_gb, 2),
                "warning_events": self._warning_count,
                "critical_events": self._critical_count,
                "thresholds": {
                    "warning": self.WARNING_THRESHOLD,
                    "critical": self.CRITICAL_THRESHOLD,
                    "fatal": self.FATAL_THRESHOLD,
                },
            }

    def prometheus_metrics(self) -> str:
        s = self.stats()
        if s.get("status") == "not_sampled":
            return ""
        return (
            "# HELP exo_memory_ram_pressure Current RAM pressure 0.0-1.0\n"
            "# TYPE exo_memory_ram_pressure gauge\n"
            f"exo_memory_ram_pressure {s['ram_pressure']:.4f}\n"
            "# HELP exo_memory_ram_available_gb Available RAM in GB\n"
            "# TYPE exo_memory_ram_available_gb gauge\n"
            f"exo_memory_ram_available_gb {s['ram_available_gb']:.2f}\n"
            "# HELP exo_memory_swap_pressure Current swap pressure 0.0-1.0\n"
            "# TYPE exo_memory_swap_pressure gauge\n"
            f"exo_memory_swap_pressure {s.get('swap_pressure', 0):.4f}\n"
        )


MEMORY_MONITOR = MemoryPressureMonitor()
