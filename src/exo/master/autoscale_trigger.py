"""
autoscale_trigger.py — Production-grade autoscale signal emitter for exo.

Evaluates queue depth, SLO p99-TTFT, and admission-controller rejection rate
every 30 s.  When a scale-up or scale-down condition is satisfied the module
fires an SSE event via EVENT_BUS and records a ScaleEvent in a ring-buffer
history.  The actual scaling action (spinning up / tearing down nodes) is
performed by external orchestration; this module only fires the signal.

Environment variables
---------------------
EXO_AUTOSCALE_COOLDOWN_SECONDS   float   120   min gap between scale events
EXO_SCALE_UP_QUEUE_DEPTH         int     20    queue.size() threshold for scale-up
EXO_AUTOSCALE_CHECK_INTERVAL     float   30    loop cadence in seconds
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, cast

import anyio
from loguru import logger

from exo.master.admission_control import ADMISSION_CONTROLLER
from exo.master.event_stream import emit as emit_cluster_event
from exo.master.heartbeat_monitor import HEARTBEAT_MONITOR
from exo.master.metrics import METRICS
from exo.master.priority_queue import PRIORITY_QUEUE
from exo.master.slo_tracker import SLO_TRACKER

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ScaleDirection = Literal["up", "down", "none"]

_COOLDOWN: float = float(os.getenv("EXO_AUTOSCALE_COOLDOWN_SECONDS", "120.0"))
_SCALE_UP_QUEUE_DEPTH: int = int(os.getenv("EXO_SCALE_UP_QUEUE_DEPTH", "20"))
_CHECK_INTERVAL: float = float(os.getenv("EXO_AUTOSCALE_CHECK_INTERVAL", "30.0"))

# Latency thresholds (ms)
_P99_SCALE_UP_MS: float = 1000.0
_P99_SCALE_DOWN_MS: float = 200.0

# Consecutive-check requirements
_UP_CONSECUTIVE: int = 3
_DOWN_CONSECUTIVE: int = 5

# Admission rejection-rate window / threshold
_REJECTION_RATE_WINDOW_S: float = 60.0
_REJECTION_RATE_THRESHOLD: float = 0.10  # 10 %


@dataclass
class ScaleEvent:
    direction: ScaleDirection
    reason: str
    current_nodes: int
    recommended_nodes: int
    triggered_at: float
    metrics_snapshot: dict[str, object]


# ---------------------------------------------------------------------------
# AutoscaleTrigger
# ---------------------------------------------------------------------------


class AutoscaleTrigger:
    """
    Stateful autoscale signal emitter.

    Maintains sliding counters for scale-up / scale-down conditions and emits
    SSE events through EVENT_BUS when thresholds are crossed.  A configurable
    cooldown prevents thrashing.
    """

    def __init__(
        self,
        cooldown_seconds: float = _COOLDOWN,
        scale_up_queue_depth: int = _SCALE_UP_QUEUE_DEPTH,
        check_interval: float = _CHECK_INTERVAL,
    ) -> None:
        self._cooldown_seconds: float = cooldown_seconds
        self._scale_up_queue_depth: int = scale_up_queue_depth
        self._check_interval: float = check_interval

        self._events: deque[ScaleEvent] = deque(maxlen=100)
        self._last_scale_at: float = 0.0

        # Consecutive-check counters — scale-up conditions (any one triggers)
        self._up_queue_streak: int = 0
        self._up_latency_streak: int = 0

        # Consecutive-check counters — scale-down conditions (ALL must be true)
        self._down_queue_streak: int = 0
        self._down_latency_streak: int = 0

        # Snapshot of admission stats from last check (for rejection-rate calc)
        self._last_rejection_snapshot: dict[str, int] = {
            "rejected": 0,
            "total": 0,
            "ts": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_nodes(self) -> int:
        return len(HEARTBEAT_MONITOR.alive_nodes())

    def _global_p99_ttft_ms(self) -> float:
        summary = SLO_TRACKER.summary()
        raw = cast(float | int | str | None, summary.get("global_p99_ttft_ms", 0.0))
        return float(raw) if raw is not None else 0.0

    def _rejection_rate_recent(self) -> float:
        """
        Approximate rejection rate over the last _REJECTION_RATE_WINDOW_S.

        AdmissionController only exposes a lifetime counter, so we diff
        against the snapshot taken on the previous check.
        """
        stats = ADMISSION_CONTROLLER.stats()
        now = time.time()
        rejected_raw = cast(float | int | str | None, stats.get("rejected_total", 0))
        admitted_raw = cast(float | int | str | None, stats.get("admitted_total", 0))
        rejected_total = int(rejected_raw) if rejected_raw is not None else 0
        admitted_total = int(admitted_raw) if admitted_raw is not None else 0
        total_now = rejected_total + admitted_total

        snap = self._last_rejection_snapshot
        elapsed = now - snap.get("ts", 0.0)

        # Only use the window diff if the snapshot is recent enough.
        if 0 < elapsed <= _REJECTION_RATE_WINDOW_S * 3:
            delta_rejected = rejected_total - snap.get("rejected", 0)
            delta_total = total_now - snap.get("total", 0)
            rate = delta_rejected / max(delta_total, 1)
        else:
            # Fall back to lifetime rate on first call.
            rate_raw = cast(
                float | int | str | None, stats.get("rejection_rate", 0.0)
            )
            rate = float(rate_raw) if rate_raw is not None else 0.0

        # Update snapshot.
        self._last_rejection_snapshot = {
            "rejected": rejected_total,
            "total": total_now,
            "ts": int(now),
        }
        return rate

    def _metrics_snapshot(self) -> dict[str, object]:
        return {
            "queue_depth": PRIORITY_QUEUE.size(),
            "global_p99_ttft_ms": self._global_p99_ttft_ms(),
            "rejection_rate": round(self._rejection_rate_recent(), 4),
            "requests_active": int(METRICS.requests_active.get()),
            "alive_nodes": self._current_nodes(),
            "timestamp": time.time(),
        }

    def _in_cooldown(self) -> bool:
        return (time.monotonic() - self._last_scale_at) < self._cooldown_seconds

    def _emit_sse(self, event: ScaleEvent) -> None:
        sse_type = "scale_up" if event.direction == "up" else "scale_down"
        emit_cluster_event(
            sse_type,
            {
                "direction": event.direction,
                "reason": event.reason,
                "current_nodes": event.current_nodes,
                "recommended_nodes": event.recommended_nodes,
                "triggered_at": event.triggered_at,
                "metrics": event.metrics_snapshot,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> ScaleEvent | None:
        """
        Evaluate all scale conditions once.

        Returns a ScaleEvent if a signal fires (and records it), or None if
        still in cooldown or no condition is met.
        """
        queue_depth = PRIORITY_QUEUE.size()
        p99_ms = self._global_p99_ttft_ms()
        rejection_rate = self._rejection_rate_recent()
        requests_active = int(METRICS.requests_active.get())
        current_nodes = self._current_nodes()

        # ----------------------------------------------------------------
        # Advance scale-UP streak counters
        # ----------------------------------------------------------------
        if queue_depth > self._scale_up_queue_depth:
            self._up_queue_streak += 1
        else:
            self._up_queue_streak = 0

        if p99_ms > _P99_SCALE_UP_MS:
            self._up_latency_streak += 1
        else:
            self._up_latency_streak = 0

        # ----------------------------------------------------------------
        # Advance scale-DOWN streak counters
        # ----------------------------------------------------------------
        if queue_depth < 2:
            self._down_queue_streak += 1
        else:
            self._down_queue_streak = 0

        if p99_ms < _P99_SCALE_DOWN_MS:
            self._down_latency_streak += 1
        else:
            self._down_latency_streak = 0

        # ----------------------------------------------------------------
        # Determine direction (scale-up takes precedence)
        # ----------------------------------------------------------------
        direction: ScaleDirection = "none"
        reason = ""

        queue_breach = self._up_queue_streak >= _UP_CONSECUTIVE
        latency_breach = self._up_latency_streak >= _UP_CONSECUTIVE
        admission_breach = rejection_rate > _REJECTION_RATE_THRESHOLD

        if queue_breach or latency_breach or admission_breach:
            direction = "up"
            parts: list[str] = []
            if queue_breach:
                parts.append(
                    f"queue_depth={queue_depth}>{self._scale_up_queue_depth} "
                    f"for {self._up_queue_streak} consecutive checks"
                )
            if latency_breach:
                parts.append(
                    f"p99_ttft={p99_ms:.0f}ms>{_P99_SCALE_UP_MS:.0f}ms "
                    f"for {self._up_latency_streak} consecutive checks"
                )
            if admission_breach:
                parts.append(
                    f"rejection_rate={rejection_rate:.1%}>{_REJECTION_RATE_THRESHOLD:.0%} "
                    f"in last {_REJECTION_RATE_WINDOW_S:.0f}s"
                )
            reason = "; ".join(parts)

        elif (
            self._down_queue_streak >= _DOWN_CONSECUTIVE
            and self._down_latency_streak >= _DOWN_CONSECUTIVE
            and requests_active == 0
            and current_nodes > 1
        ):
            direction = "down"
            reason = (
                f"queue_depth<2 for {self._down_queue_streak} checks; "
                f"p99_ttft<{_P99_SCALE_DOWN_MS:.0f}ms for {self._down_latency_streak} checks; "
                f"requests_active=0; current_nodes={current_nodes}"
            )

        if direction == "none":
            return None

        # ----------------------------------------------------------------
        # Cooldown gate
        # ----------------------------------------------------------------
        if self._in_cooldown():
            logger.debug(
                f"[autoscale] {direction} condition met but in cooldown "
                f"({self._cooldown_seconds - (time.monotonic() - self._last_scale_at):.0f}s remaining)"
            )
            return None

        # ----------------------------------------------------------------
        # Compute recommended node count
        # ----------------------------------------------------------------
        if direction == "up":
            recommended = max(current_nodes + 1, 1)
        else:
            recommended = max(current_nodes - 1, 1)

        snapshot = self._metrics_snapshot()
        event = ScaleEvent(
            direction=direction,
            reason=reason,
            current_nodes=current_nodes,
            recommended_nodes=recommended,
            triggered_at=time.time(),
            metrics_snapshot=snapshot,
        )

        self._events.append(event)
        self._last_scale_at = time.monotonic()

        # Reset streaks so the same condition doesn't re-fire next cycle.
        if direction == "up":
            self._up_queue_streak = 0
            self._up_latency_streak = 0
        else:
            self._down_queue_streak = 0
            self._down_latency_streak = 0

        self._emit_sse(event)
        logger.warning(
            f"[autoscale] SCALE_{direction.upper()} fired — "
            f"nodes {current_nodes}→{recommended}: {reason}"
        )
        return event

    async def run_autoscale_loop(self) -> None:
        """Runs check() every _check_interval seconds indefinitely."""
        logger.info(
            f"[autoscale] loop started — interval={self._check_interval}s "
            f"cooldown={self._cooldown_seconds}s "
            f"up_queue_threshold={self._scale_up_queue_depth}"
        )
        while True:
            await anyio.sleep(self._check_interval)
            try:
                event = self.check()
                if event:
                    logger.info(
                        f"[autoscale] emitted direction={event.direction} "
                        f"nodes={event.current_nodes}→{event.recommended_nodes}"
                    )
            except Exception as exc:
                logger.error(f"[autoscale] check() raised: {exc}")

    def status(self) -> dict[str, object]:
        """Alias for stats() — satisfies the /v1/autoscale/status endpoint."""
        return self.stats()

    def history(self, n: int = 20) -> list[ScaleEvent]:
        """Return the last *n* scale events (most recent last)."""
        events = list(self._events)
        return events[-n:]

    def stats(self) -> dict[str, object]:
        now = time.monotonic()
        cooldown_remaining = max(
            0.0, self._cooldown_seconds - (now - self._last_scale_at)
        )
        last_event = self._events[-1] if self._events else None
        return {
            "total_scale_events": len(self._events),
            "last_scale_direction": last_event.direction if last_event else "none",
            "last_scale_at": last_event.triggered_at if last_event else None,
            "cooldown_remaining_s": round(cooldown_remaining, 1),
            "in_cooldown": self._in_cooldown(),
            "up_queue_streak": self._up_queue_streak,
            "up_latency_streak": self._up_latency_streak,
            "down_queue_streak": self._down_queue_streak,
            "down_latency_streak": self._down_latency_streak,
            "thresholds": {
                "scale_up_queue_depth": self._scale_up_queue_depth,
                "scale_up_p99_ttft_ms": _P99_SCALE_UP_MS,
                "scale_down_p99_ttft_ms": _P99_SCALE_DOWN_MS,
                "rejection_rate_threshold": _REJECTION_RATE_THRESHOLD,
                "cooldown_seconds": self._cooldown_seconds,
                "check_interval_s": self._check_interval,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

AUTOSCALE_TRIGGER = AutoscaleTrigger()
