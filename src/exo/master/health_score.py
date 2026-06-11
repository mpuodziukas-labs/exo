"""
health_score.py — Cluster health score aggregator.

Single 0-100 score synthesised from six weighted signals:
    circuit_breakers · memory_pressure · error_rate
    p99_ttft_slo · queue_depth · link_health

Grade bands:  A ≥ 90 | B ≥ 75 | C ≥ 60 | D ≥ 40 | F < 40
SSE events:  score < 75 → "health_warn" | score < 50 → "health_critical"
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from exo.master.adaptive_health_interval import ADAPTIVE_HEALTH_INTERVAL
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from exo.master.circuit_breaker import CIRCUIT_BREAKERS, CircuitState  # CircuitState used for .value string comparisons
from exo.master.event_stream import emit as emit_cluster_event
from exo.master.link_health import LINK_MONITOR
from exo.master.memory_monitor import MEMORY_MONITOR
from exo.master.metrics import METRICS
from exo.master.priority_queue import PRIORITY_QUEUE
from exo.master.slo_tracker import SLO_TRACKER

# ---------------------------------------------------------------------------
# Configuration knobs (override via env if needed)
# ---------------------------------------------------------------------------
_SCORE_INTERVAL_S: float = 15.0   # how often compute() is called in the loop
_HISTORY_MAXLEN: int = 100         # HealthReport ring-buffer depth

# TTFT SLO breakpoints (ms)
_TTFT_PERFECT_MS: float = 300.0
_TTFT_ZERO_MS: float = 2_000.0

# Error-rate breakpoints
_ERR_PERFECT: float = 0.01   # below this → 100
_ERR_ZERO: float = 0.10      # above this → 0

# Queue depth breakpoints
_QUEUE_PERFECT: int = 10
_QUEUE_ZERO: int = 100

# Alert thresholds
_WARN_THRESHOLD: float = 75.0
_CRITICAL_THRESHOLD: float = 50.0
_DEGRADE_THRESHOLD: float = 60.0   # factor score below this → degraded


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthFactor:
    name: str
    score: float       # 0-100
    weight: float      # contribution weight (all weights sum to 1.0)
    detail: str        # human-readable explanation


@dataclass(frozen=True)
class HealthReport:
    overall_score: float
    grade: Literal["A", "B", "C", "D", "F"]
    factors: list[HealthFactor]
    timestamp: float
    degraded_factors: list[str]


# ---------------------------------------------------------------------------
# Helper: linear interpolation clamp
# ---------------------------------------------------------------------------

def _lerp_score(value: float, perfect: float, zero: float) -> float:
    """Return 100 when value ≤ perfect, 0 when value ≥ zero, linear in between."""
    if value <= perfect:
        return 100.0
    if value >= zero:
        return 0.0
    return 100.0 * (1.0 - (value - perfect) / (zero - perfect))


def _assign_grade(score: float) -> Literal["A", "B", "C", "D", "F"]:
    if score >= 90.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 60.0:
        return "C"
    if score >= 40.0:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class ClusterHealthScorer:
    """
    Aggregates cluster signals into a single 0-100 health score.

    Designed to be instantiated once as a module-level singleton and driven
    by ``run_score_loop()`` inside the API task group.
    """

    def __init__(self) -> None:
        self._history: deque[HealthReport] = deque(maxlen=_HISTORY_MAXLEN)
        self._prev_grade: Literal["A", "B", "C", "D", "F"] | None = None

    # ------------------------------------------------------------------
    # Factor scorers
    # ------------------------------------------------------------------

    def _score_circuit_breakers(self) -> HealthFactor:
        # all_states() is lock-safe and returns a snapshot list of dicts
        all_states = CIRCUIT_BREAKERS.all_states()
        if not all_states:
            return HealthFactor(
                name="circuit_breakers",
                score=100.0,
                weight=0.25,
                detail="no breakers registered",
            )
        open_ids = [s["worker_id"] for s in all_states if s["state"] == CircuitState.OPEN.value]
        half_ids = [s["worker_id"] for s in all_states if s["state"] == CircuitState.HALF_OPEN.value]
        if open_ids:
            score = 0.0
            detail = f"OPEN: {', '.join(open_ids)}"
        elif half_ids:
            score = 50.0
            detail = f"HALF_OPEN: {', '.join(half_ids)}"
        else:
            score = 100.0
            detail = f"all {len(all_states)} breaker(s) CLOSED"
        return HealthFactor(name="circuit_breakers", score=score, weight=0.25, detail=detail)

    def _score_memory_pressure(self) -> HealthFactor:
        pressure = MEMORY_MONITOR.current_pressure  # 0.0-1.0
        score = max(0.0, min(100.0, 100.0 - pressure * 100.0))
        return HealthFactor(
            name="memory_pressure",
            score=score,
            weight=0.20,
            detail=f"ram_pressure={pressure:.1%} → score={score:.1f}",
        )

    def _score_error_rate(self) -> HealthFactor:
        total = METRICS.requests_total.get()
        errors = METRICS.errors_total.get()
        error_rate = errors / max(total, 1.0)
        score = _lerp_score(error_rate, _ERR_PERFECT, _ERR_ZERO)
        return HealthFactor(
            name="error_rate",
            score=score,
            weight=0.20,
            detail=f"error_rate={error_rate:.2%} (errors={errors:.0f}/total={total:.0f})",
        )

    def _score_p99_ttft(self) -> HealthFactor:
        summary = SLO_TRACKER.summary()
        p99_ms: float = summary.get("global_p99_ttft_ms", 0.0)
        if p99_ms == 0.0:
            # No samples yet — treat as perfect; we can't penalise what hasn't happened
            return HealthFactor(
                name="p99_ttft_slo",
                score=100.0,
                weight=0.15,
                detail="no TTFT samples recorded yet",
            )
        score = _lerp_score(p99_ms, _TTFT_PERFECT_MS, _TTFT_ZERO_MS)
        return HealthFactor(
            name="p99_ttft_slo",
            score=score,
            weight=0.15,
            detail=f"p99_ttft={p99_ms:.1f}ms (perfect<{_TTFT_PERFECT_MS:.0f}ms zero>{_TTFT_ZERO_MS:.0f}ms)",
        )

    def _score_queue_depth(self) -> HealthFactor:
        depth = PRIORITY_QUEUE.size()
        score = _lerp_score(float(depth), float(_QUEUE_PERFECT), float(_QUEUE_ZERO))
        return HealthFactor(
            name="queue_depth",
            score=score,
            weight=0.10,
            detail=f"queue_depth={depth} (perfect<{_QUEUE_PERFECT} zero>{_QUEUE_ZERO})",
        )

    def _score_link_health(self) -> HealthFactor:
        _status_to_score: dict[str, float] = {
            "healthy": 100.0,
            "warning": 50.0,
            "degraded": 0.0,
            "unknown": 30.0,
        }
        node_stats = LINK_MONITOR.get_stats()
        if not node_stats:
            return HealthFactor(
                name="link_health",
                score=100.0,
                weight=0.10,
                detail="no link nodes registered",
            )
        scores = [_status_to_score.get(n["status"], 30.0) for n in node_stats]
        avg = sum(scores) / len(scores)
        statuses = ", ".join(f"{n['node_id']}:{n['status']}" for n in node_stats)
        return HealthFactor(
            name="link_health",
            score=avg,
            weight=0.10,
            detail=f"avg={avg:.1f} [{statuses}]",
        )

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute(self) -> HealthReport:
        factors: list[HealthFactor] = [
            self._score_circuit_breakers(),
            self._score_memory_pressure(),
            self._score_error_rate(),
            self._score_p99_ttft(),
            self._score_queue_depth(),
            self._score_link_health(),
        ]

        overall = sum(f.score * f.weight for f in factors)
        grade = _assign_grade(overall)
        degraded = [f.name for f in factors if f.score < _DEGRADE_THRESHOLD]

        report = HealthReport(
            overall_score=round(overall, 2),
            grade=grade,
            factors=list(factors),
            timestamp=time.time(),
            degraded_factors=degraded,
        )

        self._history.append(report)

        # SSE alerts on state transitions
        self._maybe_emit_alert(report)

        logger.debug(
            f"[health_score] score={overall:.1f} grade={grade} "
            f"degraded={degraded or 'none'}"
        )
        return report

    def _maybe_emit_alert(self, report: HealthReport) -> None:
        score = report.overall_score
        grade = report.grade
        if self._prev_grade == grade:
            self._prev_grade = grade
            return  # no transition — skip noise

        if score < _CRITICAL_THRESHOLD:
            emit_cluster_event(
                "health_critical",
                {
                    "score": score,
                    "grade": grade,
                    "degraded_factors": report.degraded_factors,
                },
            )
            logger.error(
                f"[health_score] CRITICAL score={score:.1f} grade={grade} "
                f"degraded={report.degraded_factors}"
            )
        elif score < _WARN_THRESHOLD:
            emit_cluster_event(
                "health_warn",
                {
                    "score": score,
                    "grade": grade,
                    "degraded_factors": report.degraded_factors,
                },
            )
            logger.warning(
                f"[health_score] WARN score={score:.1f} grade={grade} "
                f"degraded={report.degraded_factors}"
            )
        self._prev_grade = grade

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run_score_loop(self) -> None:
        """Run compute() every ``_SCORE_INTERVAL_S`` seconds indefinitely."""
        logger.info(f"[health_score] score loop started (interval={_SCORE_INTERVAL_S}s)")
        while True:
            try:
                self.compute()
            except Exception as exc:
                logger.warning(f"[health_score] compute() raised: {exc}")
            await asyncio.sleep(ADAPTIVE_HEALTH_INTERVAL.get_interval())

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def current(self) -> HealthReport | None:
        """Return the most recent HealthReport, or None if no score computed yet."""
        return self._history[-1] if self._history else None

    def trend(self, n: int = 10) -> list[float]:
        """Return the last *n* overall scores (oldest first)."""
        reports = list(self._history)
        return [r.overall_score for r in reports[-n:]]

    def stats(self) -> dict[str, object]:
        report = self.current()
        if report is None:
            return {"status": "not_computed"}
        return {
            "overall_score": report.overall_score,
            "grade": report.grade,
            "timestamp": report.timestamp,
            "degraded_factors": report.degraded_factors,
            "history_depth": len(self._history),
            "factors": [
                {
                    "name": f.name,
                    "score": f.score,
                    "weight": f.weight,
                    "detail": f.detail,
                }
                for f in report.factors
            ],
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

HEALTH_SCORER: ClusterHealthScorer = ClusterHealthScorer()
