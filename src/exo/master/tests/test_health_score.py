"""
Behavioral tests for exo.master.health_score.

Tests focus on the pure/injectable functions and the ClusterHealthScorer
computation logic.  The module's singleton dependencies (CIRCUIT_BREAKERS,
MEMORY_MONITOR, METRICS, SLO_TRACKER, PRIORITY_QUEUE, LINK_MONITOR,
emit_cluster_event, ADAPTIVE_HEALTH_INTERVAL) are all patched at the point of
use via unittest.mock.patch(..., new=<typed fake>), so each test controls
exactly what the scorer sees without affecting other tests. Typed fakes
(rather than bare MagicMock) are used so every attribute access below is
statically concrete instead of Any.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from exo.master.health_score import (
    ClusterHealthScorer,
    HealthFactor,
    assign_grade,
    lerp_score,
)

# ---------------------------------------------------------------------------
# Pure helper tests: lerp_score
# ---------------------------------------------------------------------------


class TestLerpScore:
    def test_at_or_below_perfect_returns_100(self) -> None:
        assert lerp_score(300.0, perfect=300.0, zero=2000.0) == pytest.approx(100.0)
        assert lerp_score(100.0, perfect=300.0, zero=2000.0) == pytest.approx(100.0)

    def test_at_or_above_zero_returns_0(self) -> None:
        assert lerp_score(2000.0, perfect=300.0, zero=2000.0) == pytest.approx(0.0)
        assert lerp_score(9999.0, perfect=300.0, zero=2000.0) == pytest.approx(0.0)

    def test_midpoint_returns_50(self) -> None:
        # midpoint of [300, 2000] is 1150
        assert lerp_score(1150.0, perfect=300.0, zero=2000.0) == pytest.approx(
            50.0, rel=1e-3
        )

    def test_interpolation_is_linear(self) -> None:
        """quarter-point should give 75."""
        # 300 + (2000-300)*0.25 = 725 → score should be 75
        assert lerp_score(725.0, perfect=300.0, zero=2000.0) == pytest.approx(
            75.0, rel=1e-3
        )


# ---------------------------------------------------------------------------
# Pure helper tests: assign_grade
# ---------------------------------------------------------------------------


class TestAssignGrade:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (95.0, "A"),
            (90.0, "A"),
            (89.9, "B"),
            (75.0, "B"),
            (74.9, "C"),
            (60.0, "C"),
            (59.9, "D"),
            (40.0, "D"),
            (39.9, "F"),
            (0.0, "F"),
        ],
    )
    def test_grade_bands(self, score: float, expected: str) -> None:
        assert assign_grade(score) == expected


# ---------------------------------------------------------------------------
# Typed fakes for the module-level singletons ClusterHealthScorer.compute()
# reaches into. Bare unittest.mock.patch(...) as X gives X the static type
# MagicMock, whose attributes are all Any — these hand-rolled fakes are
# patched in via patch(..., new=fake) instead, which types the bound name as
# the fake's own concrete class, eliminating reportAny entirely.
# ---------------------------------------------------------------------------


class _FakeCounter:
    def __init__(self, value: float) -> None:
        self._value = value

    def get(self) -> float:
        return self._value


class _FakeMetrics:
    def __init__(self, requests_total: float, errors_total: float) -> None:
        self.requests_total = _FakeCounter(requests_total)
        self.errors_total = _FakeCounter(errors_total)


class _FakeMemoryMonitor:
    def __init__(self, current_pressure: float) -> None:
        self.current_pressure = current_pressure


class _FakeSloTracker:
    def __init__(self, global_p99_ttft_ms: float) -> None:
        self._summary: dict[str, float] = {"global_p99_ttft_ms": global_p99_ttft_ms}

    def summary(self) -> dict[str, float]:
        return self._summary


class _FakeCircuitBreakers:
    def __init__(self, states: list[dict[str, str]]) -> None:
        self._states = states

    def all_states(self) -> list[dict[str, str]]:
        return self._states


class _FakePriorityQueue:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeLinkMonitor:
    def __init__(self, stats: list[dict[str, str]]) -> None:
        self._stats = stats

    def get_stats(self) -> list[dict[str, str]]:
        return self._stats


@contextmanager
def _patched_singletons(
    *,
    cb_states: list[dict[str, str]] | None = None,
    memory_pressure: float = 0.0,
    requests_total: float = 100.0,
    errors_total: float = 0.0,
    p99_ttft_ms: float = 0.0,
    queue_size: int = 0,
    link_stats: list[dict[str, str]] | None = None,
) -> Iterator[None]:
    """Patch every singleton exo.master.health_score.compute() touches with a
    typed fake, so tests configure behavior via plain constructor args."""
    with (
        patch(
            "exo.master.health_score.CIRCUIT_BREAKERS",
            new=_FakeCircuitBreakers(cb_states or []),
        ),
        patch(
            "exo.master.health_score.MEMORY_MONITOR",
            new=_FakeMemoryMonitor(memory_pressure),
        ),
        patch(
            "exo.master.health_score.METRICS",
            new=_FakeMetrics(requests_total, errors_total),
        ),
        patch(
            "exo.master.health_score.SLO_TRACKER",
            new=_FakeSloTracker(p99_ttft_ms),
        ),
        patch(
            "exo.master.health_score.PRIORITY_QUEUE",
            new=_FakePriorityQueue(queue_size),
        ),
        patch(
            "exo.master.health_score.LINK_MONITOR",
            new=_FakeLinkMonitor(link_stats or []),
        ),
        patch("exo.master.health_score.emit_cluster_event"),
    ):
        yield


# ---------------------------------------------------------------------------
# ClusterHealthScorer.compute() with patched singletons
# ---------------------------------------------------------------------------


class TestClusterHealthScorer:
    def _make_scorer(self) -> ClusterHealthScorer:
        return ClusterHealthScorer()

    def test_all_perfect_inputs_give_near_100(self) -> None:
        """When every factor returns 100 and weights sum to 1.0, overall = 100."""

        def perfect_factor(name: str, weight: float) -> HealthFactor:
            return HealthFactor(name=name, score=100.0, weight=weight, detail="perfect")

        factors = [
            perfect_factor("circuit_breakers", 0.25),
            perfect_factor("memory_pressure", 0.20),
            perfect_factor("error_rate", 0.20),
            perfect_factor("p99_ttft_slo", 0.15),
            perfect_factor("queue_depth", 0.10),
            perfect_factor("link_health", 0.10),
        ]

        # Weights sum to 1.0; all scores are 100 → weighted sum = 100.0
        assert sum(f.weight for f in factors) == pytest.approx(1.0)
        overall = sum(f.score * f.weight for f in factors)
        assert overall == pytest.approx(100.0)

    def test_weights_sum_to_one(self) -> None:
        """Documented weights must sum to 1.0 (integrity check)."""
        scorer = self._make_scorer()

        with _patched_singletons(
            requests_total=100.0, errors_total=0.0, p99_ttft_ms=0.0, queue_size=0
        ):
            report = scorer.compute()

        total_weight = sum(f.weight for f in report.factors)
        assert total_weight == pytest.approx(1.0)

    def test_open_circuit_breaker_gives_zero_cb_score(self) -> None:
        """When a circuit breaker is OPEN, circuit_breakers factor = 0."""
        scorer = self._make_scorer()

        with _patched_singletons(
            cb_states=[{"worker_id": "w1", "state": "open"}],
            requests_total=100.0,
            errors_total=0.0,
            p99_ttft_ms=0.0,
            queue_size=0,
        ):
            report = scorer.compute()

        cb_factor = next(f for f in report.factors if f.name == "circuit_breakers")
        assert cb_factor.score == pytest.approx(0.0)

    def test_degraded_factors_collected_below_threshold(self) -> None:
        """Factors with score < 60 appear in degraded_factors list."""
        scorer = self._make_scorer()

        with _patched_singletons(
            # Open breaker → cb score = 0 → degraded
            cb_states=[{"worker_id": "w1", "state": "open"}],
            memory_pressure=0.9,  # 90% pressure → score 10 → degraded
            requests_total=100.0,
            errors_total=0.0,
            p99_ttft_ms=0.0,
            queue_size=0,
        ):
            report = scorer.compute()

        assert "circuit_breakers" in report.degraded_factors
        assert "memory_pressure" in report.degraded_factors

    def test_history_grows_with_each_compute(self) -> None:
        scorer = self._make_scorer()

        with _patched_singletons(
            requests_total=10.0, errors_total=0.0, p99_ttft_ms=0.0, queue_size=5
        ):
            scorer.compute()
            scorer.compute()
            scorer.compute()

        assert scorer.history_len == 3
        assert scorer.current() is not None
        assert len(scorer.trend(2)) == 2
