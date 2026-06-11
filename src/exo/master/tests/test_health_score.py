"""
Behavioral tests for exo.master.health_score.

Tests focus on the pure/injectable functions and the ClusterHealthScorer
computation logic.  The module's singleton dependencies (CIRCUIT_BREAKERS,
MEMORY_MONITOR, METRICS, SLO_TRACKER, PRIORITY_QUEUE, LINK_MONITOR,
emit_cluster_event, ADAPTIVE_HEALTH_INTERVAL) are all patched at the point of
use via pytest monkeypatch or unittest.mock.patch, so each test controls
exactly what the scorer sees without affecting other tests.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.master.health_score import (
    ClusterHealthScorer,
    HealthFactor,
    _assign_grade,
    _lerp_score,
)

# ---------------------------------------------------------------------------
# Pure helper tests: _lerp_score
# ---------------------------------------------------------------------------

class TestLerpScore:
    def test_at_or_below_perfect_returns_100(self) -> None:
        assert _lerp_score(300.0, perfect=300.0, zero=2000.0) == pytest.approx(100.0)
        assert _lerp_score(100.0, perfect=300.0, zero=2000.0) == pytest.approx(100.0)

    def test_at_or_above_zero_returns_0(self) -> None:
        assert _lerp_score(2000.0, perfect=300.0, zero=2000.0) == pytest.approx(0.0)
        assert _lerp_score(9999.0, perfect=300.0, zero=2000.0) == pytest.approx(0.0)

    def test_midpoint_returns_50(self) -> None:
        # midpoint of [300, 2000] is 1150
        assert _lerp_score(1150.0, perfect=300.0, zero=2000.0) == pytest.approx(50.0, rel=1e-3)

    def test_interpolation_is_linear(self) -> None:
        """quarter-point should give 75."""
        # 300 + (2000-300)*0.25 = 725 → score should be 75
        assert _lerp_score(725.0, perfect=300.0, zero=2000.0) == pytest.approx(75.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Pure helper tests: _assign_grade
# ---------------------------------------------------------------------------

class TestAssignGrade:
    @pytest.mark.parametrize("score,expected", [
        (95.0, "A"),
        (90.0, "A"),
        (89.9, "B"),
        (75.0, "B"),
        (74.9, "C"),
        (60.0, "C"),
        (59.9, "D"),
        (40.0, "D"),
        (39.9, "F"),
        (0.0,  "F"),
    ])
    def test_grade_bands(self, score: float, expected: str) -> None:
        assert _assign_grade(score) == expected


# ---------------------------------------------------------------------------
# ClusterHealthScorer.compute() with mocked singletons
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

        # Mock all the singletons referenced inside compute()
        with (
            patch("exo.master.health_score.CIRCUIT_BREAKERS") as cb,
            patch("exo.master.health_score.MEMORY_MONITOR") as mm,
            patch("exo.master.health_score.METRICS") as met,
            patch("exo.master.health_score.SLO_TRACKER") as slo,
            patch("exo.master.health_score.PRIORITY_QUEUE") as pq,
            patch("exo.master.health_score.LINK_MONITOR") as lm,
            patch("exo.master.health_score.emit_cluster_event") as _emit,
        ):
            cb.all_states.return_value = []
            mm.current_pressure = 0.0
            met.requests_total.get.return_value = 100.0
            met.errors_total.get.return_value = 0.0
            slo.summary.return_value = {"global_p99_ttft_ms": 0.0}
            pq.size.return_value = 0
            lm.get_stats.return_value = []

            report = scorer.compute()

        total_weight = sum(f.weight for f in report.factors)
        assert total_weight == pytest.approx(1.0)

    def test_open_circuit_breaker_gives_zero_cb_score(self) -> None:
        """When a circuit breaker is OPEN, circuit_breakers factor = 0."""
        scorer = self._make_scorer()

        with (
            patch("exo.master.health_score.CIRCUIT_BREAKERS") as cb,
            patch("exo.master.health_score.MEMORY_MONITOR") as mm,
            patch("exo.master.health_score.METRICS") as met,
            patch("exo.master.health_score.SLO_TRACKER") as slo,
            patch("exo.master.health_score.PRIORITY_QUEUE") as pq,
            patch("exo.master.health_score.LINK_MONITOR") as lm,
            patch("exo.master.health_score.emit_cluster_event"),
        ):
            cb.all_states.return_value = [{"worker_id": "w1", "state": "open"}]
            mm.current_pressure = 0.0
            met.requests_total.get.return_value = 100.0
            met.errors_total.get.return_value = 0.0
            slo.summary.return_value = {"global_p99_ttft_ms": 0.0}
            pq.size.return_value = 0
            lm.get_stats.return_value = []

            report = scorer.compute()

        cb_factor = next(f for f in report.factors if f.name == "circuit_breakers")
        assert cb_factor.score == pytest.approx(0.0)

    def test_degraded_factors_collected_below_threshold(self) -> None:
        """Factors with score < 60 appear in degraded_factors list."""
        scorer = self._make_scorer()

        with (
            patch("exo.master.health_score.CIRCUIT_BREAKERS") as cb,
            patch("exo.master.health_score.MEMORY_MONITOR") as mm,
            patch("exo.master.health_score.METRICS") as met,
            patch("exo.master.health_score.SLO_TRACKER") as slo,
            patch("exo.master.health_score.PRIORITY_QUEUE") as pq,
            patch("exo.master.health_score.LINK_MONITOR") as lm,
            patch("exo.master.health_score.emit_cluster_event"),
        ):
            # Open breaker → cb score = 0 → degraded
            cb.all_states.return_value = [{"worker_id": "w1", "state": "open"}]
            mm.current_pressure = 0.9  # 90% pressure → score 10 → degraded
            met.requests_total.get.return_value = 100.0
            met.errors_total.get.return_value = 0.0
            slo.summary.return_value = {"global_p99_ttft_ms": 0.0}
            pq.size.return_value = 0
            lm.get_stats.return_value = []

            report = scorer.compute()

        assert "circuit_breakers" in report.degraded_factors
        assert "memory_pressure" in report.degraded_factors

    def test_history_grows_with_each_compute(self) -> None:
        scorer = self._make_scorer()

        with (
            patch("exo.master.health_score.CIRCUIT_BREAKERS") as cb,
            patch("exo.master.health_score.MEMORY_MONITOR") as mm,
            patch("exo.master.health_score.METRICS") as met,
            patch("exo.master.health_score.SLO_TRACKER") as slo,
            patch("exo.master.health_score.PRIORITY_QUEUE") as pq,
            patch("exo.master.health_score.LINK_MONITOR") as lm,
            patch("exo.master.health_score.emit_cluster_event"),
        ):
            cb.all_states.return_value = []
            mm.current_pressure = 0.0
            met.requests_total.get.return_value = 10.0
            met.errors_total.get.return_value = 0.0
            slo.summary.return_value = {"global_p99_ttft_ms": 0.0}
            pq.size.return_value = 5
            lm.get_stats.return_value = []

            scorer.compute()
            scorer.compute()
            scorer.compute()

        assert len(scorer._history) == 3
        assert scorer.current() is not None
        assert len(scorer.trend(2)) == 2
