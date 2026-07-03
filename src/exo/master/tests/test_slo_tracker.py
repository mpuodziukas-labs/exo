"""
Behavioral tests for exo.master.slo_tracker.

Focuses on:
- Percentile computation accuracy (p50/p99) with controlled samples
- SLO violation detection at the p99 boundary
- Per-client isolation
- Global summary aggregation across multiple clients
- Violation counter increments
- Window size cap (deque maxlen)
"""

from __future__ import annotations

import math

from exo.master.slo_tracker import ClientSloStats, SloTracker

# ---------------------------------------------------------------------------
# ClientSloStats percentile math
# ---------------------------------------------------------------------------


class TestClientSloStatsPercentiles:
    def test_empty_stats_return_zero(self) -> None:
        stats = ClientSloStats("empty_client")
        assert stats.p50_ttft_ms == 0.0
        assert stats.p99_ttft_ms == 0.0
        assert stats.p50_total_ms == 0.0
        assert stats.p99_total_ms == 0.0

    def test_p50_ttft_is_median(self) -> None:
        """p50 index = int(n * 0.50)."""
        stats = ClientSloStats("c1")
        for v in [100.0, 200.0, 300.0, 400.0, 500.0]:  # 5 samples
            stats.add_sample(ttft_ms=v, total_ms=v, tokens=10)
        # sorted: [100, 200, 300, 400, 500]  int(5*0.50)=2 → 300
        assert math.isclose(stats.p50_ttft_ms, 300.0, rel_tol=1e-6, abs_tol=1e-12)

    def test_p99_ttft_clamps_to_last_index(self) -> None:
        """With 5 samples, int(5*0.99)=4 → last element."""
        stats = ClientSloStats("c2")
        for v in [10.0, 20.0, 30.0, 40.0, 999.0]:
            stats.add_sample(ttft_ms=v, total_ms=v, tokens=5)
        assert math.isclose(stats.p99_ttft_ms, 999.0, rel_tol=1e-6, abs_tol=1e-12)

    def test_avg_tokens_per_request(self) -> None:
        stats = ClientSloStats("avg_test")
        stats.add_sample(ttft_ms=100.0, total_ms=200.0, tokens=10)
        stats.add_sample(ttft_ms=100.0, total_ms=200.0, tokens=20)
        assert math.isclose(
            stats.avg_tokens_per_request, 15.0, rel_tol=1e-6, abs_tol=1e-12
        )

    def test_slo_passes_when_p99_under_threshold(self) -> None:
        stats = ClientSloStats("ok_client")
        for _ in range(100):
            stats.add_sample(ttft_ms=100.0, total_ms=200.0, tokens=10)
        # All samples at 100ms; default SLO = 500ms → should pass
        assert stats.check_slo(500.0) is True

    def test_slo_violation_increments_counter(self) -> None:
        """p99 above threshold triggers violation and increments counter."""
        stats = ClientSloStats("slow_client")
        # 100 samples, all at 600ms > 500ms SLO
        for _ in range(100):
            stats.add_sample(ttft_ms=600.0, total_ms=1000.0, tokens=10)
        initial = stats.violations
        result = stats.check_slo(500.0)
        assert result is False
        assert stats.violations == initial + 1

    def test_to_dict_includes_all_keys(self) -> None:
        stats = ClientSloStats("dict_test")
        stats.add_sample(ttft_ms=50.0, total_ms=100.0, tokens=8)
        d = stats.to_dict()
        for key in (
            "client_key",
            "sample_count",
            "p50_ttft_ms",
            "p99_ttft_ms",
            "p50_total_ms",
            "p99_total_ms",
            "avg_tokens_per_request",
            "violations",
        ):
            assert key in d


# ---------------------------------------------------------------------------
# SloTracker aggregation
# ---------------------------------------------------------------------------


class TestSloTrackerAggregation:
    def test_record_creates_client_on_first_call(self) -> None:
        tracker = SloTracker()
        tracker.record("new_client", ttft_ms=50.0, total_ms=100.0, tokens=5)
        assert tracker.get("new_client") is not None

    def test_summary_empty_tracker(self) -> None:
        tracker = SloTracker()
        s = tracker.summary()
        assert s["client_count"] == 0
        assert s["total_violations"] == 0

    def test_summary_global_p99_across_clients(self) -> None:
        """global_p99 should be the 99th percentile across ALL samples from all clients."""
        tracker = SloTracker()
        # client_a: 50 samples at 100ms; client_b: 50 samples at 900ms
        for _ in range(50):
            tracker.record("client_a", ttft_ms=100.0, total_ms=200.0, tokens=1)
        for _ in range(50):
            tracker.record("client_b", ttft_ms=900.0, total_ms=1500.0, tokens=1)

        s = tracker.summary()
        assert s["client_count"] == 2
        # int(100 * 0.99) = 99 → 99th out of 100 sorted values
        # sorted: 50×100 then 50×900 → index 99 = 900
        assert isinstance(s["global_p99_ttft_ms"], float)
        assert math.isclose(s["global_p99_ttft_ms"], 900.0, rel_tol=1e-6, abs_tol=1e-12)

    def test_violating_clients_returns_correct_subset(self) -> None:
        tracker = SloTracker()
        # client_ok: all fast samples
        for _ in range(100):
            tracker.record("client_ok", ttft_ms=100.0, total_ms=200.0, tokens=5)
        # client_slow: all slow samples above 500ms
        for _ in range(100):
            tracker.record("client_slow", ttft_ms=800.0, total_ms=1500.0, tokens=5)

        violators = tracker.violating_clients(ttft_slo_ms=500.0)
        assert "client_slow" in violators
        assert "client_ok" not in violators
