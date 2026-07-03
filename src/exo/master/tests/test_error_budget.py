"""
Behavioral tests for exo.master.error_budget.

Focuses on:
- Budget consumption calculation (error_rate / (1 - SLO_target))
- Alert threshold boundary (>50% consumed triggers alert, <=50% resets)
- SLO-met flag at 0.1% error boundary
- budget_consumed_pct clamped at 100 on full exhaustion
- Minimum-sample guard (no alert below 10 samples)
- ErrorBudgetManager multi-endpoint isolation
- Rolling window trimming (time-based cutoff)
"""

from __future__ import annotations

import math
import time
from unittest.mock import patch

from exo.master.error_budget import EndpointErrorBudget, ErrorBudgetManager


class TestEndpointErrorBudgetMath:
    def test_zero_errors_gives_full_budget(self) -> None:
        eb = EndpointErrorBudget("/api/v1/generate")
        for _ in range(20):
            eb.record(success=True)
        d = eb.to_dict()
        assert isinstance(d["error_rate_pct"], float)
        assert isinstance(d["budget_consumed_pct"], float)
        assert math.isclose(d["error_rate_pct"], 0.0, rel_tol=1e-6, abs_tol=1e-12)
        assert math.isclose(d["budget_consumed_pct"], 0.0, rel_tol=1e-6, abs_tol=1e-12)
        assert d["slo_met"] is True

    def test_budget_consumed_pct_formula(self) -> None:
        """budget_consumed = error_rate / (1 - 0.999) → at 0.05% err = 50%."""
        eb = EndpointErrorBudget("/ep")
        # 10 000 requests, 5 errors → error_rate = 0.05% → consumed = 50%
        for _ in range(9995):
            eb.record(success=True)
        for _ in range(5):
            eb.record(success=False)
        d = eb.to_dict()
        assert isinstance(d["error_rate_pct"], float)
        assert isinstance(d["budget_consumed_pct"], float)
        assert math.isclose(d["error_rate_pct"], 0.05, rel_tol=1e-3, abs_tol=1e-12)
        assert math.isclose(d["budget_consumed_pct"], 50.0, rel_tol=1e-3, abs_tol=1e-12)

    def test_slo_violated_above_0_1_percent_errors(self) -> None:
        """error_rate > 0.1% means SLO not met."""
        eb = EndpointErrorBudget("/ep")
        for _ in range(999):
            eb.record(success=True)
        for _ in range(2):  # 2/1001 ≈ 0.2% > 0.1%
            eb.record(success=False)
        d = eb.to_dict()
        assert d["slo_met"] is False

    def test_budget_consumed_capped_at_100(self) -> None:
        """At 50% error rate (500× over budget), cap at 100%."""
        eb = EndpointErrorBudget("/ep")
        for _ in range(10):
            eb.record(success=False)  # 100% errors → consumed = 10000%, capped 100
        d = eb.to_dict()
        assert isinstance(d["budget_consumed_pct"], float)
        assert math.isclose(
            d["budget_consumed_pct"], 100.0, rel_tol=1e-6, abs_tol=1e-12
        )

    def test_no_alert_below_min_samples(self) -> None:
        """Budget alert is suppressed when fewer than 10 samples recorded."""
        eb = EndpointErrorBudget("/ep")
        # 9 samples, all errors — should NOT trigger alert (guard at total < 10)
        for _ in range(9):
            eb.record(success=False)
        # If alert had fired, _budget_alert_sent would be True
        # The guard prevents firing, so it stays False even with 100% errors
        assert eb.budget_alert_sent is False

    def test_alert_sent_flag_resets_when_budget_recovers(self) -> None:
        """_budget_alert_sent resets to False once error rate drops below 50% of budget.

        Strategy: record 50% consumed to trigger the flag, then expire ALL those
        samples by fast-forwarding time past the 1-hour window, then add clean
        successful requests within the new window.
        """
        eb = EndpointErrorBudget("/ep")

        t_start = time.time()
        # Record enough requests to cross 50% budget consumed:
        # Need error_rate > 0.05% → e.g. 1 error in 10 total = 10% → consumed 100%
        with patch("time.time", return_value=t_start):
            for _ in range(9):
                eb.record(success=True)
            eb.record(success=False)  # 10% error → flag set

        assert eb.budget_alert_sent is True

        # Fast-forward past window (3600 s) so old samples are evicted
        t_new = t_start + 3700.0
        with patch("time.time", return_value=t_new):
            # Add 10 successful requests in the new window — clears the flag
            for _ in range(10):
                eb.record(success=True)

        assert eb.budget_alert_sent is False

    def test_rolling_window_excludes_old_samples(self) -> None:
        """Samples older than _WINDOW_SECONDS are trimmed from calculations."""
        eb = EndpointErrorBudget("/ep_window")
        old_time = time.time() - 7200.0  # 2 hours ago (outside 1-hour window)

        with patch("time.time", return_value=old_time):
            for _ in range(50):
                eb.record(success=False)  # 50 old errors

        # Now add 10 recent successes
        for _ in range(10):
            eb.record(success=True)

        # to_dict() calls _trim() internally; old errors should be gone
        d = eb.to_dict()
        assert d["total_requests"] == 10
        assert d["errors"] == 0


class TestErrorBudgetManager:
    def test_separate_endpoints_isolated(self) -> None:
        mgr = ErrorBudgetManager()
        for _ in range(100):
            mgr.record("/a", success=True)
        for _ in range(50):
            mgr.record("/b", success=False)
        for _ in range(50):
            mgr.record("/b", success=True)

        budgets = {b["endpoint"]: b for b in mgr.get_all()}
        assert budgets["/a"]["errors"] == 0
        assert budgets["/b"]["errors"] == 50

    def test_get_summary_reports_violation_count(self) -> None:
        mgr = ErrorBudgetManager()
        # endpoint that clearly violates SLO (>0.1% error rate)
        for _ in range(900):
            mgr.record("/bad", success=True)
        for _ in range(100):
            mgr.record("/bad", success=False)  # 10% error rate

        s = mgr.get_summary()
        assert s["slo_violations"] >= 1
        assert s["endpoints"] == 1
