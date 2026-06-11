"""Behavioral tests for exo.master.admission_control.

AdmissionController gates requests on four dimensions:
  1. active_requests  >= max_concurrent
  2. memory_pressure  >  memory_pressure_threshold
  3. kv_cache_pressure > kv_cache_threshold
  4. queue_depth      >  max_queue_depth

Each rejected request increments rejected_total; admitted increments admitted_total.
"""

from __future__ import annotations

import pytest

from exo.master.admission_control import AdmissionController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _controller(
    *,
    max_concurrent: int = 16,
    memory_threshold: float = 0.90,
    kv_threshold: float = 0.95,
    max_queue: int = 32,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> AdmissionController:
    """Return a fresh AdmissionController with env-vars patched."""
    ctrl = AdmissionController()
    if monkeypatch is not None:
        monkeypatch.setenv("EXO_MAX_CONCURRENT_REQUESTS", str(max_concurrent))
        monkeypatch.setenv("EXO_ADMISSION_MEMORY_THRESHOLD", str(memory_threshold))
        monkeypatch.setenv("EXO_ADMISSION_KV_THRESHOLD", str(kv_threshold))
        monkeypatch.setenv("EXO_MAX_QUEUE_DEPTH", str(max_queue))
    return ctrl


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_admit_when_all_pressures_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request is admitted when no threshold is breached."""
    ctrl = _controller(max_concurrent=16, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=5, memory_pressure=0.5, kv_cache_pressure=0.5, queue_depth=2)
    assert decision.admitted is True
    assert decision.reason == "ok"


# ---------------------------------------------------------------------------
# Concurrent request limit
# ---------------------------------------------------------------------------


def test_reject_on_concurrent_request_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """active_requests >= max_concurrent triggers 429."""
    ctrl = _controller(max_concurrent=8, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=8)
    assert decision.admitted is False
    assert "8/8" in decision.reason
    assert decision.retry_after_seconds > 0.0


def test_boundary_just_below_concurrent_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """active_requests == max_concurrent - 1 is still admitted."""
    ctrl = _controller(max_concurrent=8, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=7)
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# Memory pressure
# ---------------------------------------------------------------------------


def test_reject_on_memory_pressure_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_pressure > threshold rejects the request."""
    ctrl = _controller(memory_threshold=0.80, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=0, memory_pressure=0.85)
    assert decision.admitted is False
    assert "Memory pressure" in decision.reason
    assert decision.retry_after_seconds == 5.0


def test_memory_pressure_exactly_at_threshold_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressure exactly equal to threshold is NOT over the threshold — admitted."""
    ctrl = _controller(memory_threshold=0.90, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=0, memory_pressure=0.90)
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# KV cache pressure
# ---------------------------------------------------------------------------


def test_reject_on_kv_cache_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """kv_cache_pressure > threshold rejects the request."""
    ctrl = _controller(kv_threshold=0.95, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=0, kv_cache_pressure=0.96)
    assert decision.admitted is False
    assert "KV cache" in decision.reason
    assert decision.retry_after_seconds == 2.0


# ---------------------------------------------------------------------------
# Queue depth
# ---------------------------------------------------------------------------


def test_reject_on_queue_depth_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """queue_depth > max_queue_depth rejects the request."""
    ctrl = _controller(max_queue=10, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=0, queue_depth=11)
    assert decision.admitted is False
    assert "Queue full" in decision.reason
    assert decision.retry_after_seconds == 1.0


def test_queue_depth_at_limit_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """queue_depth == max_queue_depth is not strictly over the limit."""
    ctrl = _controller(max_queue=10, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=0, queue_depth=10)
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# Priority ordering: concurrent > memory > kv > queue
# ---------------------------------------------------------------------------


def test_concurrent_limit_takes_priority_over_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both concurrent limit and memory are breached, concurrent fires first."""
    ctrl = _controller(max_concurrent=4, memory_threshold=0.5, monkeypatch=monkeypatch)
    decision = ctrl.check(active_requests=4, memory_pressure=0.9)
    assert decision.admitted is False
    assert "concurrent" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Stats counter accuracy
# ---------------------------------------------------------------------------


def test_stats_counters_increment_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """admitted_total and rejected_total stay accurate across multiple calls."""
    ctrl = _controller(max_concurrent=2, monkeypatch=monkeypatch)
    ctrl.check(active_requests=0)   # admitted
    ctrl.check(active_requests=0)   # admitted
    ctrl.check(active_requests=2)   # rejected

    s = ctrl.stats()
    assert s["admitted_total"] == 2
    assert s["rejected_total"] == 1
    assert abs(s["rejection_rate"] - 1 / 3) < 1e-3
