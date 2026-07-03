"""Behavioral tests for exo.master.worker_circuit_breaker.

Key difference from circuit_breaker.py: this uses a ROLLING TIME WINDOW for
failure counting, not consecutive failures.  Burst detection works even when
successes are interspersed between failures.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from exo.master.worker_circuit_breaker import (
    WorkerCircuitBreaker,
    WorkerCircuitBreakerRegistry,
    WorkerCircuitState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcb(**kwargs: object) -> WorkerCircuitBreaker:
    defaults: dict[str, object] = {
        "worker_id": "node-a",
        "failure_threshold": 3,
        "failure_window_seconds": 60.0,
        "open_duration_seconds": 30.0,
    }
    defaults.update(kwargs)
    return WorkerCircuitBreaker(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLOSED → OPEN via rolling window
# ---------------------------------------------------------------------------


def test_trips_on_threshold_failures_in_window() -> None:
    """Three failures within the window trips the circuit to OPEN."""
    wcb = _make_wcb(failure_threshold=3)
    for _ in range(2):
        wcb.record_failure()
        assert wcb.state == WorkerCircuitState.CLOSED

    wcb.record_failure()
    assert wcb.state == WorkerCircuitState.OPEN
    assert wcb.total_trips == 1


def test_failures_outside_window_do_not_trip() -> None:
    """Failures that have expired from the window should not count toward threshold."""
    wcb = _make_wcb(failure_threshold=3, failure_window_seconds=10.0)

    old_time = time.monotonic() - 20.0  # well outside 10s window
    # Inject a stale timestamp directly
    wcb.failure_timestamps.append(old_time)
    wcb.failure_timestamps.append(old_time)
    # These two old failures should be pruned; adding one fresh failure stays below threshold
    wcb.record_failure()
    assert wcb.state == WorkerCircuitState.CLOSED  # 1 in-window < threshold=3


def test_interspersed_successes_still_trip_on_window_failures() -> None:
    """Successes between failures don't stop window-based trip detection."""
    wcb = _make_wcb(failure_threshold=3)
    wcb.record_failure()
    wcb.record_success()  # doesn't reset the window timestamps
    wcb.record_failure()
    wcb.record_success()
    wcb.record_failure()
    # 3 failures are within the window — circuit should open
    assert wcb.state == WorkerCircuitState.OPEN


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN via cooldown (time injection)
# ---------------------------------------------------------------------------


def test_open_rejects_during_open_duration() -> None:
    """allow_request returns False while the circuit is OPEN and not cooled."""
    wcb = _make_wcb(failure_threshold=1, open_duration_seconds=30.0)
    wcb.record_failure()
    assert wcb.state == WorkerCircuitState.OPEN
    assert wcb.allow_request() is False


def test_open_transitions_to_half_open_after_cooldown() -> None:
    """After open_duration_seconds, allow_request promotes OPEN → HALF_OPEN."""
    wcb = _make_wcb(failure_threshold=1, open_duration_seconds=30.0)
    wcb.record_failure()

    future = time.monotonic() + 31.0
    with patch("exo.master.worker_circuit_breaker.time.monotonic", return_value=future):
        allowed = wcb.allow_request()

    assert allowed is True
    assert wcb.state == WorkerCircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# HALF_OPEN probe outcomes
# ---------------------------------------------------------------------------


def test_probe_success_closes_circuit() -> None:
    """A successful probe in HALF_OPEN closes the circuit and clears history."""
    wcb = _make_wcb(failure_threshold=1, open_duration_seconds=0.0)
    wcb.record_failure()  # → OPEN
    wcb.opened_at = 0.0  # force cooldown expired

    wcb.allow_request()  # → HALF_OPEN
    wcb.record_success()  # → CLOSED

    assert wcb.state == WorkerCircuitState.CLOSED
    assert len(wcb.failure_timestamps) == 0


def test_probe_failure_reopens_circuit() -> None:
    """A failed probe in HALF_OPEN re-opens the circuit."""
    wcb = _make_wcb(failure_threshold=1, open_duration_seconds=0.0)
    wcb.record_failure()  # → OPEN
    wcb.opened_at = 0.0

    wcb.allow_request()  # → HALF_OPEN
    wcb.record_failure()  # → OPEN
    assert wcb.state == WorkerCircuitState.OPEN


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_workers_are_isolated() -> None:
    """Each node gets its own independent breaker."""
    reg = WorkerCircuitBreakerRegistry(failure_threshold=2)
    reg.record_failure("node-x")
    reg.record_failure("node-x")  # trips node-x

    assert reg.get("node-x").is_open is True
    assert reg.get("node-y").is_open is False  # unrelated


def test_registry_any_open_reflects_open_breaker() -> None:
    """any_open correctly returns True when at least one breaker is open."""
    reg = WorkerCircuitBreakerRegistry(failure_threshold=1)
    assert reg.any_open() is False

    reg.record_failure("node-z")
    assert reg.any_open() is True
