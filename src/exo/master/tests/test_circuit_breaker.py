"""Behavioral tests for exo.master.circuit_breaker.

State machine under test:
  CLOSED -> OPEN  : failure_count >= failure_threshold
  OPEN -> HALF_OPEN: cooldown_seconds elapsed (via time.monotonic mock)
  HALF_OPEN -> CLOSED: success_streak >= success_threshold
  HALF_OPEN -> OPEN  : any failure resets
"""

from __future__ import annotations

import time
from unittest.mock import patch

from exo.master.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cb(**kwargs: object) -> CircuitBreaker:
    """Create a fresh CircuitBreaker with sensible fast-trip defaults."""
    defaults: dict[str, object] = {
        "worker_id": "test-worker",
        "failure_threshold": 3,
        "success_threshold": 2,
        "cooldown_seconds": 30.0,
        "window_seconds": 60.0,
    }
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLOSED → OPEN transition
# ---------------------------------------------------------------------------


def test_closed_trips_on_failure_threshold() -> None:
    """Reaching failure_threshold consecutive failures opens the circuit."""
    cb = _make_cb(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED

    for _ in range(2):
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # not yet

    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_closed_allows_requests_before_threshold() -> None:
    """Circuit remains CLOSED and allows requests until threshold is hit."""
    cb = _make_cb(failure_threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.allow_request() is True
    assert cb.state == CircuitState.CLOSED


def test_success_resets_failure_count() -> None:
    """A success in CLOSED state resets the consecutive-failure counter."""
    cb = _make_cb(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()        # resets failure_count
    cb.record_failure()        # starts from 1 again
    cb.record_failure()        # 2 — still closed
    assert cb.state == CircuitState.CLOSED

    cb.record_failure()        # 3 — trips now
    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN via cooldown (time injection)
# ---------------------------------------------------------------------------


def test_open_rejects_requests_during_cooldown() -> None:
    """While OPEN and cooldown not expired, allow_request returns False."""
    cb = _make_cb(failure_threshold=1, cooldown_seconds=30.0)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False


def test_open_transitions_to_half_open_after_cooldown() -> None:
    """After cooldown_seconds, the first allow_request call enters HALF_OPEN."""
    cb = _make_cb(failure_threshold=1, cooldown_seconds=30.0)
    cb.record_failure()

    # Simulate time passing beyond cooldown
    future = time.monotonic() + 31.0
    with patch("exo.master.circuit_breaker.time.monotonic", return_value=future):
        allowed = cb.allow_request()

    assert allowed is True
    assert cb.state == CircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED or OPEN
# ---------------------------------------------------------------------------


def test_half_open_closes_after_success_streak() -> None:
    """success_threshold consecutive successes in HALF_OPEN closes the circuit."""
    cb = _make_cb(failure_threshold=1, success_threshold=2, cooldown_seconds=0.0)
    cb.record_failure()  # → OPEN
    cb.last_failure_time = 0.0  # force cooldown expired

    cb.allow_request()  # → HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN  # one more needed

    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_reopens_on_failure() -> None:
    """Any failure in HALF_OPEN sends circuit back to OPEN."""
    cb = _make_cb(failure_threshold=1, cooldown_seconds=0.0)
    cb.record_failure()  # → OPEN
    cb.last_failure_time = 0.0

    cb.allow_request()   # → HALF_OPEN
    cb.record_failure()  # → OPEN again
    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Counters and error_rate
# ---------------------------------------------------------------------------


def test_error_rate_reflects_all_outcomes() -> None:
    """error_rate = total_failures / total_requests."""
    cb = _make_cb(failure_threshold=100)
    cb.record_success()
    cb.record_success()
    cb.record_failure()
    # 1/3 failures
    assert abs(cb.error_rate - 1 / 3) < 1e-9


def test_error_rate_zero_division_guard() -> None:
    """error_rate returns 0.0 when no requests recorded."""
    cb = _make_cb()
    assert cb.error_rate == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_creates_independent_breakers() -> None:
    """Each worker_id gets its own independent circuit breaker."""
    reg = CircuitBreakerRegistry()
    # Lower the threshold on w1's breaker before recording failures
    reg.get("w1").failure_threshold = 3
    reg.record_failure("w1")
    reg.record_failure("w1")
    reg.record_failure("w1")

    assert reg.get("w1").state == CircuitState.OPEN
    assert reg.get("w2").state == CircuitState.CLOSED  # unrelated worker


def test_registry_any_open() -> None:
    """any_open returns True iff at least one breaker is OPEN."""
    reg = CircuitBreakerRegistry()
    assert reg.any_open() is False

    # Force open by direct state manipulation
    reg.get("wA").failure_threshold = 1
    reg.record_failure("wA")
    assert reg.any_open() is True
