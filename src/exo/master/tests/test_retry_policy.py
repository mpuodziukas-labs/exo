"""Behavioral tests for exo.master.retry_policy.

RetryManager decides whether to retry a trace_id based on:
  - timeout errors  → never retry
  - max_attempts reached → stop
  - non-retryable HTTP status → stop
  - 429 + quota-exceeded keyword → stop
  - retryable statuses (502, 503, 429) → retry unless quota
  - None status (network error) → retry
"""

from __future__ import annotations

import pytest

from exo.master.retry_policy import RetryManager, RetryPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(
    max_attempts: int = 3,
    base_delay_ms: float = 10.0,
    max_delay_ms: float = 500.0,
    backoff_factor: float = 2.0,
) -> RetryManager:
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
        backoff_factor=backoff_factor,
    )
    return RetryManager(policy=policy)


# ---------------------------------------------------------------------------
# Timeout is never retried
# ---------------------------------------------------------------------------


def test_timeout_error_never_retried() -> None:
    """An error_type containing 'timeout' is immediately rejected."""
    mgr = _mgr(max_attempts=5)
    assert (
        mgr.should_retry("t1", status_code=503, error_type="connection_timeout")
        is False
    )


# ---------------------------------------------------------------------------
# max_attempts guard
# ---------------------------------------------------------------------------


def test_stop_after_max_attempts() -> None:
    """should_retry returns False once the attempt count equals max_attempts."""
    mgr = _mgr(max_attempts=2)
    tid = "t-max"
    assert mgr.should_retry(tid, status_code=503, error_type="") is True
    mgr.record_attempt(tid, 503, None)  # attempt 1

    assert mgr.should_retry(tid, status_code=503, error_type="") is True
    mgr.record_attempt(tid, 503, None)  # attempt 2

    assert mgr.should_retry(tid, status_code=503, error_type="") is False


# ---------------------------------------------------------------------------
# Retryable vs non-retryable HTTP statuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [502, 503, 429])
def test_retryable_statuses_are_allowed(status: int) -> None:
    """HTTP 502, 503, and 429 (non-quota) are retryable."""
    mgr = _mgr()
    assert mgr.should_retry("t-ret", status_code=status, error_type="") is True


@pytest.mark.parametrize("status", [400, 404, 500, 200])
def test_non_retryable_statuses_are_rejected(status: int) -> None:
    """HTTP 400, 404, 500, 200 are NOT retried."""
    mgr = _mgr()
    assert mgr.should_retry("t-noret", status_code=status, error_type="") is False


# ---------------------------------------------------------------------------
# 429 quota exhaustion — no retry
# ---------------------------------------------------------------------------


def test_429_quota_keyword_stops_retry() -> None:
    """429 with a quota-exceeded keyword in error_type is not retried."""
    mgr = _mgr()
    assert (
        mgr.should_retry(
            "t-quota", status_code=429, error_type="daily token quota exceeded"
        )
        is False
    )


def test_429_without_quota_keyword_retries() -> None:
    """429 without the quota keyword (plain rate-limit) is retried."""
    mgr = _mgr()
    assert mgr.should_retry("t-rl", status_code=429, error_type="rate_limited") is True


# ---------------------------------------------------------------------------
# None status (network/internal error)
# ---------------------------------------------------------------------------


def test_none_status_retries_when_not_timeout() -> None:
    """A None status code without 'timeout' in error_type is retried."""
    mgr = _mgr()
    assert (
        mgr.should_retry("t-net", status_code=None, error_type="connection_reset")
        is True
    )


# ---------------------------------------------------------------------------
# Exponential backoff capping
# ---------------------------------------------------------------------------


def test_delay_respects_max_delay() -> None:
    """next_delay_ms never exceeds max_delay_ms + jitter ceiling (100 ms)."""
    mgr = _mgr(base_delay_ms=1000.0, max_delay_ms=500.0, backoff_factor=4.0)
    for _ in range(5):
        delay = mgr.next_delay_ms("t-delay")
    # raw = 1000 * 4^5 >> 500; capped at 500; plus up to 100 jitter
    assert delay <= 600.0 + 1e-6


def test_delay_grows_with_attempt_count() -> None:
    """Delay for attempt 0 is smaller than delay for attempt 2 (no jitter collision)."""
    mgr = _mgr(base_delay_ms=10.0, max_delay_ms=10_000.0, backoff_factor=4.0)
    # Attempt 0 (no record_attempt yet): base * 4^0 = 10 ms + jitter
    delay_a0 = mgr.next_delay_ms("t-grow")
    mgr.record_attempt("t-grow", 503, None)
    mgr.record_attempt("t-grow", 503, None)
    # Attempt 2: base * 4^2 = 160 ms + jitter — definitely larger
    delay_a2 = mgr.next_delay_ms("t-grow")
    assert delay_a2 > delay_a0


# ---------------------------------------------------------------------------
# record_attempt + clear lifecycle
# ---------------------------------------------------------------------------


def test_clear_removes_trace_from_active() -> None:
    """clear() removes the trace from active tracking."""
    mgr = _mgr()
    mgr.record_attempt("t-clr", 503, "err")
    assert "t-clr" in mgr._active
    mgr.clear("t-clr")
    assert "t-clr" not in mgr._active


def test_stats_track_total_retries() -> None:
    """stats() reflects cumulative retry counts."""
    mgr = _mgr(max_attempts=5)
    for i in range(3):
        mgr.record_attempt(f"t-st-{i}", 503, None)

    s = mgr.stats()
    assert s["total_retries"] == 3
    assert s["active_traces"] == 3
