"""
Behavioral tests for exo.master.rate_limiter.

Focuses on:
- Token-bucket refill math (elapsed * refill_rate)
- Burst capacity enforcement
- Per-client isolation (anonymous vs API-key vs IP)
- Rate-limiter disabled path
- Stats counters (allowed / rejected / rejection_rate)
- wait_time_seconds formula
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from exo.master.rate_limiter import RateLimiter, TokenBucket

# ---------------------------------------------------------------------------
# TokenBucket unit tests
# ---------------------------------------------------------------------------


class TestTokenBucketRefillMath:
    def test_full_bucket_allows_first_consume(self) -> None:
        """A fresh bucket at capacity allows a cost-1 consume."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, tokens=10.0)
        with patch("time.monotonic", return_value=1000.0):
            bucket.last_refill = 1000.0
            result = bucket.consume(1.0)
        assert result is True
        assert bucket.tokens == pytest.approx(9.0)

    def test_refill_adds_elapsed_times_rate(self) -> None:
        """After 2 s with refill_rate=3 tok/s, tokens increase by 6."""
        bucket = TokenBucket(capacity=20.0, refill_rate=3.0, tokens=0.0)
        t0 = 1000.0
        bucket.last_refill = t0
        with patch("time.monotonic", return_value=t0 + 2.0):
            result = bucket.consume(5.0)  # needs 5, gets 6 from refill
        assert result is True
        # After refill: 0 + 6 = 6; after consuming 5: 1 remaining
        assert bucket.tokens == pytest.approx(1.0)

    def test_capacity_cap_prevents_over_fill(self) -> None:
        """Even with a long elapsed, tokens never exceed capacity."""
        bucket = TokenBucket(capacity=5.0, refill_rate=2.0, tokens=0.0)
        t0 = 1000.0
        bucket.last_refill = t0
        with patch("time.monotonic", return_value=t0 + 100.0):  # would add 200 tokens
            bucket.consume(0.0)  # trigger refill without spending
        assert bucket.tokens == pytest.approx(5.0)

    def test_insufficient_tokens_returns_false(self) -> None:
        """Consume fails when tokens < cost."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, tokens=0.5)
        t0 = 1000.0
        bucket.last_refill = t0
        with patch("time.monotonic", return_value=t0):
            result = bucket.consume(1.0)
        assert result is False
        # tokens unchanged on failure
        assert bucket.tokens == pytest.approx(0.5)

    def test_wait_time_zero_when_token_available(self) -> None:
        bucket = TokenBucket(capacity=5.0, refill_rate=1.0, tokens=3.0)
        assert bucket.wait_time_seconds == pytest.approx(0.0)

    def test_wait_time_formula(self) -> None:
        """wait_time = (1 - tokens) / refill_rate."""
        bucket = TokenBucket(capacity=5.0, refill_rate=2.0, tokens=0.0)
        # needs 1.0 token; rate = 2.0/s → wait = 0.5 s
        assert bucket.wait_time_seconds == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RateLimiter integration tests
# ---------------------------------------------------------------------------


class TestRateLimiterPerClientIsolation:
    def test_separate_clients_dont_interfere(self) -> None:
        """Exhausting one client's bucket doesn't affect another's."""
        rl = RateLimiter()
        # Exhaust client_a entirely
        rl._get_or_create_bucket("client_a").tokens = 0.0
        # client_b starts fresh
        rl._get_or_create_bucket("client_b")

        allowed_a, _ = rl.check("client_a")
        allowed_b, _ = rl.check("client_b")

        assert allowed_a is False
        assert allowed_b is True

    def test_anonymous_uses_lower_rpm(self) -> None:
        """Anonymous bucket is created with the anonymous RPM, not the default."""
        with patch.dict(
            os.environ,
            {"EXO_RATE_LIMIT_ANONYMOUS_RPM": "6", "EXO_RATE_LIMIT_RPM": "60"},
        ):
            rl = RateLimiter()
            bucket = rl._get_or_create_bucket("anonymous")
        # refill_rate = 6/60 = 0.1 tok/s
        assert bucket.refill_rate == pytest.approx(0.1)

    def test_api_key_client_id_hashed(self) -> None:
        """client_id_from_request with API key returns 'key:' + 8-char hex."""
        rl = RateLimiter()
        cid = rl.client_id_from_request("secret-key", None)
        assert cid.startswith("key:")
        assert len(cid) == 4 + 8  # "key:" + 8 hex chars

    def test_ip_client_id(self) -> None:
        rl = RateLimiter()
        cid = rl.client_id_from_request(None, "192.0.2.1")
        assert cid == "ip:192.0.2.1"

    def test_no_api_key_no_ip_gives_anonymous(self) -> None:
        rl = RateLimiter()
        assert rl.client_id_from_request(None, None) == "anonymous"

    def test_disabled_rate_limiter_always_allows(self) -> None:
        with patch.dict(os.environ, {"EXO_RATE_LIMIT_ENABLED": "0"}):
            rl = RateLimiter()
            # Even with an empty bucket it should pass
            rl._get_or_create_bucket("any_client").tokens = 0.0
            allowed, wait = rl.check("any_client")
        assert allowed is True
        assert wait == 0.0

    def test_stats_rejection_rate(self) -> None:
        """Stats rejection_rate is rejected / total.

        Default bucket capacity is 1.5 (60 rpm * 1.5 burst / 60 = 1.5 tokens).
        One allowed call drains it to ~0.5, then we force tokens=0 to trigger rejection.
        """
        rl = RateLimiter()
        # First call: bucket starts full (capacity ~1.5), consume 1 → allowed
        rl._get_or_create_bucket("c")
        rl.check("c")  # allowed
        # Force bucket to zero for the rejection
        rl._buckets["c"].tokens = 0.0
        rl.check("c")  # rejected

        s = rl.stats()
        assert s["allowed_total"] == 1
        assert s["rejected_total"] == 1
        assert s["rejection_rate"] == pytest.approx(0.5, rel=1e-4)

    def test_reset_client_removes_bucket(self) -> None:
        rl = RateLimiter()
        rl._get_or_create_bucket("to_remove")
        assert "to_remove" in rl._buckets
        rl.reset_client("to_remove")
        assert "to_remove" not in rl._buckets


def test_bucket_store_does_not_autocreate_with_wrong_rpm() -> None:
    """Regression: _buckets was a defaultdict whose factory hardcoded the
    authenticated RPM — any direct subscript silently created an anonymous
    bucket with 6x the intended capacity. The store must not auto-create."""
    limiter = RateLimiter()
    with pytest.raises(KeyError):
        _ = limiter._buckets["anonymous"]  # pyright: ignore[reportPrivateUsage]
