"""
Behavioral tests for exo.master.request_dedup.

Tests cover TWO classes in the module:
  1. RequestDedup  — LRU idempotency cache (sync, used for X-Idempotency-Key)
  2. RequestDeduplicator — async in-flight dedup

Focuses on:
- Idempotency cache: HIT path, MISS on unknown key, expiry eviction, LRU cap
- is_deterministic() logic
- compute_dedup_key() stability and sensitivity
- Async RequestDeduplicator: new vs hit entry, complete()/fail() propagation
- Expired entry eviction via cleanup_expired()
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from exo.master.request_dedup import (
    RequestDedup,
    RequestDeduplicator,
    compute_dedup_key,
    is_deterministic,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestIsDeterministic:
    def test_temperature_zero_is_deterministic(self) -> None:
        assert is_deterministic(temperature=0.0, seed=None) is True

    def test_temperature_none_is_deterministic(self) -> None:
        assert is_deterministic(temperature=None, seed=None) is True

    def test_positive_temperature_no_seed_is_not_deterministic(self) -> None:
        assert is_deterministic(temperature=0.7, seed=None) is False

    def test_positive_temperature_with_seed_is_deterministic(self) -> None:
        assert is_deterministic(temperature=0.7, seed=42) is True


class TestComputeDedupKey:
    def test_same_inputs_produce_same_key(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = compute_dedup_key("gpt-4", msgs, 0.0, 100, 42)
        k2 = compute_dedup_key("gpt-4", msgs, 0.0, 100, 42)
        assert k1 == k2

    def test_different_model_produces_different_key(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = compute_dedup_key("model-a", msgs, 0.0, 100, None)
        k2 = compute_dedup_key("model-b", msgs, 0.0, 100, None)
        assert k1 != k2

    def test_different_messages_produce_different_key(self) -> None:
        k1 = compute_dedup_key("m", [{"role": "user", "content": "a"}], 0.0, None, None)
        k2 = compute_dedup_key("m", [{"role": "user", "content": "b"}], 0.0, None, None)
        assert k1 != k2


# ---------------------------------------------------------------------------
# RequestDedup (LRU idempotency cache)
# ---------------------------------------------------------------------------

class TestRequestDedupCache:
    def test_miss_on_unknown_key(self) -> None:
        cache = RequestDedup()
        result = cache.check("nonexistent")
        assert result is None

    def test_hit_after_register(self) -> None:
        cache = RequestDedup()
        cache.register("idem-key-1", content_hash="abc123", response_status=200)
        entry = cache.check("idem-key-1")
        assert entry is not None
        assert entry.response_status == 200
        assert entry.idempotency_key == "idem-key-1"

    def test_expired_entry_returns_none(self) -> None:
        """Entry whose cached_at is older than TTL should be treated as a miss."""
        cache = RequestDedup()
        cache.register("old-key", content_hash="xyz", response_status=200)
        # Fast-forward time past TTL (default 300 s)
        with patch("time.monotonic", return_value=time.monotonic() + 400.0):
            result = cache.check("old-key")
        assert result is None

    def test_lru_eviction_at_capacity(self) -> None:
        """When cache is full, the oldest entry is evicted on next register."""
        from exo.master import request_dedup as _mod

        original_max = _mod._MAX_ENTRIES
        try:
            _mod._MAX_ENTRIES = 3
            cache = RequestDedup()
            cache.register("k1", "h1", 200)
            cache.register("k2", "h2", 200)
            cache.register("k3", "h3", 200)
            # Adding k4 should evict k1 (oldest/LRU)
            cache.register("k4", "h4", 200)
            assert cache.check("k1") is None
            assert cache.check("k4") is not None
        finally:
            _mod._MAX_ENTRIES = original_max

    def test_stats_hit_rate(self) -> None:
        cache = RequestDedup()
        cache.register("k", "h", 200)
        cache.check("k")          # hit
        cache.check("missing")    # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_content_hash_uses_first_512_bytes(self) -> None:
        body = b"x" * 600
        h = RequestDedup.content_hash(body)
        # Must equal sha256 of first 512 bytes, 16-char hex
        import hashlib
        expected = hashlib.sha256(body[:512]).hexdigest()[:16]
        assert h == expected


# ---------------------------------------------------------------------------
# RequestDeduplicator (async in-flight dedup)
# ---------------------------------------------------------------------------

class TestRequestDeduplicator:
    def test_first_caller_gets_is_new_true(self) -> None:
        async def run() -> None:
            dedup = RequestDeduplicator()
            is_new, _ = await dedup.get_or_create("key-a", "trace-1")
            assert is_new is True

        asyncio.run(run())

    def test_second_caller_same_key_gets_is_new_false(self) -> None:
        async def run() -> None:
            dedup = RequestDeduplicator()
            is_new_1, _ = await dedup.get_or_create("key-b", "trace-1")
            is_new_2, _ = await dedup.get_or_create("key-b", "trace-2")
            assert is_new_1 is True
            assert is_new_2 is False
            assert dedup.stats()["hits_total"] == 1

        asyncio.run(run())

    def test_complete_propagates_result_to_waiters(self) -> None:
        async def run() -> None:
            dedup = RequestDeduplicator()
            _, primary = await dedup.get_or_create("key-c", "trace-1")
            _, waiter = await dedup.get_or_create("key-c", "trace-2")
            dedup.complete("key-c", result={"answer": 42})
            assert primary.result() == {"answer": 42}
            assert waiter.result() == {"answer": 42}

        asyncio.run(run())

    def test_fail_propagates_exception_to_waiters(self) -> None:
        async def run() -> None:
            dedup = RequestDeduplicator()
            _, primary = await dedup.get_or_create("key-d", "trace-1")
            _, waiter = await dedup.get_or_create("key-d", "trace-2")
            err = RuntimeError("inference failed")
            dedup.fail("key-d", err)
            with pytest.raises(RuntimeError, match="inference failed"):
                primary.result()
            with pytest.raises(RuntimeError, match="inference failed"):
                waiter.result()

        asyncio.run(run())

    def test_expired_entry_is_cleaned_up(self) -> None:
        async def run() -> None:
            dedup = RequestDeduplicator()
            await dedup.get_or_create("key-exp", "trace-1")
            # Simulate expiry by backdating the entry's created_at
            entry = dedup._entries["key-exp"]
            entry.created_at = time.monotonic() - 9999.0
            count = dedup.cleanup_expired()
            assert count == 1
            assert "key-exp" not in dedup._entries

        asyncio.run(run())
