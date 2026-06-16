"""
Behavioral tests for exo.master.response_cache.

Focuses on:
- make_key() returns None for non-deterministic requests (temp>0, no seed)
- make_key() returns stable sha256 key for deterministic requests
- get() returns None on miss and miss counter increments
- get() returns entry on hit and hit counter increments; touch() increments hit_count
- TTL expiry: get() returns None and removes expired entry
- LRU eviction: oldest entry (by created_at) evicted when max_entries reached
- invalidate_model() removes entries for a specific model only
- stats() hit_rate math
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from exo.master.response_cache import ResponseCache


class TestMakeKey:
    def test_temperature_zero_with_no_seed_is_cacheable(self) -> None:
        key = ResponseCache.make_key(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            seed=None,
        )
        assert key is not None
        assert len(key) == 64  # sha256 hex digest

    def test_positive_temperature_without_seed_returns_none(self) -> None:
        key = ResponseCache.make_key(
            model="gpt-4",
            messages=[],
            temperature=0.9,
            max_tokens=None,
            top_p=1.0,
            seed=None,
        )
        assert key is None

    def test_positive_temperature_with_seed_is_cacheable(self) -> None:
        key = ResponseCache.make_key(
            model="m",
            messages=[],
            temperature=1.0,
            max_tokens=None,
            top_p=1.0,
            seed=7,
        )
        assert key is not None

    def test_same_inputs_produce_identical_key(self) -> None:
        msgs = [{"role": "user", "content": "test"}]
        k1 = ResponseCache.make_key("m", msgs, 0.0, 50, 0.95, None)
        k2 = ResponseCache.make_key("m", msgs, 0.0, 50, 0.95, None)
        assert k1 == k2

    def test_different_models_produce_different_keys(self) -> None:
        msgs: list[dict] = []
        k1 = ResponseCache.make_key("model-a", msgs, 0.0, None, 1.0, None)
        k2 = ResponseCache.make_key("model-b", msgs, 0.0, None, 1.0, None)
        assert k1 != k2


class TestResponseCacheGetPut:
    def test_miss_on_empty_cache(self) -> None:
        cache = ResponseCache(max_entries=10)
        result = cache.get("nonexistent-key")
        assert result is None
        assert cache.stats()["misses"] == 1

    def test_hit_after_put(self) -> None:
        cache = ResponseCache(max_entries=10)
        cache.put("key1", "gpt-4", '{"text":"hi"}', 10, 20)
        entry = cache.get("key1")
        assert entry is not None
        assert entry.model == "gpt-4"
        assert entry.hit_count == 1
        assert cache.stats()["hits"] == 1

    def test_ttl_expiry_removes_entry(self) -> None:
        cache = ResponseCache(max_entries=10, default_ttl_seconds=60.0)
        cache.put("expiring", "m", "{}", 5, 5)
        # Simulate time past TTL
        with patch("time.time", return_value=time.time() + 120.0):
            result = cache.get("expiring")
        assert result is None
        # Entry must have been deleted
        assert "expiring" not in cache._cache

    def test_lru_eviction_removes_oldest_created_at(self) -> None:
        cache = ResponseCache(max_entries=2, default_ttl_seconds=600.0)

        t_base = 1_000_000.0
        with patch("time.time", return_value=t_base):
            cache.put("old", "m", "{}", 1, 1)
        with patch("time.time", return_value=t_base + 1.0):
            cache.put("newer", "m", "{}", 1, 1)
        # Adding a third entry triggers LRU eviction of "old"
        with patch("time.time", return_value=t_base + 2.0):
            cache.put("newest", "m", "{}", 1, 1)

        assert cache.get("old") is None
        assert cache.get("newer") is not None
        assert cache.get("newest") is not None

    def test_custom_ttl_overrides_default(self) -> None:
        cache = ResponseCache(max_entries=10, default_ttl_seconds=300.0)
        cache.put("custom", "m", "{}", 1, 1, ttl_seconds=10.0)
        entry = cache._cache["custom"]
        assert entry.ttl_seconds == pytest.approx(10.0)

    def test_invalidate_model_removes_only_target_model(self) -> None:
        cache = ResponseCache(max_entries=20)
        cache.put("k1", "model-a", "{}", 1, 1)
        cache.put("k2", "model-a", "{}", 1, 1)
        cache.put("k3", "model-b", "{}", 1, 1)

        removed = cache.invalidate_model("model-a")
        assert removed == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        # model-b entry untouched
        assert cache.get("k3") is not None

    def test_stats_hit_rate_formula(self) -> None:
        cache = ResponseCache(max_entries=10)
        cache.put("k", "m", "{}", 1, 1)
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("missing")  # miss

        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3, rel=1e-4)

    def test_prometheus_metrics_contains_expected_lines(self) -> None:
        cache = ResponseCache(max_entries=5)
        metrics = cache.prometheus_metrics()
        assert "exo_response_cache_hit_rate" in metrics
        assert "exo_response_cache_entries" in metrics
