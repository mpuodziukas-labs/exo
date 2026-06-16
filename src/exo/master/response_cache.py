from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger


@dataclass
class CachedResponse:
    cache_key: str
    model: str
    response_body: str  # JSON-serialized full response
    prompt_tokens: int
    completion_tokens: int
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        self.hit_count += 1


class ResponseCache:
    """
    Exact-match response cache keyed by SHA256(model + serialized_messages + params).

    Streaming responses are NOT cached (cache_streaming=False by default).
    Only non-streaming requests with deterministic params (temperature=0 or
    seed is set) are cached.

    LRU eviction + TTL expiry.
    """

    def __init__(
        self,
        max_entries: int = 256,
        default_ttl_seconds: float = 300.0,
        cache_streaming: bool = False,
    ) -> None:
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.cache_streaming = cache_streaming
        self._cache: dict[str, CachedResponse] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        top_p: float,
        seed: int | None,
    ) -> str | None:
        """
        Return cache key if request is cacheable (deterministic params), else None.
        Only cache when temperature=0 (deterministic output) or seed is set.
        """
        if temperature > 0 and seed is None:
            return None  # non-deterministic — don't cache
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, cache_key: str) -> CachedResponse | None:
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._cache[cache_key]
                self._misses += 1
                return None
            entry.touch()
            self._hits += 1
            logger.debug(
                f"Response cache HIT key={cache_key[:8]} hits={entry.hit_count}"
            )
            return entry

    def put(
        self,
        cache_key: str,
        model: str,
        response_body: str,
        prompt_tokens: int,
        completion_tokens: int,
        ttl_seconds: float | None = None,
    ) -> None:
        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._evict_lru()
            self._cache[cache_key] = CachedResponse(
                cache_key=cache_key,
                model=model,
                response_body=response_body,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
            )
            logger.debug(f"Response cache STORE key={cache_key[:8]} model={model}")

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        # Evict oldest by created_at
        lru_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[lru_key]

    def invalidate_model(self, model: str) -> int:
        with self._lock:
            keys = [k for k, v in self._cache.items() if v.model == model]
            for k in keys:
                del self._cache[k]
            return len(keys)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 4),
                "default_ttl_seconds": self.default_ttl_seconds,
            }

    def prometheus_metrics(self) -> str:
        s = self.stats()
        return (
            "# HELP exo_response_cache_hit_rate Response cache hit rate\n"
            "# TYPE exo_response_cache_hit_rate gauge\n"
            f"exo_response_cache_hit_rate {s['hit_rate']:.4f}\n"
            "# HELP exo_response_cache_entries Current response cache entries\n"
            "# TYPE exo_response_cache_entries gauge\n"
            f"exo_response_cache_entries {s['entries']}\n"
        )


RESPONSE_CACHE = ResponseCache(
    max_entries=int(os.getenv("EXO_RESPONSE_CACHE_MAX_ENTRIES", "256")),
    default_ttl_seconds=float(os.getenv("EXO_RESPONSE_CACHE_TTL", "300")),
)
