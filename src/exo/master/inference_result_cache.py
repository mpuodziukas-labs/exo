from __future__ import annotations

import hashlib
import json
import os as _os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_DEFAULT_TTL = 600.0  # 10 minutes
_MAX_ENTRIES = 5_000
_MAX_VALUE_BYTES = 128 * 1024  # 128 KB max cached response size


@dataclass
class CachedResult:
    cache_key: str
    model_id: str
    response_text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.cached_at > _DEFAULT_TTL

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key[:16],
            "model_id": self.model_id,
            "age_s": round(time.monotonic() - self.cached_at, 1),
            "hit_count": self.hit_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "expired": self.expired,
        }


def _make_cache_key(
    model_id: str, messages: list[dict[str, Any]], temperature: float, max_tokens: int
) -> str:
    """SHA-256 of canonical request parameters."""
    canon = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode()).hexdigest()


class InferenceResultCache:
    """
    LRU cache for inference results keyed by (model, messages, temperature, max_tokens).
    Only caches temperature=0 results (deterministic) unless EXO_CACHE_ALL_TEMPS=1.
    """

    def __init__(self, cache_deterministic_only: bool = True) -> None:
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()
        self._cache_deterministic_only = cache_deterministic_only
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _evict_expired(self) -> None:
        expired = [k for k, v in self._cache.items() if v.expired]
        for k in expired:
            del self._cache[k]
            self._evictions += 1

    def _enforce_cap(self) -> None:
        while len(self._cache) >= _MAX_ENTRIES:
            self._cache.popitem(last=False)
            self._evictions += 1

    def get(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> CachedResult | None:
        if self._cache_deterministic_only and temperature > 0.0:
            return None
        self._evict_expired()
        key = _make_cache_key(model_id, messages, temperature, max_tokens)
        entry = self._cache.get(key)
        if entry is None or entry.expired:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        logger.debug(f"InferenceResultCache HIT key={key[:16]} model={model_id}")
        return entry

    def put(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_text: str,
        finish_reason: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> bool:
        if self._cache_deterministic_only and temperature > 0.0:
            return False
        if len(response_text.encode()) > _MAX_VALUE_BYTES:
            return False
        self._evict_expired()
        self._enforce_cap()
        key = _make_cache_key(model_id, messages, temperature, max_tokens)
        self._cache[key] = CachedResult(
            cache_key=key,
            model_id=model_id,
            response_text=response_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._cache.move_to_end(key)
        return True

    def stats(self) -> dict[str, Any]:
        self._evict_expired()
        return {
            "entries": len(self._cache),
            "max_entries": _MAX_ENTRIES,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 4),
            "deterministic_only": self._cache_deterministic_only,
            "ttl_s": _DEFAULT_TTL,
        }

    def flush(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count


INFERENCE_RESULT_CACHE = InferenceResultCache(
    cache_deterministic_only=_os.getenv("EXO_CACHE_ALL_TEMPS", "0") != "1"
)
