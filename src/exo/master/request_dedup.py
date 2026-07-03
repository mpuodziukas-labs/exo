from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import Any

from loguru import logger

_DEDUP_TTL: float = float(os.getenv("EXO_DEDUP_TTL", "30"))


@dataclass
class DedupEntry:
    key: str
    first_trace_id: str
    waiters: list[asyncio.Future[Any]] = field(default_factory=list)
    result: object = None
    exception: BaseException | None = None
    created_at: float = field(default_factory=time.monotonic)
    completed: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_expired(self, ttl_seconds: float = _DEDUP_TTL) -> bool:
        return (time.monotonic() - self.created_at) > ttl_seconds


def compute_dedup_key(
    model_id: str,
    messages: list[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    seed: int | None,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def is_deterministic(temperature: float | None, seed: int | None) -> bool:
    """Only dedup when output is reproducible: temp=0 or seed is pinned."""
    return (temperature is None or temperature == 0.0) or (seed is not None)


class RequestDeduplicator:
    """
    In-flight request deduplicator.

    Identical concurrent deterministic requests share one inference pass.
    The first caller gets `is_new=True` and must drive inference to completion
    then call `complete()` or `fail()`.  Subsequent callers with the same key
    get `is_new=False` and an already-registered Future they can await.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DedupEntry] = {}
        self._global_lock: asyncio.Lock = asyncio.Lock()
        self._hits: int = 0
        self._ttl_seconds: float = _DEDUP_TTL

    def configure(self, *, ttl_seconds: float) -> None:
        """Override the in-flight entry TTL (e.g. from hot-reloaded config)."""
        self._ttl_seconds = ttl_seconds

    def get_entry(self, key: str) -> DedupEntry | None:
        """Public accessor for the in-flight entry registered under `key`."""
        return self._entries.get(key)

    async def get_or_create(
        self, key: str, trace_id: str
    ) -> tuple[bool, asyncio.Future[Any]]:
        """
        Returns (is_new, future).

        is_new=True  → caller is responsible for running inference and calling
                        complete()/fail() when done.
        is_new=False → caller should await the future; result was already in-flight.
        """
        async with self._global_lock:
            entry = self._entries.get(key)

            if entry is not None and not entry.is_expired(self._ttl_seconds):
                # Already in-flight: subscribe a new waiter
                loop = asyncio.get_event_loop()
                waiter: asyncio.Future[Any] = loop.create_future()
                if entry.completed:
                    # Resolved between the lock grab and now — hand result directly
                    if entry.exception is not None:
                        waiter.set_exception(entry.exception)
                    else:
                        waiter.set_result(entry.result)
                else:
                    entry.waiters.append(waiter)
                self._hits += 1
                logger.debug(
                    f"[dedup] HIT key={key[:8]} first_trace={entry.first_trace_id} subscriber={trace_id}"
                )
                return False, waiter

            # New entry
            loop = asyncio.get_event_loop()
            primary: asyncio.Future[Any] = loop.create_future()
            new_entry = DedupEntry(key=key, first_trace_id=trace_id)
            new_entry.waiters.append(primary)
            self._entries[key] = new_entry
            logger.debug(f"[dedup] NEW key={key[:8]} trace={trace_id}")
            return True, primary

    def complete(self, key: str, result: object) -> None:
        """Resolve all waiters with result and mark entry completed."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.result = result
        entry.completed = True
        for waiter in entry.waiters:
            if not waiter.done():
                waiter.set_result(result)
        logger.debug(f"[dedup] COMPLETE key={key[:8]} waiters={len(entry.waiters)}")

    def fail(self, key: str, exc: BaseException) -> None:
        """Propagate exception to all waiters."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.exception = exc
        entry.completed = True
        for waiter in entry.waiters:
            if not waiter.done():
                waiter.set_exception(exc)
        logger.debug(f"[dedup] FAIL key={key[:8]} exc={exc!r}")

    def cleanup_expired(self) -> int:
        expired = [
            k for k, e in self._entries.items() if e.is_expired(self._ttl_seconds)
        ]
        for k in expired:
            entry = self._entries.pop(k)
            # Cancel any still-pending waiters so callers don't hang
            for waiter in entry.waiters:
                if not waiter.done():
                    waiter.cancel()
        if expired:
            logger.debug(f"[dedup] evicted {len(expired)} expired entries")
        return len(expired)

    def stats(self) -> dict[str, Any]:
        return {
            "active_entries": len(self._entries),
            "hits_total": self._hits,
            "ttl_seconds": self._ttl_seconds,
        }


DEDUP = RequestDeduplicator()


# ---------------------------------------------------------------------------
# In-memory LRU idempotency cache (DD4)
# Uses X-Idempotency-Key header value as cache key.
# Falls back to SHA-256(request_body[:512]) if no key provided.
# ---------------------------------------------------------------------------


_DEFAULT_TTL = 300.0  # 5 minutes
_MAX_ENTRIES = 10_000
_CONTENT_HASH_BYTES = 32  # bytes


@_dataclass
class IdempotencyEntry:
    idempotency_key: str
    content_hash: str
    response_status: int
    cached_at: float = _field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.cached_at > _DEFAULT_TTL

    def to_dict(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "content_hash": self.content_hash,
            "response_status": self.response_status,
            "age_s": round(time.monotonic() - self.cached_at, 1),
            "expired": self.expired,
        }


class RequestDedup:
    """
    In-memory LRU idempotency cache.
    Uses X-Idempotency-Key header value as cache key.
    Falls back to SHA-256(request_body[:512]) if no key provided.

    Evicts expired entries lazily on every check() call.
    Hard cap at _MAX_ENTRIES via LRU eviction.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._cache: OrderedDict[str, IdempotencyEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._max_entries: int = _MAX_ENTRIES if max_entries is None else max_entries

    def _evict_expired(self) -> None:
        expired_keys = [k for k, v in self._cache.items() if v.expired]
        for k in expired_keys:
            del self._cache[k]

    def _enforce_cap(self) -> None:
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)  # LRU eviction

    @staticmethod
    def content_hash(body: bytes) -> str:
        return hashlib.sha256(body[:512]).hexdigest()[:16]

    def check(self, key: str) -> IdempotencyEntry | None:
        """Return cached entry if key exists and is not expired, else None."""
        self._evict_expired()
        entry = self._cache.get(key)
        if entry is None or entry.expired:
            self._misses += 1
            return None
        # Move to end (LRU touch)
        self._cache.move_to_end(key)
        self._hits += 1
        logger.debug(f"RequestDedup HIT key={key[:16]}")
        return entry

    def register(self, key: str, content_hash: str, response_status: int) -> None:
        """Store a completed request in the dedup cache."""
        self._evict_expired()
        self._enforce_cap()
        self._cache[key] = IdempotencyEntry(
            idempotency_key=key,
            content_hash=content_hash,
            response_status=response_status,
        )
        self._cache.move_to_end(key)

    def stats(self) -> dict[str, Any]:
        self._evict_expired()
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 4),
            "ttl_s": _DEFAULT_TTL,
        }


REQUEST_DEDUP = RequestDedup()
