"""
Session affinity: for multi-turn conversations, routes subsequent turns to the
same worker node that handled the first turn (KV cache locality).
Tracks session → node_id mapping with TTL expiry and LRU eviction.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_DEFAULT_TTL = 1800.0   # 30 minutes
_DEFAULT_MAX_SESSIONS = 10_000


@dataclass
class AffinityEntry:
    session_id: str
    node_id: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ttl: float = _DEFAULT_TTL
    hit_count: int = 0

    def is_expired(self) -> bool:
        return time.time() - self.last_used > self.ttl

    def touch(self) -> None:
        self.last_used = time.time()
        self.hit_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "hit_count": self.hit_count,
            "age_s": round(time.time() - self.created_at, 1),
            "expires_in_s": round(max(0, self.ttl - (time.time() - self.last_used)), 1),
        }


class SessionAffinityManager:
    """
    bind / record(session_id, node_id): assigns a session to a node.
    get / get_node(session_id): returns node_id if bound and not expired, else None.
    evict / clear(session_id): removes binding.
    LRU eviction enforces max_sessions cap.
    """

    def __init__(self, max_sessions: int = _DEFAULT_MAX_SESSIONS) -> None:
        self.max_sessions = max_sessions
        self._entries: collections.OrderedDict[str, AffinityEntry] = collections.OrderedDict()
        self._total_sessions: int = 0  # monotonically increasing unique session count

    # ------------------------------------------------------------------ bind / record
    def bind(self, session_id: str, node_id: str, ttl: float = _DEFAULT_TTL) -> None:
        """Assign session to node (no-op if already bound; use record() to update)."""
        if session_id not in self._entries:
            self._evict_if_full()
            self._entries[session_id] = AffinityEntry(
                session_id=session_id, node_id=node_id, ttl=ttl
            )
            self._total_sessions += 1
            logger.debug(f"SessionAffinity: bound session={session_id} → node={node_id}")
        else:
            # Move to end (most-recently-used)
            self._entries.move_to_end(session_id)

    def record(self, session_id: str, node_id: str, ttl: float = _DEFAULT_TTL) -> None:
        """Bind or update an existing session mapping (strengthens affinity)."""
        if session_id in self._entries:
            entry = self._entries[session_id]
            entry.node_id = node_id
            entry.touch()
            self._entries.move_to_end(session_id)
            logger.debug(f"SessionAffinity: updated session={session_id} → node={node_id}")
        else:
            self._evict_if_full()
            self._entries[session_id] = AffinityEntry(
                session_id=session_id, node_id=node_id, ttl=ttl
            )
            self._total_sessions += 1
            logger.debug(f"SessionAffinity: recorded session={session_id} → node={node_id}")

    # ------------------------------------------------------------------ get / get_node
    def get(self, session_id: str) -> str | None:
        entry = self._entries.get(session_id)
        if entry is None:
            return None
        if entry.is_expired():
            del self._entries[session_id]
            logger.debug(f"SessionAffinity: session={session_id} expired")
            return None
        entry.touch()
        self._entries.move_to_end(session_id)
        return entry.node_id

    def get_node(self, session_id: str) -> str | None:
        """Alias for get(); preferred name in the inference handler."""
        return self.get(session_id)

    # ------------------------------------------------------------------ evict / clear
    def evict(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def clear(self, session_id: str) -> None:
        """Alias for evict()."""
        self.evict(session_id)

    # ------------------------------------------------------------------ helpers
    def _evict_if_full(self) -> None:
        """LRU evict oldest entry when at capacity."""
        while len(self._entries) >= self.max_sessions:
            oldest_sid, _ = next(iter(self._entries.items()))
            del self._entries[oldest_sid]
            logger.debug(f"SessionAffinity: LRU evicted session={oldest_sid}")

    def session_count(self) -> int:
        """Total unique sessions ever registered (monotonic)."""
        return self._total_sessions

    def list_sessions(self) -> list[str]:
        """Return all active (non-expired) session IDs."""
        return [sid for sid, e in list(self._entries.items()) if not e.is_expired()]

    def purge_expired(self) -> int:
        expired = [sid for sid, e in list(self._entries.items()) if e.is_expired()]
        for sid in expired:
            del self._entries[sid]
        return len(expired)

    def stats(self) -> dict[str, Any]:
        """Alias for get_stats() with additional total_sessions field."""
        active = [e for e in self._entries.values() if not e.is_expired()]
        return {
            "total_sessions": self._total_sessions,
            "active_sessions": len(active),
            "total_hits": sum(e.hit_count for e in active),
            "sessions": [e.to_dict() for e in active[:20]],
        }

    def get_stats(self) -> dict[str, Any]:
        return self.stats()


SESSION_AFFINITY = SessionAffinityManager()
