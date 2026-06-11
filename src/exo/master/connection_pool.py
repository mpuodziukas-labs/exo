from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

from loguru import logger

ConnectionState = Literal["connecting", "connected", "draining", "closed", "error"]

_ERROR_THRESHOLD = 3


@dataclass
class PooledConnection:
    """
    Represents one persistent TCP slot in the pool.

    Lifecycle:
      connecting -> connected  (happy path)
      connected  -> draining   (graceful eviction)
      connected  -> error      (too many failures)
      draining   -> closed     (after drain completes)
      error      -> connecting (on reconnect attempt)
    """

    conn_id: str
    node_id: str
    host: str
    port: int
    state: ConnectionState
    created_at: float
    last_used: float
    bytes_sent: int = 0
    bytes_recv: int = 0
    errors: int = 0

    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "conn_id": self.conn_id,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "state": self.state,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "errors": self.errors,
            "age_seconds": round(self.age_seconds(), 2),
            "idle_seconds": round(self.idle_seconds(), 2),
        }


class ConnectionPool:
    """
    Inter-node connection pool with backpressure, health tracking, and
    automatic reconnect bookkeeping.

    One pool instance is shared process-wide (CONNECTION_POOL singleton).

    Pool sizing:
      - Up to _max_per_node connections per remote node.
      - Selection policy: least-recently-used among "connected" entries.

    Backpressure:
      - backpressure_active(node_id) returns True when every connection to
        that node is in "error" state — callers should stop sending and wait.

    Stats / observability:
      - stats()         → structured dict for /v1/connection-pool JSON.
      - to_prometheus() → Prometheus text for /metrics.
    """

    def __init__(
        self,
        max_per_node: int = 4,
        backpressure_threshold: int = 100,
    ) -> None:
        self._pool: dict[str, list[PooledConnection]] = {}
        self._max_per_node: int = max_per_node
        self._backpressure_threshold: int = backpressure_threshold
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register_node(self, node_id: str, host: str, port: int) -> None:
        """
        Seed the pool with _max_per_node connection slots for *node_id*.

        Idempotent: existing entries are left untouched; only missing slots
        are added up to the cap.
        """
        with self._lock:
            existing = self._pool.setdefault(node_id, [])
            to_create = self._max_per_node - len(existing)
            if to_create <= 0:
                return
            now = time.monotonic()
            for _ in range(to_create):
                conn = PooledConnection(
                    conn_id=str(uuid.uuid4()),
                    node_id=node_id,
                    host=host,
                    port=port,
                    state="connecting",
                    created_at=now,
                    last_used=now,
                )
                existing.append(conn)
            logger.info(
                f"[connection_pool] registered node={node_id} host={host}:{port} "
                f"slots_created={to_create} total={len(existing)}"
            )

    # ------------------------------------------------------------------ #
    # Acquisition                                                          #
    # ------------------------------------------------------------------ #

    def get_connection(self, node_id: str) -> PooledConnection | None:
        """
        Return the least-recently-used **connected** connection for *node_id*.

        Returns None if the node is unknown or has no healthy connections.
        The caller is responsible for marking sends/recvs via record_send /
        record_recv.
        """
        with self._lock:
            conns = self._pool.get(node_id)
            if not conns:
                logger.debug(f"[connection_pool] get_connection: unknown node={node_id}")
                return None
            healthy = [c for c in conns if c.state == "connected"]
            if not healthy:
                logger.warning(
                    f"[connection_pool] no connected slot for node={node_id} "
                    f"(states={[c.state for c in conns]})"
                )
                return None
            # LRU: pick the one used longest ago
            lru = min(healthy, key=lambda c: c.last_used)
            lru.last_used = time.monotonic()
            return lru

    # ------------------------------------------------------------------ #
    # Stats mutation                                                       #
    # ------------------------------------------------------------------ #

    def record_send(self, conn_id: str, bytes_sent: int) -> None:
        """Accumulate bytes sent on *conn_id*."""
        conn = self._find(conn_id)
        if conn is None:
            logger.debug(f"[connection_pool] record_send: unknown conn_id={conn_id}")
            return
        with self._lock:
            conn.bytes_sent += bytes_sent
            conn.last_used = time.monotonic()

    def record_recv(self, conn_id: str, bytes_recv: int) -> None:
        """Accumulate bytes received on *conn_id*."""
        conn = self._find(conn_id)
        if conn is None:
            logger.debug(f"[connection_pool] record_recv: unknown conn_id={conn_id}")
            return
        with self._lock:
            conn.bytes_recv += bytes_recv
            conn.last_used = time.monotonic()

    def record_error(self, conn_id: str) -> None:
        """
        Increment error counter for *conn_id*.

        Transitions state to "error" once errors exceed _ERROR_THRESHOLD,
        which will trigger backpressure for that node if all slots degrade.
        """
        conn = self._find(conn_id)
        if conn is None:
            logger.debug(f"[connection_pool] record_error: unknown conn_id={conn_id}")
            return
        with self._lock:
            conn.errors += 1
            if conn.errors > _ERROR_THRESHOLD and conn.state not in ("draining", "closed"):
                logger.error(
                    f"[connection_pool] conn={conn_id} node={conn.node_id} "
                    f"errors={conn.errors} -> state=error"
                )
                conn.state = "error"

    # ------------------------------------------------------------------ #
    # Backpressure                                                         #
    # ------------------------------------------------------------------ #

    def backpressure_active(self, node_id: str) -> bool:
        """
        Returns True if **every** connection to *node_id* is in error state
        (or the node has no slots at all).

        Callers should treat this as a signal to pause sending to this node
        and wait for reconnect / recovery.
        """
        with self._lock:
            conns = self._pool.get(node_id)
            if not conns:
                return True
            return all(c.state == "error" for c in conns)

    # ------------------------------------------------------------------ #
    # Drain / close                                                        #
    # ------------------------------------------------------------------ #

    def drain_node(self, node_id: str) -> None:
        """
        Gracefully drain all connections to *node_id*.

        Sets every non-closed connection to "draining" so in-flight requests
        can finish before the connections are torn down.
        """
        with self._lock:
            conns = self._pool.get(node_id, [])
            for conn in conns:
                if conn.state not in ("closed",):
                    conn.state = "draining"
            logger.info(
                f"[connection_pool] drain_node node={node_id} "
                f"drained={sum(1 for c in conns if c.state == 'draining')}"
            )

    # ------------------------------------------------------------------ #
    # Observability                                                        #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """
        Return per-node summary dict suitable for JSON serialisation.

        Shape:
          {
            "<node_id>": {
              "connection_count": int,
              "total_bytes_sent": int,
              "total_bytes_recv": int,
              "error_count": int,
              "backpressure": bool,
              "connections": [PooledConnection.to_dict(), ...]
            },
            ...
          }
        """
        result: dict[str, Any] = {}
        with self._lock:
            for node_id, conns in self._pool.items():
                result[node_id] = {
                    "connection_count": len(conns),
                    "total_bytes_sent": sum(c.bytes_sent for c in conns),
                    "total_bytes_recv": sum(c.bytes_recv for c in conns),
                    "error_count": sum(c.errors for c in conns),
                    "backpressure": all(c.state == "error" for c in conns) if conns else True,
                    "connections": [c.to_dict() for c in conns],
                }
        return result

    def to_prometheus(self) -> str:
        """
        Emit Prometheus gauge: exo_pool_connections_active{node="<node_id>"}.

        Value = number of connections in "connected" state for that node.
        """
        lines = [
            "# HELP exo_pool_connections_active Active (connected) pool connections per node",
            "# TYPE exo_pool_connections_active gauge",
        ]
        with self._lock:
            for node_id, conns in self._pool.items():
                active = sum(1 for c in conns if c.state == "connected")
                lines.append(f'exo_pool_connections_active{{node="{node_id}"}} {active}')
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _find(self, conn_id: str) -> PooledConnection | None:
        """O(N) scan across all nodes; pool sizes are small (max_per_node * nodes)."""
        with self._lock:
            for conns in self._pool.values():
                for conn in conns:
                    if conn.conn_id == conn_id:
                        return conn
        return None

    # ------------------------------------------------------------------ #
    # State promotion helpers (for reconnect logic in callers)            #
    # ------------------------------------------------------------------ #

    def mark_connected(self, conn_id: str) -> None:
        """Transition a connecting/error slot to connected."""
        conn = self._find(conn_id)
        if conn is None:
            return
        with self._lock:
            if conn.state in ("connecting", "error"):
                logger.info(
                    f"[connection_pool] conn={conn_id} node={conn.node_id} "
                    f"{conn.state} -> connected"
                )
                conn.state = "connected"
                conn.errors = 0

    def mark_closed(self, conn_id: str) -> None:
        """Transition any slot to closed (used after drain completes)."""
        conn = self._find(conn_id)
        if conn is None:
            return
        with self._lock:
            prev = conn.state
            conn.state = "closed"
            logger.info(
                f"[connection_pool] conn={conn_id} node={conn.node_id} {prev} -> closed"
            )


CONNECTION_POOL = ConnectionPool()
