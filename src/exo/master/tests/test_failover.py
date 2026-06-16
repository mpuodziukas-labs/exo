"""Behavioral tests for exo.master.failover.

failover.py imports two real modules:
  - exo.master.heartbeat_monitor  (HEARTBEAT_MONITOR)
  - exo.master.event_stream       (emit)

Both modules now exist on disk.  Tests use pytest monkeypatch to replace the
singletons that failover.py references at call time, so each test controls
exactly what the coordinator sees without mutating global state permanently.

FailoverCoordinator lifecycle under test:
  - trigger()   → creates FailoverEvent, idempotent on duplicate
  - complete()  → finalises event, moves to history, ignores unknown task_id
  - select_failover_node() → picks lowest error-rate alive node
  - should_failover() → gates on circuit-breaker state + heartbeat eviction
  - stats()     → tracks total/successful/active
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from exo.master.circuit_breaker import CIRCUIT_BREAKERS, CircuitState
from exo.master.failover import FailoverCoordinator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> FailoverCoordinator:
    """Return a new coordinator instance."""
    return FailoverCoordinator()


# ---------------------------------------------------------------------------
# trigger()
# ---------------------------------------------------------------------------


def test_trigger_creates_event_and_increments_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trigger() creates a FailoverEvent and increments _total_failovers."""
    fake_hb = MagicMock()
    fake_hb.evicted_nodes.return_value = set()
    fake_hb.alive_nodes.return_value = []
    fake_emit = MagicMock()
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", fake_emit)

    coord = _fresh()
    event = coord.trigger("task-1", "node-dead", "heartbeat_evicted")

    assert event.task_id == "task-1"
    assert event.failed_node_id == "node-dead"
    assert event.reason == "heartbeat_evicted"
    assert event.completed_at is None
    assert event.success is False
    assert coord.stats()["total"] == 1
    assert coord.stats()["active"] == 1


def test_trigger_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling trigger twice for the same task returns the same event."""
    fake_hb = MagicMock()
    fake_hb.evicted_nodes.return_value = set()
    fake_hb.alive_nodes.return_value = []
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    coord = _fresh()
    e1 = coord.trigger("task-dup", "node-x", "cb_open")
    e2 = coord.trigger("task-dup", "node-x", "cb_open")

    assert e1 is e2
    assert coord.stats()["total"] == 1  # counted only once


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


def test_complete_moves_event_to_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """complete() finalises an active event and adds it to history."""
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", MagicMock())
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    coord = _fresh()
    coord.trigger("task-c", "node-f", "reason")
    coord.complete("task-c", "node-g", success=True)

    assert coord.stats()["active"] == 0
    assert coord.stats()["successful"] == 1
    hist = coord.history()
    assert len(hist) == 1
    assert hist[0].success is True
    assert hist[0].failover_node_id == "node-g"


def test_complete_with_unknown_task_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """complete() for a non-existent task_id silently does nothing."""
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", MagicMock())
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    coord = _fresh()
    coord.complete("no-such-task", "node-x", success=True)
    assert coord.stats()["total"] == 0
    assert coord.stats()["successful"] == 0


def test_complete_failure_does_not_increment_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed failover (success=False) does not count as successful."""
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", MagicMock())
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    coord = _fresh()
    coord.trigger("task-fail", "node-dead", "cb")
    coord.complete("task-fail", None, success=False)

    assert coord.stats()["successful"] == 0
    assert coord.stats()["total"] == 1


# ---------------------------------------------------------------------------
# select_failover_node()
# ---------------------------------------------------------------------------


def test_select_failover_node_returns_none_when_no_alive_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns None when heartbeat has no alive nodes."""
    fake_hb = MagicMock()
    fake_hb.alive_nodes.return_value = []
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    coord = _fresh()
    result = coord.select_failover_node("dead-node")
    assert result is None


def test_select_failover_node_excludes_failed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failed node itself is never selected as failover target."""
    fake_hb = MagicMock()
    fake_hb.alive_nodes.return_value = ["dead-node", "node-b"]
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    # Ensure node-b has a closed circuit breaker
    CIRCUIT_BREAKERS.get("node-b").state = CircuitState.CLOSED

    coord = _fresh()
    result = coord.select_failover_node("dead-node")
    assert result == "node-b"


def test_select_failover_node_prefers_healthy_over_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nodes with CLOSED circuit breakers are preferred over OPEN ones."""
    fake_hb = MagicMock()
    fake_hb.alive_nodes.return_value = ["node-open", "node-closed"]
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)
    monkeypatch.setattr("exo.master.failover.emit_cluster_event", MagicMock())

    CIRCUIT_BREAKERS.get("node-open").state = CircuitState.OPEN
    CIRCUIT_BREAKERS.get("node-closed").state = CircuitState.CLOSED

    coord = _fresh()
    result = coord.select_failover_node("dead")
    assert result == "node-closed"


# ---------------------------------------------------------------------------
# should_failover()
# ---------------------------------------------------------------------------


def test_should_failover_true_when_cb_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """should_failover returns True if the node's circuit breaker is OPEN."""
    fake_hb = MagicMock()
    fake_hb.evicted_nodes.return_value = set()
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)

    coord = _fresh()
    CIRCUIT_BREAKERS.get("dying-node").state = CircuitState.OPEN
    assert coord.should_failover("dying-node") is True


def test_should_failover_true_when_heartbeat_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """should_failover returns True if heartbeat has evicted the node."""
    fake_hb = MagicMock()
    fake_hb.evicted_nodes.return_value = {"hb-node"}
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)

    coord = _fresh()
    CIRCUIT_BREAKERS.get("hb-node").state = CircuitState.CLOSED
    assert coord.should_failover("hb-node") is True


def test_should_failover_false_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """should_failover returns False for a healthy node with no eviction."""
    fake_hb = MagicMock()
    fake_hb.evicted_nodes.return_value = set()
    monkeypatch.setattr("exo.master.failover.HEARTBEAT_MONITOR", fake_hb)

    coord = _fresh()
    CIRCUIT_BREAKERS.get("healthy-node").state = CircuitState.CLOSED
    assert coord.should_failover("healthy-node") is False
