"""
drift_detector.py — Cluster config drift detection for exo.

Detects when nodes disagree on:
  - exo software version
  - config hash (SHA-256 of ExoConfig.__dict__)
  - loaded model IDs

Alerts via SSE when the cluster is in a split-brain / drifted state.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Final

import anyio
from loguru import logger

from exo.master.config_watcher import CONFIG_WATCHER
from exo.master.event_stream import emit as emit_cluster_event
from exo.master.node_registry import NODE_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIFT_LOOP_INTERVAL: Final[float] = 60.0
EVENT_CLUSTER_DRIFT: Final[str] = "cluster_drift"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class NodeSnapshot:
    """Point-in-time observation of a single node's versioning state."""

    node_id: str
    exo_version: str
    python_version: str
    model_ids: list[str]
    config_hash: str  # SHA-256 hex of json.dumps(ExoConfig.__dict__)
    captured_at: float  # time.time()


@dataclass
class DriftReport:
    """Result of a single cross-node drift check."""

    checked_at: float
    nodes_checked: int
    drifted_fields: list[str]  # e.g. ["exo_version", "config_hash"]
    consensus_value: dict[str, str]  # majority value per field
    outlier_nodes: dict[str, list[str]]  # node_id → list of fields where it differs
    is_drifted: bool


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Maintains per-node snapshots and periodically computes drift reports.

    Drift is defined as any field whose value is not unanimous across all
    registered snapshots.  Majority-vote determines the "consensus" value;
    nodes that deviate are recorded as outliers.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, NodeSnapshot] = {}
        self._reports: deque[DriftReport] = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def register_snapshot(self, snapshot: NodeSnapshot) -> None:
        """Store (or replace) the latest snapshot for snapshot.node_id."""
        self._snapshots[snapshot.node_id] = snapshot
        logger.debug(
            f"[drift] registered snapshot node={snapshot.node_id!r} "
            f"exo={snapshot.exo_version!r} "
            f"cfg_hash={snapshot.config_hash[:12]}…"
        )

    def local_snapshot(self, node_id: str) -> NodeSnapshot:
        """Capture a fresh snapshot for this process (master node)."""
        # --- exo version --------------------------------------------------
        try:
            from importlib.metadata import version as _ver

            exo_version = _ver("exo")
        except Exception:
            exo_version = "dev"

        # --- python version -----------------------------------------------
        python_version = sys.version.split()[0]

        # --- loaded models ------------------------------------------------
        cap = NODE_REGISTRY.get(node_id)
        model_ids: list[str] = list(cap.loaded_models) if cap is not None else []

        # --- config hash --------------------------------------------------
        cfg = CONFIG_WATCHER.get()
        raw = json.dumps(cfg.__dict__, sort_keys=True, default=str)
        config_hash = hashlib.sha256(raw.encode()).hexdigest()

        return NodeSnapshot(
            node_id=node_id,
            exo_version=exo_version,
            python_version=python_version,
            model_ids=sorted(model_ids),
            config_hash=config_hash,
            captured_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------

    def check_drift(self) -> DriftReport:
        """
        Compare all registered snapshots for unanimity across:
          - exo_version
          - config_hash
          - model_ids (as a frozenset, serialised to sorted JSON for hashing)

        Returns a DriftReport and appends it to the rolling history.
        Emits an SSE event when is_drifted is True.
        """
        snapshots = list(self._snapshots.values())
        now = time.time()

        if len(snapshots) < 2:
            report = DriftReport(
                checked_at=now,
                nodes_checked=len(snapshots),
                drifted_fields=[],
                consensus_value={},
                outlier_nodes={},
                is_drifted=False,
            )
            self._reports.append(report)
            return report

        drifted_fields: list[str] = []
        consensus_value: dict[str, str] = {}
        outlier_nodes: dict[str, list[str]] = {}

        # Helper: given a mapping of node_id → value string, determine
        # majority value and collect outliers.
        def _analyse_field(field_name: str, values: dict[str, str]) -> None:
            counts: Counter[str] = Counter(values.values())
            majority_val, _ = counts.most_common(1)[0]
            consensus_value[field_name] = majority_val

            is_unanimous = len(counts) == 1
            if not is_unanimous:
                drifted_fields.append(field_name)
                for node_id, val in values.items():
                    if val != majority_val:
                        outlier_nodes.setdefault(node_id, []).append(field_name)

        # --- exo_version --------------------------------------------------
        _analyse_field(
            "exo_version",
            {s.node_id: s.exo_version for s in snapshots},
        )

        # --- config_hash --------------------------------------------------
        _analyse_field(
            "config_hash",
            {s.node_id: s.config_hash for s in snapshots},
        )

        # --- model_ids (normalised as sorted JSON) -------------------------
        _analyse_field(
            "model_ids",
            {s.node_id: json.dumps(sorted(s.model_ids)) for s in snapshots},
        )

        is_drifted = bool(drifted_fields)
        report = DriftReport(
            checked_at=now,
            nodes_checked=len(snapshots),
            drifted_fields=drifted_fields,
            consensus_value=consensus_value,
            outlier_nodes=outlier_nodes,
            is_drifted=is_drifted,
        )
        self._reports.append(report)

        if is_drifted:
            logger.warning(
                f"[drift] DRIFT DETECTED — fields={drifted_fields} "
                f"outliers={list(outlier_nodes.keys())}"
            )
            emit_cluster_event(
                EVENT_CLUSTER_DRIFT,
                {
                    "drifted_fields": drifted_fields,
                    "outlier_nodes": outlier_nodes,
                    "nodes_checked": len(snapshots),
                    "consensus_value": consensus_value,
                },
            )
        else:
            logger.debug(f"[drift] cluster clean — {len(snapshots)} node(s) unanimous")

        return report

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run_drift_loop(self) -> None:
        """Register local snapshot and check drift every 60 s indefinitely."""
        # Derive a stable local node_id from the hostname.
        import socket

        local_node_id = socket.gethostname()

        logger.info(
            f"[drift] loop started — interval={_DRIFT_LOOP_INTERVAL}s "
            f"local_node={local_node_id!r}"
        )
        while True:
            await anyio.sleep(_DRIFT_LOOP_INTERVAL)
            try:
                snap = self.local_snapshot(local_node_id)
                self.register_snapshot(snap)
                self.check_drift()
            except Exception as exc:
                logger.error(f"[drift] loop error: {exc}")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def latest_report(self) -> DriftReport | None:
        """Return the most recent DriftReport, or None if none have run yet."""
        return self._reports[-1] if self._reports else None

    def stats(self) -> dict[str, object]:
        report = self.latest_report()
        return {
            "snapshots_registered": len(self._snapshots),
            "reports_stored": len(self._reports),
            "latest_is_drifted": report.is_drifted if report else None,
            "latest_drifted_fields": report.drifted_fields if report else [],
            "latest_checked_at": report.checked_at if report else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

DRIFT_DETECTOR: Final[DriftDetector] = DriftDetector()
