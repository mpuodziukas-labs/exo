"""
Cluster snapshot export: periodically writes a full cluster state snapshot
to ~/.exo/snapshots/{timestamp}.json for offline debugging and audit.
Keeps only the last N snapshots (default 24 — one per hour if run hourly).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

_SNAPSHOT_DIR = Path.home() / ".exo" / "snapshots"
_MAX_SNAPSHOTS = 24


@dataclass
class ClusterSnapshot:
    timestamp: float
    node_id: str
    state_summary: dict[str, Any]
    health: dict[str, Any]
    metrics_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "state_summary": self.state_summary,
            "health": self.health,
            "metrics_summary": self.metrics_summary,
        }


class ClusterSnapshotManager:
    """
    take(node_id, state_summary, health, metrics_summary): writes snapshot to disk.
    list_snapshots(): returns metadata for all stored snapshots, newest first.
    load(filename): reads a snapshot by filename.
    """

    def __init__(self) -> None:
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def take(
        self,
        node_id: str,
        state_summary: dict[str, Any],
        health: dict[str, Any],
        metrics_summary: dict[str, Any],
    ) -> str:
        snap = ClusterSnapshot(
            timestamp=time.time(),
            node_id=node_id,
            state_summary=state_summary,
            health=health,
            metrics_summary=metrics_summary,
        )
        filename = f"{int(snap.timestamp)}.json"
        path = _SNAPSHOT_DIR / filename
        try:
            path.write_text(json.dumps(snap.to_dict(), ensure_ascii=False))
            logger.debug(f"ClusterSnapshot: wrote {filename}")
        except Exception as exc:
            logger.warning(f"ClusterSnapshot write failed: {exc}")
        self._rotate()
        return filename

    def _rotate(self) -> None:
        snapshots = sorted(_SNAPSHOT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in snapshots[_MAX_SNAPSHOTS:]:
            try:
                old.unlink()
            except Exception as exc:
                logger.debug(f"ClusterSnapshot rotate failed: {exc}")

    def list_snapshots(self) -> list[dict[str, Any]]:
        snapshots = sorted(_SNAPSHOT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        result = []
        for p in snapshots[:_MAX_SNAPSHOTS]:
            try:
                data = json.loads(p.read_text())
                result.append({
                    "filename": p.name,
                    "timestamp": data.get("timestamp", 0),
                    "node_id": data.get("node_id", ""),
                })
            except Exception as exc:
                logger.debug(f"ClusterSnapshot list skip {p.name}: {exc}")
        return result

    def load(self, filename: str) -> dict[str, Any] | None:
        path = _SNAPSHOT_DIR / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning(f"ClusterSnapshot load failed {filename}: {exc}")
            return None


CLUSTER_SNAPSHOT = ClusterSnapshotManager()
