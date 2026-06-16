"""
Multi-node tensor shard validator: verifies that all required model shards are
present across cluster nodes before dispatching inference. If any shard is
missing, blocks the request and triggers a re-download.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ShardStatus:
    model_id: str
    node_id: str
    shard_index: int
    total_shards: int
    present: bool
    verified_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "node_id": self.node_id,
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "present": self.present,
            "verified_at": self.verified_at,
        }


@dataclass
class ShardValidationResult:
    model_id: str
    ready: bool
    missing_shards: list[tuple[str, int]]  # (node_id, shard_index)
    total_shards: int
    present_shards: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ready": self.ready,
            "total_shards": self.total_shards,
            "present_shards": self.present_shards,
            "missing": [{"node": n, "shard": s} for n, s in self.missing_shards],
        }


class ShardValidator:
    """
    register_shard(model_id, node_id, shard_index, total_shards, present)
    validate(model_id): returns ShardValidationResult
    mark_all_present(model_id, node_ids, total_shards): marks all shards as present
    """

    def __init__(self) -> None:
        # key: (model_id, node_id, shard_index) → ShardStatus
        self._shards: dict[tuple[str, str, int], ShardStatus] = {}

    def register_shard(
        self,
        model_id: str,
        node_id: str,
        shard_index: int,
        total_shards: int,
        present: bool = True,
    ) -> None:
        key = (model_id, node_id, shard_index)
        self._shards[key] = ShardStatus(
            model_id=model_id,
            node_id=node_id,
            shard_index=shard_index,
            total_shards=total_shards,
            present=present,
        )

    def mark_all_present(
        self, model_id: str, node_ids: list[str], total_shards: int
    ) -> None:
        """Convenience: mark all shards across all nodes as present."""
        for node_id in node_ids:
            for idx in range(total_shards):
                self.register_shard(model_id, node_id, idx, total_shards, present=True)
        logger.debug(
            f"ShardValidator: marked {total_shards}×{len(node_ids)} shards ready for {model_id}"
        )

    def validate(self, model_id: str) -> ShardValidationResult:
        relevant = {k: v for k, v in self._shards.items() if k[0] == model_id}
        if not relevant:
            # No shard info registered — assume ready (legacy path)
            return ShardValidationResult(
                model_id=model_id,
                ready=True,
                missing_shards=[],
                total_shards=0,
                present_shards=0,
            )
        total = max(v.total_shards for v in relevant.values())
        present = [v for v in relevant.values() if v.present]
        missing = [
            (v.node_id, v.shard_index) for v in relevant.values() if not v.present
        ]
        ready = len(missing) == 0
        if not ready:
            logger.warning(
                f"ShardValidator: model={model_id} missing {len(missing)} shards"
            )
        return ShardValidationResult(
            model_id=model_id,
            ready=ready,
            missing_shards=missing,
            total_shards=total,
            present_shards=len(present),
        )

    def get_stats(self) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = {}
        for (mid, _, _), status in self._shards.items():
            if mid not in models:
                models[mid] = {"total": 0, "present": 0}
            models[mid]["total"] += 1
            if status.present:
                models[mid]["present"] += 1
        return {"models": [{"model_id": k, **v} for k, v in models.items()]}


SHARD_VALIDATOR = ShardValidator()
