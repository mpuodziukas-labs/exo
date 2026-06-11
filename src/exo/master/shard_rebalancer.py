from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ShardAssignment:
    model_id: str
    node_id: str
    shard_index: int
    total_shards: int
    assigned_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "node_id": self.node_id,
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "assigned_at": self.assigned_at,
        }


@dataclass
class RebalanceOperation:
    operation_id: str
    model_id: str
    from_node: str
    to_node: str
    shard_index: int
    reason: str
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    status: str = "pending"  # pending / in_progress / completed / failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "model_id": self.model_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "shard_index": self.shard_index,
            "reason": self.reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
        }


class ShardRebalancer:
    """
    Tracks shard assignments across nodes and manages rebalance operations.
    Rebalance is triggered by: node join/leave, memory pressure, or manual request.
    """

    def __init__(self) -> None:
        self._assignments: dict[tuple[str, int], ShardAssignment] = {}  # (model_id, shard_index) -> assignment
        self._operations: list[RebalanceOperation] = []
        self._op_counter = 0

    def assign_shard(self, model_id: str, node_id: str, shard_index: int, total_shards: int) -> ShardAssignment:
        key = (model_id, shard_index)
        assignment = ShardAssignment(
            model_id=model_id,
            node_id=node_id,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        self._assignments[key] = assignment
        return assignment

    def plan_rebalance(
        self,
        model_id: str,
        node_ids: list[str],
        total_shards: int,
        reason: str = "manual",
    ) -> list[RebalanceOperation]:
        """
        Plan rebalance: distribute shards evenly across available nodes.
        Returns list of move operations needed.
        """
        if not node_ids:
            return []

        ops: list[RebalanceOperation] = []
        for shard_index in range(total_shards):
            target_node = node_ids[shard_index % len(node_ids)]
            current = self._assignments.get((model_id, shard_index))
            if current is not None and current.node_id == target_node:
                continue  # already in right place
            self._op_counter += 1
            op = RebalanceOperation(
                operation_id=f"rebal_{self._op_counter:04d}",
                model_id=model_id,
                from_node=current.node_id if current else "none",
                to_node=target_node,
                shard_index=shard_index,
                reason=reason,
            )
            ops.append(op)
            self._operations.append(op)

        if ops:
            logger.info(
                f"ShardRebalancer planned {len(ops)} moves for model={model_id} "
                f"nodes={len(node_ids)} shards={total_shards} reason={reason}"
            )
        return ops

    def complete_operation(self, operation_id: str, success: bool = True) -> bool:
        for op in self._operations:
            if op.operation_id == operation_id:
                op.status = "completed" if success else "failed"
                op.completed_at = time.time()
                if success:
                    self.assign_shard(op.model_id, op.to_node, op.shard_index, 0)
                return True
        return False

    def current_assignments(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._assignments.values()]

    def recent_operations(self, limit: int = 20) -> list[dict[str, Any]]:
        return [op.to_dict() for op in self._operations[-limit:]]

    def stats(self) -> dict[str, Any]:
        pending = sum(1 for op in self._operations if op.status == "pending")
        in_progress = sum(1 for op in self._operations if op.status == "in_progress")
        completed = sum(1 for op in self._operations if op.status == "completed")
        failed = sum(1 for op in self._operations if op.status == "failed")
        return {
            "shard_count": len(self._assignments),
            "total_operations": len(self._operations),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "failed": failed,
        }


SHARD_REBALANCER = ShardRebalancer()
