"""shard_placement.py — production-grade model shard placement optimizer for exo.

Decides which model shards land on which node to minimise inter-node data
transfer.  Uses NODE_REGISTRY capability data for greedy RAM-aware placement.

Two-node (MacBook + Mini) behaviour
------------------------------------
* MacBook is master — gets the first half of shards (lower first-token latency).
* Mini gets the second half.
* Transfer cost = shards that cross the node boundary × shard_size_gb.
* Link label: "remote_tb4" (TB4 / RDMA at 40 Gbps) for cross-node shards,
  "local" for shards that stay on the requesting node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Final

from loguru import logger

from exo.master.node_registry import NODE_REGISTRY, NodeCapability

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShardPlacement:
    model_id: str
    shard_index: int
    total_shards: int
    node_id: str
    shard_size_gb: float
    placement_reason: str  # "local" | "remote_tb4"


@dataclass
class PlacementPlan:
    model_id: str
    placements: list[ShardPlacement]
    total_transfer_gb: float
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class ShardPlacementOptimizer:
    """Greedy, RAM-aware shard placement optimizer.

    Thread-safe plan cache keyed by model_id.  Plans are invalidated
    whenever node topology changes (call ``invalidate`` or let the caller
    decide based on heartbeat events).
    """

    def __init__(self) -> None:
        self._plans: dict[str, PlacementPlan] = {}
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Core placement logic
    # ------------------------------------------------------------------

    def compute_plan(
        self,
        model_id: str,
        total_shards: int,
        model_size_gb: float,
    ) -> PlacementPlan:
        """Compute and cache an optimised shard placement plan.

        Parameters
        ----------
        model_id:       Unique identifier for the model (e.g. "llama3-70b").
        total_shards:   Number of pipeline shards to distribute.
        model_size_gb:  Total model size in GB (used to derive per-shard size).
        """
        if total_shards < 1:
            raise ValueError(f"total_shards must be ≥ 1, got {total_shards}")
        if model_size_gb <= 0:
            raise ValueError(f"model_size_gb must be > 0, got {model_size_gb}")

        shard_size_gb = model_size_gb / total_shards
        nodes: list[NodeCapability] = NODE_REGISTRY.all_nodes()

        if not nodes:
            raise RuntimeError(
                "NODE_REGISTRY is empty — no alive nodes to place shards on"
            )

        placements: list[ShardPlacement] = []
        transfer_shards: int = 0

        if len(nodes) == 1:
            # ── Single-node: all shards local ─────────────────────────────
            node = nodes[0]
            for idx in range(total_shards):
                placements.append(
                    ShardPlacement(
                        model_id=model_id,
                        shard_index=idx,
                        total_shards=total_shards,
                        node_id=node.node_id,
                        shard_size_gb=shard_size_gb,
                        placement_reason="local",
                    )
                )
            logger.info(
                f"[shard_placement] {model_id!r}: all {total_shards} shards "
                f"→ single node {node.node_id!r}"
            )

        elif len(nodes) == 2:
            # ── Two-node: MacBook (master) gets first half, Mini gets second ─
            # Identify master by highest FLOPS (MacBook M1 Max >> Mini M4).
            master, secondary = (
                (nodes[0], nodes[1])
                if nodes[0].compute_flops_tflops >= nodes[1].compute_flops_tflops
                else (nodes[1], nodes[0])
            )
            split = total_shards // 2 + (total_shards % 2)  # master gets ceil half

            for idx in range(total_shards):
                target = master if idx < split else secondary
                # A shard "transfers" when it is not on the master (first-token) node.
                # We count master→secondary boundary crossings.
                is_remote = idx >= split
                if is_remote:
                    transfer_shards += 1
                reason = "remote_tb4" if is_remote else "local"
                placements.append(
                    ShardPlacement(
                        model_id=model_id,
                        shard_index=idx,
                        total_shards=total_shards,
                        node_id=target.node_id,
                        shard_size_gb=shard_size_gb,
                        placement_reason=reason,
                    )
                )
            logger.info(
                f"[shard_placement] {model_id!r}: {split} shards → master "
                f"{master.node_id!r}, {total_shards - split} shards → "
                f"secondary {secondary.node_id!r} (TB4 link)"
            )

        else:
            # ── N-node: greedy — node with most available RAM gets next shard ─
            # Track remaining capacity per node as we assign shards.
            available: dict[str, float] = {
                cap.node_id: cap.ram_available_gb for cap in nodes
            }
            # Sort nodes by flops descending so the first shard lands on the
            # fastest node (best first-token latency).
            sorted_nodes = sorted(
                nodes,
                key=lambda c: c.compute_flops_tflops,
                reverse=True,
            )
            master_node_id = sorted_nodes[0].node_id
            prev_node_id: str | None = None

            for idx in range(total_shards):
                # Pick node with most remaining RAM.
                best_id = max(available, key=lambda nid: available[nid])
                available[best_id] = max(0.0, available[best_id] - shard_size_gb)

                # Count boundary crossings (adjacent shards on different nodes).
                if prev_node_id is not None and best_id != prev_node_id:
                    transfer_shards += 1

                reason = "local" if best_id == master_node_id else "remote_tb4"
                placements.append(
                    ShardPlacement(
                        model_id=model_id,
                        shard_index=idx,
                        total_shards=total_shards,
                        node_id=best_id,
                        shard_size_gb=shard_size_gb,
                        placement_reason=reason,
                    )
                )
                prev_node_id = best_id

            logger.info(
                f"[shard_placement] {model_id!r}: {total_shards} shards across "
                f"{len(nodes)} nodes, boundary crossings={transfer_shards}"
            )

        total_transfer_gb = round(transfer_shards * shard_size_gb, 4)
        plan = PlacementPlan(
            model_id=model_id,
            placements=placements,
            total_transfer_gb=total_transfer_gb,
        )

        with self._lock:
            self._plans[model_id] = plan

        logger.debug(
            f"[shard_placement] cached plan for {model_id!r}: "
            f"transfer={total_transfer_gb:.3f}GB"
        )
        return plan

    # ------------------------------------------------------------------
    # Cache accessors
    # ------------------------------------------------------------------

    def get_plan(self, model_id: str) -> PlacementPlan | None:
        """Return cached plan or None if not yet computed."""
        with self._lock:
            return self._plans.get(model_id)

    def invalidate(self, model_id: str) -> None:
        """Drop a cached plan, forcing recompute on next access."""
        with self._lock:
            dropped = self._plans.pop(model_id, None)
        if dropped is not None:
            logger.debug(
                f"[shard_placement] invalidated cached plan for {model_id!r}"
            )

    def invalidate_all(self) -> None:
        """Drop all cached plans (e.g. after node topology change)."""
        with self._lock:
            count = len(self._plans)
            self._plans.clear()
        logger.info(f"[shard_placement] invalidated all {count} cached plans")

    def optimal_node_for_shard(
        self,
        model_id: str,
        shard_index: int,
    ) -> str | None:
        """Return the node_id assigned to *shard_index* in the cached plan.

        Returns None if no plan exists for *model_id* or *shard_index* is
        out of range.
        """
        plan = self.get_plan(model_id)
        if plan is None:
            return None
        for p in plan.placements:
            if p.shard_index == shard_index:
                return p.node_id
        return None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return a serialisable summary of all cached plans."""
        with self._lock:
            plans_snapshot = list(self._plans.values())

        return {
            "cached_plan_count": len(plans_snapshot),
            "plans": [
                {
                    "model_id": plan.model_id,
                    "total_shards": len(plan.placements),
                    "total_transfer_gb": plan.total_transfer_gb,
                    "created_at": plan.created_at,
                    "age_seconds": round(time.time() - plan.created_at, 1),
                    "placements": [
                        {
                            "shard_index": p.shard_index,
                            "node_id": p.node_id,
                            "shard_size_gb": p.shard_size_gb,
                            "placement_reason": p.placement_reason,
                        }
                        for p in sorted(plan.placements, key=lambda x: x.shard_index)
                    ],
                }
                for plan in plans_snapshot
            ],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

SHARD_OPTIMIZER: Final[ShardPlacementOptimizer] = ShardPlacementOptimizer()
