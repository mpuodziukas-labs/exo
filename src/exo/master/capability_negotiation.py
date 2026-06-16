"""
Node capability negotiation: when assigning a model shard to a node, verifies
the node has sufficient capabilities (RAM, FLOPS, MLX version, features).
Returns negotiation result explaining why a node was accepted or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Minimum requirements per model size tier (rough heuristics)
_MIN_RAM_GB_FOR_TIER: dict[str, float] = {
    "small": 4.0,  # < 4B params
    "medium": 8.0,  # 4-13B params
    "large": 16.0,  # 13-70B params
    "xlarge": 48.0,  # 70B+ params
}

_MIN_TFLOPS_FOR_TIER: dict[str, float] = {
    "small": 1.0,
    "medium": 5.0,
    "large": 10.0,
    "xlarge": 20.0,
}


@dataclass
class NodeCapabilityProfile:
    node_id: str
    ram_gb: float
    tflops: float
    mlx_version: str = ""
    features: list[str] = field(default_factory=list)  # e.g. ["FAST_SYNCH", "TB4"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "tflops": self.tflops,
            "mlx_version": self.mlx_version,
            "features": self.features,
        }


@dataclass
class NegotiationResult:
    node_id: str
    model_tier: str
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_tier": self.model_tier,
            "accepted": self.accepted,
            "reason": self.reason,
        }


class CapabilityNegotiator:
    """
    register(profile): stores a node's capability profile.
    negotiate(node_id, model_tier): checks if node can handle the model tier.
    best_node(model_tier): returns node_id of best candidate for a tier.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, NodeCapabilityProfile] = {}

    def register(self, profile: NodeCapabilityProfile) -> None:
        self._profiles[profile.node_id] = profile
        logger.debug(
            f"CapabilityNegotiator: registered node={profile.node_id} "
            f"ram={profile.ram_gb}GB tflops={profile.tflops}"
        )

    def negotiate(self, node_id: str, model_tier: str) -> NegotiationResult:
        profile = self._profiles.get(node_id)
        if profile is None:
            return NegotiationResult(
                node_id=node_id,
                model_tier=model_tier,
                accepted=False,
                reason="node not registered",
            )
        min_ram = _MIN_RAM_GB_FOR_TIER.get(model_tier, 8.0)
        min_tflops = _MIN_TFLOPS_FOR_TIER.get(model_tier, 5.0)
        if profile.ram_gb < min_ram:
            return NegotiationResult(
                node_id=node_id,
                model_tier=model_tier,
                accepted=False,
                reason=f"insufficient RAM: {profile.ram_gb}GB < {min_ram}GB required",
            )
        if profile.tflops < min_tflops:
            return NegotiationResult(
                node_id=node_id,
                model_tier=model_tier,
                accepted=False,
                reason=f"insufficient FLOPS: {profile.tflops}T < {min_tflops}T required",
            )
        return NegotiationResult(
            node_id=node_id, model_tier=model_tier, accepted=True, reason="capable"
        )

    def best_node(self, model_tier: str) -> str | None:
        """Returns node_id of highest-RAM node that passes negotiation."""
        candidates = [
            p
            for p in self._profiles.values()
            if self.negotiate(p.node_id, model_tier).accepted
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.tflops).node_id

    def get_all(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._profiles.values()]

    def get_stats(self) -> dict[str, Any]:
        return {
            "registered_nodes": len(self._profiles),
            "profiles": self.get_all(),
        }


CAPABILITY_NEGOTIATOR = CapabilityNegotiator()
