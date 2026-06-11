from __future__ import annotations

import time
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AutoscalerMetrics:
    queue_depth: int = 0
    active_nodes: int = 0
    tokens_per_second: float = 0.0
    last_scale_recommendation: str = "none"
    last_checked: float = field(default_factory=time.monotonic)

    def update(self, queue_depth: int, active_nodes: int, tps: float) -> None:
        self.queue_depth = queue_depth
        self.active_nodes = active_nodes
        self.tokens_per_second = tps
        self.last_checked = time.monotonic()
        self._evaluate()

    def _evaluate(self) -> None:
        if self.active_nodes == 0:
            self.last_scale_recommendation = "scale_up"
            logger.warning("Autoscaler: no active nodes — scale_up needed")
            return

        requests_per_node = self.queue_depth / max(self.active_nodes, 1)

        if requests_per_node > 3:
            self.last_scale_recommendation = "scale_up"
            logger.warning(
                f"Autoscaler: queue_depth={self.queue_depth} nodes={self.active_nodes} "
                f"ratio={requests_per_node:.1f} → scale_up"
            )
        elif requests_per_node < 0.1 and self.active_nodes > 1:
            self.last_scale_recommendation = "scale_down"
            logger.info(
                f"Autoscaler: queue_depth={self.queue_depth} nodes={self.active_nodes} "
                f"ratio={requests_per_node:.1f} → scale_down"
            )
        else:
            self.last_scale_recommendation = "stable"

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_depth": self.queue_depth,
            "active_nodes": self.active_nodes,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "recommendation": self.last_scale_recommendation,
        }


AUTOSCALER = AutoscalerMetrics()
