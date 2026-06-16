"""
Inference pipeline health score: computes a single composite 0-100 score
for the end-to-end inference pipeline by combining scores from all subsystems.
Used as the primary dashboard signal and for automated gate decisions.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class SubsystemScore:
    name: str
    score: float  # 0.0 - 100.0
    weight: float  # relative weight in composite
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "detail": self.detail,
        }


# (module, attr, score_extractor_fn, weight)
_SUBSYSTEM_EXTRACTORS: list[tuple[str, str, str, float]] = [
    # (module_path, attr, method_or_key, weight)
    ("exo.master.health_score", "HEALTH_SCORER", "current().overall_score", 0.30),
    ("exo.master.quorum_check", "QUORUM_CHECKER", "quorum_met", 0.20),
    (
        "exo.master.graceful_degradation",
        "DEGRADATION_CONTROLLER",
        "blocks_inference",
        0.20,
    ),
    ("exo.master.circuit_breaker", "CIRCUIT_BREAKERS", "_all_open_check", 0.15),
    ("exo.master.admission_control", "ADMISSION_CONTROLLER", "_capacity_check", 0.15),
]


def _safe_get_score(module_path: str, attr: str, expr: str) -> float:
    """Safely extract a score from a singleton. Returns 0-100."""
    try:
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr, None)
        if obj is None:
            return 0.0
        # Use known patterns
        if expr == "current().overall_score":
            return float(obj.current().overall_score)
        elif expr == "quorum_met":
            return 100.0 if obj.quorum_met else 0.0
        elif expr == "blocks_inference":
            # blocks_inference True = bad = 0 score
            return 0.0 if obj.blocks_inference else 100.0
        elif expr == "_all_open_check":
            # Check if any CB is open — if so, partial score
            cbs = list(obj._breakers.values()) if hasattr(obj, "_breakers") else []
            if not cbs:
                return 100.0
            open_count = sum(1 for cb in cbs if cb.state.value == "open")
            return 100.0 * (1 - open_count / len(cbs))
        elif expr == "_capacity_check":
            # Admission controller: score based on how far from max
            max_c = getattr(obj, "max_concurrent", 20)
            current = getattr(obj, "_current_concurrent", 0)
            return 100.0 * max(0, 1 - current / max(max_c, 1))
        return 50.0
    except Exception as exc:
        logger.debug(
            f"PipelineHealth: score extraction failed {module_path}.{attr}: {exc}"
        )
        return 50.0  # neutral on failure


class PipelineHealthScorer:
    def __init__(self) -> None:
        self._last_score: float = 100.0
        self._last_computed: float = 0.0

    def compute(self) -> dict[str, Any]:
        subsystems: list[SubsystemScore] = []
        for module_path, attr, expr, weight in _SUBSYSTEM_EXTRACTORS:
            score = _safe_get_score(module_path, attr, expr)
            subsystems.append(
                SubsystemScore(
                    name=attr, score=score, weight=weight, detail=f"{module_path}"
                )
            )

        total_weight = sum(s.weight for s in subsystems)
        composite = sum(s.score * s.weight for s in subsystems) / max(
            total_weight, 0.001
        )
        self._last_score = composite
        self._last_computed = time.time()

        grade = (
            "A"
            if composite >= 90
            else "B"
            if composite >= 75
            else "C"
            if composite >= 60
            else "D"
            if composite >= 40
            else "F"
        )

        return {
            "composite_score": round(composite, 2),
            "grade": grade,
            "computed_at": self._last_computed,
            "subsystems": [s.to_dict() for s in subsystems],
        }

    def get_quick_score(self) -> float:
        return self._last_score


PIPELINE_HEALTH = PipelineHealthScorer()
