"""
Inference pipeline health score: computes a single composite 0-100 score
for the end-to-end inference pipeline by combining scores from all subsystems.
Used as the primary dashboard signal and for automated gate decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

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


def _health_scorer_score() -> float:
    """Composite cluster health score (0-100), neutral (50) on extraction failure."""
    try:
        from exo.master.health_score import HEALTH_SCORER

        report = HEALTH_SCORER.current()
        if report is None:
            return 50.0
        return float(report.overall_score)
    except Exception as exc:
        logger.debug(f"PipelineHealth: health_score extraction failed: {exc}")
        return 50.0  # neutral on failure


def _quorum_score() -> float:
    try:
        from exo.master.quorum_check import QUORUM_CHECKER

        return 100.0 if QUORUM_CHECKER.quorum_met else 0.0
    except Exception as exc:
        logger.debug(f"PipelineHealth: quorum extraction failed: {exc}")
        return 50.0


def _degradation_score() -> float:
    try:
        from exo.master.graceful_degradation import DEGRADATION_CONTROLLER

        # blocks_inference True = bad = 0 score
        return 0.0 if DEGRADATION_CONTROLLER.blocks_inference else 100.0
    except Exception as exc:
        logger.debug(f"PipelineHealth: degradation extraction failed: {exc}")
        return 50.0


def _circuit_breaker_score() -> float:
    """Score based on the fraction of circuit breakers currently open."""
    try:
        from exo.master.circuit_breaker import CIRCUIT_BREAKERS

        states = CIRCUIT_BREAKERS.all_states()
        if not states:
            return 100.0
        open_count = sum(1 for s in states if s.get("state") == "open")
        return 100.0 * (1 - open_count / len(states))
    except Exception as exc:
        logger.debug(f"PipelineHealth: circuit_breaker extraction failed: {exc}")
        return 50.0


def _admission_score() -> float:
    """AdmissionController does not yet expose a live concurrency gauge, so
    this subsystem contributes a neutral-max score until that lands."""
    try:
        from exo.master.admission_control import ADMISSION_CONTROLLER

        _ = ADMISSION_CONTROLLER.max_concurrent  # touch to surface import failures
        return 100.0
    except Exception as exc:
        logger.debug(f"PipelineHealth: admission extraction failed: {exc}")
        return 50.0


# (module_path, attr, score_extractor_fn, weight)
_SUBSYSTEM_EXTRACTORS: list[tuple[str, str, Callable[[], float], float]] = [
    ("exo.master.health_score", "HEALTH_SCORER", _health_scorer_score, 0.30),
    ("exo.master.quorum_check", "QUORUM_CHECKER", _quorum_score, 0.20),
    (
        "exo.master.graceful_degradation",
        "DEGRADATION_CONTROLLER",
        _degradation_score,
        0.20,
    ),
    ("exo.master.circuit_breaker", "CIRCUIT_BREAKERS", _circuit_breaker_score, 0.15),
    ("exo.master.admission_control", "ADMISSION_CONTROLLER", _admission_score, 0.15),
]


class PipelineHealthScorer:
    def __init__(self) -> None:
        self._last_score: float = 100.0
        self._last_computed: float = 0.0

    def compute(self) -> dict[str, Any]:
        subsystems: list[SubsystemScore] = []
        for module_path, attr, extractor, weight in _SUBSYSTEM_EXTRACTORS:
            score = extractor()
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
