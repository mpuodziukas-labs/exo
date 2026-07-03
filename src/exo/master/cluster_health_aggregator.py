from __future__ import annotations

import time
from typing import Any, Callable, TypeVar, cast

from loguru import logger

_T = TypeVar("_T")


def _safe(fn: Callable[[], _T], default: _T | None = None) -> _T | None:
    try:
        return fn()
    except Exception as exc:
        logger.debug(f"ClusterHealthAggregator safe call failed: {exc}")
        return default


class ClusterHealthAggregator:
    """
    Aggregates health data from all subsystem singletons into a single report.
    Used by the /v1/cluster/health endpoint.
    Each subsystem is queried with a safe wrapper — one failure doesn't break the report.
    """

    def collect(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "timestamp": time.time(),
            "subsystems": {},
        }

        # Import lazily to avoid circular imports
        subsystems = self._collect_subsystems()
        report["subsystems"] = subsystems

        # Compute overall health grade
        ok_count = sum(
            1
            for v in subsystems.values()
            if isinstance(v, dict)
            and cast(dict[str, object], v).get("status") != "error"
        )
        total = len(subsystems)
        report["overall_health"] = (
            "healthy"
            if ok_count == total
            else ("degraded" if ok_count > total // 2 else "critical")
        )
        report["subsystem_count"] = total
        report["healthy_count"] = ok_count

        return report

    def _collect_subsystems(self) -> dict[str, object]:
        results: dict[str, object] = {}

        try:
            from exo.master.pipeline_health import PIPELINE_HEALTH

            results["pipeline"] = _safe(lambda: PIPELINE_HEALTH.compute())
        except ImportError as exc:
            logger.debug(
                "ClusterHealthAggregator: pipeline_health not available: {}", exc
            )

        try:
            from exo.master.error_budget import ERROR_BUDGET

            results["error_budget"] = _safe(lambda: ERROR_BUDGET.get_summary())
        except ImportError as exc:
            logger.debug("ClusterHealthAggregator: error_budget not available: {}", exc)

        try:
            from exo.master.graceful_degradation import DEGRADATION_CONTROLLER

            results["degradation"] = _safe(lambda: DEGRADATION_CONTROLLER.get_status())
        except ImportError as exc:
            logger.debug(
                "ClusterHealthAggregator: graceful_degradation not available: {}", exc
            )

        try:
            from exo.master.quorum_check import QUORUM_CHECKER

            results["quorum"] = _safe(lambda: QUORUM_CHECKER.get_status())
        except ImportError as exc:
            logger.debug("ClusterHealthAggregator: quorum_check not available: {}", exc)

        try:
            from exo.master.anomaly_detector import ANOMALY_DETECTOR

            results["anomaly_detector"] = _safe(lambda: ANOMALY_DETECTOR.get_stats())
        except ImportError as exc:
            logger.debug(
                "ClusterHealthAggregator: anomaly_detector not available: {}", exc
            )

        try:
            from exo.master.link_health import LINK_MONITOR

            results["link_health"] = _safe(lambda: {"nodes": LINK_MONITOR.get_stats()})
        except ImportError as exc:
            logger.debug("ClusterHealthAggregator: link_health not available: {}", exc)

        try:
            from exo.master.autoscale_trigger import AUTOSCALE_TRIGGER

            results["autoscale"] = _safe(lambda: AUTOSCALE_TRIGGER.stats())
        except ImportError as exc:
            logger.debug(
                "ClusterHealthAggregator: autoscale_trigger not available: {}", exc
            )

        return results


CLUSTER_HEALTH_AGGREGATOR = ClusterHealthAggregator()
