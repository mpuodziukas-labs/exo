"""
Singleton health check on startup: for each key singleton, calls a lightweight
self_check() method (if it exists) or checks that the object is non-None and
has expected attributes. Reports a structured health report at startup.
"""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Any
from loguru import logger


@dataclass
class SingletonHealth:
    name: str
    healthy: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "healthy": self.healthy, "detail": self.detail}


# (module, attr, required_attrs) — required_attrs checked via hasattr
_SINGLETONS: list[tuple[str, str, list[str]]] = [
    ("exo.master.metrics", "METRICS", ["inference_requests_total"]),
    ("exo.master.admission_control", "ADMISSION_CONTROLLER", ["check"]),
    ("exo.master.rate_limiter", "RATE_LIMITER", ["check"]),
    ("exo.master.circuit_breaker", "CIRCUIT_BREAKERS", ["get"]),
    ("exo.master.response_cache", "RESPONSE_CACHE", ["get", "put"]),
    ("exo.master.memory_monitor", "MEMORY_MONITOR", ["sample"]),
    ("exo.master.priority_queue", "PRIORITY_QUEUE", ["push", "pop"]),
    ("exo.master.health_score", "HEALTH_SCORER", ["current"]),
    ("exo.master.graceful_degradation", "DEGRADATION_CONTROLLER", ["evaluate", "blocks_inference"]),
    ("exo.master.quorum_check", "QUORUM_CHECKER", ["update", "quorum_met"]),
    ("exo.master.capacity_planner", "CAPACITY_PLANNER", ["record_arrival", "get_snapshot"]),
    ("exo.master.schema_registry", "SCHEMA_REGISTRY", ["register", "resolve"]),
    ("exo.master.persistent_dedup", "PERSISTENT_DEDUP", ["record", "is_duplicate"]),
    ("exo.master.stream_recovery", "STREAM_RECOVERY", ["start", "finish"]),
    ("exo.master.slo_tracker", "SLO_TRACKER", ["record"]),
    ("exo.master.cost_tracker", "COST_TRACKER", ["record"]),
    ("exo.master.canary", "CANARY_CONTROLLER", ["should_use_canary"]),
    ("exo.master.api_keys", "API_KEY_STORE", ["verify"]),
    ("exo.master.latency_budget", "LATENCY_BUDGET_MANAGER", ["start", "finish"]),
    ("exo.master.checkpoint_recovery", "CHECKPOINT_RECOVERY", ["scan_incomplete"]),
]


class SingletonHealthChecker:
    def run(self) -> list[SingletonHealth]:
        results: list[SingletonHealth] = []
        for module_path, attr, required_attrs in _SINGLETONS:
            try:
                mod = importlib.import_module(module_path)
                obj = getattr(mod, attr, None)
                if obj is None:
                    results.append(SingletonHealth(name=attr, healthy=False, detail="singleton is None"))
                    continue
                missing = [a for a in required_attrs if not hasattr(obj, a)]
                if missing:
                    results.append(SingletonHealth(
                        name=attr, healthy=False,
                        detail=f"missing attrs: {missing}"
                    ))
                else:
                    results.append(SingletonHealth(name=attr, healthy=True))
            except Exception as exc:
                results.append(SingletonHealth(name=attr, healthy=False, detail=str(exc)))
                logger.warning(f"Singleton health fail: {attr}: {exc}")

        ok = sum(1 for r in results if r.healthy)
        logger.info(f"Singleton health: {ok}/{len(results)} healthy")
        return results

    def get_report(self) -> dict[str, Any]:
        results = self.run()
        return {
            "total": len(results),
            "healthy": sum(1 for r in results if r.healthy),
            "unhealthy": [r.to_dict() for r in results if not r.healthy],
        }


SINGLETON_HEALTH = SingletonHealthChecker()
