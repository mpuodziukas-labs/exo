from __future__ import annotations

import time
from typing import Any

from loguru import logger


class PrometheusAggregator:
    """
    Aggregates Prometheus text metrics from all registered sources.
    Each source must implement prometheus_metrics() -> str.
    """

    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}

    def register(self, name: str, source: Any) -> None:
        if hasattr(source, "prometheus_metrics"):
            self._sources[name] = source
            logger.debug(f"PrometheusAggregator registered source={name}")
        else:
            logger.warning(f"PrometheusAggregator: {name} has no prometheus_metrics() method, skipping")

    def aggregate(self) -> str:
        """Alias for collect() — Prometheus scrape-format text of all registered sources."""
        return self.collect()

    def collect(self) -> str:
        """Collect and concatenate all Prometheus metrics."""
        parts: list[str] = []

        # Add scrape timestamp
        parts.append(f"# HELP exo_scrape_timestamp_seconds Last scrape timestamp\n")
        parts.append(f"# TYPE exo_scrape_timestamp_seconds gauge\n")
        parts.append(f"exo_scrape_timestamp_seconds {time.time():.3f}\n\n")

        for name, source in self._sources.items():
            try:
                metrics = source.prometheus_metrics()
                if metrics:
                    parts.append(f"# Source: {name}\n")
                    parts.append(metrics)
                    parts.append("\n")
            except Exception as exc:
                logger.debug(f"PrometheusAggregator source={name} failed: {exc}")

        return "".join(parts)

    def auto_register_all(self) -> int:
        """Auto-discover and register all known singletons that have prometheus_metrics()."""
        registered = 0

        sources_to_try = [
            ("link_health", "exo.master.link_health", "LINK_MONITOR"),
            ("slo_tracker", "exo.master.slo_tracker", "SLO_TRACKER"),
            ("slo_budget", "exo.master.slo_budget_tracker", "SLO_BUDGET_TRACKER"),
            ("batch_efficiency", "exo.master.batch_efficiency", "BATCH_EFFICIENCY"),
            ("mfu_reporter", "exo.master.mfu_reporter", "MFU_REPORTER"),
            ("speculative", "exo.master.speculative_monitor", "SPECULATIVE_MONITOR"),
            ("kvcache", "exo.master.kv_cache_tier", "KV_CACHE_TIER"),
            ("token_velocity", "exo.master.token_velocity_limiter", "TOKEN_VELOCITY_LIMITER"),
            ("prefill_decode", "exo.master.prefill_decode_scheduler", "PREFILL_DECODE_SCHEDULER"),
        ]

        for name, module_path, attr in sources_to_try:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                source = getattr(mod, attr)
                self.register(name, source)
                registered += 1
            except Exception as exc:
                logger.debug(f"PrometheusAggregator auto-register {name} failed: {exc}")

        logger.info(f"PrometheusAggregator auto-registered {registered} sources")
        return registered


PROMETHEUS_AGGREGATOR = PrometheusAggregator()
