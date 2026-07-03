"""
config_watcher.py — hot-reload for exo runtime config.

Polls ~/.exo/config.json every 5 s via mtime comparison (no kqueue/inotify
dependency), diffs changed fields, and applies them to the relevant singletons
without a restart.

Singletons touched:
  RATE_LIMITER          rate_limit_rpm_authenticated / rate_limit_rpm_anonymous
  SLO_TRACKER           ttft_slo_ms
  ADMISSION_CONTROLLER  admission_max_concurrent
  CIRCUIT_BREAKERS      failure_threshold / cooldown_seconds on each breaker
  DEDUP                 _ttl_seconds (module-level override via attribute)
  QUORUM_CHECKER        quorum_min
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from exo.master.config_version import CONFIG_VERSION

# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


@dataclass
class ExoConfig:
    rate_limit_rpm_authenticated: int = 60
    rate_limit_rpm_anonymous: int = 10
    ttft_slo_ms: float = 500.0
    admission_max_concurrent: int = 32
    canary_percent: float = 0.0
    canary_model: str = ""
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0
    checkpoint_every_n_tokens: int = 50
    record_rate: float = 0.0
    dedup_ttl_seconds: float = 30.0
    quorum_min: int = 1


_CONFIG_PATH: Path = Path(os.path.expanduser("~/.exo/config.json"))


def _load_from_dict(raw: dict[str, Any]) -> ExoConfig:
    """Merge raw JSON dict with ExoConfig defaults; unknown keys are silently dropped."""
    d = ExoConfig()
    return ExoConfig(
        rate_limit_rpm_authenticated=int(
            cast(
                int,
                raw.get(
                    "rate_limit_rpm_authenticated", d.rate_limit_rpm_authenticated
                ),
            )
        ),
        rate_limit_rpm_anonymous=int(
            cast(
                int, raw.get("rate_limit_rpm_anonymous", d.rate_limit_rpm_anonymous)
            )
        ),
        ttft_slo_ms=float(cast(float, raw.get("ttft_slo_ms", d.ttft_slo_ms))),
        admission_max_concurrent=int(
            cast(
                int,
                raw.get("admission_max_concurrent", d.admission_max_concurrent),
            )
        ),
        canary_percent=float(
            cast(float, raw.get("canary_percent", d.canary_percent))
        ),
        canary_model=str(cast(str, raw.get("canary_model", d.canary_model))),
        circuit_breaker_failure_threshold=int(
            cast(
                int,
                raw.get(
                    "circuit_breaker_failure_threshold",
                    d.circuit_breaker_failure_threshold,
                ),
            )
        ),
        circuit_breaker_cooldown_seconds=float(
            cast(
                float,
                raw.get(
                    "circuit_breaker_cooldown_seconds",
                    d.circuit_breaker_cooldown_seconds,
                ),
            )
        ),
        checkpoint_every_n_tokens=int(
            cast(
                int,
                raw.get("checkpoint_every_n_tokens", d.checkpoint_every_n_tokens),
            )
        ),
        record_rate=float(cast(float, raw.get("record_rate", d.record_rate))),
        dedup_ttl_seconds=float(
            cast(float, raw.get("dedup_ttl_seconds", d.dedup_ttl_seconds))
        ),
        quorum_min=int(cast(int, raw.get("quorum_min", d.quorum_min))),
    )


# ---------------------------------------------------------------------------
# ConfigWatcher
# ---------------------------------------------------------------------------


class ConfigWatcher:
    def __init__(
        self,
        config_path: Path = _CONFIG_PATH,
        poll_interval: float = 5.0,
    ) -> None:
        self._config_path: Path = config_path
        self._poll_interval: float = poll_interval
        self._last_mtime: float = 0.0
        self._config: ExoConfig = self._boot()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> ExoConfig:
        """Return the current active config (zero-copy — callers must not mutate)."""
        return self._config

    def load(self) -> ExoConfig:
        """Read config file and return a fresh ExoConfig merged with defaults."""
        from exo.master.config_validator import CONFIG_VALIDATOR

        try:
            raw: dict[str, Any] = cast(
                "dict[str, Any]", json.loads(self._config_path.read_text())
            )
        except FileNotFoundError:
            logger.warning(
                f"[config_watcher] {self._config_path} not found; using defaults"
            )
            return ExoConfig()
        except json.JSONDecodeError as exc:
            logger.error(
                f"[config_watcher] JSON parse error: {exc}; keeping current config"
            )
            return self._config
        raw = CONFIG_VERSION.migrate(raw)
        candidate = _load_from_dict(raw)
        errors = CONFIG_VALIDATOR.validate(candidate)
        if errors:
            for err in errors:
                logger.warning(f"[config_watcher] validation error: {err}")
            logger.warning(
                "[config_watcher] config rejected — keeping previous valid config"
            )
            return self._config
        return candidate

    def save(self, config: ExoConfig) -> None:
        """Persist config to disk (creates parent dirs if needed)."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = CONFIG_VERSION.stamp(asdict(config))
        self._config_path.write_text(json.dumps(config_dict, indent=2))
        logger.info(f"[config_watcher] saved config to {self._config_path}")

    async def run_watch_loop(self) -> None:
        """Poll loop — runs until cancelled.  Start in the API task group."""
        logger.info(
            f"[config_watcher] watching {self._config_path} "
            f"(poll_interval={self._poll_interval}s)"
        )
        while True:
            await asyncio.sleep(self._poll_interval)
            self._check_and_reload()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _boot(self) -> ExoConfig:
        """Called once at construction: create default file if absent, then load."""
        if not self._config_path.exists():
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            default = ExoConfig()
            self._config_path.write_text(json.dumps(asdict(default), indent=2))
            logger.info(
                f"[config_watcher] created default config at {self._config_path}"
            )
        cfg = self.load()
        try:
            self._last_mtime = self._config_path.stat().st_mtime
        except OSError:
            self._last_mtime = 0.0
        self._apply(cfg)
        return cfg

    def force_reload(self) -> ExoConfig:
        """Force an immediate reload from disk regardless of mtime.

        Applies the new config to all singletons and returns the active config
        after reload.
        Never raises: errors are logged and the previous config is kept.
        """
        new_config = self.load()
        changed = self._diff(self._config, new_config)
        if changed:
            logger.info(f"[config_watcher] force_reload — changed fields: {changed}")
            self._apply(new_config)
            self._config = new_config
            with contextlib.suppress(OSError):
                self._last_mtime = self._config_path.stat().st_mtime
        else:
            logger.info("[config_watcher] force_reload — config unchanged")
        return self._config

    def _check_and_reload(self) -> None:
        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            return  # file disappeared; keep current config
        if mtime <= self._last_mtime:
            return

        self._last_mtime = mtime
        new_config = self.load()
        changed = self._diff(self._config, new_config)
        if not changed:
            return

        logger.info(f"[config_watcher] reloading config — changed fields: {changed}")
        self._apply(new_config)
        self._config = new_config

    @staticmethod
    def _diff(old: ExoConfig, new: ExoConfig) -> list[str]:
        """Return names of fields that changed."""
        return [
            f.name
            for f in dataclasses.fields(ExoConfig)
            if getattr(old, f.name) != getattr(new, f.name)
        ]

    def _apply(self, cfg: ExoConfig) -> None:  # noqa: C901 (acceptable length)
        """Push every config field to the relevant singleton."""
        from exo.master.config_validator import CONFIG_VALIDATOR

        try:
            CONFIG_VALIDATOR.validate_and_raise(cfg)
        except ValueError as exc:
            logger.error(f"[config_watcher] _apply aborted — {exc}")
            return
        # ---- Rate limiter -----------------------------------------------
        try:
            from exo.master.rate_limiter import RATE_LIMITER

            RATE_LIMITER.configure(
                authenticated_rpm=cfg.rate_limit_rpm_authenticated,
                anonymous_rpm=cfg.rate_limit_rpm_anonymous,
            )
            logger.debug(
                f"[config_watcher] rate_limiter auth={cfg.rate_limit_rpm_authenticated} "
                f"anon={cfg.rate_limit_rpm_anonymous}"
            )
        except Exception as exc:
            logger.warning(f"[config_watcher] rate_limiter apply failed: {exc}")

        # ---- SLO tracker ------------------------------------------------
        try:
            from exo.master.slo_tracker import SLO_TRACKER

            SLO_TRACKER.configure(slo_threshold_ms=cfg.ttft_slo_ms)
            logger.debug(f"[config_watcher] slo_tracker threshold={cfg.ttft_slo_ms}ms")
        except Exception as exc:
            logger.warning(f"[config_watcher] slo_tracker apply failed: {exc}")

        # ---- Admission controller ---------------------------------------
        try:
            from exo.master.admission_control import ADMISSION_CONTROLLER

            ADMISSION_CONTROLLER.configure(
                max_concurrent_requests=cfg.admission_max_concurrent
            )
            logger.debug(
                f"[config_watcher] admission_controller max_concurrent={cfg.admission_max_concurrent}"
            )
        except Exception as exc:
            logger.warning(f"[config_watcher] admission_controller apply failed: {exc}")

        # ---- Circuit breakers -------------------------------------------
        try:
            from exo.master.circuit_breaker import CIRCUIT_BREAKERS

            CIRCUIT_BREAKERS.configure(
                failure_threshold=cfg.circuit_breaker_failure_threshold,
                cooldown_seconds=cfg.circuit_breaker_cooldown_seconds,
            )
            logger.debug(
                f"[config_watcher] circuit_breakers failure_threshold="
                f"{cfg.circuit_breaker_failure_threshold} "
                f"cooldown={cfg.circuit_breaker_cooldown_seconds}s"
            )
        except Exception as exc:
            logger.warning(f"[config_watcher] circuit_breakers apply failed: {exc}")

        # ---- Request deduplicator ---------------------------------------
        try:
            from exo.master.request_dedup import DEDUP

            DEDUP.configure(ttl_seconds=cfg.dedup_ttl_seconds)
            logger.debug(f"[config_watcher] dedup ttl={cfg.dedup_ttl_seconds}s")
        except Exception as exc:
            logger.warning(f"[config_watcher] dedup apply failed: {exc}")

        # ---- Quorum checker ---------------------------------------------
        try:
            from exo.master.quorum_check import QUORUM_CHECKER

            QUORUM_CHECKER.configure(min_quorum=cfg.quorum_min)
            logger.info(f"[config_watcher] quorum_checker min_quorum={cfg.quorum_min}")
        except Exception as exc:
            logger.warning(f"[config_watcher] quorum_checker apply failed: {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

CONFIG_WATCHER = ConfigWatcher()
