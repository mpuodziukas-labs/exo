"""
Config schema versioning: tracks the config schema version and migrates
old config files forward. Each migration is a pure function from old dict → new dict.
Ensures backward compatibility when config fields change.
"""
from __future__ import annotations
from typing import Any, Callable
from loguru import logger

_CURRENT_VERSION = 3

# Migration functions: version N → N+1
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

def _migrate_v1_to_v2(cfg: dict[str, Any]) -> dict[str, Any]:
    """v1 had flat keys; v2 groups them under sections."""
    result = dict(cfg)
    result.setdefault("admission", {})
    result.setdefault("slo", {})
    result.setdefault("canary", {})
    # Move top-level rate_limit_rpm into admission section
    if "rate_limit_rpm" in result:
        result["admission"]["rate_limit_rpm"] = result.pop("rate_limit_rpm")
    result["schema_version"] = 2
    return result

def _migrate_v2_to_v3(cfg: dict[str, Any]) -> dict[str, Any]:
    """v3 adds memory_reclaim and health sections."""
    result = dict(cfg)
    result.setdefault("memory_reclaim", {"enabled": True})
    result.setdefault("health", {"probe_interval_s": 5.0})
    result["schema_version"] = 3
    return result

_MIGRATIONS: dict[int, MigrationFn] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


class ConfigVersionManager:
    """
    migrate(cfg): applies all needed migrations to bring cfg to current version.
    validate_version(cfg): returns True if cfg is current version.
    current_version: int property.
    """

    @property
    def current_version(self) -> int:
        return _CURRENT_VERSION

    def migrate(self, cfg: dict[str, Any]) -> dict[str, Any]:
        version = cfg.get("schema_version", 1)
        if version == _CURRENT_VERSION:
            return cfg
        if version > _CURRENT_VERSION:
            logger.warning(f"ConfigVersion: config version {version} > current {_CURRENT_VERSION} — using as-is")
            return cfg

        result = dict(cfg)
        while version < _CURRENT_VERSION:
            migration = _MIGRATIONS.get(version)
            if migration is None:
                logger.error(f"ConfigVersion: no migration from v{version} → skipping")
                break
            try:
                result = migration(result)
                logger.info(f"ConfigVersion: migrated v{version} → v{version+1}")
            except Exception as exc:
                logger.warning(f"ConfigVersion: migration v{version} failed: {exc}")
                break
            version += 1

        return result

    def validate_version(self, cfg: dict[str, Any]) -> bool:
        return cfg.get("schema_version", 1) == _CURRENT_VERSION

    def stamp(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Add current version stamp to a config dict."""
        result = dict(cfg)
        result["schema_version"] = _CURRENT_VERSION
        return result


CONFIG_VERSION = ConfigVersionManager()
