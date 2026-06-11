from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exo.master.config_watcher import ExoConfig


@dataclass
class ValidationError:
    field: str
    value: Any
    reason: str

    def __str__(self) -> str:
        return f"{self.field}={self.value!r}: {self.reason}"


# (field, lo, hi) — inclusive on both ends; None = no bound
_INT_RANGES: list[tuple[str, int, int]] = [
    ("rate_limit_rpm_authenticated", 1, 10000),
    ("rate_limit_rpm_anonymous", 1, 1000),
    ("admission_max_concurrent", 1, 1000),
    ("circuit_breaker_failure_threshold", 1, 100),
    ("checkpoint_every_n_tokens", 1, 10000),
]

_FLOAT_RANGES: list[tuple[str, float, float]] = [
    ("ttft_slo_ms", 50.0, 30000.0),
    ("canary_percent", 0.0, 100.0),
    ("circuit_breaker_cooldown_seconds", 1.0, 3600.0),
    ("record_rate", 0.0, 1.0),
    ("dedup_ttl_seconds", 1.0, 3600.0),
]

SCHEMA: dict[str, dict[str, Any]] = {
    "rate_limit_rpm_authenticated": {"type": "int", "min": 1, "max": 10000},
    "rate_limit_rpm_anonymous": {"type": "int", "min": 1, "max": 1000, "constraint": "<= rate_limit_rpm_authenticated"},
    "ttft_slo_ms": {"type": "float", "min": 50.0, "max": 30000.0},
    "admission_max_concurrent": {"type": "int", "min": 1, "max": 1000},
    "canary_percent": {"type": "float", "min": 0.0, "max": 100.0},
    "circuit_breaker_failure_threshold": {"type": "int", "min": 1, "max": 100},
    "circuit_breaker_cooldown_seconds": {"type": "float", "min": 1.0, "max": 3600.0},
    "checkpoint_every_n_tokens": {"type": "int", "min": 1, "max": 10000},
    "record_rate": {"type": "float", "min": 0.0, "max": 1.0},
    "dedup_ttl_seconds": {"type": "float", "min": 1.0, "max": 3600.0},
}


class ConfigValidator:
    def validate(self, config: ExoConfig) -> list[ValidationError]:
        errors: list[ValidationError] = []

        for field, lo, hi in _INT_RANGES:
            val = getattr(config, field)
            if not isinstance(val, int) or not (lo <= val <= hi):
                errors.append(ValidationError(field, val, f"must be int in [{lo}, {hi}]"))

        for field, lo, hi in _FLOAT_RANGES:
            val = getattr(config, field)
            if not isinstance(val, (int, float)) or not (lo <= float(val) <= hi):
                errors.append(ValidationError(field, val, f"must be float in [{lo}, {hi}]"))

        anon = config.rate_limit_rpm_anonymous
        auth = config.rate_limit_rpm_authenticated
        if isinstance(anon, int) and isinstance(auth, int) and anon > auth:
            errors.append(ValidationError(
                "rate_limit_rpm_anonymous", anon,
                f"must be <= rate_limit_rpm_authenticated ({auth})"
            ))

        return errors

    def validate_and_raise(self, config: ExoConfig) -> None:
        errors = self.validate(config)
        if errors:
            raise ValueError("Invalid config:\n" + "\n".join(f"  {e}" for e in errors))

    def is_valid(self, config: ExoConfig) -> bool:
        return not self.validate(config)


CONFIG_VALIDATOR = ConfigValidator()
