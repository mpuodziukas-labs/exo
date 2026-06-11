"""Behavioral tests for exo.master.adaptive_timeout.

AdaptiveTimeoutCalculator maintains a rolling window of latency samples
per model and computes:
  - p99 latency from ≥ 10 samples
  - base_timeout = clamp(p99 * 1.5, 5.0, 300.0)
  - timeout_for(model, priority) = clamp(base * tier_multiplier, 5.0, 300.0)

Fallback (< 10 samples) returns FALLBACK_TIMEOUT_S = 60.0.
"""

from __future__ import annotations

import pytest

from exo.master.adaptive_timeout import (
    _FALLBACK_TIMEOUT_S,
    _MAX_TIMEOUT_S,
    _MIN_TIMEOUT_S,
    _SAFETY_MULTIPLIER,
    _WINDOW,
    AdaptiveTimeoutCalculator,
)

MODEL_A = "mlx-community/Llama-3b-4bit"
MODEL_B = "mlx-community/Qwen3-30B"


# ---------------------------------------------------------------------------
# Fallback when insufficient samples
# ---------------------------------------------------------------------------


def test_fallback_when_fewer_than_10_samples() -> None:
    """base_timeout returns FALLBACK when < 10 samples recorded."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(9):
        calc.record(MODEL_A, 10.0)
    assert calc.base_timeout(MODEL_A) == _FALLBACK_TIMEOUT_S


def test_fallback_for_unknown_model() -> None:
    """base_timeout returns FALLBACK for a model with zero samples."""
    calc = AdaptiveTimeoutCalculator()
    assert calc.base_timeout("no-such-model") == _FALLBACK_TIMEOUT_S


# ---------------------------------------------------------------------------
# Adaptive computation with ≥ 10 samples
# ---------------------------------------------------------------------------


def test_base_timeout_computed_from_p99() -> None:
    """With ≥ 10 uniform samples, base = clamp(p99 * 1.5, 5, 300)."""
    calc = AdaptiveTimeoutCalculator()
    # All samples identical: p99 = 20.0
    for _ in range(15):
        calc.record(MODEL_A, 20.0)
    expected = max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, 20.0 * _SAFETY_MULTIPLIER))
    assert calc.base_timeout(MODEL_A) == pytest.approx(expected, abs=0.01)


def test_base_timeout_clamped_to_min() -> None:
    """Very fast responses → base_timeout is clamped at _MIN_TIMEOUT_S."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(15):
        calc.record(MODEL_A, 0.01)  # p99 * 1.5 = 0.015 < 5.0
    assert calc.base_timeout(MODEL_A) == _MIN_TIMEOUT_S


def test_base_timeout_clamped_to_max() -> None:
    """Extremely slow model → base_timeout is clamped at _MAX_TIMEOUT_S."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(15):
        calc.record(MODEL_A, 1000.0)  # p99 * 1.5 >> 300
    assert calc.base_timeout(MODEL_A) == _MAX_TIMEOUT_S


# ---------------------------------------------------------------------------
# Priority tier multipliers
# ---------------------------------------------------------------------------


def test_critical_priority_gets_highest_timeout() -> None:
    """critical tier multiplier is 3.0 — highest among all tiers."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(15):
        calc.record(MODEL_A, 20.0)

    critical_t = calc.timeout_for(MODEL_A, "critical")
    normal_t = calc.timeout_for(MODEL_A, "normal")
    background_t = calc.timeout_for(MODEL_A, "background")

    assert critical_t > normal_t
    assert normal_t > background_t


def test_unknown_priority_defaults_to_normal_multiplier() -> None:
    """Unknown priority key defaults to multiplier 1.5 (same as 'normal')."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(15):
        calc.record(MODEL_A, 20.0)

    unknown_t = calc.timeout_for(MODEL_A, "nonexistent")
    normal_t = calc.timeout_for(MODEL_A, "normal")
    assert unknown_t == pytest.approx(normal_t, abs=0.01)


# ---------------------------------------------------------------------------
# Rolling window eviction (_WINDOW = 200)
# ---------------------------------------------------------------------------


def test_window_eviction_drops_oldest_samples() -> None:
    """After _WINDOW samples the deque drops the oldest, affecting p99."""
    calc = AdaptiveTimeoutCalculator()
    # Fill window with slow samples
    for _ in range(_WINDOW):
        calc.record(MODEL_B, 100.0)
    slow_base = calc.base_timeout(MODEL_B)

    # Overwrite the entire window with fast samples
    for _ in range(_WINDOW):
        calc.record(MODEL_B, 1.0)  # clamp will floor at _MIN_TIMEOUT_S
    fast_base = calc.base_timeout(MODEL_B)

    assert fast_base < slow_base


# ---------------------------------------------------------------------------
# all_timeouts introspection
# ---------------------------------------------------------------------------


def test_all_timeouts_includes_recorded_models() -> None:
    """all_timeouts() returns one entry per model that has been recorded."""
    calc = AdaptiveTimeoutCalculator()
    for _ in range(15):
        calc.record(MODEL_A, 5.0)
    for _ in range(15):
        calc.record(MODEL_B, 10.0)

    entries = calc.all_timeouts()
    model_ids = {e["model_id"] for e in entries}
    assert MODEL_A in model_ids
    assert MODEL_B in model_ids
