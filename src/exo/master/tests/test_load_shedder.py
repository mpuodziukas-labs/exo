"""Behavioral tests for exo.master.load_shedder.

LoadShedder controls active load shedding based on a configurable ShedLevel.
Higher levels shed lower-priority traffic cumulatively:
  LIGHT     → background only
  MODERATE  → background, low
  HEAVY     → background, low, normal
  EMERGENCY → background, low, normal, high
"""

from __future__ import annotations

from exo.master.load_shedder import LoadShedder, ShedLevel

# ---------------------------------------------------------------------------
# ShedLevel.NONE — nothing shed
# ---------------------------------------------------------------------------


def test_none_level_passes_all_priorities() -> None:
    """At NONE level every priority class is allowed through."""
    shedder = LoadShedder()
    for priority in ("critical", "high", "normal", "low", "background"):
        should_shed, retry = shedder.check(priority)
        assert should_shed is False, f"Expected no shedding for {priority!r} at NONE"
        assert retry == 0


# ---------------------------------------------------------------------------
# LIGHT level — background only
# ---------------------------------------------------------------------------


def test_light_sheds_background_only() -> None:
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.LIGHT)

    shed, retry = shedder.check("background")
    assert shed is True
    assert retry == 5

    for priority in ("critical", "high", "normal", "low"):
        shed, _ = shedder.check(priority)
        assert shed is False, f"LIGHT should not shed {priority!r}"


# ---------------------------------------------------------------------------
# MODERATE level — background + low
# ---------------------------------------------------------------------------


def test_moderate_sheds_background_and_low() -> None:
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.MODERATE)

    for priority in ("background", "low"):
        shed, retry = shedder.check(priority)
        assert shed is True
        assert retry == 15

    for priority in ("critical", "high", "normal"):
        shed, _ = shedder.check(priority)
        assert shed is False, f"MODERATE should not shed {priority!r}"


# ---------------------------------------------------------------------------
# HEAVY level — background + low + normal
# ---------------------------------------------------------------------------


def test_heavy_sheds_up_to_normal() -> None:
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.HEAVY)

    for priority in ("background", "low", "normal"):
        shed, retry = shedder.check(priority)
        assert shed is True
        assert retry == 30

    for priority in ("critical", "high"):
        shed, _ = shedder.check(priority)
        assert shed is False


# ---------------------------------------------------------------------------
# EMERGENCY level — everything except critical
# ---------------------------------------------------------------------------


def test_emergency_sheds_all_except_critical() -> None:
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.EMERGENCY)

    for priority in ("background", "low", "normal", "high"):
        shed, retry = shedder.check(priority)
        assert shed is True
        assert retry == 60

    shed, _ = shedder.check("critical")
    assert shed is False


# ---------------------------------------------------------------------------
# auto_set_from_metrics
# ---------------------------------------------------------------------------


def test_auto_set_from_metrics_queue_thresholds() -> None:
    """Queue depth thresholds map to correct shed levels."""
    shedder = LoadShedder()

    assert shedder.auto_set_from_metrics(5, "ok") == ShedLevel.NONE
    assert shedder.auto_set_from_metrics(15, "ok") == ShedLevel.LIGHT
    assert shedder.auto_set_from_metrics(25, "ok") == ShedLevel.MODERATE
    assert shedder.auto_set_from_metrics(55, "ok") == ShedLevel.HEAVY
    assert shedder.auto_set_from_metrics(101, "ok") == ShedLevel.EMERGENCY


def test_auto_set_from_metrics_memory_pressure() -> None:
    """Memory pressure keywords drive the correct shed level."""
    shedder = LoadShedder()
    assert shedder.auto_set_from_metrics(0, "warning") == ShedLevel.MODERATE
    assert shedder.auto_set_from_metrics(0, "critical") == ShedLevel.HEAVY
    assert shedder.auto_set_from_metrics(0, "fatal") == ShedLevel.EMERGENCY


# ---------------------------------------------------------------------------
# History logging
# ---------------------------------------------------------------------------


def test_set_level_records_history_on_change() -> None:
    """set_level logs a ShedEvent only when the level actually changes."""
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.LIGHT, reason="test-reason")
    shedder.set_level(ShedLevel.LIGHT)  # same level — should not add another event

    s = shedder.stats()
    assert len(s["recent_events"]) == 1
    assert s["recent_events"][0]["level"] == "light"


def test_shed_count_accumulates() -> None:
    """shed_count increments each time a request is shed."""
    shedder = LoadShedder()
    shedder.set_level(ShedLevel.HEAVY)
    shedder.check("normal")
    shedder.check("normal")
    shedder.check("critical")  # not shed

    s = shedder.stats()
    assert s["shed_count"] == 2
    assert s["total_checked"] == 3
