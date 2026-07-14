"""Unit tests for the value-quality strike/retire policy and geo helpers."""

from src.pws.distance import bearing_octant, distance_band
from src.pws.registry import (
    BAD_WINDOW_MIN_READINGS,
    IMMEDIATE_RETIRE_READINGS,
    STRIKE_LIMIT,
    assess_value_quality,
)

NOW = "2026-07-14T00:00:00+00:00"


def test_clean_window_resets_strikes():
    prior = {"value_strikes": 2}
    updates, retired = assess_value_quality(prior, bad_readings=0, now_iso=NOW)
    assert updates["value_strikes"] == 0
    assert retired is False
    assert "retired" not in updates


def test_bad_window_increments_strikes_without_retiring():
    prior = {"value_strikes": 0}
    updates, retired = assess_value_quality(prior, BAD_WINDOW_MIN_READINGS, NOW)
    assert updates["value_strikes"] == 1
    assert retired is False


def test_retire_after_strike_limit():
    prior = {"value_strikes": STRIKE_LIMIT - 1}
    updates, retired = assess_value_quality(prior, BAD_WINDOW_MIN_READINGS, NOW)
    assert updates["value_strikes"] == STRIKE_LIMIT
    assert retired is True
    assert updates["retired"] is True
    assert updates["blacklisted"] is True  # retire forces blacklist
    assert updates["retired_at"] == NOW
    assert "value-quality" in updates["retired_reason"]


def test_immediate_retire_on_egregious_window():
    """A fully stuck sensor retires in one window, no strike wait."""
    prior = {"value_strikes": 0}
    updates, retired = assess_value_quality(prior, IMMEDIATE_RETIRE_READINGS, NOW)
    assert retired is True
    assert updates["retired"] is True


def test_retire_is_sticky_and_preserves_provenance():
    """An already-retired station stays retired even on a now-clean window."""
    prior = {
        "retired": True,
        "retired_at": "2026-07-01T00:00:00+00:00",
        "retired_reason": "value-quality: 40 impossible readings",
        "replaced_by": "KNEW123",
        "value_strikes": 5,
    }
    updates, retired = assess_value_quality(prior, bad_readings=0, now_iso=NOW)
    assert retired is False  # not NEWLY retired
    assert updates["retired"] is True
    assert updates["blacklisted"] is True
    assert updates["retired_at"] == "2026-07-01T00:00:00+00:00"  # original preserved
    assert updates["replaced_by"] == "KNEW123"  # carried forward


def test_distance_band():
    assert distance_band(3.0) == (0, 10)
    assert distance_band(10.0) == (10, 25)
    assert distance_band(49.9) == (25, 50)
    assert distance_band(150.0) is None
    assert distance_band(None) is None


def test_bearing_octant():
    assert bearing_octant(0.0) == "N"
    assert bearing_octant(45.0) == "NE"
    assert bearing_octant(350.0) == "N"   # wraps
    assert bearing_octant(200.0) == "S"
    assert bearing_octant(None) is None
