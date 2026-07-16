"""Unit tests for the crowdsourced-station QC predicates."""

from __future__ import annotations

import numpy as np
import pytest

from src.pws.qc import (
    COLD_THRESHOLD_SIGMA,
    LAPSE_RATE_C_PER_M,
    WARM_THRESHOLD_SIGMA,
    buddy_check_hour,
    buddy_correlation,
    classify_station,
    elevation_adjust,
    flag_outlier,
    robust_center_spread,
    station_flag_fraction,
)


# --- elevation_adjust ------------------------------------------------------------

def test_elevation_adjust_is_a_noop_at_the_same_elevation():
    out = elevation_adjust(np.array([15.0]), np.array([100.0]), 100.0)
    assert out[0] == pytest.approx(15.0)


def test_station_above_reference_is_warmed_when_brought_down():
    # A station 100m ABOVE the reference reads ~0.65 C cold legitimately. Adjusting it
    # to the (lower) reference must add that back.
    out = elevation_adjust(np.array([14.35]), np.array([200.0]), 100.0)
    assert out[0] == pytest.approx(15.0, abs=1e-6)


def test_station_below_reference_is_cooled_when_brought_up():
    out = elevation_adjust(np.array([15.65]), np.array([0.0]), 100.0)
    assert out[0] == pytest.approx(15.0, abs=1e-6)


def test_lapse_rate_matches_the_standard_atmosphere():
    # 1 km of ascent is ~6.5 C.
    out = elevation_adjust(np.array([20.0]), np.array([0.0]), 1000.0)
    assert out[0] == pytest.approx(20.0 - 6.5, abs=1e-6)
    assert LAPSE_RATE_C_PER_M == -0.0065


def test_elevation_adjust_handles_the_networks_real_range():
    # The registry spans 0-1174 m: ~7.6 C of legitimate spread.
    out = elevation_adjust(np.array([10.0]), np.array([1174.0]), 0.0)
    assert out[0] == pytest.approx(10.0 + 7.631, abs=0.01)


# --- robust_center_spread --------------------------------------------------------

def test_robust_center_spread_ignores_an_outlier():
    # mean/SD would be dragged by the 99; median/MAD must not be.
    center, spread = robust_center_spread([15.0, 15.2, 14.8, 15.1, 99.0])
    assert center == pytest.approx(15.1, abs=0.2)
    assert spread < 1.0


def test_robust_center_spread_ignores_nan():
    center, _ = robust_center_spread([15.0, np.nan, 15.0])
    assert center == pytest.approx(15.0)


def test_robust_center_spread_all_nan_is_nan():
    center, spread = robust_center_spread([np.nan, np.nan])
    assert np.isnan(center) and np.isnan(spread)


def test_robust_center_spread_identical_values_gives_zero_spread():
    _, spread = robust_center_spread([15.0, 15.0, 15.0])
    assert spread == 0.0


# --- flag_outlier: the asymmetry ------------------------------------------------

def test_warm_outlier_is_flagged_at_the_warm_threshold():
    assert flag_outlier(15.0 + (WARM_THRESHOLD_SIGMA + 0.1), 15.0, 1.0) is True


def test_cold_deviation_of_the_same_size_is_NOT_flagged():
    # The core asymmetry: CWS radiative error runs warm, so an equal-magnitude cold
    # excursion is likelier to be real weather (cold-air pooling) than a broken sensor.
    magnitude = WARM_THRESHOLD_SIGMA + 0.1
    assert flag_outlier(15.0 + magnitude, 15.0, 1.0) is True
    assert flag_outlier(15.0 - magnitude, 15.0, 1.0) is False


def test_cold_outlier_is_flagged_past_the_laxer_cold_threshold():
    assert flag_outlier(15.0 - (COLD_THRESHOLD_SIGMA + 0.1), 15.0, 1.0) is True


def test_thresholds_are_ordered_warm_stricter_than_cold():
    assert WARM_THRESHOLD_SIGMA < COLD_THRESHOLD_SIGMA


def test_zero_spread_makes_no_outlier_call():
    # Identical buddies means a stuck/duplicated feed, not perfect agreement. Refuse to
    # divide by zero and hand the decision to the other tests.
    assert flag_outlier(99.0, 15.0, 0.0) is False


def test_nan_inputs_make_no_outlier_call():
    assert flag_outlier(np.nan, 15.0, 1.0) is False
    assert flag_outlier(15.0, np.nan, 1.0) is False


# --- buddy_check_hour ------------------------------------------------------------

def test_buddy_check_passes_a_normal_station():
    flagged, z = buddy_check_hour(15.0, 100.0, [15.1, 14.9, 15.2, 15.0], [100.0] * 4)
    assert flagged is False
    assert abs(z) < 2.0


def test_buddy_check_catches_a_radiatively_heated_sensor():
    # The canonical failure: a poorly-aspirated sensor reading high in daylight.
    flagged, z = buddy_check_hour(21.0, 100.0, [15.1, 14.9, 15.2, 15.0], [100.0] * 4)
    assert flagged is True
    assert z > 0


def test_buddy_check_does_not_punish_altitude():
    # A station at 1000 m among sea-level buddies reads 6.5 C colder for real reasons.
    # Without the lapse-rate adjustment this is a huge cold outlier; with it, it passes.
    flagged, z = buddy_check_hour(
        8.5, 1000.0, [15.1, 14.9, 15.2, 15.0], [0.0] * 4
    )
    assert flagged is False
    assert abs(z) < 2.0


def test_buddy_check_needs_a_quorum():
    flagged, z = buddy_check_hour(99.0, 100.0, [15.0, 15.1], [100.0, 100.0], min_buddies=3)
    assert flagged is False
    assert np.isnan(z)  # untestable, not passing


def test_buddy_check_with_degenerate_spread_is_untestable():
    flagged, z = buddy_check_hour(99.0, 100.0, [15.0] * 5, [100.0] * 5)
    assert flagged is False and np.isnan(z)


def test_buddy_check_drops_buddies_with_missing_elevation():
    flagged, _ = buddy_check_hour(
        21.0, 100.0, [15.1, 14.9, 15.2, 15.0], [100.0, np.nan, 100.0, 100.0], min_buddies=3
    )
    assert flagged is True  # still 3 usable buddies


# --- station_flag_fraction -------------------------------------------------------

def test_station_flag_fraction_counts_flags():
    assert station_flag_fraction([True, False, False, False]) == pytest.approx(0.25)


def test_station_flag_fraction_empty_is_nan():
    assert np.isnan(station_flag_fraction([]))


# --- buddy_correlation: the indoor detector --------------------------------------

def _diurnal(n=300, amplitude=8.0, offset=15.0):
    return offset + amplitude * np.sin(np.linspace(0, 12 * np.pi, n))


def test_outdoor_station_correlates_with_its_buddies():
    ref = _diurnal()
    station = ref + np.random.default_rng(0).normal(0, 0.3, ref.size)
    assert buddy_correlation(station, ref) > 0.95


def test_indoor_station_fails_the_correlation_test():
    # An indoor sensor is plausible hour by hour -- it passes the outlier test -- but it
    # does not track the outdoor diurnal cycle. This is the test that catches it.
    ref = _diurnal()
    indoor = 21.0 + np.random.default_rng(1).normal(0, 0.4, ref.size)
    assert buddy_correlation(indoor, ref) < 0.90


def test_correlation_requires_enough_overlap():
    assert np.isnan(buddy_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], min_overlap=100))


def test_correlation_of_a_flatlined_station_is_nan():
    ref = _diurnal()
    assert np.isnan(buddy_correlation(np.full(ref.size, 20.0), ref))


# --- classify_station ------------------------------------------------------------

def test_good_station_is_ok():
    status, reason = classify_station(flag_fraction=0.02, correlation=0.98, n_buddies=8)
    assert status == "ok" and reason is None


def test_isolation_is_checked_before_anything_else():
    # An isolated station's other statistics are untrustworthy, not passing.
    status, reason = classify_station(flag_fraction=0.0, correlation=0.99, n_buddies=1)
    assert status == "isolated"
    assert "buddies" in reason


def test_low_correlation_is_suspect():
    status, reason = classify_station(flag_fraction=0.01, correlation=0.55, n_buddies=8)
    assert status == "suspect" and "indoor" in reason


def test_persistent_flagging_is_suspect():
    status, reason = classify_station(flag_fraction=0.45, correlation=0.97, n_buddies=8)
    assert status == "suspect" and "flagged" in reason


def test_nan_statistics_do_not_condemn_a_station():
    # No evidence is not evidence of badness.
    status, _ = classify_station(flag_fraction=float("nan"), correlation=float("nan"), n_buddies=8)
    assert status == "ok"
