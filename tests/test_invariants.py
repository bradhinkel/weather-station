"""Unit tests for the physical-plausibility predicates.

Each test reconstructs the *signature* of a bug this project actually shipped, so a
regression trips the same wire that caught it the first time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.invariants import (
    check_baseline_monotonic,
    check_forecast_lead,
    check_no_constant_columns,
    check_physical_bounds,
    check_rain_positive_fraction,
)


# --- check_baseline_monotonic — the 2026-07-15 forecast-join bug ------------------

def test_baseline_monotonic_accepts_growing_error():
    # Real own-station Open-Meteo MAE measured 2026-07-15 at true lead times.
    real = {1: 1.013, 3: 1.067, 6: 1.140, 12: 1.209, 24: 1.264}
    assert check_baseline_monotonic(real) == []


def test_baseline_monotonic_catches_flat_baseline():
    # The old README table: identical baseline at every horizon.
    violations = check_baseline_monotonic({1: 1.68, 3: 1.68, 24: 1.68})
    assert violations
    assert "identical" in " ".join(violations)


def test_baseline_monotonic_catches_error_falling_with_lead():
    violations = check_baseline_monotonic({1: 1.5, 24: 0.9})
    assert len(violations) == 1
    assert "cannot be more" in violations[0]


def test_baseline_monotonic_tolerates_small_wobble_within_tol():
    assert check_baseline_monotonic({1: 1.10, 3: 1.09}, tol=0.05) == []


def test_baseline_monotonic_single_horizon_is_vacuously_ok():
    assert check_baseline_monotonic({6: 1.2}) == []


# --- check_forecast_lead — the train/serve contract -------------------------------

def _lead_frame(lead_hours: list[float]) -> pd.DataFrame:
    valid = pd.Timestamp("2026-07-01 12:00", tz="UTC")
    return pd.DataFrame(
        {
            "valid_time": [valid] * len(lead_hours),
            "forecast_time": [valid - pd.Timedelta(hours=h) for h in lead_hours],
        }
    )


def test_forecast_lead_accepts_rows_at_or_beyond_horizon():
    assert check_forecast_lead(_lead_frame([24.0, 25.5, 30.0]), horizon=24) == []


def test_forecast_lead_catches_too_fresh_forecast():
    # The pre-fix behaviour: a ~1h-lead forecast used to train a +24h model.
    violations = check_forecast_lead(_lead_frame([1.0, 1.0, 24.0]), horizon=24)
    assert len(violations) == 1
    assert "2/3 rows" in violations[0]
    assert "train/serve skew" in violations[0]


def test_forecast_lead_empty_frame_is_ok():
    assert check_forecast_lead(pd.DataFrame(), horizon=6) == []


# --- check_no_constant_columns — the 2026-06-19 tz-strip --------------------------

def test_no_constant_columns_accepts_varying_features():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, -0.5, 2.0]})
    assert check_no_constant_columns(df, ["a", "b"]) == []


def test_no_constant_columns_catches_silently_zeroed_feature():
    # Exactly the tz-strip signature: the column exists, is all zeros, nothing raises.
    df = pd.DataFrame({"net_temp": [0.0, 0.0, 0.0], "ok": [1.0, 2.0, 3.0]})
    violations = check_no_constant_columns(df, ["net_temp", "ok"])
    assert len(violations) == 1
    assert "net_temp" in violations[0]


def test_no_constant_columns_catches_all_nan_feature():
    df = pd.DataFrame({"gone": [np.nan, np.nan]})
    violations = check_no_constant_columns(df, ["gone"])
    assert len(violations) == 1
    assert "NaN" in violations[0]


def test_no_constant_columns_ignores_absent_columns():
    assert check_no_constant_columns(pd.DataFrame({"a": [1.0, 2.0]}), ["missing"]) == []


# --- check_physical_bounds — the 2026-06-19 fill-0 blowup -------------------------

def test_physical_bounds_accepts_real_weather():
    df = pd.DataFrame({"f_temp_c": [12.0, 24.5], "f_pressure_hpa": [1013.0, 1021.5]})
    assert check_physical_bounds(df) == []


def test_physical_bounds_catches_fill_zero_pressure():
    # fill-0 on a field that physically sits near 1015 hPa.
    df = pd.DataFrame({"f_pressure_hpa": [1013.0, 0.0, 0.0]})
    violations = check_physical_bounds(df)
    assert len(violations) == 1
    assert "2/3 rows" in violations[0]


def test_physical_bounds_catches_stuck_rain_gauge():
    # The KWARENTO432 signature: 896 mm/h.
    df = pd.DataFrame({"lag_rain_mm_1h": [0.0, 0.4, 896.37]})
    violations = check_physical_bounds(df)
    assert len(violations) == 1
    assert "lag_rain_mm_1h" in violations[0]


# --- check_rain_positive_fraction — the 2026-06-01 rain-target bug ----------------

def test_rain_fraction_accepts_plausible_wet_rate():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(1000) < 0.12, 1.5, 0.0)
    assert check_rain_positive_fraction(y) == []


def test_rain_fraction_catches_near_zero_positives():
    # The exact shape of the bug: 1 positive row in 76k.
    y = np.zeros(76_000)
    y[0] = 2.0
    violations = check_rain_positive_fraction(y)
    assert len(violations) == 1
    assert "implausibly dry" in violations[0]


def test_rain_fraction_catches_implausibly_wet_corpus():
    violations = check_rain_positive_fraction(np.full(500, 5.0))
    assert len(violations) == 1
    assert "implausibly wet" in violations[0]


def test_rain_fraction_empty_is_flagged():
    assert check_rain_positive_fraction(np.array([])) == ["rain target is empty."]


@pytest.mark.parametrize("threshold", [0.1, 0.2])
def test_rain_fraction_respects_threshold(threshold):
    # Trace amounts below the threshold must not count as wet hours.
    y = np.full(1000, threshold - 0.01)
    violations = check_rain_positive_fraction(y, threshold_mm=threshold)
    assert violations and "implausibly dry" in violations[0]


# --- the rain-model guard (added 2026-07-15) --------------------------------------

def test_pooled_targets_keeps_rain_pooled():
    """Rain must not train own-station until a wet season supplies positives.

    Retrained own-station on 2026-07-15, all five rain horizons returned zero positive
    test hours -- the backyard had no wet hours in July and the temporal split puts July
    in test -- collapsing F1 to 0. This pins the policy so a future refactor cannot
    quietly flip rain to a target with no rain in it.
    """
    from src.ml import POOLED_TARGETS, trains_pooled

    assert trains_pooled("rain_mm_1h") is True
    assert trains_pooled("temp_c") is False
    assert "rain_mm_1h" in POOLED_TARGETS
    assert "temp_c" not in POOLED_TARGETS
