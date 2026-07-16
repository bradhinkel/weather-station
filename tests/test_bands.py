"""Unit tests for the shared upwind band-mean features."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.bands import (
    BAND_FEATURE_COLS,
    BANDS_KM,
    apply_band_fills,
    band_fill_values,
    build_band_features,
)

HOME_ELEV = 15.0

# station_id -> (distance_km, bearing_deg, elevation_m)
GEOM = {
    "near_w":   (5.0, 270.0, 15.0),    # band 0, due west
    "mid_w":    (20.0, 270.0, 15.0),   # band 1
    "far_w":    (45.0, 265.0, 15.0),   # band 2
    "vfar_w":   (80.0, 275.0, 15.0),   # band 3
    "near_e":   (5.0, 90.0, 15.0),     # band 0 but downwind of a west wind
    "hill_w":   (22.0, 268.0, 1015.0), # band 1, 1000m above home
}


def test_west_wind_selects_only_western_stations():
    temps = {"near_w": 14.0, "near_e": 25.0}
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, temps)
    # near_e is downwind and must not contribute.
    assert f["band0_temp"] == pytest.approx(14.0)
    assert f["bands_n_total"] == 1.0


def test_stations_land_in_the_right_bands():
    temps = {"near_w": 10.0, "mid_w": 11.0, "far_w": 12.0, "vfar_w": 13.0}
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, temps)
    assert f["band0_temp"] == pytest.approx(10.0)
    assert f["band1_temp"] == pytest.approx(11.0)
    assert f["band2_temp"] == pytest.approx(12.0)
    assert f["band3_temp"] == pytest.approx(13.0)
    assert f["bands_n_total"] == 4.0


def test_band_mean_averages_within_a_band():
    geom = {"a": (12.0, 270.0, 15.0), "b": (25.0, 270.0, 15.0)}
    f = build_band_features(270.0, 15.0, HOME_ELEV, geom, {"a": 10.0, "b": 20.0})
    assert f["band1_temp"] == pytest.approx(15.0)
    assert f["bands_n_total"] == 2.0


def test_elevation_is_adjusted_before_averaging():
    # hill_w sits 1000m above home and reads 6.5 C colder for real reasons. Adjusted to
    # home's elevation it should agree with mid_w, so the band mean must be ~11.0 --
    # not the raw mean of 11.0 and 4.5.
    temps = {"mid_w": 11.0, "hill_w": 4.5}
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, temps)
    assert f["band1_temp"] == pytest.approx(11.0, abs=0.02)


def test_gradient_is_incoming_air_minus_home():
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, {"mid_w": 11.0})
    assert f["band1_grad"] == pytest.approx(-4.0)


def test_empty_band_is_nan_never_zero():
    # A 0 C band mean is physically absurd and is the fill-0 defect that blew Ridge up.
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, {"mid_w": 11.0})
    assert np.isnan(f["band0_temp"])
    assert np.isnan(f["band0_grad"])
    assert f["band1_temp"] == pytest.approx(11.0)


def test_upwind_bearing_equals_wind_direction():
    # Meteorological convention: 180 deg means wind FROM the south, so southern stations
    # are the upwind ones.
    geom = {"south": (20.0, 180.0, 15.0), "north": (20.0, 0.0, 15.0)}
    f = build_band_features(180.0, 15.0, HOME_ELEV, geom, {"south": 12.0, "north": 20.0})
    assert f["band1_temp"] == pytest.approx(12.0)


def test_tolerance_excludes_crosswind_stations():
    geom = {"perp": (20.0, 0.0, 15.0)}  # 90 deg off a west wind
    f = build_band_features(270.0, 15.0, HOME_ELEV, geom, {"perp": 12.0})
    assert f["bands_n_total"] == 0.0
    assert np.isnan(f["band1_temp"])


def test_nan_wind_direction_yields_no_stations():
    f = build_band_features(float("nan"), 15.0, HOME_ELEV, GEOM, {"mid_w": 11.0})
    assert f["bands_n_total"] == 0.0


def test_missing_temperature_is_skipped():
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, {"mid_w": float("nan")})
    assert f["bands_n_total"] == 0.0


def test_output_keys_match_the_declared_feature_list():
    f = build_band_features(270.0, 15.0, HOME_ELEV, GEOM, {"mid_w": 11.0})
    assert set(f) == set(BAND_FEATURE_COLS)
    assert len(BANDS_KM) * 2 + 1 == len(BAND_FEATURE_COLS)


def test_quality_filtering_happens_at_the_boundary():
    # Callers pass only the stations they trust; the function does not second-guess.
    all_temps = {"mid_w": 11.0, "far_w": 99.0}
    screened = {"mid_w": 11.0}
    assert build_band_features(270.0, 15.0, HOME_ELEV, GEOM, screened)["bands_n_total"] == 1.0
    assert build_band_features(270.0, 15.0, HOME_ELEV, GEOM, all_temps)["bands_n_total"] == 2.0


# --- fills -----------------------------------------------------------------------

def test_band_fill_values_are_train_means_ignoring_nan():
    rows = [
        {"band0_temp": 10.0, "band0_grad": -1.0},
        {"band0_temp": 20.0, "band0_grad": float("nan")},
    ]
    fills = band_fill_values(rows)
    assert fills["band0_temp"] == pytest.approx(15.0)
    assert fills["band0_grad"] == pytest.approx(-1.0)


def test_apply_band_fills_replaces_nan_with_the_train_mean():
    feat = {c: float("nan") for c in BAND_FEATURE_COLS}
    feat["band0_temp"] = 12.0
    out = apply_band_fills(feat, {c: 5.0 for c in BAND_FEATURE_COLS})
    assert out["band0_temp"] == 12.0        # present value untouched
    assert out["band1_temp"] == 5.0         # gap filled from the train mean
    assert all(np.isfinite(out[c]) for c in BAND_FEATURE_COLS)


def test_apply_band_fills_without_fills_is_a_noop():
    feat = {"band0_temp": float("nan")}
    assert np.isnan(apply_band_fills(feat, None)["band0_temp"])
