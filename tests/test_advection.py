"""Unit tests for the dynamic-advection feature geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.advection import (
    MAX_MATCH_DISTANCE_KM,
    MIN_ADVECTION_WIND_MS,
    build_advection_features,
    impute_advection,
    nearest_station,
    project_upwind_point,
)
from src.pws.distance import bearing_deg, haversine_km

HOME_LAT, HOME_LON = 47.6944, -122.2144


# --- project_upwind_point --------------------------------------------------------

def test_projection_distance_matches_v_times_h():
    # 2.26 m/s (network median forecast wind) for 3h = 24.4 km.
    _, _, dist = project_upwind_point(HOME_LAT, HOME_LON, 2.26, 270.0, 3)
    assert dist == pytest.approx(24.4, abs=0.1)


def test_projection_scales_linearly_with_lead_time():
    _, _, d1 = project_upwind_point(HOME_LAT, HOME_LON, 3.0, 180.0, 1)
    _, _, d6 = project_upwind_point(HOME_LAT, HOME_LON, 3.0, 180.0, 6)
    assert d6 == pytest.approx(6 * d1)


def test_projection_lands_at_the_stated_distance_and_bearing():
    lat, lon, dist = project_upwind_point(HOME_LAT, HOME_LON, 5.0, 225.0, 4)
    assert haversine_km(HOME_LAT, HOME_LON, lat, lon) == pytest.approx(dist, rel=1e-3)
    assert bearing_deg(HOME_LAT, HOME_LON, lat, lon) == pytest.approx(225.0, abs=0.5)


def test_west_wind_projects_to_the_west():
    # Meteorological convention: 270 deg means wind FROM the west, so the incoming
    # parcel is currently west of home -> longitude decreases.
    lat, lon, _ = project_upwind_point(HOME_LAT, HOME_LON, 4.0, 270.0, 3)
    assert lon < HOME_LON
    assert lat == pytest.approx(HOME_LAT, abs=0.05)


def test_south_wind_projects_to_the_south():
    lat, lon, _ = project_upwind_point(HOME_LAT, HOME_LON, 4.0, 180.0, 3)
    assert lat < HOME_LAT


# --- nearest_station -------------------------------------------------------------

def _coords() -> dict[str, tuple[float, float]]:
    return {
        "near": (47.70, -122.22),
        "west_20km": (47.6944, -122.48),
        "south_far": (47.40, -122.2144),
    }


def test_nearest_station_picks_the_closest():
    sid, km = nearest_station(47.70, -122.22, _coords())
    assert sid == "near"
    assert km < 1.0


def test_nearest_station_honours_exclude():
    sid, _ = nearest_station(47.70, -122.22, _coords(), exclude={"near"})
    assert sid != "near"


def test_nearest_station_empty_registry():
    sid, km = nearest_station(47.7, -122.2, {})
    assert sid is None and km == float("inf")


# --- build_advection_features ----------------------------------------------------

def _frame(wind_ms: float, wind_dir: float = 270.0, lag_temp: float = 15.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time": [pd.Timestamp("2026-07-01 12:00", tz="UTC")],
            "f_wind_speed_ms": [wind_ms],
            "f_wind_dir_deg": [wind_dir],
            "lag_temp_c": [lag_temp],
        }
    )


def test_calm_wind_is_flagged_invalid_not_fabricated():
    # Below the threshold the direction is noise; we must not invent an upwind station.
    df = _frame(MIN_ADVECTION_WIND_MS - 0.1)
    out = build_advection_features(df, 3, HOME_LAT, HOME_LON, _coords(), {})
    assert out["adv_valid"].iloc[0] == 0.0
    assert np.isnan(out["adv_temp_c"].iloc[0])


def test_reads_the_upwind_station_at_t_minus_horizon():
    # 2.0 m/s west wind for 3h reaches ~21.6 km west -> the west_20km station.
    obs_hour = pd.Timestamp("2026-07-01 09:00", tz="UTC")  # valid_time - 3h
    lookup = {("west_20km", obs_hour): 11.0}
    out = build_advection_features(_frame(2.0), 3, HOME_LAT, HOME_LON, _coords(), lookup)
    assert out["adv_valid"].iloc[0] == 1.0
    assert out["adv_temp_c"].iloc[0] == 11.0
    # Incoming air is 4 C colder than home's current reading of 15.
    assert out["adv_temp_gradient"].iloc[0] == pytest.approx(-4.0)


def test_observation_at_the_wrong_hour_is_not_used():
    # Same station, but only a valid_time-hour reading exists. The parcel must be
    # sampled at t, not at t+h -- using t+h would leak the future.
    lookup = {("west_20km", pd.Timestamp("2026-07-01 12:00", tz="UTC")): 11.0}
    out = build_advection_features(_frame(2.0), 3, HOME_LAT, HOME_LON, _coords(), lookup)
    assert out["adv_valid"].iloc[0] == 0.0


def test_home_station_is_excluded_from_selection():
    coords = {"home": (HOME_LAT, HOME_LON), "west_20km": (47.6944, -122.48)}
    obs_hour = pd.Timestamp("2026-07-01 09:00", tz="UTC")
    lookup = {("home", obs_hour): 99.0, ("west_20km", obs_hour): 11.0}
    out = build_advection_features(
        _frame(2.0), 3, HOME_LAT, HOME_LON, coords, lookup, home_station_id="home"
    )
    assert out["adv_temp_c"].iloc[0] == 11.0


def test_projection_into_a_network_gap_is_flagged():
    # Strong wind reaches far past any station in the registry.
    obs_hour = pd.Timestamp("2026-07-01 09:00", tz="UTC")
    lookup = {("west_20km", obs_hour): 11.0}
    out = build_advection_features(_frame(30.0), 6, HOME_LAT, HOME_LON, _coords(), lookup)
    assert out["adv_match_km"].iloc[0] > MAX_MATCH_DISTANCE_KM
    assert out["adv_valid"].iloc[0] == 0.0


def test_distance_column_records_the_reach_even_when_invalid():
    out = build_advection_features(_frame(3.0), 6, HOME_LAT, HOME_LON, {}, {})
    assert out["adv_distance_km"].iloc[0] == pytest.approx(3.0 * 6 * 3.6)


# --- the build_dataset contract --------------------------------------------------

@pytest.mark.parametrize("wind_dir", [0.0, 45.0, 90.0, 180.0, 269.9, 359.0])
def test_wind_dir_round_trips_through_the_sin_cos_encoding(wind_dir):
    """build_dataset drops f_wind_dir_deg, keeping only wind_dir_sin/wind_dir_cos.

    Callers must reconstruct the bearing with atan2 rather than assume the raw column
    survives. The first run of tools/advection_experiment.py assumed it did and died on
    a live AttributeError -- the unit tests above had passed because they hand-built
    frames containing the column the author believed was there. This asserts the
    inverse actually holds.
    """
    sin_v = np.sin(np.deg2rad(wind_dir))
    cos_v = np.cos(np.deg2rad(wind_dir))
    recovered = (np.degrees(np.arctan2(sin_v, cos_v)) + 360.0) % 360.0
    assert recovered == pytest.approx(wind_dir, abs=1e-6)


# --- impute_advection ------------------------------------------------------------

def test_impute_uses_train_mean_not_zero():
    train = pd.DataFrame(
        {
            "adv_temp_c": [10.0, 20.0],
            "adv_temp_gradient": [-1.0, 1.0],
            "adv_distance_km": [20.0, 30.0],
            "adv_match_km": [1.0, 3.0],
        }
    )
    test = pd.DataFrame(
        {
            "adv_temp_c": [np.nan],
            "adv_temp_gradient": [np.nan],
            "adv_distance_km": [np.nan],
            "adv_match_km": [np.nan],
        }
    )
    out = impute_advection(test, fill_from=train)
    # Train mean, NOT 0 -- a 0 C "upwind temperature" is the fill-0 bug class.
    assert out["adv_temp_c"].iloc[0] == pytest.approx(15.0)
    assert out["adv_distance_km"].iloc[0] == pytest.approx(25.0)
    assert (out.to_numpy() != 0).any()
