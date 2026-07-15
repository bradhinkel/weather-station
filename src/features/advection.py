"""Wind-scaled ("dynamic advection") upwind features.

The existing network pipeline (`src.features.pipeline`) reduces a *cohort* of
upwind stations into a weighted mean at *fixed* lags (1/3/6/12 h). That shape
cannot express the physics it is trying to capture. Air moves. The parcel that
arrives at the home station at `t + h` is, right now at time `t`, sitting roughly
`v · h` upwind — and `v · h` is not a constant, it is whatever the wind is doing
today:

    lead   distance at median wind (2.26 m/s)   at p90 (4.18 m/s)
    +1 h    8 km                                 15 km
    +3 h   24 km                                 45 km
    +6 h   49 km                                 90 km
    +12 h  98 km                                180 km

So a fixed 5–25 km band is the right answer only at one wind speed and one lead.
Averaging twenty stations across that band dilutes the one station standing where
the incoming air actually is with nineteen that are not. This module instead
*selects* the station nearest the projected upwind point, per row, per horizon.

It also explains a finding the fixed-band ablation reported without a mechanism:
the 0–2 km band was the weakest contributor. At 2 km a neighbour is ~15 minutes of
advection away — it is sitting in the *same air mass* as home and cannot carry new
information. No amount of data fixes that; it is geometry.

**Wind source.** The advection velocity comes from the NWP forecast, never from the
station network. Observed PWS wind is unusable for this: the network's median wind
speed is 0.28 m/s against the forecast's 2.26 m/s, and 152 of 236 stations average
below 0.5 m/s. That gap is mostly measurement height — forecasts report the
meteorological standard 10 m, backyard anemometers sit at ~2 m amid fences and
trees, and the log wind profile over that roughness gives u(2)/u(10) ≈ 0.3. Feeding
a 2 m reading into a `v · h` calculation would under-reach by ~3×.

**Causality.** Every input is available at prediction time `t`: the forecast for
`t + h` (which is what a forecast *is*), and the upwind station's observation at
`t`. The feature asks "what does the air that is about to arrive look like right
now", which is a question you can actually answer at `t`.

Wind direction is meteorological — the direction wind blows *from* — so the upwind
bearing is `wind_dir_deg` itself, matching `src.features.bearing.direction_class`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.pws.distance import destination_point, haversine_km

# Below this wind speed the advection vector is meaningless: direction is noise and
# v·h collapses toward the home station itself. Rows below it get NaN features plus
# a flag, rather than a fabricated "upwind" station chosen by a random breeze.
MIN_ADVECTION_WIND_MS = 0.5

# If the nearest station to the projected point is further away than this, the
# projection landed in a gap in the network and the "upwind" reading is not
# describing the incoming parcel. Recorded per row so the model can discount it.
MAX_MATCH_DISTANCE_KM = 25.0

ADVECTION_FEATURE_COLS = [
    "adv_temp_c",
    "adv_temp_gradient",
    "adv_distance_km",
    "adv_match_km",
    "adv_valid",
]


def project_upwind_point(
    home_lat: float,
    home_lon: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    horizon_h: int,
) -> tuple[float, float, float]:
    """Where is the parcel that reaches home in `horizon_h` hours, right now?

    Returns (lat, lon, distance_km). Distance is `v · h` converted to km:
    m/s × h × 3600 s/h ÷ 1000 m/km = v · h · 3.6.
    """
    distance_km = wind_speed_ms * horizon_h * 3.6
    lat, lon = destination_point(home_lat, home_lon, distance_km, wind_dir_deg)
    return lat, lon, distance_km


def nearest_station(
    target_lat: float,
    target_lon: float,
    station_coords: dict[str, tuple[float, float]],
    exclude: Optional[set[str]] = None,
) -> tuple[Optional[str], float]:
    """Station closest to (target_lat, target_lon). Returns (station_id, km)."""
    exclude = exclude or set()
    best_id, best_km = None, float("inf")
    for sid, (lat, lon) in station_coords.items():
        if sid in exclude:
            continue
        km = haversine_km(target_lat, target_lon, lat, lon)
        if km < best_km:
            best_id, best_km = sid, km
    return best_id, best_km


def build_advection_features(
    df: pd.DataFrame,
    horizon_h: int,
    home_lat: float,
    home_lon: float,
    station_coords: dict[str, tuple[float, float]],
    obs_lookup: dict[tuple[str, pd.Timestamp], float],
    home_station_id: Optional[str] = None,
) -> pd.DataFrame:
    """Attach dynamic-advection columns to a `build_dataset` frame.

    `df` must carry `valid_time`, `f_wind_speed_ms`, `f_wind_dir_deg` (the forecast
    for the target hour — already present) and `lag_temp_c` (home's reading at `t`).
    `obs_lookup` maps (station_id, hour) -> temp_c for the network.

    Columns added:
      adv_temp_c        upwind station's temperature at t (the incoming parcel)
      adv_temp_gradient adv_temp_c - lag_temp_c (how much colder/warmer it is than here)
      adv_distance_km   v · h — how far the feature reached
      adv_match_km      distance from the projected point to the chosen station
      adv_valid         1.0 if wind was strong enough and a station was close enough
    """
    exclude = {home_station_id} if home_station_id else set()

    temps: list[float] = []
    gradients: list[float] = []
    distances: list[float] = []
    matches: list[float] = []
    valids: list[float] = []

    for row in df.itertuples(index=False):
        v = float(getattr(row, "f_wind_speed_ms"))
        theta = float(getattr(row, "f_wind_dir_deg"))
        lag_temp = float(getattr(row, "lag_temp_c"))
        # The parcel is observed NOW (t = valid_time - horizon), not at valid_time.
        obs_hour = pd.Timestamp(getattr(row, "valid_time")) - pd.Timedelta(hours=horizon_h)

        if not np.isfinite(v) or not np.isfinite(theta) or v < MIN_ADVECTION_WIND_MS:
            temps.append(np.nan); gradients.append(np.nan)
            distances.append(v * horizon_h * 3.6 if np.isfinite(v) else np.nan)
            matches.append(np.nan); valids.append(0.0)
            continue

        lat, lon, dist_km = project_upwind_point(home_lat, home_lon, v, theta, horizon_h)
        sid, match_km = nearest_station(lat, lon, station_coords, exclude=exclude)

        temp = obs_lookup.get((sid, obs_hour)) if sid else None
        ok = (
            temp is not None
            and np.isfinite(temp)
            and match_km <= MAX_MATCH_DISTANCE_KM
        )

        temps.append(float(temp) if ok else np.nan)
        gradients.append(float(temp) - lag_temp if ok and np.isfinite(lag_temp) else np.nan)
        distances.append(dist_km)
        matches.append(match_km)
        valids.append(1.0 if ok else 0.0)

    out = df.copy()
    out["adv_temp_c"] = temps
    out["adv_temp_gradient"] = gradients
    out["adv_distance_km"] = distances
    out["adv_match_km"] = matches
    out["adv_valid"] = valids
    return out


def impute_advection(df: pd.DataFrame, fill_from: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Fill NaN advection cells with the TRAIN mean, keeping `adv_valid` as the flag.

    Never fill-0: `adv_temp_c` sits near 15 °C and `adv_distance_km` near 25 km, so a
    zero is a physically absurd value that wrecks StandardScaler and Ridge. That exact
    mistake (fill-0 on pressure ~1015 hPa) produced MAE 20-30 in the 2026-06-19 sweep;
    see src/ml/invariants.check_physical_bounds. Pass `fill_from` (the training slice)
    when imputing test rows so the fill value never leaks across the split.
    """
    source = fill_from if fill_from is not None else df
    out = df.copy()
    for col in ("adv_temp_c", "adv_temp_gradient", "adv_distance_km", "adv_match_km"):
        mean = source[col].mean()
        if not np.isfinite(mean):
            mean = 0.0 if col in ("adv_temp_gradient",) else source[col].median()
        out[col] = out[col].fillna(mean)
    return out
