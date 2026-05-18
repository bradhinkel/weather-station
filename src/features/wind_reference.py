"""Wind-direction reference resolver — Phase 7.2.

Returns the bearing the wind is coming FROM at a given target hour, used to
classify network stations as upwind / crosswind / downwind. Three modes via
:class:`src.features.config.FeatureConfig`:

  - ``own``           — home Ecowitt station. Subject to a measured ~81°
                        shelter offset (see memory: project_intent /
                        modeling preferences). Available as an ablation
                        comparator to validate the shelter hypothesis.
  - ``network_mean``  — circular mean of nearby quality stations (default).
                        Falls back to ``own`` if fewer than
                        ``wind_reference_min_stations`` contribute.
  - ``nwp``           — Open-Meteo forecast at the prediction hour.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.features.bearing import circular_mean
from src.features.config import FeatureConfig


def resolve_wind_reference(
    obs_hourly: pd.DataFrame,
    forecasts: pd.DataFrame,
    own_station_obs: pd.DataFrame,
    target_time: datetime,
    config: FeatureConfig,
) -> Optional[float]:
    """Return a bearing in [0, 360) — the wind 'from' direction — or None.

    ``obs_hourly`` must contain (station_id, time_hour, wind_dir_deg,
    distance_km) for the network stations under consideration.
    ``forecasts`` must contain (valid_time, wind_dir_deg) when mode == nwp.
    ``own_station_obs`` is a (time_hour, wind_dir_deg) frame for the home
    station — used as the fallback path.
    """
    target_hour = _floor_to_hour_utc(target_time)

    if config.wind_reference == "own":
        return _own_wind(own_station_obs, target_hour)

    if config.wind_reference == "nwp":
        return _nwp_wind(forecasts, target_hour)

    # network_mean (default)
    near = obs_hourly[
        (obs_hourly["time_hour"] == target_hour)
        & (obs_hourly["distance_km"] <= config.wind_reference_radius_km)
        & (obs_hourly["wind_dir_deg"].notna())
    ]
    if len(near) < config.wind_reference_min_stations:
        # Insufficient quality data — fall back to own station.
        return _own_wind(own_station_obs, target_hour)
    return circular_mean(near["wind_dir_deg"].tolist())


def _floor_to_hour_utc(t: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.floor("h")


def _own_wind(own_obs: pd.DataFrame, target_hour: pd.Timestamp) -> Optional[float]:
    if own_obs.empty:
        return None
    row = own_obs[own_obs["time_hour"] == target_hour]
    if row.empty:
        return None
    val = row["wind_dir_deg"].iloc[0]
    if pd.isna(val):
        return None
    return float(val) % 360.0


def _nwp_wind(forecasts: pd.DataFrame, target_hour: pd.Timestamp) -> Optional[float]:
    if forecasts.empty:
        return None
    row = forecasts[forecasts["valid_time"] == target_hour]
    if row.empty:
        return None
    val = row["wind_dir_deg"].iloc[0]
    if pd.isna(val):
        return None
    return float(val) % 360.0
