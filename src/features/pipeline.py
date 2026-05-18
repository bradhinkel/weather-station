"""Feature pipeline orchestrator — Phase 7.2.

Top-level entry point: :func:`build_features` returns one row per target hour
in [start, end), with the chosen wind reference and weighted-mean aggregates
over the upwind (and optionally downwind) station cohort at each lag in
``config.lag_hours``.

DB loaders live here (rather than a separate ``data.py``) because the SQL is
tightly coupled to the feature schema and short enough to keep co-located.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env at import time so the sync DSN below can read DB_* without the
# caller needing to source .env. Matches src/database.py and src/ml/dataset.py.
load_dotenv()

from src.features.aggregation import kernel_weights, weighted_mean
from src.features.bearing import direction_class
from src.features.config import FeatureConfig
from src.features.lags import slice_at_lag
from src.features.wind_reference import resolve_wind_reference

# Fields aggregated across the upwind cohort. rain_mm_1h uses sum semantics
# at the hourly bucket level but is averaged across stations like the rest —
# that's the right metric for "how much rain is the upwind direction seeing
# right now", though the units are slightly noisy.
_AGGREGATE_FIELDS: tuple[str, ...] = (
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "rain_mm_1h",
)


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------

def _sync_dsn() -> str:
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )


_STATIONS_SQL = text("""
SELECT
    station_id,
    is_network,
    source,
    distance_km,
    bearing_deg,
    lat,
    lon,
    (quality_flags->>'blacklisted') AS blacklisted
FROM stations
""")


# Hourly aggregation in SQL — fewer rows transferred, circular wind average
# computed via atan2(avg(sin), avg(cos)) so it doesn't break across 0°/360°.
_OBSERVATIONS_HOURLY_SQL = text("""
SELECT
    station_id,
    date_trunc('hour', time) AS time_hour,
    avg(temp_c)        AS temp_c,
    avg(humidity_pct)  AS humidity_pct,
    avg(pressure_hpa)  AS pressure_hpa,
    avg(wind_speed_ms) AS wind_speed_ms,
    degrees(atan2(
        avg(sin(radians(wind_dir_deg))),
        avg(cos(radians(wind_dir_deg)))
    )) AS wind_dir_deg_raw,
    sum(rain_mm_1h)    AS rain_mm_1h
FROM observations
WHERE time >= :start AND time < :end
GROUP BY station_id, time_hour
""")


# Most-recent forecast per (valid_time) for the home station.
_FORECASTS_SQL = text("""
SELECT DISTINCT ON (valid_time)
    valid_time,
    wind_dir_deg
FROM forecasts
WHERE station_id = :sid
  AND valid_time >= :start
  AND valid_time <  :end
ORDER BY valid_time, forecast_time DESC
""")


def _load_stations(engine) -> pd.DataFrame:
    df = pd.read_sql(_STATIONS_SQL, engine)
    return df


def _load_observations_hourly(engine, start: datetime, end: datetime) -> pd.DataFrame:
    df = pd.read_sql(
        _OBSERVATIONS_HOURLY_SQL, engine, params={"start": start, "end": end}
    )
    df["time_hour"] = pd.to_datetime(df["time_hour"], utc=True)
    df["wind_dir_deg"] = (df["wind_dir_deg_raw"] + 360.0) % 360.0
    return df.drop(columns=["wind_dir_deg_raw"])


def _load_forecasts(engine, start: datetime, end: datetime, station_id: str) -> pd.DataFrame:
    df = pd.read_sql(
        _FORECASTS_SQL,
        engine,
        params={"sid": station_id, "start": start, "end": end},
    )
    if df.empty:
        return df
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _resolve_home_station(stations: pd.DataFrame) -> str:
    own = stations[stations["is_network"] == False]  # noqa: E712 — sql bool col
    if own.empty:
        raise RuntimeError("no own station registered (is_network=False)")
    return sorted(own["station_id"].tolist())[0]


def _hourly_index(start: datetime, end: datetime) -> pd.DatetimeIndex:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
    return pd.date_range(s.ceil("h"), e.floor("h"), freq="1h", inclusive="left")


def _quality_network_stations(stations: pd.DataFrame) -> pd.DataFrame:
    return stations[
        (stations["is_network"] == True)              # noqa: E712
        & (stations["blacklisted"] == "false")
    ].copy()


def build_features(
    start: datetime,
    end: datetime,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Build network features for every target hour in [start, end).

    Returns a DataFrame indexed by ``time`` (hourly UTC) with columns:
      - ``wind_ref_deg``     wind direction reference used
      - ``n_upwind``         number of stations contributing to the cohort
      - ``upwind_<field>_lag<H>h`` weighted mean of that field at T - H

    Lag 0 (current hour) is always included. NaN cells mean "not enough data."
    """
    config = config or FeatureConfig()
    max_lag = max(config.lag_hours) if config.lag_hours else 0
    load_start = start - timedelta(hours=max_lag + 1)
    load_end = end + timedelta(hours=1)

    engine = create_engine(_sync_dsn())
    try:
        stations = _load_stations(engine)
        home_id = _resolve_home_station(stations)
        obs_hourly = _load_observations_hourly(engine, load_start, load_end)
        forecasts = (
            _load_forecasts(engine, load_start, load_end, home_id)
            if config.wind_reference == "nwp"
            else pd.DataFrame(columns=["valid_time", "wind_dir_deg"])
        )
    finally:
        engine.dispose()

    # Split obs frames and merge geometry onto the network slice.
    quality = _quality_network_stations(stations)
    quality_ids = set(quality["station_id"].tolist())
    home_obs = obs_hourly[obs_hourly["station_id"] == home_id].copy()
    net_obs = obs_hourly[obs_hourly["station_id"].isin(quality_ids)].merge(
        quality[["station_id", "distance_km", "bearing_deg"]],
        on="station_id",
        how="left",
    )

    band_lo, band_hi = config.distance_band_km
    keep_dirs = {"upwind"} | ({"downwind"} if config.include_downwind else set())
    target_times = _hourly_index(start, end)

    rows: list[dict] = []
    for t in target_times:
        wind_ref = resolve_wind_reference(
            obs_hourly=net_obs,
            forecasts=forecasts,
            own_station_obs=home_obs,
            target_time=t,
            config=config,
        )
        row: dict = {"time": t, "wind_ref_deg": wind_ref, "n_upwind": 0}
        if wind_ref is None:
            rows.append(_fill_nan_features(row, config))
            continue

        # Candidate stations: distance band + valid bearing.
        cand = quality[
            (quality["distance_km"] >= band_lo)
            & (quality["distance_km"] < band_hi)
            & (quality["bearing_deg"].notna())
        ].copy()
        cand["direction"] = cand["bearing_deg"].apply(
            lambda b: direction_class(b, wind_ref, config.angular_tolerance_deg)
        )
        cand = cand[cand["direction"].isin(keep_dirs)]
        cand = cand.sort_values("distance_km").head(config.n_stations)
        cand_ids = cand["station_id"].tolist()
        row["n_upwind"] = len(cand_ids)

        if not cand_ids:
            rows.append(_fill_nan_features(row, config))
            continue

        for lag in (0, *config.lag_hours):
            sub = slice_at_lag(net_obs, t, lag, cand_ids)
            if sub.empty:
                for f in _AGGREGATE_FIELDS:
                    row[f"upwind_{f}_lag{lag}h"] = None
                continue
            w = kernel_weights(
                sub["distance_km"].tolist(),
                kernel=config.aggregation_kernel,
                gaussian_sigma_km=config.gaussian_sigma_km,
            )
            for f in _AGGREGATE_FIELDS:
                row[f"upwind_{f}_lag{lag}h"] = weighted_mean(sub[f].values, w)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("time")
    return df


def _fill_nan_features(row: dict, config: FeatureConfig) -> dict:
    """Fill all upwind_<field>_lag<H>h slots with None for skipped hours."""
    for lag in (0, *config.lag_hours):
        for f in _AGGREGATE_FIELDS:
            row[f"upwind_{f}_lag{lag}h"] = None
    return row
