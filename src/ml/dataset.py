"""Build a training dataset by joining hourly observations with the nearest
prior Open-Meteo forecast for the target hour.

Schema produced (columns in returned DataFrame):
  - valid_time, station_id
  - feature columns (see FEATURE_COLS)
  - y                     -- the target value at valid_time
  - openmeteo_baseline    -- the raw forecast value at valid_time (for comparison)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS

load_dotenv()


FEATURE_COLS = [
    "f_temp_c", "f_humidity_pct", "f_pressure_hpa",
    "f_wind_speed_ms", "wind_dir_sin", "wind_dir_cos",
    "f_precip_mm", "f_weather_code",
    "lag_temp_c", "lag_humidity_pct", "lag_pressure_hpa",
    "lag_wind_speed_ms", "lag_rain_mm_1h",
    "hod_sin", "hod_cos", "doy_sin", "doy_cos",
]


def _sync_dsn() -> str:
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )


_PAIRED_SQL = text("""\
WITH obs_hourly AS (
    SELECT
        station_id,
        date_trunc('hour', time) AS hour,
        avg(temp_c)        AS temp_c,
        avg(humidity_pct)  AS humidity_pct,
        avg(pressure_hpa)  AS pressure_hpa,
        avg(wind_speed_ms) AS wind_speed_ms,
        avg(wind_dir_deg)  AS wind_dir_deg,
        max(rain_mm_daily_total) AS daily_total_end
    FROM observations
    WHERE (CAST(:sid AS TEXT) IS NULL OR station_id = :sid)
    GROUP BY 1, 2
),
obs_with_rain AS (
    SELECT
        station_id, hour,
        temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg,
        CASE
            WHEN LAG(daily_total_end) OVER w IS NULL
                THEN NULL
            WHEN daily_total_end >= LAG(daily_total_end) OVER w
                THEN daily_total_end - LAG(daily_total_end) OVER w
            ELSE daily_total_end
        END AS rain_mm_1h
    FROM obs_hourly
    WINDOW w AS (PARTITION BY station_id ORDER BY hour)
),
nearest_forecast AS (
    SELECT DISTINCT ON (station_id, valid_time)
        station_id, valid_time, forecast_time,
        temp_c        AS f_temp_c,
        humidity_pct  AS f_humidity_pct,
        pressure_hpa  AS f_pressure_hpa,
        wind_speed_ms AS f_wind_speed_ms,
        wind_dir_deg  AS f_wind_dir_deg,
        precip_mm     AS f_precip_mm,
        weather_code  AS f_weather_code
    FROM forecasts
    WHERE forecast_time < valid_time
    ORDER BY station_id, valid_time, forecast_time DESC
)
SELECT
    f.valid_time,
    f.station_id,
    f.f_temp_c, f.f_humidity_pct, f.f_pressure_hpa,
    f.f_wind_speed_ms, f.f_wind_dir_deg,
    f.f_precip_mm, f.f_weather_code,
    o_lag.temp_c        AS lag_temp_c,
    o_lag.humidity_pct  AS lag_humidity_pct,
    o_lag.pressure_hpa  AS lag_pressure_hpa,
    o_lag.wind_speed_ms AS lag_wind_speed_ms,
    o_lag.rain_mm_1h    AS lag_rain_mm_1h,
    o_target.temp_c     AS y_temp_c,
    o_target.rain_mm_1h AS y_rain_mm_1h
FROM nearest_forecast f
JOIN obs_with_rain o_lag
  ON  o_lag.station_id = f.station_id
  AND o_lag.hour       = f.valid_time - make_interval(hours => :horizon)
JOIN obs_with_rain o_target
  ON  o_target.station_id = f.station_id
  AND o_target.hour       = f.valid_time
""")


def build_dataset(
    target: str,
    horizon_hours: int,
    station_id: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (df, feature_cols) for the given (target, horizon).

    df rows are aligned (lag_obs at hour t, forecast for hour t+horizon, obs at hour t+horizon).
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"target must be one of {SUPPORTED_TARGETS}")
    if horizon_hours not in SUPPORTED_HORIZONS:
        raise ValueError(f"horizon_hours must be one of {SUPPORTED_HORIZONS}")

    engine = create_engine(_sync_dsn())
    with engine.connect() as conn:
        df = pd.read_sql(
            _PAIRED_SQL,
            conn,
            params={"sid": station_id, "horizon": horizon_hours},
        )
    engine.dispose()

    if df.empty:
        return df, FEATURE_COLS

    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.sort_values("valid_time").reset_index(drop=True)

    # Cyclical encodings
    hod = df["valid_time"].dt.hour.to_numpy()
    doy = df["valid_time"].dt.dayofyear.to_numpy()
    df["hod_sin"] = np.sin(2 * math.pi * hod / 24)
    df["hod_cos"] = np.cos(2 * math.pi * hod / 24)
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)
    wd = df["f_wind_dir_deg"].to_numpy(dtype=float)
    df["wind_dir_sin"] = np.sin(np.deg2rad(wd))
    df["wind_dir_cos"] = np.cos(np.deg2rad(wd))

    # Rain is reported intermittently by the Ecowitt — when missing,
    # treat as zero rather than dropping the row (otherwise we lose
    # the bulk of training data on dry days).
    df["lag_rain_mm_1h"] = df["lag_rain_mm_1h"].fillna(0.0)
    if target == "rain_mm_1h":
        df["y_rain_mm_1h"] = df["y_rain_mm_1h"].fillna(0.0)

    df["y"] = df[f"y_{target}"]
    df["openmeteo_baseline"] = df["f_temp_c"] if target == "temp_c" else df["f_precip_mm"]

    needed = FEATURE_COLS + ["y", "openmeteo_baseline", "valid_time", "station_id"]
    df = df.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
    return df[needed], FEATURE_COLS
