"""Build a training dataset by joining hourly observations with the Open-Meteo
forecast that was available `horizon` hours ahead of the target hour.

A row answers: "standing at time t, holding an observation for t and the
forecast issued for t+horizon, what is the actual value at t+horizon?" Both the
lag observation AND the forecast are therefore horizon-lagged — see the lead-time
constraint in _PAIRED_SQL. The horizon is a real forecast lead time, not just the
staleness of the lag feature.

Schema produced (columns in returned DataFrame):
  - valid_time, station_id
  - feature columns (see FEATURE_COLS)
  - y                     -- the target value at valid_time
  - openmeteo_baseline    -- the horizon-lead forecast value (for comparison)
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
from src.quality_limits import RAIN_MM_1H_MAX

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


def resolve_own_station_id() -> Optional[str]:
    """The `is_network = false` station — the backyard this project exists to forecast.

    Training pooled across the whole network produces a region-average corrector, which
    measured 2.3% skill on the own station at +3h against 17.1% for the same model class
    trained on own-station rows alone. 1,089 of the right rows beat 230k of the wrong
    ones, because the pooled target is a different question.
    """
    engine = create_engine(_sync_dsn())
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT station_id FROM stations WHERE is_network = false LIMIT 1")
            ).first()
    finally:
        engine.dispose()
    return row[0] if row else None


_PAIRED_SQL = text("""\
WITH obs_hourly AS (
    SELECT
        o.station_id,
        date_trunc('hour', o.time) AS hour,
        avg(o.temp_c)        AS temp_c,
        avg(o.humidity_pct)  AS humidity_pct,
        avg(o.pressure_hpa)  AS pressure_hpa,
        avg(o.wind_speed_ms) AS wind_speed_ms,
        avg(o.wind_dir_deg)  AS wind_dir_deg,
        -- Reject physically-impossible rain at aggregation time: a single
        -- jammed sub-hourly reading (e.g. a gauge stuck at 896 mm/h) must not
        -- become the hour's max(). Valid sub-hourly rows in the same hour still
        -- aggregate normally; an hour that is ALL garbage collapses to NULL and
        -- is then treated as dry downstream (fillna 0). Persistent offenders are
        -- removed wholesale by the station filter below + the retire loop.
        max(o.rain_mm_1h) FILTER (
            WHERE o.rain_mm_1h >= 0 AND o.rain_mm_1h <= :rain_max
        ) AS rain_1h_reported,
        max(o.rain_mm_daily_total) AS daily_total_end
    FROM observations o
    LEFT JOIN stations s ON s.station_id = o.station_id
    WHERE (CAST(:sid AS TEXT) IS NULL OR o.station_id = :sid)
      -- Drop stations the quality loop has RETIRED for persistently bad values
      -- (see src.pws.registry). We deliberately do NOT filter on `blacklisted`:
      -- that flag is coverage-based (a liveness signal for live ingest/features)
      -- and is orthogonal to data quality — a low-coverage station's historical
      -- rows are still valid training pairs, and the worst value-offenders (e.g.
      -- a gauge stuck at 896 mm/h) report every hour and are never coverage-
      -- blacklisted anyway. Transient garbage is scrubbed row-by-row by the rain
      -- clip above; retire removes the persistent offenders wholesale. COALESCE
      -- keeps the own station and any not-yet-evaluated station.
      AND COALESCE(s.quality_flags->>'retired', 'false') <> 'true'
    GROUP BY 1, 2
),
obs_with_rain AS (
    SELECT
        station_id, hour,
        temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg,
        -- Prefer the station-reported trailing-hour accumulation (rain_mm_1h):
        -- both Ecowitt and the network (WU) sources populate it, whereas
        -- rain_mm_daily_total is Ecowitt-only and updates too sparsely to
        -- recover light rain via hourly deltas. Fall back to the daily-total
        -- delta only when the reported hourly value is missing.
        COALESCE(
            rain_1h_reported,
            CASE
                WHEN LAG(daily_total_end) OVER w IS NULL
                    THEN NULL
                WHEN daily_total_end >= LAG(daily_total_end) OVER w
                    THEN daily_total_end - LAG(daily_total_end) OVER w
                ELSE daily_total_end
            END
        ) AS rain_mm_1h
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
    -- Take the freshest forecast that was ALREADY ISSUED `horizon` hours before
    -- the target hour. This must mirror serving: predict.py asks for
    -- valid_time = now + horizon, so the newest forecast it can possibly see was
    -- issued at now = valid_time - horizon. Selecting purely on
    -- `forecast_time < valid_time` (as this did until 2026-07-15) hands training
    -- a ~1h-lead forecast at EVERY horizon, so the model learns to trust f_temp_c
    -- like a nowcast and then meets a 24h-lead forecast in production — train/serve
    -- skew that widens with horizon, and it also pinned the Open-Meteo baseline to
    -- a horizon-independent constant (the flat 1.68 °C in the old README table).
    WHERE forecast_time <= valid_time - make_interval(hours => :horizon)
    ORDER BY station_id, valid_time, forecast_time DESC
)
SELECT
    f.valid_time,
    f.station_id,
    f.forecast_time,
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
            params={"sid": station_id, "horizon": horizon_hours, "rain_max": RAIN_MM_1H_MAX},
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

    # forecast_time rides along (not a feature) so callers can verify the realised
    # forecast lead — see src.ml.invariants.check_forecast_lead.
    df["forecast_time"] = pd.to_datetime(df["forecast_time"], utc=True)
    needed = FEATURE_COLS + [
        "y", "openmeteo_baseline", "valid_time", "station_id", "forecast_time",
    ]
    df = df.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
    return df[needed], FEATURE_COLS
