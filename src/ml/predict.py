"""Online prediction: build a feature row from the latest observation and the
nearest forecast for the target hour, then run any persisted model bundles.

Used by the FastAPI /api/predict endpoint.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import engine
from src.icons import cond_label, pick_icon_slug
from src.ml import SUPPORTED_HORIZONS, SUPPORTED_MODELS, SUPPORTED_TARGETS

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))

# Cache holds (mtime_ns, bundle). When the file on disk has a newer mtime than
# the cached entry, we reload — so a retrain landing a fresh joblib gets picked
# up on the next predict without restarting the API.
_cache: dict[tuple[str, int, str], tuple[int, dict]] = {}


def load_bundle(target: str, horizon: int, model_name: str) -> Optional[dict]:
    key = (target, horizon, model_name)
    path = MODEL_DIR / f"{target}_{horizon}h_{model_name}.joblib"
    if not path.exists():
        _cache.pop(key, None)
        return None
    mtime_ns = path.stat().st_mtime_ns
    cached = _cache.get(key)
    if cached is None or cached[0] != mtime_ns:
        _cache[key] = (mtime_ns, joblib.load(path))
    return _cache[key][1]


def list_available() -> dict[tuple[str, int], list[str]]:
    """Return {(target, horizon): [model_name, ...]} for all on-disk bundles."""
    out: dict[tuple[str, int], list[str]] = {}
    if not MODEL_DIR.exists():
        return out
    for target in SUPPORTED_TARGETS:
        for horizon in SUPPORTED_HORIZONS:
            for model_name in SUPPORTED_MODELS:
                if (MODEL_DIR / f"{target}_{horizon}h_{model_name}.joblib").exists():
                    out.setdefault((target, horizon), []).append(model_name)
    return out


_LAG_SQL = text("""\
SELECT
    avg(temp_c)        AS temp_c,
    avg(humidity_pct)  AS humidity_pct,
    avg(pressure_hpa)  AS pressure_hpa,
    avg(wind_speed_ms) AS wind_speed_ms,
    max(rain_mm_1h)          AS rain_1h_reported,
    max(rain_mm_daily_total) AS daily_total_end
FROM observations
WHERE station_id = :sid
  AND time >= :start
  AND time <  :end_excl
""")


_FORECAST_SQL = text("""\
SELECT temp_c, humidity_pct, pressure_hpa, wind_speed_ms,
       wind_dir_deg, precip_mm, precip_prob_pct,
       weather_code, forecast_time
FROM forecasts
WHERE station_id = :sid
  AND valid_time = :target_hour
ORDER BY forecast_time DESC
LIMIT 1
""")


_LATEST_OBS_SQL = text("""\
SELECT time, temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg,
       solar_wm2, rain_mm_1h, feels_like_c
FROM observations
WHERE station_id = :sid
ORDER BY time DESC
LIMIT 1
""")


# Most recent forecast row whose valid_time covers "now" — used to pick a
# weather_code for the current-conditions icon. We round "now" down to the hour
# to match how forecasts are bucketed.
_CURRENT_FORECAST_SQL = text("""\
SELECT weather_code, precip_prob_pct
FROM forecasts
WHERE station_id = :sid
  AND valid_time = :now_hour
ORDER BY forecast_time DESC
LIMIT 1
""")


async def latest_observation(session: AsyncSession, station_id: str) -> Optional[dict]:
    row = (await session.execute(_LATEST_OBS_SQL, {"sid": station_id})).one_or_none()
    if row is None:
        return None
    return {
        "time": row.time.isoformat(),
        "temp_c": row.temp_c,
        "humidity_pct": row.humidity_pct,
        "pressure_hpa": row.pressure_hpa,
        "wind_speed_ms": row.wind_speed_ms,
        "wind_dir_deg": row.wind_dir_deg,
        "solar_wm2": row.solar_wm2,
        "rain_mm_1h": row.rain_mm_1h,
        "feels_like_c": row.feels_like_c,
    }


async def current_conditions(
    session: AsyncSession, station_id: str, lat: float, lon: float
) -> Optional[dict]:
    """Latest observation enriched with weather icon + plain-language label.

    The icon is chosen from the most recent forecast for the current hour, then
    locally overridden if the station's own rain gauge or pyranometer disagrees.
    """
    obs = await latest_observation(session, station_id)
    if obs is None:
        return None

    now = datetime.now(timezone.utc)
    now_hour = now.replace(minute=0, second=0, microsecond=0)
    fc_row = (await session.execute(
        _CURRENT_FORECAST_SQL,
        {"sid": station_id, "now_hour": now_hour},
    )).one_or_none()

    weather_code = fc_row.weather_code if fc_row is not None else None
    slug = pick_icon_slug(
        weather_code=weather_code,
        lat=lat, lon=lon, t=now,
        rain_mm_1h=obs.get("rain_mm_1h"),
        solar_wm2=obs.get("solar_wm2"),
    )
    obs["weather_code"] = weather_code
    obs["icon_slug"] = slug
    obs["cond_label"] = cond_label(slug)
    return obs


def _build_feature_dict(
    target_hour: datetime,
    forecast_row,
    lag_row,
    prev_lag_daily_total: Optional[float],
) -> dict[str, float]:
    """Mirror the feature engineering in src.ml.dataset for one row."""
    # Match dataset.py: prefer the station-reported trailing-hour accumulation,
    # fall back to the daily-total delta only when it's missing.
    lag_rain = 0.0
    if lag_row.rain_1h_reported is not None:
        lag_rain = float(lag_row.rain_1h_reported)
    elif lag_row.daily_total_end is not None and prev_lag_daily_total is not None:
        d = float(lag_row.daily_total_end) - float(prev_lag_daily_total)
        lag_rain = d if d >= 0 else float(lag_row.daily_total_end)

    wd = forecast_row.wind_dir_deg
    hod = target_hour.hour
    doy = target_hour.timetuple().tm_yday

    feat = {
        "f_temp_c":         forecast_row.temp_c,
        "f_humidity_pct":   forecast_row.humidity_pct,
        "f_pressure_hpa":   forecast_row.pressure_hpa,
        "f_wind_speed_ms":  forecast_row.wind_speed_ms,
        "wind_dir_sin":     math.sin(math.radians(wd)) if wd is not None else 0.0,
        "wind_dir_cos":     math.cos(math.radians(wd)) if wd is not None else 0.0,
        "f_precip_mm":      forecast_row.precip_mm,
        "f_weather_code":   forecast_row.weather_code,
        "lag_temp_c":       lag_row.temp_c,
        "lag_humidity_pct": lag_row.humidity_pct,
        "lag_pressure_hpa": lag_row.pressure_hpa,
        "lag_wind_speed_ms": lag_row.wind_speed_ms,
        "lag_rain_mm_1h":   lag_rain,
        "hod_sin":          math.sin(2 * math.pi * hod / 24),
        "hod_cos":          math.cos(2 * math.pi * hod / 24),
        "doy_sin":          math.sin(2 * math.pi * doy / 365),
        "doy_cos":          math.cos(2 * math.pi * doy / 365),
    }
    return feat


async def predict_one(
    target: str,
    horizon: int,
    station_id: str,
    lat: float,
    lon: float,
) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")

    now = datetime.now(timezone.utc)
    # Use the most recent *complete* hour as the lag observation.
    lag_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    target_hour = lag_hour + timedelta(hours=horizon)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        latest = await latest_observation(session, station_id)
        lag_row = (await session.execute(
            _LAG_SQL,
            {"sid": station_id, "start": lag_hour, "end_excl": lag_hour + timedelta(hours=1)},
        )).one()
        prev_row = (await session.execute(
            _LAG_SQL,
            {"sid": station_id, "start": lag_hour - timedelta(hours=1), "end_excl": lag_hour},
        )).one()
        forecast_row = (await session.execute(
            _FORECAST_SQL,
            {"sid": station_id, "target_hour": target_hour},
        )).one_or_none()

    response: dict[str, Any] = {
        "target": target,
        "horizon_hours": horizon,
        "valid_time": target_hour.isoformat(),
        "asof": now.isoformat(),
        "latest_observation": latest,
        "open_meteo": None,
        "open_meteo_forecast_time": None,
        "weather_code": None,
        "icon_slug": None,
        "cond_label": None,
        "precip_prob_pct": None,
        "linear": None,
        "xgboost": None,
        "metrics": {},
        "warnings": [],
    }

    if forecast_row is None:
        response["warnings"].append(
            f"No forecast available for {target_hour.isoformat()}"
        )
        return response

    response["open_meteo_forecast_time"] = forecast_row.forecast_time.isoformat()
    response["weather_code"] = forecast_row.weather_code
    # Forecast icon is from Open-Meteo's weather_code; no sensor overrides apply
    # to a future hour we haven't observed yet.
    slug = pick_icon_slug(
        weather_code=forecast_row.weather_code,
        lat=lat, lon=lon, t=target_hour,
    )
    response["icon_slug"] = slug
    response["cond_label"] = cond_label(slug)
    response["precip_prob_pct"] = forecast_row.precip_prob_pct

    if target == "temp_c":
        response["open_meteo"] = float(forecast_row.temp_c) if forecast_row.temp_c is not None else None
    else:
        response["open_meteo"] = float(forecast_row.precip_mm) if forecast_row.precip_mm is not None else None

    feat = _build_feature_dict(
        target_hour, forecast_row, lag_row,
        prev_lag_daily_total=prev_row.daily_total_end,
    )

    for model_name in SUPPORTED_MODELS:
        bundle = load_bundle(target, horizon, model_name)
        if bundle is None:
            response["warnings"].append(f"No persisted model for {target}_{horizon}h_{model_name}")
            continue
        feat_cols = bundle["feature_cols"]
        try:
            row = np.array([[feat.get(c) for c in feat_cols]], dtype=float)
            if np.isnan(row).any():
                response["warnings"].append(
                    f"{model_name}: missing feature values, skipped"
                )
                continue
            pred = bundle["model"].predict(row)[0]
            response[model_name] = float(pred)
            response["metrics"][model_name] = bundle.get("metrics")
        except Exception as exc:  # pragma: no cover
            response["warnings"].append(f"{model_name}: {exc}")

    return response
