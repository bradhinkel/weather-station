import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Station, engine, init_db, write_observation
from sqlalchemy import text as _sa_text
from src.analysis import get_baseline_errors, get_forecast_bias
from src.models import BaselineResponse, HealthResponse, IngestionResponse
from src.scheduler import configure_scheduler, fetch_forecasts_job
from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS
from src.ml.predict import current_conditions, list_available, predict_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _log_prefetch_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Startup forecast prefetch failed.", exc_info=exc)
    else:
        logger.info("Startup forecast prefetch complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    sched = configure_scheduler()
    sched.start()
    logger.info("Scheduler started.")
    # Prefetch forecasts in the BACKGROUND so the app binds its port
    # immediately. A synchronous prefetch over ~322 stations blocked startup
    # for minutes and 502'd the public site on every restart. Existing
    # forecasts already serve predictions during the warm-up; the scheduler's
    # hourly job also covers it. Keep a reference so the task isn't GC'd.
    prefetch_task = asyncio.create_task(fetch_forecasts_job())
    prefetch_task.add_done_callback(_log_prefetch_done)
    app.state.prefetch_task = prefetch_task
    logger.info("Startup forecast prefetch scheduled (background).")
    yield
    if not prefetch_task.done():
        prefetch_task.cancel()
    sched.shutdown()
    logger.info("Scheduler shut down.")


app = FastAPI(title="Weather Station API", lifespan=lifespan)

# Static UI: served at /static/* and the comparison dashboard at /
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "weather-station API. UI not bundled."}
    return FileResponse(index_path)


# --- Validation bounds (in original units) ---
TEMP_MIN_F = -58.0
TEMP_MAX_F = 140.0
WIND_DIR_MIN = 0
WIND_DIR_MAX = 360
PRESSURE_MIN_HPA = 870.0
PRESSURE_MAX_HPA = 1085.0


# --- Unit conversions ---
def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9

def mph_to_ms(mph: float) -> float:
    return mph * 0.44704

def inhg_to_hpa(inhg: float) -> float:
    return inhg * 33.8639

def inches_to_mm(inches: float) -> float:
    return inches * 25.4


def _float(value: Optional[str], name: str) -> Optional[float]:
    """Parse a string to float, returning None and logging on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning("Could not parse field '%s' with value %r", name, value)
        return None


def validate_and_convert(data: dict) -> Optional[dict]:
    """
    Validate raw Ecowitt fields and convert to metric.
    Returns a converted dict on success, or None if any required field is invalid.
    """
    errors = []

    tempf = _float(data.get("tempf"), "tempf")
    if tempf is None:
        errors.append("tempf missing or unparseable")
    elif not (TEMP_MIN_F <= tempf <= TEMP_MAX_F):
        errors.append(f"tempf {tempf} out of range [{TEMP_MIN_F}, {TEMP_MAX_F}]")

    humidity = _float(data.get("humidity"), "humidity")

    winddir = _float(data.get("winddir"), "winddir")
    if winddir is not None and not (WIND_DIR_MIN <= winddir <= WIND_DIR_MAX):
        errors.append(f"winddir {winddir} out of range [{WIND_DIR_MIN}, {WIND_DIR_MAX}]")

    windspeedmph = _float(data.get("windspeedmph"), "windspeedmph")
    windgustmph = _float(data.get("windgustmph"), "windgustmph")

    baromrelin = _float(data.get("baromrelin"), "baromrelin")
    baromabsin = _float(data.get("baromabsin"), "baromabsin")

    for field_name, raw in (("baromrelin", baromrelin), ("baromabsin", baromabsin)):
        if raw is not None:
            hpa = inhg_to_hpa(raw)
            if not (PRESSURE_MIN_HPA <= hpa <= PRESSURE_MAX_HPA):
                errors.append(
                    f"{field_name} converts to {hpa:.1f} hPa, "
                    f"out of range [{PRESSURE_MIN_HPA}, {PRESSURE_MAX_HPA}]"
                )

    rainratein = _float(data.get("rainratein"), "rainratein")
    hourlyrainin = _float(data.get("hourlyrainin"), "hourlyrainin")
    dailyrainin = _float(data.get("dailyrainin"), "dailyrainin")
    solarradiation = _float(data.get("solarradiation"), "solarradiation")
    uv = _float(data.get("uv"), "uv")

    if errors:
        logger.warning(
            "Invalid reading from PASSKEY=%s dateutc=%r — skipping. Errors: %s",
            data.get("PASSKEY", "unknown"),
            data.get("dateutc"),
            "; ".join(errors),
        )
        return None

    return {
        "passkey": data.get("PASSKEY"),
        "dateutc": data.get("dateutc"),
        "temp_c": round(f_to_c(tempf), 2) if tempf is not None else None,
        "humidity": humidity,
        "wind_dir_deg": winddir,
        "wind_speed_ms": round(mph_to_ms(windspeedmph), 3) if windspeedmph is not None else None,
        "wind_gust_ms": round(mph_to_ms(windgustmph), 3) if windgustmph is not None else None,
        "pressure_rel_hpa": round(inhg_to_hpa(baromrelin), 2) if baromrelin is not None else None,
        "pressure_abs_hpa": round(inhg_to_hpa(baromabsin), 2) if baromabsin is not None else None,
        "rain_rate_mm_hr": round(inches_to_mm(rainratein), 2) if rainratein is not None else None,
        "rain_hourly_mm": round(inches_to_mm(hourlyrainin), 2) if hourlyrainin is not None else None,
        "rain_daily_mm": round(inches_to_mm(dailyrainin), 2) if dailyrainin is not None else None,
        "solar_radiation": solarradiation,
        "uv_index": uv,
    }


@app.post("/api/ecowitt")
async def receive_ecowitt(
    PASSKEY: Optional[str] = Form(None),
    dateutc: Optional[str] = Form(None),
    tempf: Optional[str] = Form(None),
    humidity: Optional[str] = Form(None),
    baromrelin: Optional[str] = Form(None),
    baromabsin: Optional[str] = Form(None),
    winddir: Optional[str] = Form(None),
    windspeedmph: Optional[str] = Form(None),
    windgustmph: Optional[str] = Form(None),
    rainratein: Optional[str] = Form(None),
    hourlyrainin: Optional[str] = Form(None),
    dailyrainin: Optional[str] = Form(None),
    solarradiation: Optional[str] = Form(None),
    uv: Optional[str] = Form(None),
):
    raw = {
        "PASSKEY": PASSKEY,
        "dateutc": dateutc,
        "tempf": tempf,
        "humidity": humidity,
        "baromrelin": baromrelin,
        "baromabsin": baromabsin,
        "winddir": winddir,
        "windspeedmph": windspeedmph,
        "windgustmph": windgustmph,
        "rainratein": rainratein,
        "hourlyrainin": hourlyrainin,
        "dailyrainin": dailyrainin,
        "solarradiation": solarradiation,
        "uv": uv,
    }
    logger.info("Received Ecowitt payload from PASSKEY=%s dateutc=%r", PASSKEY, dateutc)

    reading = validate_and_convert(raw)
    if reading is None:
        return IngestionResponse(status="skipped")

    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            await write_observation(session, reading)
    logger.info("Stored observation for station=%s dateutc=%r", PASSKEY, dateutc)
    return IngestionResponse(status="ok")


@app.get("/api/stations/{station_id}/baseline", response_model=BaselineResponse)
async def station_baseline(station_id: str, days: int = 30):
    baseline = await get_baseline_errors(station_id, days=days)
    bias = await get_forecast_bias(station_id, days=days)
    return BaselineResponse(
        station_id=station_id,
        days=days,
        baseline_errors=baseline,
        forecast_bias=bias,
    )


async def _resolve_station(
    session: AsyncSession, station_id: Optional[str]
) -> Station:
    """Look up the requested station — or the first registered if omitted."""
    if station_id is None:
        row = (await session.execute(select(Station).limit(1))).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="No stations registered")
        return row
    row = (await session.execute(
        select(Station).where(Station.station_id == station_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station_id}")
    return row


@app.get("/api/current")
async def current_observation(station_id: Optional[str] = None):
    """Latest observation for a station, enriched with weather icon + feels-like.

    If station_id is omitted, picks the first registered station.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        station = await _resolve_station(session, station_id)
        current = await current_conditions(
            session, station.station_id, station.lat, station.lon
        )
    if current is None:
        raise HTTPException(
            status_code=404, detail=f"No observations for {station.station_id}"
        )
    return {"station_id": station.station_id, **current}


@app.get("/api/predict")
async def predict(
    target: str = Query("temp_c"),
    horizon: int = Query(1, ge=1),
    station_id: Optional[str] = None,
):
    """Compare three forecasts (Open-Meteo, linear, XGBoost) for a (target, horizon)."""
    if target not in SUPPORTED_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"target must be one of {list(SUPPORTED_TARGETS)}",
        )
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be one of {list(SUPPORTED_HORIZONS)}",
        )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        station = await _resolve_station(session, station_id)

    return await predict_one(target, horizon, station.station_id, station.lat, station.lon)


@app.get("/api/metrics_history")
async def metrics_history(
    target: str = Query("temp_c"),
    horizon: int = Query(1, ge=1),
    model: Optional[str] = Query(None, description="Filter by model name. Defaults to all."),
    limit: int = Query(200, ge=1, le=2000),
):
    """Time-series of training-run metrics for plotting MAE-over-time."""
    sql = _sa_text("""\
        SELECT trained_at, target, horizon, model,
               mae, rmse, n_train, n_test,
               openmeteo_mae, openmeteo_rmse
        FROM model_metrics
        WHERE target = :target AND horizon = :horizon
          AND (CAST(:model AS TEXT) IS NULL OR model = :model)
        ORDER BY trained_at ASC
        LIMIT :limit
    """)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        rows = (await session.execute(
            sql,
            {"target": target, "horizon": horizon, "model": model, "limit": limit},
        )).all()
    return {
        "target": target,
        "horizon": horizon,
        "model": model,
        "points": [
            {
                "trained_at": r.trained_at.isoformat(),
                "model": r.model,
                "mae": r.mae,
                "rmse": r.rmse,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "openmeteo_mae": r.openmeteo_mae,
                "openmeteo_rmse": r.openmeteo_rmse,
            }
            for r in rows
        ],
    }


@app.get("/api/heartbeat/latest")
async def heartbeat_latest(station_id: Optional[str] = None):
    """Most recent heartbeat row for a station (or the first registered)."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        station = await _resolve_station(session, station_id)
        row = (await session.execute(_sa_text("""
            SELECT run_time, station_id, window_days,
                   obs_hours_covered, obs_hours_expected, obs_gap_pct,
                   nwp_hours_covered, nwp_hours_expected, nwp_gap_pct,
                   rain_positive_hours, frontal_passage_hours, stable_period_hours,
                   network_coverage_pct, sensor_drift_flags, notes
            FROM heartbeat_runs
            WHERE station_id = :sid
            ORDER BY run_time DESC
            LIMIT 1
        """), {"sid": station.station_id})).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No heartbeat runs yet for {station.station_id}",
        )
    return _heartbeat_row_to_dict(row)


@app.get("/api/heartbeat")
async def heartbeat_history(
    station_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
):
    """Heartbeat runs for a station over the past `days` days, oldest first."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        station = await _resolve_station(session, station_id)
        rows = (await session.execute(_sa_text("""
            SELECT run_time, station_id, window_days,
                   obs_hours_covered, obs_hours_expected, obs_gap_pct,
                   nwp_hours_covered, nwp_hours_expected, nwp_gap_pct,
                   rain_positive_hours, frontal_passage_hours, stable_period_hours,
                   network_coverage_pct, sensor_drift_flags, notes
            FROM heartbeat_runs
            WHERE station_id = :sid
              AND run_time >= now() - make_interval(days => :days)
            ORDER BY run_time ASC
        """), {"sid": station.station_id, "days": days})).all()
    return {
        "station_id": station.station_id,
        "days": days,
        "runs": [_heartbeat_row_to_dict(r) for r in rows],
    }


def _heartbeat_row_to_dict(row) -> dict:
    return {
        "run_time": row.run_time.isoformat(),
        "station_id": row.station_id,
        "window_days": row.window_days,
        "obs_hours_covered": row.obs_hours_covered,
        "obs_hours_expected": row.obs_hours_expected,
        "obs_gap_pct": row.obs_gap_pct,
        "nwp_hours_covered": row.nwp_hours_covered,
        "nwp_hours_expected": row.nwp_hours_expected,
        "nwp_gap_pct": row.nwp_gap_pct,
        "rain_positive_hours": row.rain_positive_hours,
        "frontal_passage_hours": row.frontal_passage_hours,
        "stable_period_hours": row.stable_period_hours,
        "network_coverage_pct": row.network_coverage_pct,
        "sensor_drift_flags": row.sensor_drift_flags,
        "notes": row.notes,
    }


@app.get("/api/models")
async def list_models():
    """Inventory of (target, horizon, model) bundles currently loadable."""
    available = list_available()
    return {
        "available": [
            {"target": t, "horizon": h, "models": models}
            for (t, h), models in sorted(available.items())
        ],
        "supported_targets": list(SUPPORTED_TARGETS),
        "supported_horizons": list(SUPPORTED_HORIZONS),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
