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
from src.analysis import get_baseline_errors, get_forecast_bias
from src.models import BaselineResponse, HealthResponse, IngestionResponse
from src.scheduler import configure_scheduler, fetch_forecasts_job
from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS
from src.ml.predict import latest_observation, list_available, predict_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    sched = configure_scheduler()
    sched.start()
    logger.info("Scheduler started.")
    await fetch_forecasts_job()
    yield
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


@app.get("/api/current")
async def current_observation(station_id: Optional[str] = None):
    """Return the most recent raw observation for a station.

    If station_id is omitted, picks the first registered station.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        if station_id is None:
            row = (await session.execute(select(Station).limit(1))).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail="No stations registered")
            station_id = row.station_id
        latest = await latest_observation(session, station_id)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"No observations for {station_id}")
    return {"station_id": station_id, **latest}


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
    if station_id is None:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            row = (await session.execute(select(Station).limit(1))).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail="No stations registered")
            station_id = row.station_id

    return await predict_one(target, horizon, station_id)


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
