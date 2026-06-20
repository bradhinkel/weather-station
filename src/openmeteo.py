"""Fetch and store hourly forecasts from the Open-Meteo API."""

import asyncio
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import INTEGER, TEXT, Column, Float, PrimaryKeyConstraint, text
from sqlalchemy.dialects.postgresql import TIMESTAMP, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Base

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]

HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "relative_humidity_2m",
    "cloud_cover",
    "weather_code",
    # Added 2026-06-19 for the irrigation/ET use case + general forecasting
    # experiments. shortwave_radiation (W/m^2) and the model's own FAO-56
    # reference ET (mm) let us compare forecast solar/ET against the station;
    # cloud_cover + wind_gusts were already fetched but discarded — now stored.
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
]


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        PrimaryKeyConstraint("valid_time", "station_id", "forecast_time"),
    )

    forecast_time   = Column(TIMESTAMP(timezone=True), nullable=False)
    valid_time      = Column(TIMESTAMP(timezone=True), nullable=False)
    station_id      = Column(TEXT, nullable=False)
    model           = Column(TEXT, nullable=False, server_default=text("'open-meteo'"))
    temp_c          = Column(Float)
    precip_mm       = Column(Float)
    precip_prob_pct = Column(INTEGER)
    wind_speed_ms   = Column(Float)
    wind_dir_deg    = Column(Float)
    pressure_hpa    = Column(Float)
    humidity_pct    = Column(Float)
    weather_code    = Column(INTEGER)
    wind_gust_ms    = Column(Float)
    cloud_cover_pct = Column(Float)
    solar_wm2       = Column(Float)   # shortwave_radiation
    et0_mm          = Column(Float)   # et0_fao_evapotranspiration (FAO-56)


# ---------------------------------------------------------------------------
# Fetch + store
# ---------------------------------------------------------------------------

async def _get_with_retries(url: str, params: dict) -> dict:
    """GET *url* with exponential-backoff retries on HTTP errors."""
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError:
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Open-Meteo returned %s (attempt %d/%d), retrying in %ds …",
                        resp.status_code, attempt + 1, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise


async def fetch_forecast(
    lat: float,
    lon: float,
    station_id: str,
    session: AsyncSession,
) -> int:
    """Fetch a 2-day hourly forecast from Open-Meteo and persist rows.

    All rows are bulk-inserted in a single transaction managed by the caller.
    Returns the number of forecast rows written.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": 2,
        "wind_speed_unit": "ms",
        # Use UTC so the ISO strings we get back are already true UTC; the
        # naive datetime we parse goes into a TIMESTAMPTZ column without an
        # implicit local-time → UTC reinterpretation.
        "timezone": "UTC",
    }

    payload = await _get_with_retries(OPEN_METEO_URL, params)

    hourly = payload["hourly"]
    times = hourly["time"]
    fetch_time = datetime.now(timezone.utc)

    rows = [
        {
            "forecast_time":   fetch_time,
            "valid_time":      datetime.fromisoformat(times[i]),
            "station_id":      station_id,
            "temp_c":          hourly["temperature_2m"][i],
            "precip_mm":       hourly["precipitation"][i],
            "precip_prob_pct": hourly["precipitation_probability"][i],
            "wind_speed_ms":   hourly["wind_speed_10m"][i],
            "wind_dir_deg":    hourly["wind_direction_10m"][i],
            "pressure_hpa":    hourly["pressure_msl"][i],
            "humidity_pct":    hourly["relative_humidity_2m"][i],
            "weather_code":    hourly["weather_code"][i],
            "wind_gust_ms":    hourly["wind_gusts_10m"][i],
            "cloud_cover_pct": hourly["cloud_cover"][i],
            "solar_wm2":       hourly["shortwave_radiation"][i],
            "et0_mm":          hourly["et0_fao_evapotranspiration"][i],
        }
        for i in range(len(times))
    ]

    stmt = pg_insert(Forecast).values(rows).on_conflict_do_nothing()
    await session.execute(stmt)
    logger.info(
        "Upserted %d forecast rows for station=%s (fetch_time=%s)",
        len(rows), station_id, fetch_time,
    )
    return len(rows)
