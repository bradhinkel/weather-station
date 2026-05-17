import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import BOOLEAN, INTEGER, TEXT, Column, Float, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base

load_dotenv()
logger = logging.getLogger(__name__)

Base = declarative_base()


def _build_dsn() -> str:
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


engine = create_async_engine(_build_dsn(), echo=False, future=True)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Station(Base):
    __tablename__ = "stations"

    station_id    = Column(TEXT, primary_key=True)
    name          = Column(TEXT)
    lat           = Column(Float)
    lon           = Column(Float)
    elevation_m   = Column(Float)
    timezone      = Column(TEXT)
    # Phase 7.1 network-source columns. is_network=False marks the own
    # station (Ecowitt) and any future co-located comparison stations;
    # network rows carry source/distance/bearing/quality info.
    is_network    = Column(BOOLEAN, nullable=False, server_default=text("FALSE"))
    source        = Column(TEXT, nullable=False, server_default=text("'ecowitt'"))
    distance_km   = Column(Float)
    bearing_deg   = Column(Float)
    quality_flags = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    last_seen     = Column(TIMESTAMP(timezone=True))
    created_at    = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )


class Observation(Base):
    __tablename__ = "observations"

    time          = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    station_id    = Column(TEXT, primary_key=True, nullable=False)
    temp_c        = Column(Float)
    humidity_pct  = Column(Float)
    pressure_hpa  = Column(Float)
    wind_speed_ms = Column(Float)
    wind_dir_deg  = Column(Float)
    wind_gust_ms  = Column(Float)
    rain_mm_1h          = Column(Float)
    rain_mm_daily_total = Column(Float)
    rain_rate_mm_hr     = Column(Float)
    solar_wm2           = Column(Float)
    uv_index            = Column(Float)
    feels_like_c        = Column(Float)
    # Provider tag — 'ecowitt' for own-station obs, 'wu' / 'pwsweather' for
    # network rows. Phase 7.1.
    source              = Column(TEXT, nullable=False, server_default=text("'ecowitt'"))


class ModelMetric(Base):
    """One row per (training run × model) — used to plot MAE/RMSE over time."""
    __tablename__ = "model_metrics"

    trained_at    = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    target        = Column(TEXT, primary_key=True, nullable=False)
    horizon       = Column(INTEGER, primary_key=True, nullable=False)
    model         = Column(TEXT, primary_key=True, nullable=False)
    mae           = Column(Float)
    rmse          = Column(Float)
    n_train       = Column(INTEGER)
    n_test        = Column(INTEGER)
    openmeteo_mae  = Column(Float)
    openmeteo_rmse = Column(Float)


class ExcludedWindow(Base):
    """Time windows to exclude from train/val/holdout splits.

    Phase 7.1 deliverable. Outages, sensor anomalies, and calibration periods
    get a row here so the 7.2 feature pipeline and 7.3 gate check don't
    silently include known-bad data. Heartbeat metrics stay raw — exclusion
    is applied downstream by the data-consuming code.
    """
    __tablename__ = "excluded_windows"

    id          = Column(INTEGER, primary_key=True, autoincrement=True)
    station_id  = Column(TEXT, nullable=False)
    start_time  = Column(TIMESTAMP(timezone=True), nullable=False)
    end_time    = Column(TIMESTAMP(timezone=True), nullable=False)
    reason      = Column(TEXT, nullable=False)
    source      = Column(TEXT, nullable=False, server_default=text("'manual'"))
    created_at  = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )


class HeartbeatRun(Base):
    """One row per data-sufficiency heartbeat run.

    Phase 7.1 deliverable. Heartbeat mode answers 'are we getting enough
    data, and is it good?' on a rolling window. The same metrics are reused
    by the 7.3 gate-mode check at holdout lock.
    """
    __tablename__ = "heartbeat_runs"

    run_time              = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    station_id            = Column(TEXT, primary_key=True, nullable=False)
    window_days           = Column(INTEGER, primary_key=True, nullable=False)

    obs_hours_covered     = Column(INTEGER)
    obs_hours_expected    = Column(INTEGER)
    obs_gap_pct           = Column(Float)
    nwp_hours_covered     = Column(INTEGER)
    nwp_hours_expected    = Column(INTEGER)
    nwp_gap_pct           = Column(Float)

    rain_positive_hours   = Column(INTEGER)
    frontal_passage_hours = Column(INTEGER)
    stable_period_hours   = Column(INTEGER)

    # Network coverage stays NULL until WU/PWSWeather ingest lands later in 7.1.
    network_coverage_pct  = Column(Float)

    sensor_drift_flags    = Column(JSONB)
    notes                 = Column(TEXT)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create tables (if absent), then promote to TimescaleDB hypertables.

    Uses a postgres advisory lock so multiple uvicorn workers don't race on
    CREATE TABLE during startup — without the lock, one worker briefly fails
    with `duplicate key in pg_type` and gets respawned by uvicorn.
    """
    import src.openmeteo  # noqa: F401 — registers Forecast on Base.metadata

    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(7142531123)"))
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created / verified.")

        # Idempotent ALTERs for columns added after initial schema creation.
        # We don't use Alembic, so this is the migration story for additive changes.
        await conn.execute(text(
            "ALTER TABLE observations ADD COLUMN IF NOT EXISTS feels_like_c FLOAT"
        ))
        await conn.execute(text(
            "ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS precip_prob_pct INTEGER"
        ))
        # Phase 7.1 — network-source columns on stations + source tag on obs.
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS is_network BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'ecowitt'"
        ))
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS distance_km FLOAT"
        ))
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS bearing_deg FLOAT"
        ))
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS quality_flags JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE stations ADD COLUMN IF NOT EXISTS last_seen TIMESTAMPTZ"
        ))
        await conn.execute(text(
            "ALTER TABLE observations ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'ecowitt'"
        ))
        logger.info("Schema migrations applied.")

        await conn.execute(
            text(
                "SELECT create_hypertable('observations', 'time', if_not_exists => TRUE)"
            )
        )
        await conn.execute(
            text(
                "SELECT create_hypertable('forecasts', 'valid_time', if_not_exists => TRUE)"
            )
        )
        logger.info("Hypertables initialised.")


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------

async def write_observation(session: AsyncSession, data: dict) -> None:
    """
    Insert one Observation row.

    `data` is the converted dict produced by api.validate_and_convert().
    The caller is responsible for committing (or using an async context manager).
    """
    # Resolve the observation timestamp.
    raw_time = data.get("dateutc")
    if raw_time:
        try:
            obs_time = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            logger.warning("Unparseable dateutc %r — using server time.", raw_time)
            obs_time = datetime.now(timezone.utc)
    else:
        obs_time = datetime.now(timezone.utc)

    from src.units import apparent_temperature

    temp_c = data.get("temp_c")
    humidity = data.get("humidity")
    wind_ms = data.get("wind_speed_ms")

    obs = Observation(
        time          = obs_time,
        station_id    = data.get("passkey", "unknown"),
        temp_c        = temp_c,
        humidity_pct  = humidity,
        pressure_hpa  = data.get("pressure_rel_hpa"),
        wind_speed_ms = wind_ms,
        wind_dir_deg  = data.get("wind_dir_deg"),
        wind_gust_ms  = data.get("wind_gust_ms"),
        rain_mm_1h          = data.get("rain_hourly_mm"),
        rain_mm_daily_total = data.get("rain_daily_mm"),
        rain_rate_mm_hr     = data.get("rain_rate_mm_hr"),
        solar_wm2           = data.get("solar_radiation"),
        uv_index            = data.get("uv_index"),
        feels_like_c        = apparent_temperature(temp_c, humidity, wind_ms),
    )
    session.add(obs)
    logger.debug("Queued observation for station=%s time=%s", obs.station_id, obs.time)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(init_db())
