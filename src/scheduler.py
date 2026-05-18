"""Background job scheduler using APScheduler."""

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Station, Observation, engine
from src.heartbeat import run_heartbeat
from src.openmeteo import fetch_forecast
from src.pws.ingest import ingest_recent
from src.pws.registry import evaluate_quality
from src.pws.wu import WUKeyMissing, WUSource

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


# ---------------------------------------------------------------------------
# Job 1 — fetch forecasts (every 60 min + once at startup)
# ---------------------------------------------------------------------------

async def fetch_forecasts_job() -> None:
    """Fetch Open-Meteo forecasts for every registered station."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        result = await session.execute(select(Station))
        stations = result.scalars().all()

    if not stations:
        logger.info("fetch_forecasts_job: no stations registered — skipping.")
        return

    for station in stations:
        try:
            async with AsyncSession(engine, expire_on_commit=False) as session:
                async with session.begin():
                    n = await fetch_forecast(
                        station.lat, station.lon, station.station_id, session,
                    )
            logger.info(
                "fetch_forecasts_job: station=%s — %d rows inserted.",
                station.station_id, n,
            )
        except Exception:
            logger.exception(
                "fetch_forecasts_job: station=%s — failed.", station.station_id,
            )


# ---------------------------------------------------------------------------
# Job 2 — data-quality check (every 5 min)
# ---------------------------------------------------------------------------

async def data_quality_job() -> None:
    """Warn if any station has not reported observations recently."""
    now = datetime.now(timezone.utc)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        result = await session.execute(select(Station))
        stations = result.scalars().all()

    if not stations:
        return

    async with AsyncSession(engine, expire_on_commit=False) as session:
        for station in stations:
            row = await session.execute(
                select(func.max(Observation.time)).where(
                    Observation.station_id == station.station_id
                )
            )
            latest = row.scalar()
            if latest is None:
                logger.warning(
                    "Station %s has never reported an observation.",
                    station.station_id,
                )
                continue

            age_minutes = (now - latest).total_seconds() / 60
            if age_minutes > 10:
                logger.warning(
                    "Station %s has not reported in %.0f minutes.",
                    station.station_id, age_minutes,
                )


# ---------------------------------------------------------------------------
# Job 3 — cleanup old data (daily at 03:00)
# ---------------------------------------------------------------------------

async def cleanup_job() -> None:
    """Delete observations older than 365 days and forecasts older than 30 days."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            obs_result = await session.execute(
                text("DELETE FROM observations WHERE time < now() - interval '365 days'")
            )
            fc_result = await session.execute(
                text("DELETE FROM forecasts WHERE forecast_time < now() - interval '30 days'")
            )
    logger.info(
        "cleanup_job: deleted %d old observations, %d old forecasts.",
        obs_result.rowcount, fc_result.rowcount,
    )


# ---------------------------------------------------------------------------
# Job 4 — WU network ingest + quality rescore (daily at 00:15 UTC)
# ---------------------------------------------------------------------------

# WU /hourly/7day returns up to 7 days, so a daily run with hours=30 backfills
# any gap from a missed run + the new day. ON CONFLICT DO NOTHING dedups.
_WU_INGEST_HOURS = 30
_QUALITY_WINDOW_DAYS = 7


async def wu_ingest_job() -> None:
    """Daily WU pull, then re-score quality_flags so blacklists stay current."""
    try:
        src = WUSource()
    except WUKeyMissing:
        logger.warning("wu_ingest_job: WU_API_KEY not set — skipping.")
        return

    try:
        summary = await ingest_recent(src, hours=_WU_INGEST_HOURS, only_active=True)
        logger.info(
            "wu_ingest_job: stations=%d rows=%d failures=%d",
            summary["stations"], summary["rows"], summary["failures"],
        )
    except Exception:
        logger.exception("wu_ingest_job: ingest failed.")
        return

    try:
        await evaluate_quality(window_days=_QUALITY_WINDOW_DAYS)
    except Exception:
        logger.exception("wu_ingest_job: quality rescore failed.")


# ---------------------------------------------------------------------------
# Job 5 — data-sufficiency heartbeat (daily at 00:30 UTC)
# ---------------------------------------------------------------------------

async def heartbeat_job() -> None:
    """Phase 7.1 data-sufficiency heartbeat over a rolling 30-day window."""
    try:
        await run_heartbeat()
    except Exception:
        logger.exception("heartbeat_job: failed.")


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------

def configure_scheduler() -> AsyncIOScheduler:
    """Register all jobs and return the scheduler (not yet started)."""
    scheduler.add_job(
        fetch_forecasts_job,
        trigger=IntervalTrigger(minutes=60),
        id="fetch_forecasts",
        replace_existing=True,
        misfire_grace_time=300,
        coalesce=True,
    )
    scheduler.add_job(
        data_quality_job,
        trigger=IntervalTrigger(minutes=5),
        id="data_quality",
        replace_existing=True,
        misfire_grace_time=120,
        coalesce=True,
    )
    scheduler.add_job(
        cleanup_job,
        trigger=CronTrigger(hour=3, minute=0),
        id="cleanup",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        wu_ingest_job,
        trigger=CronTrigger(hour=0, minute=15),
        id="wu_ingest",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        heartbeat_job,
        trigger=CronTrigger(hour=0, minute=30),
        id="heartbeat",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    return scheduler
