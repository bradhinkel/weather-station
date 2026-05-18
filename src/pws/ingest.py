"""Network ingest entry point — Phase 7.1.

Single function used by both the CLI (`python -m src.pws.cli ingest`) and the
APScheduler job in `src/scheduler.py`. Keeping the loop in one place means the
scheduler and ad-hoc runs can't drift apart on filter logic.

The "active" filter requires an explicit ``quality_flags->>'blacklisted' = 'false'``
stamp — unevaluated stations are skipped. Run ``evaluate-quality`` after a fresh
discover so new stations become eligible.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Observation, Station, engine
from src.pws.base import NetworkObservation, PWSSource

logger = logging.getLogger(__name__)


async def _persist_observations(rows: list[NetworkObservation]) -> int:
    """Bulk-upsert into the observations hypertable.

    Uses ON CONFLICT DO NOTHING on (time, station_id) so re-fetching the same
    hour from WU's /hourly/7day endpoint is a no-op.
    """
    if not rows:
        return 0
    payload = [
        {
            "time": r.time,
            "station_id": r.station_id,
            "temp_c": r.temp_c,
            "humidity_pct": r.humidity_pct,
            "pressure_hpa": r.pressure_hpa,
            "wind_speed_ms": r.wind_speed_ms,
            "wind_dir_deg": r.wind_dir_deg,
            "wind_gust_ms": r.wind_gust_ms,
            "rain_mm_1h": r.rain_mm_1h,
            "rain_rate_mm_hr": r.rain_rate_mm_hr,
            "solar_wm2": r.solar_wm2,
            "uv_index": r.uv_index,
            "source": r.source,
        }
        for r in rows
    ]
    stmt = pg_insert(Observation).values(payload).on_conflict_do_nothing(
        index_elements=["time", "station_id"],
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            await session.execute(stmt)
    return len(payload)


async def _select_targets(source_name: str, only_active: bool) -> list[str]:
    """Return station_ids for ``source_name`` to ingest from."""
    stmt = select(Station.station_id).where(
        Station.is_network.is_(True),
        Station.source == source_name,
    )
    if only_active:
        # JSONB blacklist check — text form because quality_flags is JSONB and
        # the JSON value is a real boolean. ``->>'blacklisted'`` returns 'true'
        # or 'false' as text. Unevaluated stations (NULL) are excluded.
        stmt = stmt.where(Station.quality_flags["blacklisted"].astext == "false")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return list((await session.execute(stmt)).scalars().all())


async def ingest_recent(
    src: PWSSource,
    hours: int = 24,
    only_active: bool = True,
    station_ids: Optional[list[str]] = None,
) -> dict:
    """Fetch the last ``hours`` from each target station and persist.

    Returns a summary dict {stations, rows, failures} so callers (CLI + job)
    can log a single line instead of replaying per-station detail.
    """
    from src.pws.registry import mark_station_seen  # avoid circular import

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    if station_ids is None:
        station_ids = await _select_targets(src.name, only_active=only_active)
    if not station_ids:
        logger.info("ingest_recent: no %r stations to ingest from.", src.name)
        return {"stations": 0, "rows": 0, "failures": 0}

    total_rows = 0
    failures = 0
    for sid in station_ids:
        try:
            obs = await src.fetch_observations(sid, start, end)
        except Exception:
            logger.exception("ingest_recent: station=%s fetch failed", sid)
            failures += 1
            continue
        try:
            n = await _persist_observations(obs)
        except Exception:
            logger.exception("ingest_recent: station=%s persist failed", sid)
            failures += 1
            continue
        await mark_station_seen(sid)
        total_rows += n

    logger.info(
        "ingest_recent: source=%s stations=%d rows=%d failures=%d window=%dh",
        src.name, len(station_ids), total_rows, failures, hours,
    )
    return {"stations": len(station_ids), "rows": total_rows, "failures": failures}
