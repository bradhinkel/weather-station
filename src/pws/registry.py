"""Station-registry helpers — Phase 7.1.

The registry is the extended `stations` table:
  - own station: is_network=False, source='ecowitt', no distance/bearing
  - network station: is_network=True, source='wu'|'pwsweather', distance/bearing set

These helpers do the CRUD; discovery (calling a PWSSource) lives in src.pws.cli.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Station, engine
from src.pws.base import StationInfo


async def upsert_network_station(info: StationInfo) -> None:
    """Insert or update a single network station. Idempotent on station_id.

    Re-running discover_stations should never duplicate rows. We update
    location, distance, bearing, sensor_flags on each call so registry
    refreshes itself if a station moves or upgrades sensors.
    """
    stmt = pg_insert(Station).values(
        station_id=info.station_id,
        name=info.name,
        lat=info.lat,
        lon=info.lon,
        elevation_m=info.elevation_m,
        is_network=True,
        source=info.source,
        distance_km=info.distance_km,
        bearing_deg=info.bearing_deg,
        quality_flags=info.sensor_flags or {},
    ).on_conflict_do_update(
        index_elements=["station_id"],
        set_={
            "name": info.name,
            "lat": info.lat,
            "lon": info.lon,
            "elevation_m": info.elevation_m,
            "distance_km": info.distance_km,
            "bearing_deg": info.bearing_deg,
            "quality_flags": info.sensor_flags or {},
        },
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            await session.execute(stmt)


async def list_network_stations(
    source: Optional[str] = None,
    max_distance_km: Optional[float] = None,
) -> list[Station]:
    """Return registered network stations, optionally filtered. Ordered by distance."""
    stmt = select(Station).where(Station.is_network.is_(True))
    if source is not None:
        stmt = stmt.where(Station.source == source)
    if max_distance_km is not None:
        stmt = stmt.where(Station.distance_km <= max_distance_km)
    stmt = stmt.order_by(Station.distance_km.asc())
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return list((await session.execute(stmt)).scalars().all())


async def mark_station_seen(station_id: str, ts: Optional[datetime] = None) -> None:
    """Bump `last_seen` after a successful observation fetch."""
    ts = ts or datetime.utcnow()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            await session.execute(
                update(Station).where(Station.station_id == station_id).values(last_seen=ts)
            )


async def set_quality_flag(station_id: str, key: str, value) -> None:
    """Merge a single key into the station's quality_flags JSONB.

    Quality keys (subject to refinement during 7.1 calibration):
      - uptime_pct    : rolling-window uptime
      - drift_temp_c  : observed bias vs nearby cluster
      - blacklisted   : true => exclude from features (with optional reason)
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            station = (await session.execute(
                select(Station).where(Station.station_id == station_id)
            )).scalar_one_or_none()
            if station is None:
                raise ValueError(f"unknown station: {station_id}")
            flags = dict(station.quality_flags or {})
            flags[key] = value
            station.quality_flags = flags
