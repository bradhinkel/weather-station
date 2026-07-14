"""Station-registry helpers — Phase 7.1.

The registry is the extended `stations` table:
  - own station: is_network=False, source='ecowitt', no distance/bearing
  - network station: is_network=True, source='wu'|'pwsweather', distance/bearing set

These helpers do the CRUD; discovery (calling a PWSSource) lives in src.pws.cli.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Station, engine
from src.pws.base import StationInfo
from src.quality_limits import (
    PRESSURE_HPA_MAX,
    PRESSURE_HPA_MIN,
    RAIN_MM_1H_MAX,
    RAIN_RATE_MM_HR_MAX,
    TEMP_C_MAX,
    TEMP_C_MIN,
)

# --- value-quality retire policy -------------------------------------------
# Value quality is scored over a short RECENT window (not the 7-day coverage
# window) so a one-off glitch ages out before it can retire a station, while a
# persistently-broken sensor scores bad every day and accrues strikes.
VALUE_WINDOW_DAYS: int = 2
# A window with at least this many physically-impossible readings is "bad".
BAD_WINDOW_MIN_READINGS: int = 3
# Retire after this many consecutive bad windows ("continues to provide poor
# data") — with daily evaluation that's ~3 days of sustained garbage.
STRIKE_LIMIT: int = 3
# ...or retire immediately when a window is egregiously bad (a fully stuck
# sensor emitting garbage nearly every hour), no need to wait out the strikes.
IMMEDIATE_RETIRE_READINGS: int = 24


def assess_value_quality(
    prior_flags: dict,
    bad_readings: int,
    now_iso: str,
) -> tuple[dict, bool]:
    """Pure strike/retire decision — no I/O, so it's unit-testable.

    Takes the station's existing flags and this window's count of impossible
    readings; returns (durable_updates, newly_retired). ``durable_updates`` are
    the keys to merge into quality_flags: the strike counter, the recent
    bad-reading count, and — once tripped — the sticky ``retired`` marker. Retire
    is one-way: a station already retired stays retired regardless of this
    window (its garbage may have aged out of the DB, but we don't resurrect it).
    """
    already_retired = bool(prior_flags.get("retired", False))
    strikes = int(prior_flags.get("value_strikes", 0))

    bad_window = bad_readings >= BAD_WINDOW_MIN_READINGS
    strikes = strikes + 1 if bad_window else 0

    updates: dict = {"value_strikes": strikes, "bad_readings_recent": bad_readings}
    newly_retired = False

    retire_now = bad_readings >= IMMEDIATE_RETIRE_READINGS or strikes >= STRIKE_LIMIT
    if already_retired or retire_now:
        updates["retired"] = True
        # Force blacklisted too so every coverage-based consumer excludes it.
        updates["blacklisted"] = True
        if already_retired:
            # Preserve the original retire provenance.
            updates["retired_at"] = prior_flags.get("retired_at", now_iso)
            updates["retired_reason"] = prior_flags.get("retired_reason", "value-quality")
        else:
            newly_retired = True
            updates["retired_at"] = now_iso
            updates["retired_reason"] = (
                f"value-quality: {bad_readings} impossible readings in "
                f"{VALUE_WINDOW_DAYS}d (strikes={strikes})"
            )
        # Carry a replaced_by marker forward if one was already recorded.
        if "replaced_by" in prior_flags:
            updates["replaced_by"] = prior_flags["replaced_by"]

    return updates, newly_retired


_VALUE_QUALITY_SQL = text("""\
SELECT count(*) FILTER (
    WHERE rain_mm_1h > :rain_max OR rain_mm_1h < 0
       OR rain_rate_mm_hr > :rate_max
       OR temp_c < :temp_min OR temp_c > :temp_max
       OR pressure_hpa < :p_min OR pressure_hpa > :p_max
)::int AS bad_readings
FROM observations
WHERE source <> 'ecowitt'
  AND station_id = :sid
  AND time >= now() - make_interval(days => :vdays)
""")


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


_QUALITY_SQL = text("""\
SELECT
    count(*)::int                                     AS rows_n,
    bool_or(pressure_hpa  IS NOT NULL)                AS has_pressure,
    bool_or(solar_wm2     IS NOT NULL)                AS has_solar,
    bool_or(rain_mm_1h    IS NOT NULL)                AS has_rain_data,
    bool_or(temp_c        IS NOT NULL)                AS has_temp
FROM observations
WHERE source <> 'ecowitt'
  AND station_id = :sid
  AND time >= now() - make_interval(days => :days)
""")


async def evaluate_quality(
    window_days: int = 7,
    min_coverage_pct: float = 50.0,
) -> dict:
    """Score every network station based on the last `window_days` of obs.

    Writes a JSONB blob to ``stations.quality_flags`` keyed by:
      rows_<N>d, coverage_<N>d_pct, has_pressure, has_solar, has_rain_data,
      has_temp, blacklisted, evaluated_at, plus the durable value-quality keys
      value_strikes / bad_readings_recent / retired[/_at/_reason].

    A station is `blacklisted=True` when its coverage_pct is below the
    threshold OR it has been retired for bad values. Downstream feature/ingest
    code filters with ``quality_flags->>'blacklisted' = 'false'``; training
    filters on ``retired`` (see src.ml.dataset).

    Coverage flags are recomputed each run; the value-quality keys are carried
    forward and updated so a retire (or accrued strikes) survives the rescan.

    Returns a one-shot summary; pretty-print upstream in the CLI.
    """
    max_rows = window_days * 24
    rows_key = f"rows_{window_days}d"
    cov_key = f"coverage_{window_days}d_pct"

    summary = {
        "total": 0,
        "active": 0,            # rows_n > 0
        "blacklisted": 0,
        "retired": 0,
        "newly_retired": 0,
        "with_pressure": 0,
        "with_solar": 0,
        "with_rain_data": 0,
    }

    async with AsyncSession(engine, expire_on_commit=False) as session:
        stations = (await session.execute(
            select(Station).where(Station.is_network.is_(True))
        )).scalars().all()

        now = datetime.now(timezone.utc).isoformat()
        for s in stations:
            row = (await session.execute(
                _QUALITY_SQL, {"sid": s.station_id, "days": window_days}
            )).one()
            rows_n = row.rows_n or 0
            coverage = round(rows_n * 100.0 / max_rows, 1) if max_rows else 0.0
            blacklisted = coverage < min_coverage_pct

            # Value-quality: count impossible readings over the recent window,
            # then apply the strike/retire policy against the existing flags.
            vq = (await session.execute(
                _VALUE_QUALITY_SQL,
                {
                    "sid": s.station_id, "vdays": VALUE_WINDOW_DAYS,
                    "rain_max": RAIN_MM_1H_MAX, "rate_max": RAIN_RATE_MM_HR_MAX,
                    "temp_min": TEMP_C_MIN, "temp_max": TEMP_C_MAX,
                    "p_min": PRESSURE_HPA_MIN, "p_max": PRESSURE_HPA_MAX,
                },
            )).one()
            prior = dict(s.quality_flags or {})
            vq_updates, newly_retired = assess_value_quality(
                prior, vq.bad_readings or 0, now
            )

            flags = {
                rows_key:        rows_n,
                cov_key:         coverage,
                "has_pressure":  bool(row.has_pressure or False),
                "has_solar":     bool(row.has_solar or False),
                "has_rain_data": bool(row.has_rain_data or False),
                "has_temp":      bool(row.has_temp or False),
                "blacklisted":   blacklisted,
                "evaluated_at":  now,
            }
            # Value-quality updates win — notably `retired` forces blacklisted=True.
            flags.update(vq_updates)
            await session.execute(
                update(Station).where(Station.station_id == s.station_id).values(quality_flags=flags)
            )

            summary["total"] += 1
            if rows_n > 0:
                summary["active"] += 1
            if flags["blacklisted"]:
                summary["blacklisted"] += 1
            if flags.get("retired"):
                summary["retired"] += 1
            if newly_retired:
                summary["newly_retired"] += 1
            if flags["has_pressure"]:
                summary["with_pressure"] += 1
            if flags["has_solar"]:
                summary["with_solar"] += 1
            if flags["has_rain_data"]:
                summary["with_rain_data"] += 1

        await session.commit()

    return summary
