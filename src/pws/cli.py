"""CLI for the PWS network — Phase 7.1.

Subcommands:
  list      — print registered network stations
  discover  — call source.discover_stations and upsert into the registry
  ingest    — call source.fetch_observations and persist into the observations
              hypertable; also bumps stations.last_seen on success.

`discover` and `ingest` require the provider's API key in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Observation, Station, engine
from src.pws.base import NetworkObservation, PWSSource
from src.pws.registry import (
    list_network_stations,
    mark_station_seen,
    upsert_network_station,
)
from src.pws.wu import WUKeyMissing, WUSource

logger = logging.getLogger(__name__)


def _build_source(name: str) -> PWSSource:
    """Factory. Add new providers here as they land."""
    if name == "wu":
        return WUSource()
    raise ValueError(f"unknown source: {name!r} (known: wu)")


async def _resolve_home() -> tuple[float, float]:
    """Locate the home (own) station; its lat/lon is the discovery centre.

    If there's more than one is_network=False station, the lowest station_id
    by sort order wins — adequate while we have a single Ecowitt.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (await session.execute(
            select(Station)
            .where(Station.is_network.is_(False))
            .order_by(Station.station_id)
            .limit(1)
        )).scalar_one_or_none()
    if row is None:
        raise RuntimeError(
            "No own station registered (no row with is_network=False). "
            "Register the Ecowitt station before running discover."
        )
    if row.lat is None or row.lon is None:
        raise RuntimeError(f"station {row.station_id} has no lat/lon")
    return float(row.lat), float(row.lon)


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

async def _cli_list(args) -> None:
    rows = await list_network_stations(
        source=args.source,
        max_distance_km=args.max_distance,
    )
    if not rows:
        print("(no network stations registered)")
        return
    header = f"{'station_id':<20}  {'source':<10}  {'km':>6}  {'bearing':>7}  last_seen  name"
    print(header)
    print("-" * len(header))
    for s in rows:
        km = f"{s.distance_km:.1f}" if s.distance_km is not None else "?"
        brg = f"{s.bearing_deg:.0f}" if s.bearing_deg is not None else "?"
        seen = s.last_seen.strftime("%Y-%m-%d %H:%M") if s.last_seen else "(never)"
        print(f"{s.station_id:<20}  {s.source:<10}  {km:>6}  {brg:>7}  {seen}  {s.name or ''}")


# --------------------------------------------------------------------------
# discover
# --------------------------------------------------------------------------

async def _cli_discover(args) -> None:
    src = _build_source(args.source)
    home_lat, home_lon = await _resolve_home()
    logger.info("discover: source=%s home=(%.4f, %.4f) radius=%.0fkm",
                src.name, home_lat, home_lon, args.radius)

    stations = await src.discover_stations(home_lat, home_lon, args.radius)
    logger.info("discover: %d stations returned within %.0fkm", len(stations), args.radius)

    if args.dry_run:
        for info in stations:
            print(f"{info.station_id:<20}  {info.distance_km:>6.1f}km  "
                  f"{info.bearing_deg:>5.0f}°  {info.name or ''}")
        return

    for info in stations:
        await upsert_network_station(info)
    logger.info("discover: upserted %d stations into registry", len(stations))


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

async def _persist_observations(rows: list[NetworkObservation]) -> int:
    """Bulk-upsert into observations table. Returns count of rows attempted.

    Uses ON CONFLICT DO NOTHING on (time, station_id) — the existing PK.
    Network rows that collide with prior fetches are silently skipped, which
    matches the WU hourly cadence (same hour-bucket = same row).
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


async def _cli_ingest(args) -> None:
    src = _build_source(args.source)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    if args.station_id:
        targets = [args.station_id]
    else:
        rows = await list_network_stations(source=src.name)
        targets = [s.station_id for s in rows]
        if not targets:
            raise RuntimeError(
                f"no registered {src.name!r} stations to ingest from — run `discover` first."
            )

    total = 0
    for sid in targets:
        try:
            obs = await src.fetch_observations(sid, start, end)
        except Exception:
            logger.exception("ingest: station=%s failed", sid)
            continue
        n = await _persist_observations(obs)
        await mark_station_seen(sid)
        logger.info("ingest: station=%s rows=%d", sid, n)
        total += n
    logger.info("ingest: %d total rows across %d stations", total, len(targets))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PWS network — registry + ingestion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list registered network stations")
    p_list.add_argument("--source", default=None, help="filter by provider")
    p_list.add_argument("--max-distance", type=float, default=None)
    p_list.set_defaults(func=_cli_list)

    p_disc = sub.add_parser("discover", help="enumerate provider stations + upsert registry")
    p_disc.add_argument("--source", default="wu", choices=["wu"])
    p_disc.add_argument("--radius", type=float, default=100.0, help="km from own station")
    p_disc.add_argument("--dry-run", action="store_true", help="print, don't write")
    p_disc.set_defaults(func=_cli_discover)

    p_ing = sub.add_parser("ingest", help="fetch + persist recent observations")
    p_ing.add_argument("--source", default="wu", choices=["wu"])
    p_ing.add_argument("--hours", type=int, default=24)
    p_ing.add_argument("--station-id", default=None,
                       help="single station; default = all registered for source")
    p_ing.set_defaults(func=_cli_ingest)

    args = parser.parse_args()
    try:
        asyncio.run(args.func(args))
    except WUKeyMissing as e:
        print(f"error: {e}")
        raise SystemExit(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
