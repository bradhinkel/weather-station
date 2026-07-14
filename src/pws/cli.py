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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Station, engine
from src.pws.base import PWSSource, StationInfo
from src.pws.distance import (
    bearing_deg,
    bearing_octant,
    destination_point,
    distance_band,
    haversine_km,
)
from src.pws.ingest import ingest_recent
from src.pws.registry import (
    evaluate_quality,
    list_network_stations,
    set_quality_flag,
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

# Grid for the --wide sweep. WU's /v3/location/near returns its top-N
# nearest to a single origin, which in urban areas collapses to a sub-km
# cluster. Querying from a ring of synthetic origins gives access to
# stations that would otherwise be invisible.
_WIDE_RINGS_KM = (5.0, 20.0, 50.0, 90.0)
_WIDE_BEARINGS_DEG = tuple(range(0, 360, 45))  # 8 compass points


async def _discover_wide(
    src: PWSSource,
    home_lat: float,
    home_lon: float,
    radius_km: float,
) -> list[StationInfo]:
    """Grid sweep that calls discover_stations from multiple origins, then
    re-grounds each station's distance/bearing to TRUE home and filters to
    `radius_km` of home (not of the query origin).
    """
    origins: list[tuple[float, float]] = [(home_lat, home_lon)]
    for ring in _WIDE_RINGS_KM:
        for brg in _WIDE_BEARINGS_DEG:
            origins.append(destination_point(home_lat, home_lon, ring, brg))

    seen: dict[str, StationInfo] = {}
    for i, (lat, lon) in enumerate(origins, start=1):
        try:
            results = await src.discover_stations(lat, lon, radius_km * 2)
        except Exception:
            logger.exception("discover-wide: origin (%.4f, %.4f) failed", lat, lon)
            results = []
        new = 0
        for info in results:
            if info.station_id in seen:
                continue
            info.distance_km = haversine_km(home_lat, home_lon, info.lat, info.lon)
            info.bearing_deg = bearing_deg(home_lat, home_lon, info.lat, info.lon)
            seen[info.station_id] = info
            new += 1
        logger.info(
            "discover-wide: %2d/%d origin=(%.3f, %.3f) +%d new (total %d)",
            i, len(origins), lat, lon, new, len(seen),
        )
        await asyncio.sleep(0.1)  # gentle pacing; WU rate budget is fine here

    return [s for s in seen.values() if s.distance_km is not None and s.distance_km <= radius_km]


def _coverage_summary(stations: list[StationInfo]) -> None:
    """Print station counts per distance band and per bearing octant. Cheap
    diagnostic for whether grid coverage is sufficient for Q3/Q4 sweeps.
    """
    bands = [(0, 10), (10, 25), (25, 50), (50, 100)]
    octants = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    print("\ncoverage by distance band:")
    for lo, hi in bands:
        n = sum(1 for s in stations if s.distance_km is not None and lo <= s.distance_km < hi)
        print(f"  {lo:>3}–{hi:<3} km   {n:>4}")

    print("\ncoverage by bearing octant (centered on cardinal/intercardinal):")
    for i, name in enumerate(octants):
        center = i * 45.0
        lo = (center - 22.5) % 360.0
        hi = (center + 22.5) % 360.0
        if lo > hi:  # wraps through 360°
            n = sum(1 for s in stations if s.bearing_deg is not None and (s.bearing_deg >= lo or s.bearing_deg < hi))
        else:
            n = sum(1 for s in stations if s.bearing_deg is not None and lo <= s.bearing_deg < hi)
        print(f"  {name:<2}   {n:>4}")


async def _cli_discover(args) -> None:
    src = _build_source(args.source)
    home_lat, home_lon = await _resolve_home()
    mode = "wide" if args.wide else "near"
    logger.info(
        "discover: source=%s mode=%s home=(%.4f, %.4f) radius=%.0fkm",
        src.name, mode, home_lat, home_lon, args.radius,
    )

    if args.wide:
        stations = await _discover_wide(src, home_lat, home_lon, args.radius)
    else:
        stations = await src.discover_stations(home_lat, home_lon, args.radius)
    logger.info("discover: %d unique stations within %.0fkm of home", len(stations), args.radius)

    if args.dry_run:
        for info in sorted(stations, key=lambda s: s.distance_km or 0):
            print(f"{info.station_id:<20}  {info.distance_km:>6.1f}km  "
                  f"{info.bearing_deg:>5.0f}°  {info.name or ''}")
        _coverage_summary(stations)
        return

    for info in stations:
        await upsert_network_station(info)
    logger.info("discover: upserted %d stations into registry", len(stations))
    _coverage_summary(stations)


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

async def _cli_ingest(args) -> None:
    src = _build_source(args.source)
    targets = [args.station_id] if args.station_id else None
    summary = await ingest_recent(
        src,
        hours=args.hours,
        only_active=not args.all,
        station_ids=targets,
    )
    print(
        f"ingest: source={src.name} stations={summary['stations']} "
        f"rows={summary['rows']} failures={summary['failures']}"
    )


# --------------------------------------------------------------------------
# evaluate-quality
# --------------------------------------------------------------------------

async def _cli_evaluate_quality(args) -> None:
    summary = await evaluate_quality(
        window_days=args.days,
        min_coverage_pct=args.min_coverage,
    )
    print(
        f"evaluate-quality (window={args.days}d, min_coverage={args.min_coverage}%):\n"
        f"  stations:       {summary['total']:>4d}\n"
        f"  active:         {summary['active']:>4d}  (rows > 0 in window)\n"
        f"  blacklisted:    {summary['blacklisted']:>4d}  (coverage < {args.min_coverage}% or retired)\n"
        f"  retired:        {summary['retired']:>4d}  (+{summary['newly_retired']} this run — bad values)\n"
        f"  has_pressure:   {summary['with_pressure']:>4d}\n"
        f"  has_solar:      {summary['with_solar']:>4d}\n"
        f"  has_rain_data:  {summary['with_rain_data']:>4d}"
    )


# --------------------------------------------------------------------------
# swap — replace retired stations with a similar one (same band + octant)
# --------------------------------------------------------------------------

# Backfill window for a freshly promoted replacement. WU's /hourly/7day serves
# up to 7 days, which is enough for the station to clear the coverage bar on the
# next evaluate-quality run and enter the active pool.
_REPLACEMENT_BACKFILL_HOURS = 24 * 7


async def swap_retired_stations(
    src: PWSSource,
    radius_km: float = 100.0,
    dry_run: bool = False,
) -> dict:
    """Find a fresh replacement for each retired-but-not-yet-replaced station.

    "Similar" = same distance band AND bearing octant from home, so the network
    keeps its spatial coverage as broken sensors are rotated out. The nearest
    qualifying station that isn't already registered wins. The replacement is
    upserted, backfilled with its recent history (so it clears coverage on the
    next quality rescan), and the retired station is stamped with ``replaced_by``
    so it is only swapped once.
    """
    home_lat, home_lon = await _resolve_home()

    async with AsyncSession(engine, expire_on_commit=False) as session:
        all_stations = list((await session.execute(select(Station))).scalars().all())
    existing_ids = {s.station_id for s in all_stations}
    pending = [
        s for s in all_stations
        if s.is_network
        and (s.quality_flags or {}).get("retired")
        and not (s.quality_flags or {}).get("replaced_by")
    ]

    swapped: list[dict] = []
    for st in pending:
        band = distance_band(st.distance_km)
        octant = bearing_octant(st.bearing_deg)
        if band is None or octant is None:
            logger.info("swap: retired %s has no band/octant — skipping", st.station_id)
            continue
        if st.lat is None or st.lon is None:
            logger.info("swap: retired %s has no lat/lon — skipping", st.station_id)
            continue

        # Search around the retired station's own location for a like-for-like.
        try:
            candidates = await src.discover_stations(st.lat, st.lon, radius_km)
        except Exception:
            logger.exception("swap: discover failed for retired %s", st.station_id)
            continue

        best = None
        best_d = float("inf")
        for info in candidates:
            if info.station_id in existing_ids:
                continue
            d = haversine_km(home_lat, home_lon, info.lat, info.lon)
            b = bearing_deg(home_lat, home_lon, info.lat, info.lon)
            if distance_band(d) != band or bearing_octant(b) != octant:
                continue
            if d < best_d:
                best_d, best = d, (info, d, b)

        if best is None:
            logger.info(
                "swap: no fresh candidate in band=%s octant=%s for retired %s",
                band, octant, st.station_id,
            )
            continue

        info, d, b = best
        info.distance_km = d
        info.bearing_deg = b
        entry = {
            "retired": st.station_id,
            "replacement": info.station_id,
            "band": f"{band[0]}-{band[1]}km",
            "octant": octant,
            "distance_km": round(d, 1),
        }
        if dry_run:
            swapped.append(entry)
            continue

        await upsert_network_station(info)
        existing_ids.add(info.station_id)
        try:
            await ingest_recent(
                src, hours=_REPLACEMENT_BACKFILL_HOURS,
                only_active=False, station_ids=[info.station_id],
            )
        except Exception:
            logger.exception("swap: backfill failed for replacement %s", info.station_id)
        await set_quality_flag(st.station_id, "replaced_by", info.station_id)
        logger.info(
            "swap: retired %s → %s (%.1fkm, band=%s octant=%s)",
            st.station_id, info.station_id, d, band, octant,
        )
        swapped.append(entry)

    return {"retired_pending": len(pending), "swapped": len(swapped), "details": swapped}


async def _cli_swap(args) -> None:
    src = _build_source(args.source)
    summary = await swap_retired_stations(
        src, radius_km=args.radius, dry_run=args.dry_run,
    )
    print(
        f"swap: pending={summary['retired_pending']} swapped={summary['swapped']}"
        + ("  (dry-run)" if args.dry_run else "")
    )
    for d in summary["details"]:
        print(f"  {d['retired']} → {d['replacement']}  "
              f"[{d['band']} {d['octant']} {d['distance_km']}km]")


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
    p_disc.add_argument("--wide", action="store_true",
                        help="grid-sweep multiple origins; needed because WU's /near caps to ~10")
    p_disc.add_argument("--dry-run", action="store_true", help="print, don't write")
    p_disc.set_defaults(func=_cli_discover)

    p_ing = sub.add_parser("ingest", help="fetch + persist recent observations")
    p_ing.add_argument("--source", default="wu", choices=["wu"])
    p_ing.add_argument("--hours", type=int, default=24)
    p_ing.add_argument("--station-id", default=None,
                       help="single station; default = all active registered for source")
    p_ing.add_argument("--all", action="store_true",
                       help="include blacklisted/unevaluated stations (default: only quality_flags->>'blacklisted' = 'false')")
    p_ing.set_defaults(func=_cli_ingest)

    p_q = sub.add_parser("evaluate-quality", help="rescore quality_flags per station from recent obs")
    p_q.add_argument("--days", type=int, default=7, help="evaluation window (default 7d)")
    p_q.add_argument("--min-coverage", type=float, default=50.0,
                     help="coverage %% below which a station is blacklisted (default 50)")
    p_q.set_defaults(func=_cli_evaluate_quality)

    p_swap = sub.add_parser("swap", help="replace retired stations with a similar fresh one")
    p_swap.add_argument("--source", default="wu", choices=["wu"])
    p_swap.add_argument("--radius", type=float, default=100.0,
                        help="km search radius around each retired station")
    p_swap.add_argument("--dry-run", action="store_true", help="print picks, don't write")
    p_swap.set_defaults(func=_cli_swap)

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
