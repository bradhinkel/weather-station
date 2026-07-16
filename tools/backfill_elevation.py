"""Backfill stations.elevation_m from Open-Meteo's DEM.

    python -m tools.backfill_elevation --dry-run
    python -m tools.backfill_elevation

Every station in the registry has a NULL elevation, including the home station. That
blocks quality control: air cools ~6.5 C per km of altitude, so within this network's
100 km radius -- Puget Sound at sea level to the Cascade foothills -- two stations can
legitimately differ by several degrees with nothing wrong with either. A QC pass that
cannot subtract the lapse rate would read terrain as sensor error and throw away the
foothills.

Open-Meteo returns `elevation` at the top level of every forecast response and the
project has been discarding it since day one, exactly as it discarded wind_gusts and
cloud_cover until 2026-06-19. It is a 90 m DEM lookup, not the model grid cell:
Snoqualmie Pass reads 928 m (true ~920), Mt Rainier 4380 m (true 4392).

Elevation is static, so this is a one-time backfill rather than a scheduled job; new
stations from `pws discover` need a rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import httpx
from sqlalchemy import create_engine, text

from src.ml.dataset import _sync_dsn

logger = logging.getLogger("backfill_elevation")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# The API accepts comma-separated coordinates and returns a list. Batching keeps this
# to a handful of calls instead of one per station.
BATCH_SIZE = 100


def fetch_elevations(coords: list[tuple[str, float, float]]) -> dict[str, float]:
    """Map station_id -> elevation for a batch, preserving request order."""
    lats = ",".join(f"{lat:.6f}" for _, lat, _ in coords)
    lons = ",".join(f"{lon:.6f}" for _, _, lon in coords)
    resp = httpx.get(
        OPEN_METEO_URL,
        params={
            "latitude": lats,
            "longitude": lons,
            # `hourly` is required by the API, but only the top-level elevation is used.
            "hourly": "temperature_2m",
            "forecast_days": 1,
            "timezone": "UTC",
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    # A single coordinate returns an object; multiples return a list.
    entries = payload if isinstance(payload, list) else [payload]
    if len(entries) != len(coords):
        raise RuntimeError(
            f"Open-Meteo returned {len(entries)} entries for {len(coords)} coordinates; "
            f"refusing to zip misaligned results onto station ids."
        )
    return {
        sid: float(entry["elevation"])
        for (sid, _, _), entry in zip(coords, entries)
        if entry.get("elevation") is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refetch stations that already have one.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = create_engine(_sync_dsn())
    where = "" if args.force else "AND elevation_m IS NULL"
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT station_id, lat, lon FROM stations
                WHERE lat IS NOT NULL AND lon IS NOT NULL {where}
                ORDER BY station_id
            """)
        ).fetchall()

    coords = [(r.station_id, r.lat, r.lon) for r in rows]
    if not coords:
        print("Nothing to backfill.")
        return 0
    logger.info("Fetching elevation for %d stations in %d batches",
                len(coords), (len(coords) + BATCH_SIZE - 1) // BATCH_SIZE)

    elevations: dict[str, float] = {}
    for i in range(0, len(coords), BATCH_SIZE):
        batch = coords[i:i + BATCH_SIZE]
        elevations.update(fetch_elevations(batch))
        logger.info("  %d/%d", min(i + BATCH_SIZE, len(coords)), len(coords))
        if i + BATCH_SIZE < len(coords):
            time.sleep(1.0)  # be polite to a free API

    vals = sorted(elevations.values())
    print(f"\nGot {len(elevations)} elevations. "
          f"min {vals[0]:.0f} m / median {vals[len(vals)//2]:.0f} m / max {vals[-1]:.0f} m")

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0

    with engine.begin() as conn:
        for sid, elev in elevations.items():
            conn.execute(
                text("UPDATE stations SET elevation_m = :e WHERE station_id = :s"),
                {"e": elev, "s": sid},
            )
    engine.dispose()
    print(f"Wrote elevation_m for {len(elevations)} stations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
