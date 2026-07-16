"""Run the crowdsourced-station QC pass over the live network.

    python -m tools.run_qc --dry-run          # report only
    python -m tools.run_qc                    # write qc_status into stations.quality_flags

Applies src.pws.qc to every network station: k-nearest elevation-adjusted buddies,
asymmetric per-hour outlier test, then a station-level verdict from the flagged
fraction and the correlation against the buddy median.

Writes `quality_flags.qc_status` ("ok" | "suspect" | "isolated"), `qc_reason`,
`qc_flag_fraction`, `qc_correlation`, `qc_n_buddies`. It deliberately does NOT set
`blacklisted` or `retired` — those already mean coverage and stuck-sensor respectively,
and conflating a third meaning into them would silently change what train/serve filter
on. Consumers opt in by reading qc_status.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.ml.dataset import _sync_dsn
from src.pws.distance import haversine_km
from src.pws.qc import (
    BUDDY_K,
    BUDDY_MAX_RADIUS_KM,
    BUDDY_MIN_COUNT,
    buddy_check_hour,
    buddy_correlation,
    classify_station,
    elevation_adjust,
    robust_center_spread,
    station_flag_fraction,
)

logger = logging.getLogger("run_qc")


def load_stations(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT station_id, lat, lon, elevation_m, distance_km, bearing_deg
                FROM stations
                WHERE is_network = true AND lat IS NOT NULL AND lon IS NOT NULL
                  AND elevation_m IS NOT NULL
            """),
            conn,
        )


def load_hourly(engine, start: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT o.station_id,
                       date_trunc('hour', o.time) AS hour,
                       avg(o.temp_c) AS temp_c
                FROM observations o
                JOIN stations s USING (station_id)
                WHERE s.is_network = true AND o.temp_c IS NOT NULL
                  AND o.time >= :start
                GROUP BY 1, 2
            """),
            conn,
            params={"start": start},
        )
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df


def compute_buddies(stations: pd.DataFrame) -> dict[str, list[str]]:
    """k-nearest neighbours within the max radius, per station."""
    ids = stations["station_id"].tolist()
    lats = stations["lat"].to_numpy()
    lons = stations["lon"].to_numpy()
    buddies: dict[str, list[str]] = {}
    for i, sid in enumerate(ids):
        dists = [
            (haversine_km(lats[i], lons[i], lats[j], lons[j]), ids[j])
            for j in range(len(ids)) if j != i
        ]
        near = sorted(d for d in dists if d[0] <= BUDDY_MAX_RADIUS_KM)[:BUDDY_K]
        buddies[sid] = [sid_j for _, sid_j in near]
    return buddies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-05-20", help="Earliest observation hour to score.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = create_engine(_sync_dsn())
    stations = load_stations(engine)
    logger.info("%d network stations with coordinates + elevation", len(stations))

    obs = load_hourly(engine, args.start)
    logger.info("%d station-hours since %s", len(obs), args.start)

    elev = dict(zip(stations["station_id"], stations["elevation_m"]))
    buddies = compute_buddies(stations)
    n_isolated = sum(1 for b in buddies.values() if len(b) < BUDDY_MIN_COUNT)
    logger.info("buddy graph: k<=%d within %.0fkm; %d stations isolated (<%d buddies)",
                BUDDY_K, BUDDY_MAX_RADIUS_KM, n_isolated, BUDDY_MIN_COUNT)

    # hour -> {station_id: temp}
    by_hour: dict[pd.Timestamp, dict[str, float]] = defaultdict(dict)
    for r in obs.itertuples(index=False):
        by_hour[r.hour][r.station_id] = r.temp_c

    flags: dict[str, list[bool]] = defaultdict(list)
    station_series: dict[str, list[float]] = defaultdict(list)
    buddy_series: dict[str, list[float]] = defaultdict(list)

    for hour, temps in by_hour.items():
        for sid, temp in temps.items():
            blist = buddies.get(sid, [])
            bt = [temps[b] for b in blist if b in temps]
            be = [elev[b] for b in blist if b in temps]
            if len(bt) < BUDDY_MIN_COUNT:
                continue
            flagged, _ = buddy_check_hour(temp, elev[sid], bt, be)
            flags[sid].append(flagged)
            # Buddy median at the station's elevation — the reference series for the
            # correlation (indoor) test.
            adj = elevation_adjust(np.asarray(bt), np.asarray(be), elev[sid])
            center, _ = robust_center_spread(adj)
            station_series[sid].append(temp)
            buddy_series[sid].append(center)

    results = []
    for sid in stations["station_id"]:
        frac = station_flag_fraction(flags.get(sid, []))
        corr = buddy_correlation(station_series.get(sid, []), buddy_series.get(sid, []))
        status, reason = classify_station(frac, corr, len(buddies.get(sid, [])))
        results.append({
            "station_id": sid,
            "qc_status": status,
            "qc_reason": reason,
            "qc_flag_fraction": None if not np.isfinite(frac) else round(float(frac), 4),
            "qc_correlation": None if not np.isfinite(corr) else round(float(corr), 4),
            "qc_n_buddies": len(buddies.get(sid, [])),
            "n_hours_tested": len(flags.get(sid, [])),
        })

    res = pd.DataFrame(results)
    counts = res["qc_status"].value_counts().to_dict()
    total_hours = int(sum(len(v) for v in flags.values()))
    total_flagged = int(sum(sum(v) for v in flags.values()))

    print()
    print(f"QC over {len(res)} stations, {total_hours} testable station-hours since {args.start}")
    print()
    for status in ("ok", "suspect", "isolated"):
        n = counts.get(status, 0)
        print(f"  {status:>9}: {n:>4}  ({100*n/len(res):.1f}%)")
    print()
    print(f"  hourly readings flagged: {total_flagged}/{total_hours} "
          f"({100*total_flagged/max(total_hours,1):.1f}%)")
    print(f"  stations that would be dropped (suspect+isolated): "
          f"{counts.get('suspect',0) + counts.get('isolated',0)}/{len(res)} "
          f"({100*(counts.get('suspect',0)+counts.get('isolated',0))/len(res):.1f}%)")
    print()
    print("  For scale, published CWS studies discard: Nipen 21% of readings,")
    print("  Meier 53% of data, CrowdQC+ ~70%. This project previously discarded ~0%.")

    worst = res[res["qc_status"] == "suspect"].sort_values(
        "qc_flag_fraction", ascending=False, na_position="last"
    ).head(10)
    if not worst.empty:
        print("\nWorst offenders:")
        for r in worst.itertuples(index=False):
            print(f"  {r.station_id:>14}  {r.qc_reason}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        engine.dispose()
        return 0

    with engine.begin() as conn:
        for r in results:
            conn.execute(
                text("""
                    UPDATE stations
                    SET quality_flags = quality_flags || CAST(:patch AS jsonb)
                    WHERE station_id = :sid
                """),
                {
                    "sid": r["station_id"],
                    "patch": json.dumps({
                        "qc_status": r["qc_status"],
                        "qc_reason": r["qc_reason"],
                        "qc_flag_fraction": r["qc_flag_fraction"],
                        "qc_correlation": r["qc_correlation"],
                        "qc_n_buddies": r["qc_n_buddies"],
                    }),
                },
            )
    engine.dispose()
    print(f"\nWrote qc_status for {len(results)} stations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
