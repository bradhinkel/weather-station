"""Data-sufficiency heartbeat — Phase 7.1.

Runs daily on the droplet. One row per (station, window) per run, written to
the ``heartbeat_runs`` table. Answers: are we getting enough data, and is it
good? The same metric set is reused at 7.3 lock time as a gate check against
the frozen snapshot.

Regime heuristics here are intentionally simple — PHASE_7_PLAN.md says to
finalize them during 7.1. Treat the first weeks of output as calibration
data: if the frontal/stable counts disagree with the user's eyeball read of
the same window, tune the thresholds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import HeartbeatRun, Station, engine

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30

# Regime thresholds — heuristic; tune during 7.1 calibration.
FRONTAL_WIND_SHIFT_DEG = 60.0
FRONTAL_PRESSURE_DROP_HPA = 2.0
STABLE_WIND_SHIFT_DEG = 30.0
STABLE_PRESSURE_CHANGE_HPA = 0.5


@dataclass
class HeartbeatReport:
    run_time: datetime
    station_id: str
    window_days: int
    obs_hours_covered: int
    obs_hours_expected: int
    obs_gap_pct: float
    nwp_hours_covered: int
    nwp_hours_expected: int
    nwp_gap_pct: float
    rain_positive_hours: int
    frontal_passage_hours: int
    stable_period_hours: int
    network_coverage_pct: Optional[float]
    sensor_drift_flags: dict
    notes: Optional[str]


_OBS_COVERAGE_SQL = text("""\
WITH hours AS (
    SELECT generate_series(
        date_trunc('hour', now() - make_interval(days => :days)),
        date_trunc('hour', now()) - interval '1 hour',
        interval '1 hour'
    ) AS hour
),
covered AS (
    SELECT DISTINCT date_trunc('hour', time) AS hour
    FROM observations
    WHERE station_id = :sid
      AND time >= now() - make_interval(days => :days)
)
SELECT
    count(h.hour)::int          AS expected,
    count(c.hour)::int          AS covered
FROM hours h
LEFT JOIN covered c USING (hour)
""")


_NWP_COVERAGE_SQL = text("""\
WITH hours AS (
    SELECT generate_series(
        date_trunc('hour', now() - make_interval(days => :days)),
        date_trunc('hour', now()) - interval '1 hour',
        interval '1 hour'
    ) AS hour
),
covered AS (
    SELECT DISTINCT date_trunc('hour', valid_time) AS hour
    FROM forecasts
    WHERE station_id = :sid
      AND valid_time >= now() - make_interval(days => :days)
      AND valid_time <  now()
)
SELECT
    count(h.hour)::int AS expected,
    count(c.hour)::int AS covered
FROM hours h
LEFT JOIN covered c USING (hour)
""")


_RAIN_POSITIVE_SQL = text("""\
SELECT count(DISTINCT date_trunc('hour', time))::int AS rain_hours
FROM observations
WHERE station_id = :sid
  AND time >= now() - make_interval(days => :days)
  AND (
      (rain_rate_mm_hr IS NOT NULL AND rain_rate_mm_hr > 0)
      OR (rain_mm_1h IS NOT NULL AND rain_mm_1h > 0)
  )
""")


# Network coverage = realized station-hours / expected station-hours, where
# expected = (count of non-blacklisted network stations) × (window hours).
# Stations without an evaluate_quality stamp are excluded from the denominator
# so the metric doesn't get diluted by stations that may not be usable.
_NETWORK_COVERAGE_SQL = text("""\
WITH usable AS (
    SELECT station_id
    FROM stations
    WHERE is_network = true
      AND quality_flags->>'blacklisted' = 'false'
)
SELECT
    (SELECT count(*) FROM usable)::int                              AS station_count,
    (
        SELECT count(*)::int FROM observations o
        JOIN usable u USING (station_id)
        WHERE time >= now() - make_interval(days => :days)
          AND time <  now()
    )                                                                AS row_count
""")


# Regime detection: bucket to hourly means (circular for wind direction via
# sin/cos averaging), then look at the 6-hour change. Front = wind veers > 60°
# AND pressure falls > 2 hPa. Stable = wind drift < 30° AND |Δp| < 0.5 hPa.
_REGIME_SQL = text("""\
WITH hourly AS (
    SELECT
        date_trunc('hour', time)                                AS hour,
        avg(pressure_hpa)                                       AS pressure_hpa,
        avg(sin(radians(wind_dir_deg)))                         AS wsin,
        avg(cos(radians(wind_dir_deg)))                         AS wcos
    FROM observations
    WHERE station_id = :sid
      AND time >= now() - make_interval(days => :days + 1)
      AND wind_dir_deg IS NOT NULL
      AND pressure_hpa IS NOT NULL
    GROUP BY 1
),
with_dir AS (
    SELECT
        hour,
        pressure_hpa,
        mod(degrees(atan2(wsin, wcos))::numeric + 360, 360)::float AS wind_dir_deg
    FROM hourly
),
lagged AS (
    SELECT
        hour,
        pressure_hpa,
        wind_dir_deg,
        LAG(pressure_hpa, 6) OVER (ORDER BY hour) AS pressure_hpa_6h_ago,
        LAG(wind_dir_deg, 6) OVER (ORDER BY hour) AS wind_dir_6h_ago
    FROM with_dir
),
diffs AS (
    SELECT
        hour,
        LEAST(
            abs(wind_dir_deg - wind_dir_6h_ago),
            360 - abs(wind_dir_deg - wind_dir_6h_ago)
        ) AS wind_shift_deg,
        (pressure_hpa_6h_ago - pressure_hpa) AS pressure_drop_hpa
    FROM lagged
    WHERE pressure_hpa_6h_ago IS NOT NULL
      AND wind_dir_6h_ago IS NOT NULL
      AND hour >= now() - make_interval(days => :days)
)
SELECT
    count(*) FILTER (
        WHERE wind_shift_deg > :frontal_wind
          AND pressure_drop_hpa > :frontal_drop
    )::int AS frontal_hours,
    count(*) FILTER (
        WHERE wind_shift_deg < :stable_wind
          AND abs(pressure_drop_hpa) < :stable_press
    )::int AS stable_hours
FROM diffs
""")


async def compute_heartbeat(
    session: AsyncSession,
    station_id: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> HeartbeatReport:
    """Compute one heartbeat snapshot for a station without persisting it."""
    obs_row = (
        await session.execute(
            _OBS_COVERAGE_SQL, {"sid": station_id, "days": window_days}
        )
    ).one()
    nwp_row = (
        await session.execute(
            _NWP_COVERAGE_SQL, {"sid": station_id, "days": window_days}
        )
    ).one()
    rain_row = (
        await session.execute(
            _RAIN_POSITIVE_SQL, {"sid": station_id, "days": window_days}
        )
    ).one()
    regime_row = (
        await session.execute(
            _REGIME_SQL,
            {
                "sid": station_id,
                "days": window_days,
                "frontal_wind": FRONTAL_WIND_SHIFT_DEG,
                "frontal_drop": FRONTAL_PRESSURE_DROP_HPA,
                "stable_wind": STABLE_WIND_SHIFT_DEG,
                "stable_press": STABLE_PRESSURE_CHANGE_HPA,
            },
        )
    ).one()
    net_row = (
        await session.execute(_NETWORK_COVERAGE_SQL, {"days": window_days})
    ).one()

    obs_expected = obs_row.expected or 0
    obs_covered = obs_row.covered or 0
    nwp_expected = nwp_row.expected or 0
    nwp_covered = nwp_row.covered or 0

    obs_gap_pct = (
        100.0 * (obs_expected - obs_covered) / obs_expected
        if obs_expected else 0.0
    )
    nwp_gap_pct = (
        100.0 * (nwp_expected - nwp_covered) / nwp_expected
        if nwp_expected else 0.0
    )

    net_stations = net_row.station_count or 0
    net_rows = net_row.row_count or 0
    net_expected = net_stations * window_days * 24
    network_coverage_pct: Optional[float]
    if net_expected:
        network_coverage_pct = round(100.0 * net_rows / net_expected, 2)
    else:
        network_coverage_pct = None  # no usable network stations yet

    return HeartbeatReport(
        run_time=datetime.now(timezone.utc),
        station_id=station_id,
        window_days=window_days,
        obs_hours_covered=obs_covered,
        obs_hours_expected=obs_expected,
        obs_gap_pct=round(obs_gap_pct, 2),
        nwp_hours_covered=nwp_covered,
        nwp_hours_expected=nwp_expected,
        nwp_gap_pct=round(nwp_gap_pct, 2),
        rain_positive_hours=rain_row.rain_hours or 0,
        frontal_passage_hours=regime_row.frontal_hours or 0,
        stable_period_hours=regime_row.stable_hours or 0,
        network_coverage_pct=network_coverage_pct,
        sensor_drift_flags={},      # finalized later in 7.1 calibration
        notes=None,
    )


async def _persist(session: AsyncSession, report: HeartbeatReport) -> None:
    """Idempotent insert; (run_time, station_id, window_days) is the PK."""
    await session.execute(
        text("""
            INSERT INTO heartbeat_runs (
                run_time, station_id, window_days,
                obs_hours_covered, obs_hours_expected, obs_gap_pct,
                nwp_hours_covered, nwp_hours_expected, nwp_gap_pct,
                rain_positive_hours, frontal_passage_hours, stable_period_hours,
                network_coverage_pct, sensor_drift_flags, notes
            ) VALUES (
                :run_time, :station_id, :window_days,
                :obs_hours_covered, :obs_hours_expected, :obs_gap_pct,
                :nwp_hours_covered, :nwp_hours_expected, :nwp_gap_pct,
                :rain_positive_hours, :frontal_passage_hours, :stable_period_hours,
                :network_coverage_pct, CAST(:sensor_drift_flags AS JSONB), :notes
            )
            ON CONFLICT (run_time, station_id, window_days) DO NOTHING
        """),
        {
            **{
                k: v for k, v in asdict(report).items()
                if k != "sensor_drift_flags"
            },
            "sensor_drift_flags": json.dumps(report.sensor_drift_flags),
        },
    )


async def run_heartbeat(
    window_days: int = DEFAULT_WINDOW_DAYS,
    station_id: Optional[str] = None,
) -> list[HeartbeatReport]:
    """Run heartbeat for one station (if given) or every registered station.

    Returns the reports it wrote, in registration order.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        if station_id is None:
            # Heartbeat reports on own-station data sufficiency (NWP coverage,
            # frontal passages, etc.). Network stations are summarized via
            # network_coverage_pct on the own-station row, not their own rows.
            stations = (
                await session.execute(
                    select(Station).where(Station.is_network.is_(False))
                )
            ).scalars().all()
        else:
            stations = (
                await session.execute(
                    select(Station).where(Station.station_id == station_id)
                )
            ).scalars().all()

    if not stations:
        logger.warning("run_heartbeat: no stations registered — skipping.")
        return []

    reports: list[HeartbeatReport] = []
    for station in stations:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            async with session.begin():
                report = await compute_heartbeat(
                    session, station.station_id, window_days
                )
                await _persist(session, report)
        logger.info(
            "heartbeat station=%s obs_gap=%.1f%% rain_h=%d frontal_h=%d stable_h=%d",
            report.station_id,
            report.obs_gap_pct,
            report.rain_positive_hours,
            report.frontal_passage_hours,
            report.stable_period_hours,
        )
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# CLI: python -m src.heartbeat [--days N] [--station-id SID]
# ---------------------------------------------------------------------------

def _print_report(r: HeartbeatReport) -> None:
    print(f"\nstation: {r.station_id}   window: {r.window_days}d   run: {r.run_time.isoformat()}")
    print(f"  obs coverage:      {r.obs_hours_covered:>5d}/{r.obs_hours_expected:<5d} "
          f"hours  ({r.obs_gap_pct:5.1f}% gap)")
    print(f"  nwp coverage:      {r.nwp_hours_covered:>5d}/{r.nwp_hours_expected:<5d} "
          f"hours  ({r.nwp_gap_pct:5.1f}% gap)")
    print(f"  rain-positive:     {r.rain_positive_hours:>5d} hours")
    print(f"  frontal-passage:   {r.frontal_passage_hours:>5d} hours")
    print(f"  stable-period:     {r.stable_period_hours:>5d} hours")
    if r.network_coverage_pct is not None:
        print(f"  network coverage:  {r.network_coverage_pct:.1f}%")


async def _main(window_days: int, station_id: Optional[str]) -> None:
    reports = await run_heartbeat(window_days=window_days, station_id=station_id)
    if not reports:
        print("no reports produced (no stations registered?)")
        return
    for r in reports:
        _print_report(r)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Data-sufficiency heartbeat")
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--station-id", default=None)
    args = parser.parse_args()
    asyncio.run(_main(args.days, args.station_id))
