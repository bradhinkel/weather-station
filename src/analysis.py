"""Forecast accuracy analysis — compare Open-Meteo forecasts with local observations."""

import asyncio
import argparse
import logging
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function 1 — baseline errors by lead time
# ---------------------------------------------------------------------------

_BASELINE_SQL = text("""\
WITH obs_hourly AS (
    SELECT
        station_id,
        date_trunc('hour', time) AS hour,
        avg(temp_c) AS temp_c
    FROM observations
    WHERE station_id = :sid
      AND time >= now() - make_interval(days => :days)
      AND temp_c IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    avg(abs(o.temp_c - f.temp_c))             AS mae,
    sqrt(avg(power(o.temp_c - f.temp_c, 2)))  AS rmse,
    avg(f.temp_c - o.temp_c)                  AS bias,
    count(*)                                  AS n
FROM obs_hourly o
JOIN forecasts f
  ON  f.station_id  = o.station_id
  AND f.valid_time  = o.hour
  AND f.forecast_time BETWEEN o.hour - make_interval(hours => :lead) - interval '30 minutes'
                          AND o.hour - make_interval(hours => :lead) + interval '30 minutes'
WHERE f.temp_c IS NOT NULL
""")


async def get_baseline_errors(
    station_id: str,
    days: int = 30,
    lead_hours: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return MAE / RMSE / bias for temperature at each lead-hour bucket.

    Returns ``{lead_hour: {"mae": …, "rmse": …, "bias": …, "n": …}, …}``.
    """
    if lead_hours is None:
        lead_hours = [1, 6, 12, 24]

    results: dict[int, dict[str, Any]] = {}
    async with AsyncSession(engine, expire_on_commit=False) as session:
        for lead in lead_hours:
            row = (
                await session.execute(
                    _BASELINE_SQL,
                    {"sid": station_id, "days": days, "lead": lead},
                )
            ).one()
            results[lead] = {
                "mae": round(row.mae, 3) if row.mae is not None else None,
                "rmse": round(row.rmse, 3) if row.rmse is not None else None,
                "bias": round(row.bias, 3) if row.bias is not None else None,
                "n": row.n,
            }
    return results


# ---------------------------------------------------------------------------
# Function 2 — multi-variable forecast bias
# ---------------------------------------------------------------------------

_BIAS_SQL = text("""\
WITH obs_hourly AS (
    SELECT
        station_id,
        date_trunc('hour', time) AS hour,
        avg(temp_c)        AS temp_c,
        avg(humidity_pct)  AS humidity_pct,
        avg(pressure_hpa)  AS pressure_hpa,
        avg(wind_speed_ms) AS wind_speed_ms,
        max(rain_mm_daily_total) AS daily_total_end
    FROM observations
    WHERE station_id = :sid
      AND time >= now() - make_interval(days => :days)
    GROUP BY 1, 2
),
obs_with_rain AS (
    SELECT
        station_id, hour,
        temp_c, humidity_pct, pressure_hpa, wind_speed_ms,
        CASE
            WHEN LAG(daily_total_end) OVER w IS NULL
                THEN NULL
            WHEN daily_total_end >= LAG(daily_total_end) OVER w
                THEN daily_total_end - LAG(daily_total_end) OVER w
            ELSE daily_total_end
        END AS rain_mm_in_hour
    FROM obs_hourly
    WINDOW w AS (PARTITION BY station_id ORDER BY hour)
)
SELECT
    avg(f.temp_c        - o.temp_c)            AS temp_bias,
    avg(f.precip_mm     - o.rain_mm_in_hour)   AS precip_bias,
    avg(f.wind_speed_ms - o.wind_speed_ms)     AS wind_speed_bias,
    avg(f.pressure_hpa  - o.pressure_hpa)      AS pressure_bias,
    count(*)                                   AS n
FROM obs_with_rain o
JOIN (
    SELECT DISTINCT ON (station_id, valid_time)
        station_id, valid_time, forecast_time,
        temp_c, precip_mm, wind_speed_ms, wind_dir_deg, pressure_hpa
    FROM forecasts
    WHERE forecast_time < valid_time
    ORDER BY station_id, valid_time, forecast_time DESC
) f
  ON  f.station_id = o.station_id
  AND f.valid_time = o.hour
""")


async def get_forecast_bias(
    station_id: str,
    days: int = 30,
) -> dict[str, Any]:
    """Return mean (forecast − observed) for temp, precip, wind_speed, pressure.

    A positive temp_bias means Open-Meteo runs warm at this location.
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = (
            await session.execute(_BIAS_SQL, {"sid": station_id, "days": days})
        ).one()

    def _r(v: float | None) -> float | None:
        return round(v, 3) if v is not None else None

    return {
        "temp_bias": _r(row.temp_bias),
        "precip_bias": _r(row.precip_bias),
        "wind_speed_bias": _r(row.wind_speed_bias),
        "pressure_bias": _r(row.pressure_bias),
        "n": row.n,
    }


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.analysis --station-id MY_STATION
# ---------------------------------------------------------------------------

def _fmt(v: float | None, width: int = 8) -> str:
    if v is None:
        return "N/A".rjust(width)
    return f"{v:>{width}.3f}"


async def _main(station_id: str, days: int) -> None:
    baseline = await get_baseline_errors(station_id, days=days)
    bias = await get_forecast_bias(station_id, days=days)

    print(f"\n{'=' * 56}")
    print(f"  Forecast accuracy — station {station_id}  (last {days} days)")
    print(f"{'=' * 56}")

    print(f"\n  {'Lead (h)':>8}  {'MAE °C':>8}  {'RMSE °C':>8}  {'Bias °C':>8}  {'N':>6}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}")
    for lead in sorted(baseline):
        b = baseline[lead]
        print(
            f"  {lead:>8d}  {_fmt(b['mae'])}  {_fmt(b['rmse'])}  "
            f"{_fmt(b['bias'])}  {b['n']:>6d}"
        )

    print(f"\n  Multi-variable bias (forecast − observed):")
    print(f"  {'Variable':>16}  {'Bias':>10}  {'Unit':>8}")
    print(f"  {'-' * 16}  {'-' * 10}  {'-' * 8}")
    labels = [
        ("temp",       bias["temp_bias"],       "°C"),
        ("precip",     bias["precip_bias"],      "mm"),
        ("wind_speed", bias["wind_speed_bias"],  "m/s"),
        ("pressure",   bias["pressure_bias"],    "hPa"),
    ]
    for name, val, unit in labels:
        print(f"  {name:>16}  {_fmt(val, 10)}  {unit:>8}")

    print(f"\n  Paired observations: {bias['n']}")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Forecast accuracy report")
    parser.add_argument("--station-id", required=True, help="Station identifier")
    parser.add_argument("--days", type=int, default=30, help="Look-back window (default 30)")
    args = parser.parse_args()
    asyncio.run(_main(args.station_id, args.days))
