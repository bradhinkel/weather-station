"""Watering-decision divergence: hyperlocal station vs regional forecast.

Directional product analysis for the garden/yard-watering use case. Quantifies
how often a watering decision driven by THIS yard's own station diverges from
one driven by the regional Open-Meteo forecast — that divergence rate is the
hyperlocal value proposition.

Method — a daily soil-water balance (the standard for irrigation scheduling):

    deficit_t = clip0(deficit_{t-1} + Kc * ET0_t - effective_rain_t)
    irrigate when deficit >= TRIGGER, refilling by APPLY mm.

ET0 is **Hargreaves** (temperature-only: Tmin/Tmax + computed extraterrestrial
radiation). Two reasons: (1) the forecast table has no solar, so temp-only is
the only apples-to-apples ET both sides can compute; (2) it doubles as a
minimal-sensor test — if temp+rain alone moves the decision, that's the cheap
device's sensor floor. Penman-Monteith (needs solar/wind/humidity) is the
refinement once forecast solar is collected.

CAVEAT: only ~30 paired days exist (forecasts retained from 2026-05-20), single
late-spring window. This is a DIRECTIONAL read, not a final analysis. The
forecast side uses the nearest-prior forecast per hour (best-case for the
forecast), so divergence here is a conservative lower bound.

Run on the droplet:  venv/bin/python -m tools.watering_divergence
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from src.ml.dataset import _sync_dsn

# --- water-balance knobs (cool-season turf defaults) ---
KC = 0.8               # crop coefficient (lawn)
TRIGGER_MM = 25.0      # deficit that triggers irrigation (MAD for turf)
APPLY_MM = 25.0        # water applied per irrigation event
RAIN_SKIP_MM = 5.0     # "meaningful rain" threshold for the skip-disagreement count
MIN_HOURS = 20         # drop partial edge days with fewer obs/forecast hours

_DAILY_SQL = text("""
WITH home AS (SELECT station_id, lat FROM stations WHERE is_network = false LIMIT 1),
obs_hourly AS (
    SELECT date_trunc('hour', time) AS hr,
           avg(temp_c) AS t, min(temp_c) AS tmn, max(temp_c) AS tmx,
           max(COALESCE(rain_mm_1h, 0)) AS rain_h,   -- hourly accumulation: max within hour
           avg(solar_wm2) AS solar
    FROM observations, home
    WHERE source = 'ecowitt' AND station_id = home.station_id
    GROUP BY 1
),
obs_daily AS (
    SELECT (hr AT TIME ZONE 'America/Los_Angeles')::date AS d,
           count(*) AS hours,
           min(tmn) AS tmin, max(tmx) AS tmax, avg(t) AS tmean,
           sum(rain_h) AS rain, avg(solar) AS solar_w
    FROM obs_hourly GROUP BY 1
),
fc_near AS (
    SELECT DISTINCT ON (valid_time) valid_time, temp_c, precip_mm
    FROM forecasts f, home
    WHERE f.station_id = home.station_id AND forecast_time < valid_time
    ORDER BY valid_time, forecast_time DESC
),
fc_daily AS (
    SELECT (valid_time AT TIME ZONE 'America/Los_Angeles')::date AS d,
           count(*) AS hours,
           min(temp_c) AS tmin, max(temp_c) AS tmax, avg(temp_c) AS tmean,
           sum(precip_mm) AS rain
    FROM fc_near GROUP BY 1
)
SELECT o.d,
       o.hours AS o_hours, f.hours AS f_hours,
       o.tmin AS o_tmin, o.tmax AS o_tmax, o.tmean AS o_tmean, o.rain AS o_rain,
       f.tmin AS f_tmin, f.tmax AS f_tmax, f.tmean AS f_tmean, f.rain AS f_rain,
       (SELECT lat FROM home) AS lat
FROM obs_daily o JOIN fc_daily f USING (d)
ORDER BY o.d
""")


def _ra_mj(lat_deg: float, doy: int) -> float:
    """FAO-56 extraterrestrial radiation (MJ/m^2/day)."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    dec = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(dec))))
    gsc = 0.0820
    return (24 * 60 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(dec)
        + math.cos(phi) * math.cos(dec) * math.sin(ws)
    )


def _hargreaves_et0(tmin, tmax, tmean, lat, doy) -> float:
    ra_mm = 0.408 * _ra_mj(lat, doy)
    return 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra_mm


def _schedule(et0, rain):
    """Run the deficit bucket; return (per-day watered bool list, events)."""
    deficit = 0.0
    watered = []
    for e, r in zip(et0, rain):
        deficit = max(0.0, deficit + KC * e - r)
        w = deficit >= TRIGGER_MM
        if w:
            deficit = max(0.0, deficit - APPLY_MM)
        watered.append(w)
    return watered


def main() -> None:
    engine = create_engine(_sync_dsn())
    df = pd.read_sql(_DAILY_SQL, engine)
    engine.dispose()

    full = df[(df.o_hours >= MIN_HOURS) & (df.f_hours >= MIN_HOURS)].copy()
    full["doy"] = pd.to_datetime(full["d"]).dt.dayofyear
    lat = float(full["lat"].iloc[0])

    full["o_et0"] = full.apply(
        lambda r: _hargreaves_et0(r.o_tmin, r.o_tmax, r.o_tmean, lat, r.doy), axis=1)
    full["f_et0"] = full.apply(
        lambda r: _hargreaves_et0(r.f_tmin, r.f_tmax, r.f_tmean, lat, r.doy), axis=1)

    # daily net irrigation requirement (mm): demand minus rain, floored at 0
    full["o_need"] = (KC * full.o_et0 - full.o_rain).clip(lower=0)
    full["f_need"] = (KC * full.f_et0 - full.f_rain).clip(lower=0)

    o_water = _schedule(full.o_et0.tolist(), full.o_rain.tolist())
    f_water = _schedule(full.f_et0.tolist(), full.f_rain.tolist())
    full["o_water"] = o_water
    full["f_water"] = f_water
    full["disagree"] = full.o_water != full.f_water

    n = len(full)
    # rain-skip disagreements: a meaningful rain one side saw and the other missed
    skip_dis = ((full.o_rain >= RAIN_SKIP_MM) != (full.f_rain >= RAIN_SKIP_MM)).sum()

    print("=" * 64)
    print("WATERING-DECISION DIVERGENCE — hyperlocal station vs regional forecast")
    print(f"window: {full.d.min()} -> {full.d.max()}   full days: {n}   "
          f"(of {len(df)} paired; lat={lat:.2f})")
    print("DIRECTIONAL ONLY — ~1 month, late spring. Forecast = best-case (nearest-prior).")
    print("=" * 64)

    print("\n-- systematic biases (station - forecast) --")
    print(f"  Tmax bias:   {(full.o_tmax - full.f_tmax).mean():+.2f} C  "
          f"(std {(full.o_tmax - full.f_tmax).std():.2f})")
    print(f"  Tmin bias:   {(full.o_tmin - full.f_tmin).mean():+.2f} C  "
          f"(std {(full.o_tmin - full.f_tmin).std():.2f})")
    print(f"  total rain:  station {full.o_rain.sum():.1f} mm vs forecast "
          f"{full.f_rain.sum():.1f} mm  "
          f"({full.o_rain.sum() - full.f_rain.sum():+.1f} mm, "
          f"{100*(full.o_rain.sum()-full.f_rain.sum())/max(full.f_rain.sum(),0.1):+.0f}%)")
    print(f"  total ET0:   station {full.o_et0.sum():.1f} mm vs forecast "
          f"{full.f_et0.sum():.1f} mm  ({full.o_et0.sum()-full.f_et0.sum():+.1f} mm)")

    print("\n-- cumulative net irrigation requirement over window --")
    print(f"  station-driven:  {full.o_need.sum():6.1f} mm")
    print(f"  forecast-driven: {full.f_need.sum():6.1f} mm")
    diff = full.o_need.sum() - full.f_need.sum()
    print(f"  difference:      {diff:+6.1f} mm  "
          f"({100*diff/max(full.f_need.sum(),0.1):+.0f}% vs forecast)")
    print("  (>0 => regional forecast UNDER-waters this yard; <0 => over-waters)")

    print("\n-- discrete watering schedule (deficit bucket) --")
    print(f"  station watering events:  {sum(o_water)}")
    print(f"  forecast watering events: {sum(f_water)}")
    print(f"  days the decision DIFFERS: {full.disagree.sum()} / {n}  "
          f"({100*full.disagree.sum()/n:.0f}%)")
    print(f"  meaningful-rain (>= {RAIN_SKIP_MM:.0f}mm) skip disagreements: {skip_dis} days")

    print("\n-- daily detail (date | o_tmax f_tmax | o_rain f_rain | o_need f_need | water o/f) --")
    for _, r in full.iterrows():
        flag = "  <-- DIFFER" if r.disagree else ""
        print(f"  {r.d}  {r.o_tmax:4.1f}/{r.f_tmax:4.1f}  "
              f"rain {r.o_rain:4.1f}/{r.f_rain:4.1f}  "
              f"need {r.o_need:4.1f}/{r.f_need:4.1f}  "
              f"{'W' if r.o_water else '.'}/{'W' if r.f_water else '.'}{flag}")


if __name__ == "__main__":
    main()
