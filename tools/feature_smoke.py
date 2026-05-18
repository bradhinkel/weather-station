"""Phase 7.2 feature smoke / sanity tool.

Builds the last N days of features and reports:
  - pipeline shape and NaN coverage,
  - per-field stats for the headline columns,
  - the empirical own-vs-network wind diagnostics
    (validates the measured ~37° CCW shelter offset on direction;
     own/network speed parity is the second-finding check),
  - a final pass/fail line for the physical-sanity checks.

Run on the droplet against the live observations:
  sudo -u www-data sh -c 'cd /opt/weather-station &&
      venv/bin/python -m tools.feature_smoke --days 7'

No CLI sub-parser; intent is "always print everything, never fail silently."
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.features.bearing import circular_mean
from src.features.config import FeatureConfig
from src.features.pipeline import (
    _load_observations_hourly,
    _load_stations,
    _resolve_home_station,
    _sync_dsn,
    build_features,
)


# Thresholds for the pass/fail summary at the bottom. Generous on purpose —
# these flag obvious wiring bugs, not modeling-quality issues.
_NAN_ROW_PCT_MAX = 20.0          # how many target hours can have no features
_TEMP_RANGE_C = (-20.0, 45.0)     # plausible Kirkland temperature bounds
_PRESSURE_RANGE_HPA = (970.0, 1050.0)
_GRADIENT_TEMP_ABS_MAX = 15.0     # 15°C across the 25–50km band would be extraordinary
_SHELTER_BIAS_MIN_DEG = 5.0       # below this, the home station may have moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7.2 feature pipeline smoke test.")
    parser.add_argument("--days", type=int, default=7,
                        help="window to build features for (default 7)")
    parser.add_argument(
        "--wind-reference", default=None,
        choices=["own", "network_mean", "nwp"],
        help="override FeatureConfig.wind_reference (default: keep config default)",
    )
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.wind_reference is not None:
        cfg_kwargs["wind_reference"] = args.wind_reference
    config = FeatureConfig(**cfg_kwargs)

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    print(f"\n=== feature_smoke: {start.isoformat()} → {end.isoformat()} ===")
    print(f"config: wind_reference={config.wind_reference}  "
          f"radius={config.wind_reference_radius_km}km  "
          f"distance_band={config.distance_band_km}  N={config.n_stations}")

    df = build_features(start, end, config)
    failures: list[str] = []

    _print_shape(df, failures)
    _print_field_stats(df, failures)
    _print_gradient_stats(df, failures)
    _print_wind_diagnostic(start, end, config, failures)

    print("\n=== summary ===")
    if not failures:
        print("PASS — all physical-sanity checks satisfied.")
    else:
        print(f"FAIL — {len(failures)} check(s) tripped:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def _print_shape(df: pd.DataFrame, failures: list) -> None:
    print(f"\n[shape] {df.shape[0]} target hours × {df.shape[1]} feature columns")
    if df.empty:
        failures.append("pipeline returned 0 rows")
        return
    n_nan_rows = df["wind_ref_deg"].isna().sum()
    pct = 100.0 * n_nan_rows / len(df)
    print(f"[shape] hours with no wind reference: {n_nan_rows}/{len(df)} ({pct:.1f}%)")
    if pct > _NAN_ROW_PCT_MAX:
        failures.append(f"{pct:.1f}% of hours have no wind reference (max {_NAN_ROW_PCT_MAX:.0f}%)")


def _print_field_stats(df: pd.DataFrame, failures: list) -> None:
    print("\n[fields] cohort means at lag 0h:")
    fields = {
        "upwind_temp_c_lag0h":         _TEMP_RANGE_C,
        "upwind_pressure_hpa_lag0h":   _PRESSURE_RANGE_HPA,
        "upwind_humidity_pct_lag0h":   (0.0, 100.0),
        "upwind_wind_speed_ms_lag0h":  (0.0, 30.0),
    }
    for col, (lo, hi) in fields.items():
        if col not in df.columns:
            failures.append(f"missing column {col}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            failures.append(f"{col} is entirely NaN")
            continue
        n_outliers = ((s < lo) | (s > hi)).sum()
        print(f"  {col:<40}  n={s.count():>4}  "
              f"min={s.min():>7.2f}  med={s.median():>7.2f}  max={s.max():>7.2f}"
              + (f"  outliers={n_outliers}" if n_outliers else ""))
        if n_outliers:
            failures.append(
                f"{col}: {n_outliers} values outside [{lo}, {hi}]"
            )


def _print_gradient_stats(df: pd.DataFrame, failures: list) -> None:
    print("\n[gradient] far - near (lag 0h):")
    bounds = {
        "upwind_temp_c_gradient_lag0h":        (-_GRADIENT_TEMP_ABS_MAX, _GRADIENT_TEMP_ABS_MAX),
        "upwind_pressure_hpa_gradient_lag0h":  (-15.0, 15.0),  # 15 hPa across 25km is large but not impossible
        "upwind_humidity_pct_gradient_lag0h":  (-100.0, 100.0),
        "upwind_wind_speed_ms_gradient_lag0h": (-20.0, 20.0),
    }
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            failures.append(f"missing gradient column {col}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            print(f"  {col:<48}  (no usable data this window)")
            continue
        n_outliers = ((s < lo) | (s > hi)).sum()
        print(f"  {col:<48}  n={s.count():>4}  "
              f"min={s.min():>+7.2f}  med={s.median():>+7.2f}  max={s.max():>+7.2f}"
              + (f"  outliers={n_outliers}" if n_outliers else ""))
        if n_outliers:
            failures.append(f"{col}: {n_outliers} values outside [{lo}, {hi}]")


def _print_wind_diagnostic(
    start: datetime, end: datetime, config: FeatureConfig, failures: list,
) -> None:
    """Recompute the own-vs-network wind diagnostic — independent of build_features
    so any bug in the pipeline can't mask a real shelter signal."""
    engine = create_engine(_sync_dsn())
    try:
        stations = _load_stations(engine)
        home_id = _resolve_home_station(stations)
        obs = _load_observations_hourly(engine, start, end)
    finally:
        engine.dispose()

    quality_ids = set(stations[
        (stations["is_network"] == True) & (stations["blacklisted"] == "false")  # noqa: E712
    ]["station_id"])
    quality = stations[stations["station_id"].isin(quality_ids)][["station_id", "distance_km"]]
    net = obs.merge(quality, on="station_id")
    near = net[net["distance_km"] <= config.wind_reference_radius_km]
    if near.empty:
        print("\n[wind diag] no quality network stations within "
              f"{config.wind_reference_radius_km}km — skipping diagnostic.")
        return

    net_mean = near.groupby("time_hour").apply(
        lambda g: circular_mean(g["wind_dir_deg"].dropna().tolist())
    )
    net_mean.name = "network_dir"
    net_speed = near.groupby("time_hour")["wind_speed_ms"].mean().rename("network_speed")

    own = obs[obs["station_id"] == home_id].set_index("time_hour")
    if own.empty:
        print("\n[wind diag] no own-station data in window — skipping diagnostic.")
        return

    joined = pd.concat(
        [own[["wind_dir_deg", "wind_speed_ms"]].rename(
            columns={"wind_dir_deg": "own_dir", "wind_speed_ms": "own_speed"}),
         net_mean, net_speed],
        axis=1,
    ).dropna(subset=["own_dir", "network_dir"])

    # Filter to wind>=0.5 m/s on own station — calm-wind direction is random noise.
    moving = joined[joined["own_speed"].fillna(0) >= 0.5]
    if moving.empty:
        print("\n[wind diag] no hours with wind ≥ 0.5 m/s — diagnostic skipped.")
        return

    diff = (moving["own_dir"] - moving["network_dir"] + 180) % 360 - 180
    median_signed = diff.median()
    abs_median = diff.abs().median()
    std_deg = diff.std()

    print(f"\n[wind diag] own − network (wind≥0.5 m/s, n={len(moving)}):")
    print(f"  median signed = {median_signed:+.1f}°    abs median = {abs_median:.1f}°    std = {std_deg:.1f}°")

    speed_ratio = moving["own_speed"] / moving["network_speed"].replace(0, np.nan)
    print(f"  own/network speed ratio — median = {speed_ratio.median():.2f}   mean = {speed_ratio.mean():.2f}")

    if abs_median < _SHELTER_BIAS_MIN_DEG:
        # Earlier memory says ~37° shelter; if it drops below 5° something changed.
        failures.append(
            f"home wind shelter offset suspiciously small ({abs_median:.1f}° abs median) — "
            "did the anemometer move? expected ~37° CCW per project memory"
        )


if __name__ == "__main__":
    main()
