"""Physical-plausibility checks on dataset and training outputs.

Every data bug this project has shipped produced output that *ran fine* and looked
plausible in aggregate. None of them raised. None were caught by a unit test —
the tested modules (bearing, aggregation, wind_reference) are pure functions with
obvious contracts, while every real bug lived in the SQL and the joins, where
correctness is invisible. What actually caught them was noticing that a number was
physically impossible:

  * 2026-05-06  `timezone=auto` shifted every forecast by 7h. Open-Meteo looked 3x
                worse than reality and the first model "beat" it by 5x. Tell: a
                skill number too good to be true.
  * 2026-06-01  the rain target was derived from a column the network never
                populates; build_dataset returned 1 positive row in 76k. Tell: a
                positive-class rate of ~0 for a Seattle spring.
  * 2026-06-19  a tz-strip zeroed every network feature, so all ablation configs
                returned byte-identical results. Tell: identical-across-configs.
  * 2026-06-19  fill-0 on pressure (~1015 hPa) blew Ridge up to MAE 20-30. Tell: a
                magnitude no thermometer could produce.
  * 2026-07-15  the forecast join ignored the horizon, pinning the Open-Meteo
                baseline to a horizon-independent 1.68 C. Tell: forecast error that
                does not grow with lead time.

These functions turn those tells into assertions. They are pure and take plain
values so they can be unit-tested; `tools/check_invariants.py` runs them against
the live corpus. Each returns a list of human-readable violations — empty means OK.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Physical bounds for the *observable* feature space. Deliberately generous: these
# catch impossible values (a fill-0 pressure, a stuck gauge), not unusual weather.
# The tighter data-quality ceilings used at ingest live in src/quality_limits.py.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "f_temp_c": (-50.0, 60.0),
    "lag_temp_c": (-50.0, 60.0),
    "f_humidity_pct": (0.0, 100.0),
    "lag_humidity_pct": (0.0, 100.0),
    "f_pressure_hpa": (870.0, 1085.0),
    "lag_pressure_hpa": (870.0, 1085.0),
    "f_wind_speed_ms": (0.0, 113.0),
    "lag_wind_speed_ms": (0.0, 113.0),
    "f_precip_mm": (0.0, 100.0),
    "lag_rain_mm_1h": (0.0, 100.0),
}

# Seattle rain-hour rate is ~10-15% annually and ~0-5% in a dry July. A pooled
# corpus outside this band means the target is being derived wrong, not that the
# weather is remarkable.
RAIN_POSITIVE_FRAC_MIN = 0.005
RAIN_POSITIVE_FRAC_MAX = 0.50


def check_baseline_monotonic(mae_by_horizon: dict[int, float], tol: float = 0.0) -> list[str]:
    """The NWP baseline's error MUST grow with forecast lead time.

    This is the check that would have caught the 2026-07-15 join bug: a baseline
    flat at 1.68 C across +1/+3/+24h is not a forecast, it is the same forecast
    scored three times. `tol` allows a small non-monotonic wobble from sampling
    noise between adjacent horizons.
    """
    violations: list[str] = []
    horizons = sorted(mae_by_horizon)
    for shorter, longer in zip(horizons, horizons[1:]):
        mae_s, mae_l = mae_by_horizon[shorter], mae_by_horizon[longer]
        if mae_l < mae_s - tol:
            violations.append(
                f"baseline MAE fell with lead time: +{shorter}h={mae_s:.3f} -> "
                f"+{longer}h={mae_l:.3f}. A longer-lead forecast cannot be more "
                f"accurate; suspect the forecast join is ignoring the horizon."
            )
    if len(horizons) >= 2:
        spread = max(mae_by_horizon.values()) - min(mae_by_horizon.values())
        if spread == 0.0:
            violations.append(
                f"baseline MAE is identical ({mae_by_horizon[horizons[0]]:.3f}) at every "
                f"horizon {horizons}. The same forecast row is being scored at all leads."
            )
    return violations


def check_forecast_lead(df: pd.DataFrame, horizon: int) -> list[str]:
    """Every row's forecast must predate its target hour by >= `horizon`.

    Guards the train/serve contract directly: predict.py can only ever see a
    forecast issued at or before `valid_time - horizon`, so a training row with a
    fresher forecast is a leak.
    """
    if df.empty or "forecast_time" not in df or "valid_time" not in df:
        return []
    lead_h = (df["valid_time"] - df["forecast_time"]).dt.total_seconds() / 3600.0
    short = lead_h < horizon
    if not short.any():
        return []
    return [
        f"{int(short.sum())}/{len(df)} rows have a forecast lead shorter than the "
        f"+{horizon}h horizon (min {lead_h.min():.2f}h). Training would see fresher "
        f"forecasts than serving can — train/serve skew."
    ]


def check_no_constant_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """A feature that never varies carries no signal and usually means a silent zero.

    The 2026-06-19 tz-strip zeroed every network column this way; the ablation still
    ran and every config scored identically.
    """
    violations: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        if len(values) == 0 or np.all(np.isnan(values)):
            violations.append(f"feature '{col}' is entirely NaN/empty.")
            continue
        if np.nanstd(values) == 0.0:
            violations.append(
                f"feature '{col}' is constant at {np.nanmin(values)!r} across "
                f"{len(values)} rows — silently zeroed or never populated?"
            )
    return violations


def check_physical_bounds(df: pd.DataFrame) -> list[str]:
    """Flag feature values no instrument could have produced."""
    violations: list[str] = []
    for col, (low, high) in FEATURE_BOUNDS.items():
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        bad = np.isfinite(values) & ((values < low) | (values > high))
        if bad.any():
            violations.append(
                f"feature '{col}': {int(bad.sum())}/{len(values)} rows outside the "
                f"physical range [{low}, {high}] (min {np.nanmin(values):.2f}, "
                f"max {np.nanmax(values):.2f})."
            )
    return violations


def check_rain_positive_fraction(y: np.ndarray, threshold_mm: float = 0.1) -> list[str]:
    """The wet-hour rate must be climatologically plausible.

    ~0 means the target is being derived from a column nobody populates (the
    2026-06-01 bug). Implausibly high means garbage is leaking past the clip.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return ["rain target is empty."]
    frac = float((y > threshold_mm).mean())
    if frac < RAIN_POSITIVE_FRAC_MIN:
        return [
            f"wet-hour fraction is {frac:.4%} ({int((y > threshold_mm).sum())}/{y.size} "
            f"rows > {threshold_mm}mm) — implausibly dry. Suspect the rain target is "
            f"being derived from a column the source does not populate."
        ]
    if frac > RAIN_POSITIVE_FRAC_MAX:
        return [
            f"wet-hour fraction is {frac:.2%} — implausibly wet for this climate. "
            f"Suspect stuck gauges leaking past the quality clip."
        ]
    return []
