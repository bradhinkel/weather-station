"""Upwind band-mean features — the ONE implementation used by training and serving.

This module exists in this shape on purpose. Two of the defects this project has shipped
were train/serve skew: the rain target was derived one way in `dataset.py` and another in
`predict.py` (2026-06-01), and the forecast join used a ~1h-lead forecast in training
while serving got a horizon-lead one (2026-07-15). Both were produced by the same logic
existing twice. So the band features exist *once*, as a pure function over plain values,
and both paths call it. If it is wrong, it is wrong identically on both sides, which is a
bug you can find rather than a skew you cannot.

**What it computes.** For a target hour `t+h`, take every network station whose bearing
from home is within `UPWIND_TOLERANCE_DEG` of the forecast wind direction (meteorological
convention: the direction the wind blows *from*, so upwind bearing == wind_dir). Group
them by distance from home, adjust each reading to home's elevation along the standard
lapse rate, and average within each band.

**Why bands rather than one radius.** Air moves: the parcel arriving at `t+h` is now
roughly `v·h` upwind, ~11 km/h at this site's mean wind. One fixed radius is therefore
correct at exactly one wind speed and one lead. Four bands enter as separate features so
the model can learn which reach matters when — instead of trusting a hand-built `v·h`
rule, which was tried (`src/features/advection.py`) and measured null.

**Why averaging rather than selecting.** Every CWS study finds 0.5-1.0 C of residual
per-station bias surviving quality control, so picking the single "right" station
maximises exposure to precisely the error that dominates this problem. Averaging removes
the random component as sqrt(N) and leaves the systematic part — which is why the
literature saturates at ~4 stations (Nipen et al. 2020) and why this project's own sweep
plateaued at n=1. Measured: band means at +3h cut own-station MAE from 0.798 to 0.718;
single-station selection cut nothing.

**Why elevation adjustment is not optional.** The registry spans 0-1174 m, some 7.6 C of
legitimate lapse rate. Averaging raw temperatures across a band lets one foothills station
drag the mean and call it weather.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from src.features.bearing import angular_distance
from src.pws.qc import LAPSE_RATE_C_PER_M

# Bands sized to advection reach at this site's prevailing wind (SW, mean 3.14 m/s ~
# 11 km/h): roughly +1h, +3h, +6h and +9h of travel.
BANDS_KM: tuple[tuple[float, float], ...] = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0))
UPWIND_TOLERANCE_DEG = 45.0

BAND_FEATURE_COLS: list[str] = [
    f"band{i}_{k}" for i in range(len(BANDS_KM)) for k in ("temp", "grad")
] + ["bands_n_total"]


def build_band_features(
    wind_dir_deg: float,
    lag_temp_c: float,
    home_elevation_m: float,
    station_geometry: Mapping[str, tuple[float, float, float]],
    temps_at_hour: Mapping[str, float],
) -> dict[str, float]:
    """Band means for ONE row.

    Args:
        wind_dir_deg: forecast wind direction for the target hour (degrees, blowing FROM).
        lag_temp_c: home's own reading at `t` — the reference for the gradient features.
        home_elevation_m: home's elevation; every band mean is adjusted to it.
        station_geometry: station_id -> (distance_km, bearing_deg, elevation_m).
        temps_at_hour: station_id -> temperature at `t`. Callers supply only the stations
            they want considered, so quality filtering happens at the boundary, not here.

    Returns a dict keyed by BAND_FEATURE_COLS. A band with no usable station is NaN — the
    caller imputes with a train-time mean. Never fill 0: a 0 C band mean is physically
    absurd and is the fill-0 defect that blew Ridge to MAE 20-30 in 2026-06.
    """
    n_bands = len(BANDS_KM)
    out: dict[str, float] = {}
    per_band: list[list[tuple[float, float]]] = [[] for _ in range(n_bands)]

    if np.isfinite(wind_dir_deg):
        for sid, (dist_km, bearing, elev) in station_geometry.items():
            if dist_km is None or bearing is None or elev is None:
                continue
            if angular_distance(bearing, wind_dir_deg) > UPWIND_TOLERANCE_DEG:
                continue
            temp = temps_at_hour.get(sid)
            if temp is None or not np.isfinite(temp):
                continue
            for bi, (lo, hi) in enumerate(BANDS_KM):
                if lo <= dist_km < hi:
                    per_band[bi].append((float(temp), float(elev)))
                    break

    total = 0
    for bi in range(n_bands):
        vals = per_band[bi]
        if vals:
            temps = np.array([v[0] for v in vals], dtype=float)
            elevs = np.array([v[1] for v in vals], dtype=float)
            # Bring every reading to home's elevation before averaging.
            adjusted = temps + LAPSE_RATE_C_PER_M * (home_elevation_m - elevs)
            mean = float(np.mean(adjusted))
            out[f"band{bi}_temp"] = mean
            out[f"band{bi}_grad"] = (
                mean - lag_temp_c if np.isfinite(lag_temp_c) else float("nan")
            )
            total += len(vals)
        else:
            out[f"band{bi}_temp"] = float("nan")
            out[f"band{bi}_grad"] = float("nan")

    out["bands_n_total"] = float(total)
    return out


def band_fill_values(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Train-time mean per band column, for imputing gaps at serve time.

    Computed from the training rows and persisted in the model bundle so serving fills
    with the same constants training saw. Deriving them at serve time from live data
    would be a fresh train/serve skew — the thing this module exists to prevent.
    """
    rows = list(rows)
    fills: dict[str, float] = {}
    for col in BAND_FEATURE_COLS:
        vals = np.array([r.get(col, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        fills[col] = float(vals.mean()) if vals.size else 0.0
    return fills


def apply_band_fills(
    feat: dict[str, float], fills: Optional[Mapping[str, float]]
) -> dict[str, float]:
    """Replace NaN band cells with the persisted train-time means."""
    if not fills:
        return feat
    out = dict(feat)
    for col in BAND_FEATURE_COLS:
        v = out.get(col)
        if v is None or not np.isfinite(v):
            out[col] = float(fills.get(col, 0.0))
    return out
