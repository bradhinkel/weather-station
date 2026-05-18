"""Distance-weighted aggregation — Phase 7.2.

Pure functions on values + distances; no DB, no pandas requirement beyond
optional convenience. Used by the pipeline to reduce a station cohort into a
single feature value (weighted mean) per (target_time, lag, field).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from src.features.config import AggregationKernel

# Minimum distance clamp (km) for the inverse-distance kernel — prevents
# divide-by-zero if a network station ends up reporting the home station's
# exact lat/lon (rare, but happened once during the wide-discover grid sweep).
_MIN_DISTANCE_KM = 0.1


def kernel_weights(
    distances_km: Iterable[float],
    kernel: AggregationKernel,
    gaussian_sigma_km: float = 5.0,
) -> np.ndarray:
    """Return unnormalized weights, one per station, given distance in km.

    Callers normalize after dropping NaN values so we keep this stateless.
    """
    d = np.asarray(list(distances_km), dtype=float)
    if kernel == "uniform":
        return np.ones_like(d)
    if kernel == "inverse_distance":
        return 1.0 / np.maximum(d, _MIN_DISTANCE_KM)
    if kernel == "gaussian":
        if gaussian_sigma_km <= 0:
            raise ValueError("gaussian_sigma_km must be > 0")
        return np.exp(-0.5 * (d / gaussian_sigma_km) ** 2)
    raise ValueError(f"unknown kernel: {kernel!r}")


def weighted_mean(
    values: Iterable[float],
    weights: Iterable[float],
) -> Optional[float]:
    """NaN-aware weighted mean.

    Drops (value, weight) pairs where either is NaN or weight <= 0.
    Returns None if no usable pairs remain.
    """
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if v.shape != w.shape:
        raise ValueError(
            f"values and weights must have equal length, got {v.shape} vs {w.shape}"
        )
    mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not mask.any():
        return None
    return float(np.average(v[mask], weights=w[mask]))
