"""Spatial-gradient features — Phase 7.2.

For each target hour, computes ``mean(far_band) − mean(near_band)`` along
the **upwind** axis. Captures advection signal beyond what cohort means
alone can express: e.g., temperature falling along the wind is the
fingerprint of an incoming cold front.

Pure function — no DB. The caller hands in an obs slice (one row per
station, all at the same target hour) with distance_km and bearing_deg
attached.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.features.aggregation import kernel_weights, weighted_mean
from src.features.bearing import direction_class
from src.features.config import FeatureConfig

# Fields gradient is computed for. rain_mm_1h is excluded — rain is bursty
# at the hour scale and a "rain gradient" is more noise than signal. Add
# it later if a Q-question motivates.
GRADIENT_FIELDS: tuple[str, ...] = (
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
)


def compute_upwind_gradient(
    obs_at_hour: pd.DataFrame,
    wind_from_deg: float,
    config: FeatureConfig,
    fields: Iterable[str] = GRADIENT_FIELDS,
) -> dict[str, Optional[float]]:
    """Return ``{field: far_minus_near}`` per requested field, or ``None``
    where either band lacks usable data.

    ``obs_at_hour`` must have columns: ``station_id``, ``distance_km``,
    ``bearing_deg``, and each requested field. All rows are assumed to be
    from the same target hour (the caller pre-slices by time).
    """
    fields = tuple(fields)
    none_result: dict[str, Optional[float]] = {f: None for f in fields}

    if obs_at_hour.empty:
        return none_result

    upwind_mask = obs_at_hour["bearing_deg"].apply(
        lambda b: pd.notna(b)
        and direction_class(b, wind_from_deg, config.angular_tolerance_deg) == "upwind"
    )
    upwind = obs_at_hour[upwind_mask]
    if upwind.empty:
        return none_result

    near_lo, near_hi = config.gradient_near_band_km
    far_lo, far_hi = config.gradient_far_band_km
    near = upwind[
        (upwind["distance_km"] >= near_lo) & (upwind["distance_km"] < near_hi)
    ]
    far = upwind[
        (upwind["distance_km"] >= far_lo) & (upwind["distance_km"] < far_hi)
    ]
    if near.empty or far.empty:
        return none_result

    near_w = kernel_weights(
        near["distance_km"].tolist(),
        kernel=config.aggregation_kernel,
        gaussian_sigma_km=config.gaussian_sigma_km,
    )
    far_w = kernel_weights(
        far["distance_km"].tolist(),
        kernel=config.aggregation_kernel,
        gaussian_sigma_km=config.gaussian_sigma_km,
    )

    result: dict[str, Optional[float]] = {}
    for f in fields:
        near_val = weighted_mean(near[f].values, near_w)
        far_val = weighted_mean(far[f].values, far_w)
        if near_val is None or far_val is None:
            result[f] = None
        else:
            result[f] = float(far_val - near_val)
    return result
