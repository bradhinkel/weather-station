"""FeatureConfig — per-ablation knobs for the network feature pipeline.

All Phase 7.4 ablations (Q1–Q8) toggle these values. Code does not change
between ablation runs; the runner instantiates a different ``FeatureConfig``
and calls :func:`src.features.pipeline.build_features`.

Defaults are the "headline" configuration we would report if no ablation
beat the baseline — pre-registration cites these literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WindReference = Literal["own", "network_mean", "nwp"]
AggregationKernel = Literal["inverse_distance", "gaussian", "uniform"]


@dataclass(frozen=True)
class FeatureConfig:
    """Frozen so an instance can be hashed (cache keys) and accidentally
    shared between ablation runs without mutation risk."""

    # Q2 sweep: number of stations kept after sorting (by weight or distance).
    n_stations: int = 5

    # Q3 sweep: (low_km, high_km) inclusive-exclusive distance band.
    distance_band_km: tuple[float, float] = (0.0, 25.0)

    # Q4 sweep: half-width of the upwind / downwind cone (degrees).
    # Valid range: (0, 90]. At 90 every station is upwind OR downwind.
    angular_tolerance_deg: float = 30.0

    # Q5: also include downwind stations in the aggregate feature set.
    include_downwind: bool = False

    # Lag windows applied to upwind-station observations.
    lag_hours: tuple[int, ...] = (1, 3, 6, 12)

    # How to determine "current wind direction" for upwind/downwind class.
    # "own"          — home station wind_dir (subject to shelter effect; the
    #                  user's setup has ~81° measured offset, see memory).
    # "network_mean" — circular mean across nearby quality stations (default).
    # "nwp"          — Open-Meteo forecast at the prediction time.
    wind_reference: WindReference = "network_mean"

    # Radius for the "network_mean" wind reference. Defaults to 10km to stay
    # within Kirkland-local terrain; 25km spans Bellevue/Redmond/Bothell with
    # different microclimates. Ablate via this knob.
    wind_reference_radius_km: float = 10.0

    # Minimum quality stations required for network_mean to be considered
    # reliable. Below this count, pipeline falls back to "own" wind_dir.
    wind_reference_min_stations: int = 5

    # Aggregation kernel for distance-weighted means.
    aggregation_kernel: AggregationKernel = "inverse_distance"

    # Sigma for Gaussian kernel (km). Used only when kernel == "gaussian".
    gaussian_sigma_km: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 < self.angular_tolerance_deg <= 90.0):
            raise ValueError(
                f"angular_tolerance_deg must be in (0, 90], got {self.angular_tolerance_deg}"
            )
        lo, hi = self.distance_band_km
        if not (0.0 <= lo < hi):
            raise ValueError(
                f"distance_band_km must satisfy 0 <= low < high, got {self.distance_band_km}"
            )
        if self.n_stations < 1:
            raise ValueError(f"n_stations must be >= 1, got {self.n_stations}")
        if self.wind_reference_radius_km <= 0:
            raise ValueError(
                f"wind_reference_radius_km must be > 0, got {self.wind_reference_radius_km}"
            )
        if self.wind_reference_min_stations < 1:
            raise ValueError(
                f"wind_reference_min_stations must be >= 1, got {self.wind_reference_min_stations}"
            )
        if self.gaussian_sigma_km <= 0:
            raise ValueError(
                f"gaussian_sigma_km must be > 0, got {self.gaussian_sigma_km}"
            )
