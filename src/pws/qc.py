"""Reference-independent quality control for crowdsourced station data.

Every published CWS study discards a large fraction of the raw data — Nipen et al.
(2020) 21%, Meier et al. (2017) 53%, Fenner et al. (2021, CrowdQC+) ~70% — and Nipen
reports the punchline this module exists for:

    without QC, the merged citizen-observation product is only marginally better than
    raw NWP, *and worse in daytime and summer*.

This project currently discards ~nothing beyond stuck rain gauges, and its network
features measure null-to-harmful on a summer, daytime-heavy test window. That is the
exact condition the literature predicts.

Design, and where it departs from the published schemes:

* **Reference-independent.** Berlin's CrowdQC leans on the crowd median; TITAN leans on
  official stations. There is no WMO-compliant reference here, so every test compares a
  station against its neighbours ("wisdom of the crowd", Napoly et al. 2018).

* **Local buddies via k-nearest, not a fixed radius.** CrowdQC+ uses >=5 buddies within
  3 km because Berlin and Toulouse pack 500-2000 stations into a city. This network is
  322 stations over a 100 km radius — roughly one per 97 km², ~10 km mean spacing — so a
  3 km rule would isolate nearly everything. kNN adapts to density; a max radius keeps
  "neighbour" physically meaningful.

* **Elevation-adjusted, and this is not optional here.** The registry spans 0-1174 m,
  which is ~7.6 °C of legitimate lapse-rate spread. Comparing raw temperatures across it
  would flag the foothills as broken sensors. Buddies are adjusted to the target
  station's elevation before comparison.

* **Asymmetric thresholds.** CWS error is not symmetric: poorly-shielded sensors in
  still air read *high* in daylight, and every study finds a residual warm bias
  surviving QC (Nipen +0.5 °C, Napoly +0.95 K Berlin, Fenner 0.5-1.0 K daytime summer).
  So a warm deviation is more likely an artifact than an equally-sized cold one, which
  may be real cold-air pooling. TITAN encodes this as SCT thresholds of 4 warm vs 8
  cold; CrowdQC uses asymmetric tails. Same idea here.

* **Per-hour and time-independent.** Following Nipen: QC is re-run each hour with no
  memory, so a station may be rejected at 13:00 and accepted at 05:00. This is
  deliberate — it lets a badly-sited station contribute at night, when its siting does
  not bite, instead of banning it outright. Persistent-offender detection is a separate,
  explicit step (`station_flag_fraction`), not a side effect of the hourly test.

Everything here is pure: values in, flags out, no DB and no clock. The runner
(`tools/run_qc.py`) owns the I/O. That split is deliberate — every data defect this
project has shipped lived in SQL and joins, never in a pure function.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

# Standard atmosphere lapse rate. Nipen et al. use the same value in TITAN's
# observation operator. Negative: temperature falls as you climb.
LAPSE_RATE_C_PER_M = -0.0065

# Scale factor making MAD a consistent estimator of sigma for normal data.
MAD_TO_SIGMA = 1.4826

# Asymmetric outlier thresholds, in robust sigmas. Warm is stricter because CWS
# radiative error is warm-signed; a cold excursion is likelier to be real weather.
WARM_THRESHOLD_SIGMA = 3.5
COLD_THRESHOLD_SIGMA = 6.0

# Buddy geometry, tuned to a sparse 100 km network rather than a dense city.
BUDDY_K = 8
BUDDY_MAX_RADIUS_KM = 25.0
BUDDY_MIN_COUNT = 3

# A station whose readings are flagged this often is not having a bad hour; it is a bad
# station. Napoly et al. drop a station-month at >20% flagged.
STATION_BAD_FLAG_FRACTION = 0.20

# Pearson R against the buddy median. Indoor sensors track the diurnal cycle weakly or
# not at all; CrowdQC's m4 uses R < 0.9 for exactly this.
MIN_BUDDY_CORRELATION = 0.90


def elevation_adjust(
    temps_c: np.ndarray,
    elevations_m: np.ndarray,
    to_elevation_m: float,
    lapse_rate: float = LAPSE_RATE_C_PER_M,
) -> np.ndarray:
    """Move temperatures to a common elevation along the lapse rate.

    A station 100 m above the reference reads ~0.65 C cold *legitimately*; adjusting it
    down to the reference must add that back. Sign check: to_elevation > station
    elevation means the reference is higher, so the adjusted value must be cooler.
    """
    temps_c = np.asarray(temps_c, dtype=float)
    elevations_m = np.asarray(elevations_m, dtype=float)
    return temps_c + lapse_rate * (to_elevation_m - elevations_m)


def robust_center_spread(values: Iterable[float]) -> tuple[float, float]:
    """Median and MAD-derived sigma, ignoring NaN.

    Median/MAD rather than mean/SD because the outliers being hunted would otherwise
    inflate the very spread used to detect them.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(v))
    mad = float(np.median(np.abs(v - center)))
    return center, MAD_TO_SIGMA * mad


def flag_outlier(
    value: float,
    center: float,
    spread: float,
    warm_sigma: float = WARM_THRESHOLD_SIGMA,
    cold_sigma: float = COLD_THRESHOLD_SIGMA,
) -> bool:
    """Is `value` an outlier against a robust centre, asymmetrically?

    A zero spread means the buddies are identical — with real sensors at differing
    sites that indicates a stuck or duplicated feed, not perfect agreement, so no
    outlier call is made and the caller is left to the isolation/correlation tests.
    """
    if not np.isfinite(value) or not np.isfinite(center) or not np.isfinite(spread):
        return False
    if spread <= 0:
        return False
    z = (value - center) / spread
    return bool(z > warm_sigma or z < -cold_sigma)


def buddy_check_hour(
    station_temp_c: float,
    station_elevation_m: float,
    buddy_temps_c: Iterable[float],
    buddy_elevations_m: Iterable[float],
    min_buddies: int = BUDDY_MIN_COUNT,
    warm_sigma: float = WARM_THRESHOLD_SIGMA,
    cold_sigma: float = COLD_THRESHOLD_SIGMA,
) -> tuple[bool, float]:
    """One station, one hour, against its elevation-adjusted buddies.

    Returns (flagged, z_score). z is NaN when the test could not be run — too few
    buddies, or a degenerate spread. `flagged` is False in those cases: an untestable
    hour is not evidence of a bad hour.
    """
    temps = np.asarray(list(buddy_temps_c), dtype=float)
    elevs = np.asarray(list(buddy_elevations_m), dtype=float)
    ok = np.isfinite(temps) & np.isfinite(elevs)
    temps, elevs = temps[ok], elevs[ok]

    if temps.size < min_buddies or not np.isfinite(station_temp_c):
        return False, float("nan")

    adjusted = elevation_adjust(temps, elevs, station_elevation_m)
    center, spread = robust_center_spread(adjusted)
    if not np.isfinite(spread) or spread <= 0:
        return False, float("nan")

    z = (station_temp_c - center) / spread
    return flag_outlier(station_temp_c, center, spread, warm_sigma, cold_sigma), float(z)


def station_flag_fraction(flags: Iterable[bool]) -> float:
    """Share of testable hours a station was flagged. NaN if nothing was testable."""
    f = np.asarray(list(flags), dtype=bool)
    if f.size == 0:
        return float("nan")
    return float(f.mean())


def buddy_correlation(
    station_series: Iterable[float],
    buddy_median_series: Iterable[float],
    min_overlap: int = 100,
) -> float:
    """Pearson R between a station and its buddy median over time.

    The indoor-sensor detector (CrowdQC m4). An indoor station is climatologically
    plausible hour by hour — it will pass the outlier test — but it does not track the
    outdoor diurnal cycle, so its correlation collapses. Returns NaN below `min_overlap`
    paired points rather than a confident number from a handful.
    """
    a = np.asarray(list(station_series), dtype=float)
    b = np.asarray(list(buddy_median_series), dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_overlap:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def classify_station(
    flag_fraction: float,
    correlation: float,
    n_buddies: int,
    min_buddies: int = BUDDY_MIN_COUNT,
    max_flag_fraction: float = STATION_BAD_FLAG_FRACTION,
    min_correlation: float = MIN_BUDDY_CORRELATION,
) -> tuple[str, Optional[str]]:
    """Verdict for one station. Returns (status, reason).

    status is "ok", "suspect", or "isolated". Order matters: isolation is checked first
    because an isolated station's other statistics are untrustworthy, not passing.
    """
    if n_buddies < min_buddies:
        return "isolated", f"only {n_buddies} buddies within {BUDDY_MAX_RADIUS_KM:.0f}km"
    if np.isfinite(correlation) and correlation < min_correlation:
        return "suspect", f"correlation {correlation:.2f} < {min_correlation} (indoor?)"
    if np.isfinite(flag_fraction) and flag_fraction > max_flag_fraction:
        return "suspect", f"{flag_fraction:.0%} of hours flagged > {max_flag_fraction:.0%}"
    return "ok", None
