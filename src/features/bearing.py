"""Pure bearing / wind-direction helpers — Phase 7.2.

No database dependency: every function here takes plain numbers (or
sequences of them) so the test suite can cover the angular logic in
isolation. Convention throughout:

- Bearings are degrees clockwise from north, [0, 360).
- Wind direction follows the meteorological convention: ``wind_from_deg``
  is the direction the wind is *coming from*. North wind has wind_from = 0°
  and pushes air southward.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

# Numerical tolerance for diametric cancellation in circular_mean.
_CANCEL_EPS = 1e-9


def angular_distance(a_deg: float, b_deg: float) -> float:
    """Shortest unsigned angular distance between two bearings, in [0, 180]."""
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff)


def circular_mean(angles_deg: Iterable[float]) -> Optional[float]:
    """Mean of bearings via sin/cos averaging.

    Returns ``None`` if the input is empty, all-NaN, or the vectors cancel
    out (e.g., perfectly diametrically opposed pairs). The cancel case is a
    real degenerate situation, not a numerical artifact — refusing to return
    a meaningless 0° is the right call.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    n = 0
    for a in angles_deg:
        if a is None:
            continue
        try:
            af = float(a)
        except (TypeError, ValueError):
            continue
        if math.isnan(af):
            continue
        r = math.radians(af)
        sin_sum += math.sin(r)
        cos_sum += math.cos(r)
        n += 1
    if n == 0:
        return None
    if abs(sin_sum) < _CANCEL_EPS and abs(cos_sum) < _CANCEL_EPS:
        return None
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def direction_class(
    station_bearing_deg: float,
    wind_from_deg: float,
    tolerance_deg: float,
) -> str:
    """Classify a station as ``"upwind"``, ``"crosswind"``, ``"downwind"``,
    or ``"unknown"`` (when either input is NaN — calm wind for instance).

    ``tolerance_deg`` is the half-width of each cone:
      - upwind:    angular_distance(station, wind_from) <= tolerance
      - downwind:  angular_distance(station, wind_from) >= 180 - tolerance
      - crosswind: everything else.

    A station whose bearing equals ``wind_from`` sits exactly in the direction
    the wind is blowing from — i.e., upwind of home. Wind convention is
    meteorological (``wind_from``).
    """
    if math.isnan(station_bearing_deg) or math.isnan(wind_from_deg):
        return "unknown"
    diff = angular_distance(station_bearing_deg, wind_from_deg)
    if diff <= tolerance_deg:
        return "upwind"
    if diff >= 180.0 - tolerance_deg:
        return "downwind"
    return "crosswind"
