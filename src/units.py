"""Small unit / derived-quantity helpers shared across the project."""

from __future__ import annotations

from math import exp
from typing import Optional


def apparent_temperature(
    temp_c: Optional[float],
    humidity_pct: Optional[float],
    wind_ms: Optional[float],
) -> Optional[float]:
    """Australian Bureau of Meteorology apparent-temperature ("feels like"), °C.

    Single unified formula across hot/mild/cold regimes — no piecewise switching.
    Returns None if any required input is missing.

    Reference: http://www.bom.gov.au/info/thermal_stress/#atapproximation
    """
    if temp_c is None or humidity_pct is None or wind_ms is None:
        return None
    e = (humidity_pct / 100.0) * 6.105 * exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * e - 0.70 * wind_ms - 4.00
