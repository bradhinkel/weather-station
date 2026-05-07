"""WMO weather code → icon slug + plain-language label, with local-sensor overrides.

The Open-Meteo forecast already discretizes conditions via the WMO 4677 code set;
we map those codes to a small set of icon slugs that exist in basmilius/weather-icons
(static-fill set, vendored under static/icons/weather/). Day/night variants are
appended for slugs that have them.

Two optional local-sensor overrides defend against obvious mis-categorizations:
  A) actual rain in the gauge → escalate a "clear/partly" code to "rain"
  B) clear-sky solar shortfall during the day → demote a "clear/partly" code to "overcast"
"""

from __future__ import annotations

from datetime import datetime
from math import radians, sin
from typing import Optional

from pysolar.solar import get_altitude

# WMO 4677 → icon slug (without -day/-night suffix)
WMO_TO_SLUG: dict[int, str] = {
    0:  "clear",
    1:  "partly-cloudy",
    2:  "partly-cloudy",
    3:  "overcast",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle",
    56: "sleet",   57: "sleet",
    61: "rain",    63: "rain",    65: "rain",
    66: "sleet",   67: "sleet",
    71: "snow",    73: "snow",    75: "snow", 77: "snow",
    80: "rain",    81: "rain",    82: "rain",
    85: "snow",    86: "snow",
    95: "thunderstorms",
    96: "thunderstorms-rain", 99: "thunderstorms-rain",
}

# Slugs that have distinct day/night variants in the icon set
DAY_NIGHT_SLUGS = {"clear", "partly-cloudy"}

# Plain-language labels keyed by full slug (with -day/-night where applicable)
COND_LABELS: dict[str, str] = {
    "clear-day":           "Clear",
    "clear-night":         "Clear",
    "partly-cloudy-day":   "Partly cloudy",
    "partly-cloudy-night": "Partly cloudy",
    "overcast":            "Overcast",
    "fog":                 "Fog",
    "drizzle":             "Drizzle",
    "rain":                "Rain",
    "sleet":               "Sleet",
    "snow":                "Snow",
    "thunderstorms":       "Thunderstorms",
    "thunderstorms-rain":  "Thunderstorms with rain",
}


def is_day(lat: float, lon: float, t: datetime) -> bool:
    """True when the sun is above the horizon at (lat, lon, t)."""
    return get_altitude(lat, lon, t) > 0


def clear_sky_solar(lat: float, lon: float, t: datetime) -> float:
    """Watts per m² assuming a perfectly clear sky. Floors at 0 below the horizon."""
    elev = get_altitude(lat, lon, t)
    return 1361.0 * max(0.0, sin(radians(elev)))


def pick_icon_slug(
    *,
    weather_code: Optional[int],
    lat: float,
    lon: float,
    t: datetime,
    rain_mm_1h: Optional[float] = None,
    solar_wm2: Optional[float] = None,
) -> str:
    """Return an icon slug like "partly-cloudy-day" or "rain"."""
    base = WMO_TO_SLUG.get(weather_code, "overcast") if weather_code is not None else "overcast"

    # Override A: actual rain in the gauge wins over a clear-ish forecast code.
    # Gate on rain_mm_1h being a real number (the Ecowitt reports the rain field
    # intermittently — None != 0).
    if (
        rain_mm_1h is not None
        and rain_mm_1h >= 0.1
        and base in {"clear", "partly-cloudy"}
    ):
        base = "rain"

    # Override B: clear-sky solar shortfall during the day → demote to overcast.
    day = is_day(lat, lon, t)
    if (
        day
        and solar_wm2 is not None
        and base in {"clear", "partly-cloudy"}
    ):
        expected = clear_sky_solar(lat, lon, t)
        # Only check when expected is large enough that a low reading is meaningful;
        # at low sun angles the ratio is too noisy.
        if expected > 50 and (solar_wm2 / expected) < 0.4:
            base = "overcast"

    if base in DAY_NIGHT_SLUGS:
        return f"{base}-{'day' if day else 'night'}"
    return base


def cond_label(slug: str) -> str:
    """Plain-language label for an icon slug. Falls back to the slug itself."""
    return COND_LABELS.get(slug, slug.replace("-", " ").capitalize())
