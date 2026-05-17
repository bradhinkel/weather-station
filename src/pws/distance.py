"""Great-circle distance and forward bearing on the WGS-84 sphere.

Adequate for the < 100 km radius this project uses; for ablation work we
don't need ellipsoidal precision (Vincenty's, etc.).
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) pairs in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from (lat1, lon1) to (lat2, lon2), 0° = north, clockwise."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination_point(
    lat: float, lon: float, distance_km: float, bearing: float,
) -> tuple[float, float]:
    """Inverse of haversine: starting at (lat, lon), travel `distance_km` along
    forward azimuth `bearing` (degrees, 0°=N). Used to generate the synthetic
    grid origins that defeat WU's top-N response cap.
    """
    angular = distance_km / EARTH_RADIUS_KM
    brg = math.radians(bearing)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(angular)
        + math.cos(phi1) * math.sin(angular) * math.cos(brg)
    )
    lam2 = lam1 + math.atan2(
        math.sin(brg) * math.sin(angular) * math.cos(phi1),
        math.cos(angular) - math.sin(phi1) * math.sin(phi2),
    )
    # Normalize longitude to [-180, 180].
    lon_out = ((math.degrees(lam2) + 540.0) % 360.0) - 180.0
    return math.degrees(phi2), lon_out
