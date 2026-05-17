"""Source abstraction for network PWS providers.

PHASE_7_PLAN.md calls for "source-abstraction layer so WU vs PWSWeather
is swappable without touching feature code." Concrete clients (`src.pws.wu`,
`src.pws.pwsweather`) implement this base and downstream code (registry,
ingest CLI, feature pipeline) only depends on the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StationInfo:
    """One discovered station from a provider.

    distance_km / bearing_deg are computed by the discover_stations() caller
    using src.pws.distance, not by the provider — the provider only needs to
    return location, the geometry is universal.
    """
    station_id: str
    source: str
    lat: float
    lon: float
    name: Optional[str] = None
    elevation_m: Optional[float] = None
    distance_km: Optional[float] = None
    bearing_deg: Optional[float] = None
    sensor_flags: dict = field(default_factory=dict)


@dataclass
class NetworkObservation:
    """One hourly-ish observation from a network station.

    Field set mirrors the `observations` table so persistence is a 1:1 map.
    Missing fields stay None — providers vary in what they report.
    """
    time: datetime
    station_id: str
    source: str
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wind_gust_ms: Optional[float] = None
    rain_mm_1h: Optional[float] = None
    rain_rate_mm_hr: Optional[float] = None
    solar_wm2: Optional[float] = None
    uv_index: Optional[float] = None


class PWSSource(ABC):
    """Abstract provider. Concrete subclasses: WUSource, PWSWeatherSource."""

    #: Short string written into observations.source / stations.source.
    name: str

    @abstractmethod
    async def discover_stations(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
    ) -> list[StationInfo]:
        """Enumerate stations within `radius_km` of (center_lat, center_lon).

        Concrete implementations are responsible for paginating the provider
        API; the caller assumes the returned list is complete.
        """

    @abstractmethod
    async def fetch_observations(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
    ) -> list[NetworkObservation]:
        """Fetch observations in [start, end). Implementations clamp to the
        provider's historical-depth limit and may return fewer rows.
        """
