"""Network PWS (personal weather station) integration — Phase 7.1.

Provides:
  - distance / bearing helpers
  - PWSSource abstract base (so WU + PWSWeather are swappable)
  - station-registry CRUD against the extended `stations` table
  - concrete clients (WU, PWSWeather)
  - CLI: `python -m src.pws.cli {list|discover|ingest}`
"""

from src.pws.base import NetworkObservation, PWSSource, StationInfo
from src.pws.distance import bearing_deg, haversine_km

__all__ = [
    "NetworkObservation",
    "PWSSource",
    "StationInfo",
    "bearing_deg",
    "haversine_km",
]
