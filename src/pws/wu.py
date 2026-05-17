"""Weather Underground PWS client — Phase 7.1.

Requires a WU API key. The key is obtained from
https://www.wunderground.com/member/api-keys after creating an account and
registering a contributing PWS (free-tier eligibility).

API endpoints used (Sun/IBM weather.com gateway):
  - Station search:  GET https://api.weather.com/v3/location/near
                       params: geocode=lat,lon, product=PWS, format=json, apiKey=...
  - Current obs:     GET https://api.weather.com/v2/pws/observations/current
                       params: stationId=..., format=json, units=m, apiKey=...
  - Recent history:  GET https://api.weather.com/v2/pws/observations/hourly/7day
                       params: stationId=..., format=json, units=m, apiKey=...

These URLs are documented in WU's "PWS Observations" API reference. Verify
the response shape on first use — Sun/IBM has rotated field names a couple
of times. If a field name has drifted, fix it in `_parse_*` rather than
adding a translation layer.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.pws.base import NetworkObservation, PWSSource, StationInfo
from src.pws.distance import bearing_deg, haversine_km

logger = logging.getLogger(__name__)

_BASE = "https://api.weather.com"
_TIMEOUT = 30.0
_DISCOVER_PATH = "/v3/location/near"
_CURRENT_PATH = "/v2/pws/observations/current"
_HOURLY_7D_PATH = "/v2/pws/observations/hourly/7day"


class WUKeyMissing(RuntimeError):
    """Raised when WU_API_KEY isn't in the environment.

    Discover/ingest CLIs catch this and print a help message pointing at
    https://www.wunderground.com/member/api-keys.
    """


class WUSource(PWSSource):
    """Weather Underground PWS provider.

    Network calls are async (httpx). The client is stateless beyond the
    cached key + http client — instantiate fresh per CLI invocation rather
    than keeping a long-lived instance.
    """

    name = "wu"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("WU_API_KEY")
        if not self.api_key:
            raise WUKeyMissing(
                "WU_API_KEY not set. Obtain a key at "
                "https://www.wunderground.com/member/api-keys and add it to .env."
            )

    async def _get(self, path: str, params: dict) -> dict:
        params = {**params, "apiKey": self.api_key, "format": "json"}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_BASE + path, params=params)
            resp.raise_for_status()
            return resp.json()

    async def discover_stations(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
    ) -> list[StationInfo]:
        """Return all PWS within `radius_km` of the given lat/lon.

        WU's /v3/location/near returns a small set per call; we filter to the
        requested radius locally rather than trusting the provider's default.
        """
        data = await self._get(
            _DISCOVER_PATH,
            {"geocode": f"{center_lat},{center_lon}", "product": "PWS"},
        )
        return [
            info
            for info in self._parse_discover(data, center_lat, center_lon)
            if info.distance_km is not None and info.distance_km <= radius_km
        ]

    async def fetch_observations(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
    ) -> list[NetworkObservation]:
        """Fetch hourly observations from WU.

        WU's free-tier history depth is 7 days via the hourly/7day endpoint.
        Larger windows require chunked daily-history calls; we don't have
        that need today. If `start` is older than 7 days from now, we clamp
        with a warning and return what's available.
        """
        now = datetime.now(timezone.utc)
        if (now - start).days > 7:
            logger.warning(
                "WU hourly/7day clamp: start=%s is more than 7d old; using last 7d window.",
                start.isoformat(),
            )
        data = await self._get(_HOURLY_7D_PATH, {"stationId": station_id, "units": "m"})
        return [
            obs
            for obs in self._parse_hourly(data, station_id)
            if start <= obs.time < end
        ]

    # -- response parsing -------------------------------------------------
    # WU/Sun field names have rotated; if the API returns 200 but parsing
    # yields zero rows, log the raw dict in a one-off run before touching
    # these methods.

    def _parse_discover(
        self,
        data: dict,
        center_lat: float,
        center_lon: float,
    ) -> list[StationInfo]:
        """Convert /v3/location/near response into StationInfo list.

        Response shape (last verified <date_TBD>):
          {
            "location": {
              "stationId":     [...],
              "stationName":   [...],
              "latitude":      [...],
              "longitude":     [...],
              "elevation":     [...],
              ...
            }
          }
        Re-validate on first live call; correct here if the field names drift.
        """
        loc = data.get("location") or {}
        ids = loc.get("stationId") or []
        names = loc.get("stationName") or [None] * len(ids)
        lats = loc.get("latitude") or [None] * len(ids)
        lons = loc.get("longitude") or [None] * len(ids)
        elevs = loc.get("elevation") or [None] * len(ids)
        out: list[StationInfo] = []
        for i, sid in enumerate(ids):
            lat = lats[i]
            lon = lons[i]
            if lat is None or lon is None:
                continue
            out.append(StationInfo(
                station_id=sid,
                source=self.name,
                lat=float(lat),
                lon=float(lon),
                name=names[i] if i < len(names) else None,
                elevation_m=float(elevs[i]) if i < len(elevs) and elevs[i] is not None else None,
                distance_km=haversine_km(center_lat, center_lon, float(lat), float(lon)),
                bearing_deg=bearing_deg(center_lat, center_lon, float(lat), float(lon)),
            ))
        return out

    def _parse_hourly(self, data: dict, station_id: str) -> list[NetworkObservation]:
        """Convert /v2/pws/observations/hourly/7day response.

        Response shape (last verified <date_TBD>):
          {
            "observations": [
              {
                "obsTimeUtc":      "2026-05-17T05:00:00Z",
                "metric": {
                  "tempAvg":       12.3,
                  "windspeedAvg":  1.2,
                  "windgustAvg":   2.0,
                  "winddirAvg":    180,
                  "pressureMax":   1015.2,
                  "humidityAvg":   55,
                  "precipTotal":   0.0,
                  "precipRate":    0.0,
                  ...
                },
                "solarRadiationHigh": 200.0,
                "uvHigh": 5,
                ...
              }, ...
            ]
          }
        """
        out: list[NetworkObservation] = []
        for row in data.get("observations") or []:
            ts_raw = row.get("obsTimeUtc")
            if not ts_raw:
                continue
            t = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            m = row.get("metric") or {}
            out.append(NetworkObservation(
                time=t,
                station_id=station_id,
                source=self.name,
                temp_c=_f(m.get("tempAvg")),
                humidity_pct=_f(m.get("humidityAvg") or row.get("humidityAvg")),
                pressure_hpa=_f(m.get("pressureMax") or m.get("pressureTrend")),
                wind_speed_ms=_kmh_to_ms(m.get("windspeedAvg")),
                wind_dir_deg=_f(m.get("winddirAvg") or row.get("winddirAvg")),
                wind_gust_ms=_kmh_to_ms(m.get("windgustAvg")),
                rain_mm_1h=_f(m.get("precipTotal")),
                rain_rate_mm_hr=_f(m.get("precipRate")),
                solar_wm2=_f(row.get("solarRadiationHigh")),
                uv_index=_f(row.get("uvHigh")),
            ))
        return out


def _f(v) -> Optional[float]:
    return float(v) if v is not None else None


def _kmh_to_ms(v) -> Optional[float]:
    """WU returns wind speeds in km/h when units=m. Convert to m/s."""
    f = _f(v)
    return f / 3.6 if f is not None else None
