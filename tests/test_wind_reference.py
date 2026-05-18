"""Unit tests for src.features.wind_reference.

Synthetic in-memory DataFrames — no DB. Validates mode dispatch and the
fallback policy from network_mean → own.
"""

import unittest
from datetime import datetime, timezone

import pandas as pd

from src.features.config import FeatureConfig
from src.features.wind_reference import resolve_wind_reference


def _own_frame(hour: pd.Timestamp, value: float | None) -> pd.DataFrame:
    return pd.DataFrame({"time_hour": [hour], "wind_dir_deg": [value]})


def _network_frame(hour: pd.Timestamp, rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """rows is [(station_id, wind_dir_deg, distance_km), ...]."""
    return pd.DataFrame(
        {
            "station_id": [r[0] for r in rows],
            "time_hour": [hour for _ in rows],
            "wind_dir_deg": [r[1] for r in rows],
            "distance_km": [r[2] for r in rows],
        }
    )


def _nwp_frame(hour: pd.Timestamp, value: float | None) -> pd.DataFrame:
    return pd.DataFrame({"valid_time": [hour], "wind_dir_deg": [value]})


class TestWindReferenceModes(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        self.hour = pd.Timestamp("2026-05-18 12:00", tz="UTC")

    def test_own_mode(self):
        cfg = FeatureConfig(wind_reference="own")
        result = resolve_wind_reference(
            obs_hourly=pd.DataFrame(),
            forecasts=pd.DataFrame(),
            own_station_obs=_own_frame(self.hour, 200.0),
            target_time=self.t,
            config=cfg,
        )
        self.assertAlmostEqual(result, 200.0, places=5)

    def test_own_missing(self):
        cfg = FeatureConfig(wind_reference="own")
        result = resolve_wind_reference(
            obs_hourly=pd.DataFrame(),
            forecasts=pd.DataFrame(),
            own_station_obs=pd.DataFrame(columns=["time_hour", "wind_dir_deg"]),
            target_time=self.t,
            config=cfg,
        )
        self.assertIsNone(result)

    def test_nwp_mode(self):
        cfg = FeatureConfig(wind_reference="nwp")
        result = resolve_wind_reference(
            obs_hourly=pd.DataFrame(),
            forecasts=_nwp_frame(self.hour, 270.0),
            own_station_obs=pd.DataFrame(),
            target_time=self.t,
            config=cfg,
        )
        self.assertAlmostEqual(result, 270.0, places=5)

    def test_network_mean_sufficient(self):
        # 6 stations within 10km, all ~200° → mean ≈ 200°.
        cfg = FeatureConfig(
            wind_reference="network_mean",
            wind_reference_radius_km=10.0,
            wind_reference_min_stations=5,
        )
        rows = [
            ("a", 198.0, 1.0),
            ("b", 199.0, 2.0),
            ("c", 200.0, 3.0),
            ("d", 201.0, 4.0),
            ("e", 202.0, 5.0),
            ("f", 200.0, 8.0),
        ]
        result = resolve_wind_reference(
            obs_hourly=_network_frame(self.hour, rows),
            forecasts=pd.DataFrame(),
            own_station_obs=_own_frame(self.hour, 119.0),  # would be returned if fallback fired
            target_time=self.t,
            config=cfg,
        )
        self.assertIsNotNone(result)
        # Within ~2° of 200.
        self.assertLess(abs(result - 200.0), 2.0)

    def test_network_mean_falls_back_when_too_few(self):
        # Only 2 stations within 10km; min_stations=5 → fall back to own.
        cfg = FeatureConfig(
            wind_reference="network_mean",
            wind_reference_radius_km=10.0,
            wind_reference_min_stations=5,
        )
        rows = [("a", 200.0, 1.0), ("b", 210.0, 2.0)]
        result = resolve_wind_reference(
            obs_hourly=_network_frame(self.hour, rows),
            forecasts=pd.DataFrame(),
            own_station_obs=_own_frame(self.hour, 119.0),
            target_time=self.t,
            config=cfg,
        )
        self.assertAlmostEqual(result, 119.0, places=5)

    def test_network_mean_excludes_far_stations(self):
        # 10 stations exist but only 1 within 10km radius → fall back.
        cfg = FeatureConfig(
            wind_reference="network_mean",
            wind_reference_radius_km=10.0,
            wind_reference_min_stations=3,
        )
        rows = [("a", 200.0, 5.0)] + [
            (f"far{i}", 200.0, 50.0 + i) for i in range(9)
        ]
        result = resolve_wind_reference(
            obs_hourly=_network_frame(self.hour, rows),
            forecasts=pd.DataFrame(),
            own_station_obs=_own_frame(self.hour, 119.0),
            target_time=self.t,
            config=cfg,
        )
        # 1 station within radius < min 3 → fallback to own.
        self.assertAlmostEqual(result, 119.0, places=5)

    def test_naive_datetime_treated_as_utc(self):
        cfg = FeatureConfig(wind_reference="own")
        naive = datetime(2026, 5, 18, 12, 0)  # no tzinfo
        result = resolve_wind_reference(
            obs_hourly=pd.DataFrame(),
            forecasts=pd.DataFrame(),
            own_station_obs=_own_frame(self.hour, 175.0),
            target_time=naive,
            config=cfg,
        )
        self.assertAlmostEqual(result, 175.0, places=5)


if __name__ == "__main__":
    unittest.main()
