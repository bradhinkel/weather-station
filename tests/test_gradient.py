"""Unit tests for src.features.gradient.

Builds synthetic obs frames so the logic is verifiable without a DB.
"""

import unittest

import pandas as pd

from src.features.config import FeatureConfig
from src.features.gradient import compute_upwind_gradient


def _obs(rows: list[dict]) -> pd.DataFrame:
    """Build an obs frame; each row dict supplies station_id, distance_km,
    bearing_deg, and the field columns."""
    return pd.DataFrame(rows)


class TestComputeUpwindGradient(unittest.TestCase):
    """All tests use wind_from_deg=0 (wind from north) and bearing=0 for
    upwind stations, so upwind classification is straightforward."""

    def test_basic_far_minus_near(self):
        # 3 near (0–10km) at temp=20, 3 far (25–50km) at temp=15.
        # Gradient = far - near = -5 (temperature falling along the wind).
        df = _obs(
            [
                {"station_id": "n1", "distance_km": 3, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
                {"station_id": "n2", "distance_km": 5, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
                {"station_id": "n3", "distance_km": 8, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
                {"station_id": "f1", "distance_km": 30, "bearing_deg": 0,
                 "temp_c": 15, "humidity_pct": 60, "pressure_hpa": 1010,
                 "wind_speed_ms": 2.0},
                {"station_id": "f2", "distance_km": 35, "bearing_deg": 0,
                 "temp_c": 15, "humidity_pct": 60, "pressure_hpa": 1010,
                 "wind_speed_ms": 2.0},
                {"station_id": "f3", "distance_km": 40, "bearing_deg": 0,
                 "temp_c": 15, "humidity_pct": 60, "pressure_hpa": 1010,
                 "wind_speed_ms": 2.0},
            ]
        )
        result = compute_upwind_gradient(df, 0.0, FeatureConfig())
        self.assertAlmostEqual(result["temp_c"], -5.0, places=5)
        self.assertAlmostEqual(result["humidity_pct"], 10.0, places=5)
        self.assertAlmostEqual(result["pressure_hpa"], -5.0, places=5)
        self.assertAlmostEqual(result["wind_speed_ms"], 1.0, places=5)

    def test_crosswind_stations_excluded(self):
        # Three near upwind (bearing=0), three near crosswind (bearing=90).
        # No far stations → gradient should be None.
        df = _obs(
            [
                {"station_id": f"u{i}", "distance_km": 3 + i, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0}
                for i in range(3)
            ]
            + [
                {"station_id": f"x{i}", "distance_km": 30 + i, "bearing_deg": 90,
                 "temp_c": 99, "humidity_pct": 99, "pressure_hpa": 999,
                 "wind_speed_ms": 9.0}
                for i in range(3)
            ]
        )
        result = compute_upwind_gradient(df, 0.0, FeatureConfig())
        # Crosswind far stations don't count → far band is empty.
        self.assertIsNone(result["temp_c"])

    def test_empty_input(self):
        df = _obs([])
        # Even with no columns, an empty DataFrame is handled.
        result = compute_upwind_gradient(df, 0.0, FeatureConfig())
        self.assertEqual(result, {f: None for f in
                                  ("temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms")})

    def test_no_upwind_stations(self):
        # All stations crosswind → gradient None.
        df = _obs(
            [
                {"station_id": "x1", "distance_km": 5, "bearing_deg": 90,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
            ]
        )
        result = compute_upwind_gradient(df, 0.0, FeatureConfig())
        self.assertIsNone(result["temp_c"])

    def test_nan_field_returns_none_for_that_field(self):
        nan = float("nan")
        df = _obs(
            [
                {"station_id": "n1", "distance_km": 5, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": nan, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
                {"station_id": "f1", "distance_km": 35, "bearing_deg": 0,
                 "temp_c": 15, "humidity_pct": nan, "pressure_hpa": 1010,
                 "wind_speed_ms": 2.0},
            ]
        )
        result = compute_upwind_gradient(df, 0.0, FeatureConfig())
        self.assertAlmostEqual(result["temp_c"], -5.0, places=5)
        self.assertIsNone(result["humidity_pct"])

    def test_band_filtering_respects_config(self):
        cfg = FeatureConfig(
            gradient_near_band_km=(0.0, 5.0),
            gradient_far_band_km=(50.0, 100.0),
        )
        df = _obs(
            [
                # In the OLD default near band (0-10) but outside the new (0-5).
                {"station_id": "skip", "distance_km": 7, "bearing_deg": 0,
                 "temp_c": 99, "humidity_pct": 99, "pressure_hpa": 999,
                 "wind_speed_ms": 9.0},
                {"station_id": "n1", "distance_km": 3, "bearing_deg": 0,
                 "temp_c": 20, "humidity_pct": 50, "pressure_hpa": 1015,
                 "wind_speed_ms": 1.0},
                {"station_id": "f1", "distance_km": 60, "bearing_deg": 0,
                 "temp_c": 15, "humidity_pct": 60, "pressure_hpa": 1010,
                 "wind_speed_ms": 2.0},
            ]
        )
        result = compute_upwind_gradient(df, 0.0, cfg)
        # "skip" at 7km is excluded by tighter near band.
        self.assertAlmostEqual(result["temp_c"], -5.0, places=5)


class TestFeatureConfigGradientValidation(unittest.TestCase):
    def test_defaults(self):
        cfg = FeatureConfig()
        self.assertEqual(cfg.gradient_near_band_km, (0.0, 10.0))
        self.assertEqual(cfg.gradient_far_band_km, (25.0, 50.0))

    def test_rejects_overlapping_bands(self):
        with self.assertRaises(ValueError):
            FeatureConfig(
                gradient_near_band_km=(0.0, 30.0),
                gradient_far_band_km=(25.0, 50.0),
            )

    def test_rejects_inverted_near(self):
        with self.assertRaises(ValueError):
            FeatureConfig(gradient_near_band_km=(10.0, 5.0))

    def test_rejects_inverted_far(self):
        with self.assertRaises(ValueError):
            FeatureConfig(gradient_far_band_km=(50.0, 25.0))


if __name__ == "__main__":
    unittest.main()
