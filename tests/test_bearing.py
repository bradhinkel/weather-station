"""Unit tests for src.features.bearing and src.features.config.

Run with stdlib unittest (no pytest dep):
    python3 -m unittest tests.test_bearing
"""

import math
import unittest

from src.features.bearing import angular_distance, circular_mean, direction_class
from src.features.config import FeatureConfig


class TestAngularDistance(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(angular_distance(0, 0), 0)
        self.assertEqual(angular_distance(123.4, 123.4), 0)

    def test_max_is_180(self):
        self.assertEqual(angular_distance(0, 180), 180)
        self.assertEqual(angular_distance(180, 0), 180)
        self.assertEqual(angular_distance(90, 270), 180)

    def test_wraparound(self):
        # 350° and 10° are 20° apart, not 340°.
        self.assertEqual(angular_distance(350, 10), 20)
        self.assertEqual(angular_distance(10, 350), 20)

    def test_symmetric(self):
        self.assertEqual(angular_distance(45, 135), 90)
        self.assertEqual(angular_distance(135, 45), 90)

    def test_inputs_over_360(self):
        # Defensive: a caller passing 370° should be treated as 10°.
        self.assertEqual(angular_distance(370, 10), 0)
        self.assertEqual(angular_distance(370, 0), 10)


class TestCircularMean(unittest.TestCase):
    def test_single(self):
        self.assertAlmostEqual(circular_mean([42.0]), 42.0, places=5)

    def test_two_adjacent(self):
        self.assertAlmostEqual(circular_mean([10.0, 20.0]), 15.0, places=5)

    def test_wraparound(self):
        # Mean of 350° and 10° is 0°, not 180°.
        result = circular_mean([350.0, 10.0])
        self.assertIsNotNone(result)
        # Tolerate the equivalent 360° representation.
        self.assertAlmostEqual(min(result, 360.0 - result), 0.0, places=5)

    def test_diametric_returns_none(self):
        # 0° and 180° cancel; refusing to return a meaningless 0° is correct.
        self.assertIsNone(circular_mean([0.0, 180.0]))
        self.assertIsNone(circular_mean([90.0, 270.0]))

    def test_empty(self):
        self.assertIsNone(circular_mean([]))

    def test_nan_filtered(self):
        nan = float("nan")
        self.assertAlmostEqual(circular_mean([10.0, nan, 20.0]), 15.0, places=5)

    def test_all_nan_or_none(self):
        nan = float("nan")
        self.assertIsNone(circular_mean([nan, nan]))
        self.assertIsNone(circular_mean([None, None]))


class TestDirectionClass(unittest.TestCase):
    """Wind convention: wind_from is the bearing the wind is blowing FROM.

    Tolerance = 30° means upwind cone is ±30° from the wind_from bearing.
    """

    TOL = 30.0

    def test_north_wind(self):
        # Wind from N (0°); station bearing 0° (N of home) → upwind.
        self.assertEqual(direction_class(0, 0, self.TOL), "upwind")
        # Station 180° (S of home) → downwind.
        self.assertEqual(direction_class(180, 0, self.TOL), "downwind")
        # Station E or W → crosswind.
        self.assertEqual(direction_class(90, 0, self.TOL), "crosswind")
        self.assertEqual(direction_class(270, 0, self.TOL), "crosswind")

    def test_east_wind(self):
        # Wind from E (90°); upwind = E of home.
        self.assertEqual(direction_class(90, 90, self.TOL), "upwind")
        self.assertEqual(direction_class(270, 90, self.TOL), "downwind")
        self.assertEqual(direction_class(0, 90, self.TOL), "crosswind")
        self.assertEqual(direction_class(180, 90, self.TOL), "crosswind")

    def test_tolerance_boundary_inclusive(self):
        # 30° off wind axis with tolerance=30 is upwind (boundary inclusive).
        self.assertEqual(direction_class(30, 0, self.TOL), "upwind")
        # Mirror image on downwind side: 150° off wind from N → downwind.
        self.assertEqual(direction_class(150, 0, self.TOL), "downwind")

    def test_just_outside_tolerance(self):
        # 31° off wind axis → crosswind.
        self.assertEqual(direction_class(31, 0, self.TOL), "crosswind")
        # 149° off → crosswind.
        self.assertEqual(direction_class(149, 0, self.TOL), "crosswind")

    def test_wraparound(self):
        # Wind from 350°, station at bearing 10° → 20° off → upwind.
        self.assertEqual(direction_class(10, 350, self.TOL), "upwind")
        # Wind from 10°, station at 190° → downwind.
        self.assertEqual(direction_class(190, 10, self.TOL), "downwind")

    def test_nan_returns_unknown(self):
        nan = float("nan")
        # Calm wind: caller passes NaN; we don't bin against an undefined axis.
        self.assertEqual(direction_class(45, nan, self.TOL), "unknown")
        self.assertEqual(direction_class(nan, 0, self.TOL), "unknown")

    def test_wide_tolerance_90_makes_crosswind_empty(self):
        # At tolerance=90, every station is either upwind (diff<=90) or
        # downwind (diff>=90). 90° off is the boundary — falls into upwind
        # first by our ordering.
        self.assertEqual(direction_class(90, 0, 90.0), "upwind")
        self.assertEqual(direction_class(91, 0, 90.0), "downwind")


class TestFeatureConfig(unittest.TestCase):
    """Defaults are pre-registered values; validation guards ablation typos."""

    def test_defaults(self):
        cfg = FeatureConfig()
        self.assertEqual(cfg.wind_reference, "network_mean")
        self.assertEqual(cfg.wind_reference_radius_km, 10.0)
        self.assertEqual(cfg.angular_tolerance_deg, 30.0)
        self.assertEqual(cfg.distance_band_km, (0.0, 25.0))
        self.assertEqual(cfg.lag_hours, (1, 3, 6, 12))
        self.assertEqual(cfg.aggregation_kernel, "inverse_distance")

    def test_frozen(self):
        cfg = FeatureConfig()
        with self.assertRaises(Exception):
            cfg.n_stations = 99  # type: ignore[misc]

    def test_hashable(self):
        # Two configs with same values hash the same — usable as cache keys.
        self.assertEqual(hash(FeatureConfig()), hash(FeatureConfig()))

    def test_rejects_zero_tolerance(self):
        with self.assertRaises(ValueError):
            FeatureConfig(angular_tolerance_deg=0)

    def test_rejects_over_90_tolerance(self):
        with self.assertRaises(ValueError):
            FeatureConfig(angular_tolerance_deg=91)

    def test_rejects_inverted_distance_band(self):
        with self.assertRaises(ValueError):
            FeatureConfig(distance_band_km=(50.0, 10.0))

    def test_rejects_negative_low_band(self):
        with self.assertRaises(ValueError):
            FeatureConfig(distance_band_km=(-1.0, 10.0))

    def test_rejects_zero_radius(self):
        with self.assertRaises(ValueError):
            FeatureConfig(wind_reference_radius_km=0)

    def test_rejects_zero_n_stations(self):
        with self.assertRaises(ValueError):
            FeatureConfig(n_stations=0)


if __name__ == "__main__":
    unittest.main()
