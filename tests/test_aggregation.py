"""Unit tests for src.features.aggregation.

Run with:
    python3 -m unittest tests.test_aggregation
"""

import math
import unittest

import numpy as np

from src.features.aggregation import kernel_weights, weighted_mean


class TestKernelWeights(unittest.TestCase):
    def test_uniform(self):
        w = kernel_weights([1, 5, 10], "uniform")
        self.assertTrue(np.allclose(w, [1.0, 1.0, 1.0]))

    def test_inverse_distance(self):
        w = kernel_weights([1.0, 2.0, 4.0], "inverse_distance")
        self.assertTrue(np.allclose(w, [1.0, 0.5, 0.25]))

    def test_inverse_distance_clamps_zero(self):
        # 0 km would divide by zero; clamp to MIN_DISTANCE_KM = 0.1.
        w = kernel_weights([0.0], "inverse_distance")
        self.assertAlmostEqual(w[0], 10.0, places=5)

    def test_gaussian_at_zero(self):
        w = kernel_weights([0.0, 5.0], "gaussian", gaussian_sigma_km=5.0)
        # At d=0, weight = exp(0) = 1; at d=sigma, weight = exp(-0.5) ≈ 0.6065
        self.assertAlmostEqual(w[0], 1.0, places=5)
        self.assertAlmostEqual(w[1], math.exp(-0.5), places=5)

    def test_gaussian_rejects_zero_sigma(self):
        with self.assertRaises(ValueError):
            kernel_weights([1.0], "gaussian", gaussian_sigma_km=0)

    def test_unknown_kernel(self):
        with self.assertRaises(ValueError):
            kernel_weights([1.0], "bogus")  # type: ignore[arg-type]


class TestWeightedMean(unittest.TestCase):
    def test_simple(self):
        self.assertAlmostEqual(
            weighted_mean([10.0, 20.0], [1.0, 1.0]), 15.0, places=5
        )

    def test_weighted(self):
        # Mean of 10@weight3 and 20@weight1 = (30+20)/4 = 12.5
        self.assertAlmostEqual(
            weighted_mean([10.0, 20.0], [3.0, 1.0]), 12.5, places=5
        )

    def test_nan_value_dropped(self):
        nan = float("nan")
        self.assertAlmostEqual(
            weighted_mean([10.0, nan, 20.0], [1.0, 5.0, 1.0]), 15.0, places=5
        )

    def test_zero_weight_dropped(self):
        self.assertAlmostEqual(
            weighted_mean([10.0, 999.0], [1.0, 0.0]), 10.0, places=5
        )

    def test_all_nan_returns_none(self):
        nan = float("nan")
        self.assertIsNone(weighted_mean([nan, nan], [1.0, 1.0]))

    def test_empty_returns_none(self):
        self.assertIsNone(weighted_mean([], []))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            weighted_mean([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
