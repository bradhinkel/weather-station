"""Unit tests for the two-stage rain model (src.ml.rain_model)."""

import numpy as np
import pytest

from src.ml.rain_model import MIN_WET_SAMPLES, TwoStageRainModel


def _make_data(n=3000, wet_scale=0.4, seed=0):
    """Zero-inflated rain: P(rain) and amount both driven by feature 0."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    p = 1.0 / (1.0 + np.exp(-(X[:, 0] - 1.0)))
    wet = rng.rand(n) < p * wet_scale
    y = np.where(wet, np.abs(X[:, 0]) * 2.0 + rng.rand(n) * 3.0, 0.0)
    return X, y


def test_fit_predict_shapes_and_bounds():
    X, y = _make_data()
    m = TwoStageRainModel().fit(X, y)

    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert proba.min() >= 0.0 and proba.max() <= 1.0

    expected = m.predict(X)
    assert expected.shape == (len(X),)
    # Expected value = P(rain) * amount, and amount is clipped non-negative.
    assert (expected >= 0.0).all()

    # Amount is likewise never negative even where the regressor extrapolates.
    assert (m.predict_amount(X) >= 0.0).all()


def test_classifier_has_signal():
    """Wet hours should get higher mean probability than dry hours."""
    X, y = _make_data()
    m = TwoStageRainModel().fit(X, y)
    proba = m.predict_proba(X)
    wet = y > m.rain_threshold_mm
    assert proba[wet].mean() > proba[~wet].mean()


def test_gated_prediction_is_zero_when_unlikely():
    X, y = _make_data()
    m = TwoStageRainModel(decision_threshold=0.5).fit(X, y)
    gated = m.predict_gated(X)
    proba = m.predict_proba(X)
    # Below-threshold rows are forced to zero; above-threshold rows are >= 0.
    assert np.all(gated[proba < 0.5] == 0.0)
    assert np.all(gated >= 0.0)


def test_regressor_fallback_when_too_few_wet():
    """With fewer than MIN_WET_SAMPLES wet hours, stage 2 falls back to the mean."""
    rng = np.random.RandomState(1)
    n = 500
    X = rng.randn(n, 4)
    y = np.zeros(n)
    n_wet = MIN_WET_SAMPLES - 5
    y[:n_wet] = 2.0  # a handful of identical wet hours
    m = TwoStageRainModel().fit(X, y)
    assert m._reg_fitted is False
    # Fallback amount is the mean of the wet hours (2.0 here).
    amt = m.predict_amount(X)
    assert np.allclose(amt, 2.0)


def test_all_dry_is_handled():
    """A fully dry window must not raise (Seattle summer edge case)."""
    rng = np.random.RandomState(2)
    X = rng.randn(200, 4)
    y = np.zeros(200)
    m = TwoStageRainModel().fit(X, y)
    assert m._reg_fitted is False
    assert np.allclose(m.predict(X), 0.0)  # wet_mean is 0 → expected rain 0


def test_predict_before_fit_raises():
    m = TwoStageRainModel()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((1, 4)))
