"""Two-stage rain model — classifier (does it rain?) + regressor (how much?).

Hourly rain is zero-inflated: on the order of 85 % of hours are dry, and a
Seattle summer is ~100 % dry. A single regressor trained on that minimises MAE
by predicting ≈0 everywhere, so it scores well on MAE while being useless as a
forecast — and even emits small negatives. That is exactly what the earlier
single-stage rain regressor did.

The fix is to decompose ``E[rain]`` into two learners that are each trained on a
well-posed problem:

* **Stage 1 — classifier.** ``P(rain > threshold)`` for the hour. Judged on
  precision / recall / F1 / PR-AUC — real rain/no-rain skill, not MAE. Its
  probability is a first-class output: the chance of rain, which for the garden
  -watering use case matters as much as the amount.
* **Stage 2 — regressor.** Amount *given that it rains*, trained on wet hours
  only. Removing the zeros lets it learn intensity structure instead of being
  dragged toward zero.

Serving combines them as an expectation, ``E[rain] = P(rain) · E[amount|rain]``,
and also exposes the gated amount (the stage-2 estimate when ``P(rain)`` clears
a decision threshold, else 0) plus the raw probability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xgboost as xgb

# Hours with < this much rain are treated as "dry" for the stage-1 label. 0.1 mm
# is a conventional trace threshold — below it is tipping-bucket noise, not rain.
RAIN_THRESHOLD_MM: float = 0.1

# Stage 2 needs enough wet hours to fit a regressor; below this we fall back to
# the mean wet-hour amount rather than fitting an unstable model.
MIN_WET_SAMPLES: int = 30

# Probability above which the gated ("will it rain?") amount is reported.
DECISION_THRESHOLD: float = 0.5


class TwoStageRainModel:
    """Classifier + wet-only regressor. Sklearn-ish ``fit`` / ``predict`` API.

    Pickles cleanly (joblib) because it only holds two fitted XGBoost models and
    a couple of scalars — no closures or DB handles.
    """

    def __init__(
        self,
        rain_threshold_mm: float = RAIN_THRESHOLD_MM,
        decision_threshold: float = DECISION_THRESHOLD,
    ) -> None:
        self.rain_threshold_mm = rain_threshold_mm
        self.decision_threshold = decision_threshold
        self.clf: Optional[xgb.XGBClassifier] = None
        self.reg: Optional[xgb.XGBRegressor] = None
        # Fallback expected amount when stage 2 can't be fit (too few wet hours).
        self._wet_mean: float = 0.0
        self._reg_fitted: bool = False

    # -- fit -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TwoStageRainModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        wet = y > self.rain_threshold_mm
        n_pos = int(wet.sum())
        n_neg = int((~wet).sum())

        # Class imbalance: weight the positive class up so the classifier isn't
        # rewarded for always predicting "dry". scale_pos_weight = neg/pos is the
        # XGBoost-recommended default.
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0
        self.clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=2,
        )
        self.clf.fit(X, wet.astype(int))

        self._wet_mean = float(y[wet].mean()) if n_pos > 0 else 0.0
        if n_pos >= MIN_WET_SAMPLES:
            self.reg = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
            self.reg.fit(X[wet], y[wet])
            self._reg_fitted = True
        else:
            self._reg_fitted = False
        return self

    # -- predict -------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(rain > threshold) for each row, in [0, 1]."""
        if self.clf is None:
            raise RuntimeError("model not fitted")
        return self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_amount(self, X: np.ndarray) -> np.ndarray:
        """Stage-2 amount given rain (mm), clipped to be non-negative."""
        X = np.asarray(X, dtype=float)
        if self._reg_fitted and self.reg is not None:
            amt = self.reg.predict(X)
        else:
            amt = np.full(X.shape[0], self._wet_mean, dtype=float)
        return np.clip(amt, 0.0, None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expected rain (mm): ``P(rain) · E[amount|rain]``."""
        return self.predict_proba(X) * self.predict_amount(X)

    def predict_gated(self, X: np.ndarray) -> np.ndarray:
        """Amount when ``P(rain)`` clears the decision threshold, else 0."""
        p = self.predict_proba(X)
        amt = self.predict_amount(X)
        return np.where(p >= self.decision_threshold, amt, 0.0)
