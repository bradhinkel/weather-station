"""ML pipeline: dataset construction, training, and prediction.

Supports targets ∈ {temp_c, rain_mm_1h} and horizons ∈ {1, 3, 6, 12, 24}.
Models persist as joblib bundles named `{target}_{horizon}h_{model}.joblib`.

Every target trains the two regressors in ``SUPPORTED_MODELS``. Rain
additionally trains the ``twostage`` classifier+regressor (see
:mod:`src.ml.rain_model`), which is the model to read for rain skill.
"""

SUPPORTED_TARGETS = ("temp_c", "rain_mm_1h")
# 6h and 12h bracket the crossover where own-station lag features stop beating
# the NWP; 6h is also the primary horizon in experiments/phase7_preregistration.md.
# Forecast lead availability caps this at ~47h (see forecasts.forecast_time spread).
SUPPORTED_HORIZONS = (1, 3, 6, 12, 24)
SUPPORTED_MODELS = ("linear", "randomforest", "xgboost")
# Rain-only two-stage model (classifier + wet-regressor). Kept separate because
# it is not a plain regressor — it carries a probability output and its own
# classification metrics.
RAIN_MODELS = ("twostage",)


def models_for_target(target: str) -> tuple[str, ...]:
    """Model names trained/served for ``target`` (rain gets the two-stage one)."""
    if target == "rain_mm_1h":
        return SUPPORTED_MODELS + RAIN_MODELS
    return SUPPORTED_MODELS
