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


# Targets that must train on the POOLED network rather than the own station.
#
# temp_c trains own-station: pooling learns the region-average forecast->observation map
# in which the backyard is <1% of rows, and the resulting model drags this yard toward a
# regional mean it does not live at (2.3% vs 17.1% skill on own rows at +3h).
#
# rain_mm_1h must stay pooled, for now, on evidence rather than preference. Retrained
# own-station on 2026-07-15, ALL FIVE horizons returned **zero positive test hours** --
# the backyard recorded no wet hours in July and the temporal split puts July in test --
# collapsing precision, recall and F1 to 0. The pooled network supplies ~2,200 wet test
# hours and F1 0.955 at +1h. A rain model with no rain in it is not a model. Revisit once
# a wet season has given the own station enough positives to train AND score on; that is
# an October-to-March question, not an engineering one.
POOLED_TARGETS = ("rain_mm_1h",)


def trains_pooled(target: str) -> bool:
    """Does ``target`` train across the whole network instead of the own station?"""
    return target in POOLED_TARGETS


def models_for_target(target: str) -> tuple[str, ...]:
    """Model names trained/served for ``target`` (rain gets the two-stage one)."""
    if target == "rain_mm_1h":
        return SUPPORTED_MODELS + RAIN_MODELS
    return SUPPORTED_MODELS
