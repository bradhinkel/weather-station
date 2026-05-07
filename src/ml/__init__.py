"""ML pipeline: dataset construction, training, and prediction.

Supports targets ∈ {temp_c, rain_mm_1h} and horizons ∈ {1, 24}.
Models persist as joblib bundles named `{target}_{horizon}h_{model}.joblib`.
"""

SUPPORTED_TARGETS = ("temp_c", "rain_mm_1h")
SUPPORTED_HORIZONS = (1, 3, 24)
SUPPORTED_MODELS = ("linear", "xgboost")
