"""Train and persist (target, horizon, model) bundles.

Usage:
    python -m src.ml.train --target temp_c --horizon 1
    python -m src.ml.train --target temp_c --horizon 1 --no-xgb

Each run writes models to MODEL_DIR (default ./models) as
    {target}_{horizon}h_{model}.joblib
where the bundle is {"model", "feature_cols", "metrics", "target", "horizon", "trained_at"}.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.dataset import build_dataset

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))


def temporal_split(df, frac: float = 0.8):
    df = df.sort_values("valid_time").reset_index(drop=True)
    n = max(1, int(frac * len(df)))
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def train_linear(X, y) -> Pipeline:
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    return pipe


def train_xgboost(X, y) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y, baseline) -> dict:
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "openmeteo_mae": float(mean_absolute_error(y, baseline)),
        "openmeteo_rmse": float(np.sqrt(mean_squared_error(y, baseline))),
        "n_test": int(len(y)),
    }


def _save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--no-xgb", action="store_true")
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    df, feature_cols = build_dataset(args.target, args.horizon, args.station_id)
    logger.info("Dataset size: %d rows, %d features", len(df), len(feature_cols))

    if len(df) < args.min_rows:
        logger.warning("Insufficient rows (%d < %d). Skipping training.", len(df), args.min_rows)
        return

    train_df, test_df = temporal_split(df)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)
    baseline_test = test_df["openmeteo_baseline"].to_numpy(dtype=float)
    logger.info("Split: %d train / %d test", len(X_train), len(X_test))

    summary = {"target": args.target, "horizon": args.horizon}
    trained_at = datetime.now(timezone.utc).isoformat()

    logger.info("Training linear (Ridge)…")
    lin = train_linear(X_train, y_train)
    lin_metrics = evaluate(lin, X_test, y_test, baseline_test)
    logger.info(
        "  linear MAE=%.3f  RMSE=%.3f   (Open-Meteo MAE=%.3f RMSE=%.3f)",
        lin_metrics["mae"], lin_metrics["rmse"],
        lin_metrics["openmeteo_mae"], lin_metrics["openmeteo_rmse"],
    )
    _save(
        {
            "model": lin, "feature_cols": feature_cols, "metrics": lin_metrics,
            "target": args.target, "horizon": args.horizon, "trained_at": trained_at,
        },
        MODEL_DIR / f"{args.target}_{args.horizon}h_linear.joblib",
    )
    summary["linear"] = lin_metrics

    if not args.no_xgb:
        logger.info("Training XGBoost…")
        xg = train_xgboost(X_train, y_train)
        xg_metrics = evaluate(xg, X_test, y_test, baseline_test)
        logger.info(
            "  xgboost MAE=%.3f  RMSE=%.3f   (Open-Meteo MAE=%.3f RMSE=%.3f)",
            xg_metrics["mae"], xg_metrics["rmse"],
            xg_metrics["openmeteo_mae"], xg_metrics["openmeteo_rmse"],
        )
        _save(
            {
                "model": xg, "feature_cols": feature_cols, "metrics": xg_metrics,
                "target": args.target, "horizon": args.horizon, "trained_at": trained_at,
            },
            MODEL_DIR / f"{args.target}_{args.horizon}h_xgboost.joblib",
        )
        summary["xgboost"] = xg_metrics

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
