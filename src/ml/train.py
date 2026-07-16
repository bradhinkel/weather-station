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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from src.ml import trains_pooled
from src.ml.dataset import _sync_dsn, build_dataset, resolve_own_station_id
from src.ml.rain_model import TwoStageRainModel

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


def train_randomforest(X, y) -> RandomForestRegressor:
    # Sized to survive the droplet, not to win a benchmark. The API mtime-caches
    # every bundle in memory and there are now 5 horizons x 2 targets of them on a
    # 4GB box that also hosts postgres; 200 trees at depth 12 over ~230k rows is
    # ~100MB per forest, which does not fit that budget. Depth 10 / 100 trees is
    # ~13MB. n_jobs=2 matches the vCPU count — the weekly retrain runs while the API
    # is up and has already CPU-starved it once (2026-06-01).
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X, y)
    return model


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


def train_twostage(X, y) -> TwoStageRainModel:
    return TwoStageRainModel().fit(X, y)


def evaluate_twostage(model: TwoStageRainModel, X, y, baseline) -> dict:
    """Score the two-stage rain model on BOTH axes.

    Regression: the expected-value prediction (P·amount) vs. the actual amount,
    alongside the Open-Meteo baseline — comparable to the other rain models.

    Classification: stage 1's rain/no-rain decision at the model's threshold —
    precision/recall/F1, plus threshold-free PR-AUC and Brier calibration. This
    is the axis that actually reflects rain skill on a zero-inflated target.
    """
    expected = model.predict(X)
    proba = model.predict_proba(X)
    y_wet = (np.asarray(y, dtype=float) > model.rain_threshold_mm).astype(int)
    pred_wet = (proba >= model.decision_threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_wet, pred_wet, average="binary", zero_division=0
    )
    # PR-AUC and Brier need both classes present to be meaningful.
    both_classes = 0 < int(y_wet.sum()) < len(y_wet)
    pr_auc = float(average_precision_score(y_wet, proba)) if both_classes else None
    brier = float(brier_score_loss(y_wet, proba)) if both_classes else None

    return {
        "mae": float(mean_absolute_error(y, expected)),
        "rmse": float(np.sqrt(mean_squared_error(y, expected))),
        "openmeteo_mae": float(mean_absolute_error(y, baseline)),
        "openmeteo_rmse": float(np.sqrt(mean_squared_error(y, baseline))),
        "n_test": int(len(y)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "brier": brier,
        "n_pos_test": int(y_wet.sum()),
    }


def _save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def _log_metric_row(
    trained_at: datetime,
    target: str,
    horizon: int,
    model_name: str,
    metrics: dict,
    n_train: int,
) -> None:
    """Append one row to model_metrics. Best-effort — failure here is logged but
    does not fail training (the joblib bundle is already on disk)."""
    try:
        engine = create_engine(_sync_dsn())
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_metrics (
                        trained_at, target, horizon, model,
                        mae, rmse, n_train, n_test,
                        openmeteo_mae, openmeteo_rmse,
                        precision, recall, f1, pr_auc, brier, n_pos_test
                    ) VALUES (
                        :trained_at, :target, :horizon, :model,
                        :mae, :rmse, :n_train, :n_test,
                        :openmeteo_mae, :openmeteo_rmse,
                        :precision, :recall, :f1, :pr_auc, :brier, :n_pos_test
                    )
                    ON CONFLICT (trained_at, target, horizon, model) DO NOTHING
                """),
                {
                    "trained_at": trained_at,
                    "target": target,
                    "horizon": horizon,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "n_train": n_train,
                    "n_test": metrics["n_test"],
                    "openmeteo_mae": metrics["openmeteo_mae"],
                    "openmeteo_rmse": metrics["openmeteo_rmse"],
                    # Classification metrics — present only for the twostage model.
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "pr_auc": metrics.get("pr_auc"),
                    "brier": metrics.get("brier"),
                    "n_pos_test": metrics.get("n_pos_test"),
                },
            )
        engine.dispose()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write model_metrics row: %s", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--station-id", default=None)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Train across ALL network stations (the pre-2026-07-15 behaviour). This "
             "yields a region-average corrector, not a microclimate model.",
    )
    parser.add_argument("--no-xgb", action="store_true")
    parser.add_argument("--no-rf", action="store_true")
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Own-station is the DEFAULT as of 2026-07-15. Pooling all ~320 stations trains the
    # region-average forecast->observation map, in which the backyard is <1% of rows, and
    # the resulting model corrects this yard toward a regional mean it does not live at:
    # 2.3% skill on own-station rows at +3h, versus 17.1% for the same model class
    # trained on own-station rows alone. Pass --pooled to get the old behaviour.
    station_id = args.station_id
    if station_id is None and not args.pooled:
        if trains_pooled(args.target):
            # See src.ml.POOLED_TARGETS: the own station has no wet hours to train or
            # score a rain model on until the wet season arrives.
            logger.info("Target %s trains POOLED by policy (see src.ml.POOLED_TARGETS)", args.target)
        else:
            station_id = resolve_own_station_id()
            if station_id is None:
                logger.error("No own station (is_network=false) found; pass --pooled to train regionally.")
                return
            logger.info("Training on OWN station %s (pass --pooled for the regional model)", station_id)

    df, feature_cols = build_dataset(args.target, args.horizon, station_id)
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
    trained_at = datetime.now(timezone.utc)
    trained_at_iso = trained_at.isoformat()

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
            "target": args.target, "horizon": args.horizon, "trained_at": trained_at_iso,
        },
        MODEL_DIR / f"{args.target}_{args.horizon}h_linear.joblib",
    )
    _log_metric_row(trained_at, args.target, args.horizon, "linear", lin_metrics, len(X_train))
    summary["linear"] = lin_metrics

    if not args.no_rf:
        logger.info("Training RandomForest…")
        rf = train_randomforest(X_train, y_train)
        rf_metrics = evaluate(rf, X_test, y_test, baseline_test)
        logger.info(
            "  randomforest MAE=%.3f  RMSE=%.3f   (Open-Meteo MAE=%.3f RMSE=%.3f)",
            rf_metrics["mae"], rf_metrics["rmse"],
            rf_metrics["openmeteo_mae"], rf_metrics["openmeteo_rmse"],
        )
        _save(
            {
                "model": rf, "feature_cols": feature_cols, "metrics": rf_metrics,
                "target": args.target, "horizon": args.horizon, "trained_at": trained_at_iso,
            },
            MODEL_DIR / f"{args.target}_{args.horizon}h_randomforest.joblib",
        )
        _log_metric_row(
            trained_at, args.target, args.horizon, "randomforest", rf_metrics, len(X_train)
        )
        summary["randomforest"] = rf_metrics

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
                "target": args.target, "horizon": args.horizon, "trained_at": trained_at_iso,
            },
            MODEL_DIR / f"{args.target}_{args.horizon}h_xgboost.joblib",
        )
        _log_metric_row(trained_at, args.target, args.horizon, "xgboost", xg_metrics, len(X_train))
        summary["xgboost"] = xg_metrics

    # Rain gets the two-stage classifier+regressor on top of the regressors above.
    if args.target == "rain_mm_1h":
        logger.info("Training two-stage rain model (classifier + wet-regressor)…")
        ts = train_twostage(X_train, y_train)
        ts_metrics = evaluate_twostage(ts, X_test, y_test, baseline_test)
        logger.info(
            "  twostage  MAE=%.3f  P=%.3f R=%.3f F1=%.3f  PR-AUC=%s  (pos=%d/%d)",
            ts_metrics["mae"], ts_metrics["precision"], ts_metrics["recall"],
            ts_metrics["f1"],
            f"{ts_metrics['pr_auc']:.3f}" if ts_metrics["pr_auc"] is not None else "n/a",
            ts_metrics["n_pos_test"], ts_metrics["n_test"],
        )
        # A rain model scored on zero wet hours is not a model, and its F1 of 0.000 is
        # not a measurement -- it is the absence of one. Refuse to overwrite a working
        # bundle with it. This fires when the test window is dry (a Seattle July) or the
        # station is too sparse; both are conditions to wait out, not to ship. The
        # existing bundle stays on disk and keeps serving.
        if ts_metrics["n_pos_test"] == 0:
            logger.error(
                "REFUSING to save %s_%dh_twostage: 0 positive (wet) hours in the test "
                "window of %d rows. F1=0 here means 'no rain to score against', not 'no "
                "skill'. Keeping the previous bundle. See src.ml.POOLED_TARGETS.",
                args.target, args.horizon, ts_metrics["n_test"],
            )
            summary["twostage"] = {"skipped": "no positive test hours"}
            print(json.dumps(summary, indent=2))
            return
        _save(
            {
                "model": ts, "feature_cols": feature_cols, "metrics": ts_metrics,
                "target": args.target, "horizon": args.horizon, "trained_at": trained_at_iso,
            },
            MODEL_DIR / f"{args.target}_{args.horizon}h_twostage.joblib",
        )
        _log_metric_row(trained_at, args.target, args.horizon, "twostage", ts_metrics, len(X_train))
        summary["twostage"] = ts_metrics

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
