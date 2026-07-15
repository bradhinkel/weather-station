"""Score the served (pooled) models on OWN-STATION test rows only.

    python -m tools.own_station_eval --target temp_c

The live models train on `build_dataset(station_id=None)`, which pools every
network station, so the reported MAE is regional skill: the backyard is <1% of the
rows and the headline number cannot reveal microclimate accuracy either way. This
tool answers the narrower question the project actually asks — *how well does the
served model do in the backyard?* — without retraining anything.

Method: build the POOLED dataset, apply the exact temporal split train.py uses,
then filter the test half to the own station. Taking the split from the pooled
frame is what keeps this honest — own-station rows in the pooled test window were
genuinely held out of the pooled fit, whereas splitting an own-station-only frame
would land on a different boundary and leak training rows into the evaluation.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from sqlalchemy import create_engine, text

from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS, models_for_target
from src.ml.dataset import _sync_dsn, build_dataset
from src.ml.predict import load_bundle
from src.ml.train import temporal_split

logger = logging.getLogger("own_station_eval")


def own_station_id() -> str | None:
    engine = create_engine(_sync_dsn())
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT station_id FROM stations WHERE is_network = false LIMIT 1")
            ).first()
    finally:
        engine.dispose()
    return row[0] if row else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="temp_c", choices=SUPPORTED_TARGETS)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    parser.add_argument("--station-id", default=None, help="Defaults to the is_network=false station.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sid = args.station_id or own_station_id()
    if not sid:
        print("No own station found (no stations row with is_network = false).")
        return 1
    logger.info("Own station: %s", sid)

    rows: list[tuple] = []
    for horizon in sorted(args.horizons):
        df, feature_cols = build_dataset(args.target, horizon, station_id=None)
        if df.empty:
            logger.warning("+%dh: pooled dataset empty, skipping.", horizon)
            continue

        _, test_df = temporal_split(df)
        own = test_df[test_df["station_id"] == sid]
        if own.empty:
            logger.warning("+%dh: no own-station rows in the pooled test window.", horizon)
            continue

        X = own[feature_cols].to_numpy(dtype=float)
        y = own["y"].to_numpy(dtype=float)
        base = own["openmeteo_baseline"].to_numpy(dtype=float)
        pooled_n = len(test_df)

        for model_name in models_for_target(args.target):
            bundle = load_bundle(args.target, horizon, model_name)
            if not bundle:
                continue
            pred = bundle["model"].predict(X)
            rows.append(
                (
                    horizon,
                    model_name,
                    len(own),
                    pooled_n,
                    float(np.mean(np.abs(y - base))),
                    float(np.mean(np.abs(y - pred))),
                    float(bundle["metrics"]["mae"]),
                )
            )

    if not rows:
        print("No (model, horizon) pairs could be scored — are bundles present on disk?")
        return 1

    print()
    print(f"Own-station skill for target={args.target}, station={sid}")
    print("(pooled MAE is the model's reported headline metric, for contrast)")
    print()
    print(f"{'horizon':>7} {'model':>13} {'n_own':>6} {'n_pooled':>9} "
          f"{'OM MAE':>8} {'model MAE':>10} {'skill':>7} {'pooled MAE':>11}")
    for horizon, model_name, n_own, n_pooled, om_mae, mdl_mae, pooled_mae in rows:
        skill = 1.0 - (mdl_mae / om_mae) if om_mae > 0 else float("nan")
        print(
            f"{'+' + str(horizon) + 'h':>7} {model_name:>13} {n_own:>6} {n_pooled:>9} "
            f"{om_mae:>8.3f} {mdl_mae:>10.3f} {skill:>7.3f} {pooled_mae:>11.3f}"
        )
    print()
    print("skill = 1 - MAE_model/MAE_openmeteo on own-station rows. Negative means the")
    print("served model is WORSE than the raw regional forecast in the backyard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
