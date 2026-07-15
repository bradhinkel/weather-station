"""Measure MAE vs. training-set size for each model class.

    python -m tools.learning_curve --target temp_c --horizon 3
    python -m tools.learning_curve --target temp_c --horizon 3 --out experiments/lc.csv

Tests the project's standing assertion: *tree models will win once there is enough
data; the deep-learning detour is the wrong technology for this problem.* That
claim is only worth making if the crossover is measured rather than assumed, and
there is already evidence for it at two points — Ridge beat XGBoost on the ~590-row
ablation window (2026-06-19), while XGBoost led at every horizon on the ~115k-row
pooled corpus. This sweeps the fractions in between so the crossover has a number
and the trend can be extrapolated toward a full year of data.

Method: hold the temporal test split FIXED and shrink the training set, so every
point is scored on identical rows. The test half is never touched.

`--sample` picks WHICH question you are asking, and they are not the same question:

  recent (default) — take the most recent N rows: a contiguous window ending at the
      split boundary. Simulates "what if this project were younger?" But it varies
      TWO things at once: row count and calendar span. The corpus is only ~58 days,
      so a 5% subset is a ~2.3-day window in which doy_sin/doy_cos are effectively
      constant; StandardScaler then divides by a near-zero std and Ridge extrapolates
      onto a test set weeks away, scoring MAE ~26 C. That number is real output but it
      measures seasonal-feature degeneracy, not model capacity.

  random — sample N rows uniformly from the same training half. Holds calendar span
      fixed and isolates the effect of row count, which is the question "do trees need
      more data than linear?" actually asks.

Use `random` to compare model classes; use `recent` to reason about project age.
Reporting only `recent` would repeat the mistake this repo keeps making: a plausible
number that answers a question nobody asked.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS
from src.ml.dataset import build_dataset
from src.ml.train import evaluate, temporal_split, train_linear, train_randomforest, train_xgboost

logger = logging.getLogger("learning_curve")

TRAINERS = {
    "linear": train_linear,
    "randomforest": train_randomforest,
    "xgboost": train_xgboost,
}

DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="temp_c", choices=SUPPORTED_TARGETS)
    parser.add_argument("--horizon", type=int, default=3, choices=SUPPORTED_HORIZONS)
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument(
        "--sample",
        choices=("recent", "random"),
        default="recent",
        help="recent: contiguous window (varies span too). random: isolates row count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional CSV path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df, feature_cols = build_dataset(args.target, args.horizon, args.station_id)
    if df.empty:
        print("Dataset is empty — nothing to sweep.")
        return 1

    train_df, test_df = temporal_split(df)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)
    base_test = test_df["openmeteo_baseline"].to_numpy(dtype=float)
    om_mae = float(np.mean(np.abs(y_test - base_test)))

    logger.info(
        "target=%s +%dh | %d train / %d test | Open-Meteo MAE %.3f",
        args.target, args.horizon, len(train_df), len(test_df), om_mae,
    )

    rows: list[dict] = []
    for frac in sorted(args.fractions):
        n = max(1, int(frac * len(train_df)))
        if args.sample == "recent":
            # Contiguous window ending at the split boundary. Varies calendar span
            # along with n — see the module docstring.
            subset = train_df.iloc[-n:]
        else:
            subset = train_df.sample(n=n, random_state=args.seed)
        X = subset[feature_cols].to_numpy(dtype=float)
        y = subset["y"].to_numpy(dtype=float)
        span_days = float(
            (subset["valid_time"].max() - subset["valid_time"].min()).total_seconds() / 86400.0
        )

        for name, trainer in TRAINERS.items():
            try:
                model = trainer(X, y)
            except Exception as exc:  # a model class may refuse a tiny subset
                logger.warning("%s at n=%d failed: %s", name, n, exc)
                continue
            metrics = evaluate(model, X_test, y_test, base_test)
            rows.append(
                {
                    "target": args.target,
                    "horizon": args.horizon,
                    "sample": args.sample,
                    "frac": frac,
                    "n_train": n,
                    "span_days": round(span_days, 2),
                    "model": name,
                    "mae": round(metrics["mae"], 4),
                    "rmse": round(metrics["rmse"], 4),
                    "openmeteo_mae": round(om_mae, 4),
                    "skill_vs_om": round(1.0 - metrics["mae"] / om_mae, 4) if om_mae else None,
                    "n_test": len(y_test),
                }
            )
            logger.info("  n=%-7d %-13s MAE=%.3f", n, name, metrics["mae"])

    print()
    print(f"Learning curve — target={args.target} +{args.horizon}h  sample={args.sample} "
          f"(Open-Meteo MAE {om_mae:.3f}, n_test={len(y_test)})")
    print()
    models = list(TRAINERS)
    print(f"{'n_train':>8} {'span_d':>7} " + " ".join(f"{m:>13}" for m in models))
    for frac in sorted(args.fractions):
        by_model = {r["model"]: r for r in rows if r["frac"] == frac}
        if not by_model:
            continue
        first = next(iter(by_model.values()))
        n, span = first["n_train"], first["span_days"]
        cells = []
        best = min((r["mae"] for r in by_model.values()), default=None)
        for m in models:
            if m not in by_model:
                cells.append(f"{'-':>13}")
                continue
            mae = by_model[m]["mae"]
            cells.append(f"{mae:>12.3f}{'*' if mae == best else ' '}")
        print(f"{n:>8} {span:>7.1f} " + " ".join(cells))
    print()
    print("* = best at that training size. The crossover is where the marker moves")
    print("  from linear to a tree model; extrapolate it against the corpus growth rate.")
    if args.sample == "recent":
        print()
        print("NOTE: sample=recent shrinks the calendar span along with n, so small-n")
        print("  linear results conflate data volume with seasonal-feature degeneracy.")
        print("  Re-run with --sample random to isolate the effect of row count.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
