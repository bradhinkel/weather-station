"""Run the physical-plausibility checks against the live corpus.

    python -m tools.check_invariants                    # temp_c, all horizons
    python -m tools.check_invariants --target rain_mm_1h
    python -m tools.check_invariants --station-id <id>  # own-station only

Exits non-zero if any invariant is violated, so it can gate a retrain. The
per-horizon checks run on each build_dataset() output; the baseline-monotonicity
check is cross-horizon and therefore needs the whole sweep, which is why this is a
tool rather than an assertion inside train.py.

See src/ml/invariants.py for what each check encodes and which shipped bug it
corresponds to.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS
from src.ml.dataset import build_dataset
from src.ml.invariants import (
    check_baseline_monotonic,
    check_forecast_lead,
    check_no_constant_columns,
    check_physical_bounds,
    check_rain_positive_fraction,
)

logger = logging.getLogger("check_invariants")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="temp_c", choices=SUPPORTED_TARGETS)
    parser.add_argument("--station-id", default=None)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(SUPPORTED_HORIZONS),
        help="Horizons to check (default: all supported).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failures: list[str] = []
    baseline_mae: dict[int, float] = {}

    for horizon in sorted(args.horizons):
        df, feature_cols = build_dataset(args.target, horizon, args.station_id)
        if df.empty:
            failures.append(f"+{horizon}h: build_dataset returned 0 rows.")
            continue

        baseline_mae[horizon] = float(
            (df["openmeteo_baseline"] - df["y"]).abs().mean()
        )
        logger.info(
            "+%dh: %d rows, baseline MAE %.3f", horizon, len(df), baseline_mae[horizon]
        )

        violations: list[str] = []
        violations += check_forecast_lead(df, horizon)
        violations += check_no_constant_columns(df, feature_cols)
        violations += check_physical_bounds(df)
        if args.target == "rain_mm_1h":
            violations += check_rain_positive_fraction(df["y"].to_numpy(dtype=float))

        failures += [f"+{horizon}h: {v}" for v in violations]

    # Cross-horizon: forecast error must grow with lead time.
    failures += check_baseline_monotonic(baseline_mae, tol=0.01)

    print()
    if not baseline_mae:
        print("NO DATA — nothing checked.")
        return 1

    print(f"Open-Meteo baseline MAE by horizon ({args.target}):")
    for horizon in sorted(baseline_mae):
        print(f"  +{horizon:>2}h  {baseline_mae[horizon]:.3f}")
    print()

    if failures:
        print(f"FAILED — {len(failures)} invariant violation(s):")
        for failure in failures:
            print(f"  * {failure}")
        return 1

    print("OK — all invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
