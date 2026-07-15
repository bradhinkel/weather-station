"""Export the model_metrics time-series to a committed CSV.

    python -m tools.export_metrics --out experiments/model_metrics.csv

model_metrics lives only in the production database, so until now every number in
the README was unreproducible from the repository — a reader could not regenerate
the results table, or even check it. This writes the table to a CSV stamped with
the git SHA, row count, and export time, so published numbers have provenance and
a reviewer can diff them against a later retrain.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.ml.dataset import _sync_dsn

COLUMNS = [
    "trained_at", "target", "horizon", "model",
    "n_train", "n_test",
    "mae", "rmse", "openmeteo_mae", "openmeteo_rmse",
    "precision", "recall", "f1", "pr_auc", "brier", "n_pos_test",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/model_metrics.csv")
    args = parser.parse_args()

    engine = create_engine(_sync_dsn())
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT {', '.join(COLUMNS)} FROM model_metrics ORDER BY trained_at, target, horizon, model"),
                conn,
            )
    finally:
        engine.dispose()

    if df.empty:
        print("model_metrics is empty — nothing to export.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sha = git_sha()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with out.open("w", newline="") as fh:
        fh.write(f"# model_metrics export | git={sha} | rows={len(df)} | exported_at={stamp}\n")
        df.to_csv(fh, index=False)

    print(f"Wrote {len(df)} rows to {out} (git={sha})")
    latest = df.sort_values("trained_at").groupby(["target", "horizon", "model"]).tail(1)
    print(f"Latest retrain per combo: {len(latest)} rows, "
          f"trained_at max = {df['trained_at'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
