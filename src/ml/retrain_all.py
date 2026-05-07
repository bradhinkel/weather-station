"""Retrain every (target, horizon) combination for which there is a previously
trained joblib bundle on disk. Skips combinations with insufficient data.

This is the entry point for the weekly systemd timer:
    /opt/weather-station/venv/bin/python -m src.ml.retrain_all
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from src.ml import SUPPORTED_HORIZONS, SUPPORTED_TARGETS

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Retrain anything we already train (i.e. has at least one bundle on disk).
    # New combinations need to be bootstrapped manually with `python -m src.ml.train --target ... --horizon ...` once.
    combos: set[tuple[str, int]] = set()
    if MODEL_DIR.exists():
        for path in MODEL_DIR.glob("*_*h_*.joblib"):
            stem = path.stem  # e.g. temp_c_1h_linear
            for target in SUPPORTED_TARGETS:
                prefix = f"{target}_"
                if not stem.startswith(prefix):
                    continue
                rest = stem[len(prefix):]              # e.g. "1h_linear"
                horizon_part = rest.split("h_", 1)[0]  # "1"
                try:
                    horizon = int(horizon_part)
                except ValueError:
                    continue
                if horizon in SUPPORTED_HORIZONS:
                    combos.add((target, horizon))

    if not combos:
        logger.warning("No existing model bundles found in %s — nothing to retrain.", MODEL_DIR)
        return 1

    logger.info("Retraining %d combinations: %s", len(combos), sorted(combos))
    failures: list[str] = []
    for target, horizon in sorted(combos):
        cmd = [sys.executable, "-m", "src.ml.train", "--target", target, "--horizon", str(horizon)]
        logger.info("→ %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=os.environ.copy())
        except subprocess.CalledProcessError as exc:
            logger.error("Retrain failed for %s_%dh: %s", target, horizon, exc)
            failures.append(f"{target}_{horizon}h")

    if failures:
        logger.error("Retrain finished with failures: %s", failures)
        return 2

    logger.info("Retrain finished successfully for %d combinations.", len(combos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
