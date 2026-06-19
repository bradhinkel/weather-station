"""Phase 7.4 network-feature ablation sweep — exploratory harness.

Answers the operational question behind the production design: *how many
neighbor stations, at what distances, improve the +1h / +3h / +24h forecast,
and by how much?* It reuses the existing pieces rather than reinventing them:

  - :func:`src.ml.dataset.build_dataset`     — own-station + NWP baseline rows
  - :func:`src.features.pipeline.build_features` — network cohort features
  - :func:`src.ml.train` model classes       — Ridge + XGBoost, 80/20 temporal

For each (target, horizon, FeatureConfig) it trains TWO models on the *same*
rows and the *same* temporal split:

  - **base**: baseline features only (no network) — the control
  - **net** : baseline + network features        — the treatment

so the reported delta is purely the network contribution, not a data-window
artifact. A bootstrap CI on ``delta_mae`` says whether the delta is real.

CAUSALITY: a forecast issued at t for valid_time = t+H may only use neighbor
obs at times <= t. The baseline already anchors own-station lags at
``valid_time - horizon``; we anchor the network feature row the same way by
joining ``build_features`` (indexed by target hour, internal lags 0..12 all
<= its index) on ``valid_time - horizon``. So every network column references
a time <= issue time. No leakage.

This is the EXPLORATORY sweep (station-count / distance-band selection). It is
NOT the locked 7.4 campaign — it writes to ``experiments/sweep_<stamp>/`` and
does not touch the pre-registered ``phase7_results.md``. Output of this sweep
(the winning band/count mix) becomes the production real-time ingest set.

Run on the droplet (needs DB access)::

    cd /opt/weather-station &&
      venv/bin/python -m tools.ablation_sweep --sweep all --stamp 20260619T2100

Targets default to ``temp_c`` — the clean signal. Rain MAE flatters a near-zero
predictor (zero-inflated; see README), so rain is opt-in via ``--target`` and
should be read as a shakedown, not skill, until the 2-stage classifier lands.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine

from src.features.config import FeatureConfig
from src.features.pipeline import (
    _load_stations,
    _resolve_home_station,
    _sync_dsn,
    build_features,
)
from src.ml import SUPPORTED_HORIZONS
from src.ml.dataset import build_dataset
from src.ml.train import temporal_split, train_linear, train_xgboost

logger = logging.getLogger("ablation_sweep")

# Network ingest only stabilized after this date (see heartbeat: ~95% coverage
# by mid-June). Restricting the sweep window to the dense-network period keeps
# the control and treatment on identical rows without all-NaN network columns
# diluting the early span. Overridable via --window-start.
DEFAULT_WINDOW_START = "2026-05-12"

HORIZONS = (1, 3, 24)          # all three product horizons
MODEL_CLASSES = ("linear", "xgboost")
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# Mode "n": fix the headline band (0,25), vary cohort size — the plateau curve.
SWEEP_N = (1, 3, 5, 10, 20)
SWEEP_N_BAND = (0.0, 25.0)

# Mode "bands": isolate each NON-OVERLAPPING band so per-band strength is
# visible per horizon (does 5km carry +1h while 25km carries +3h?). n_stations
# is held generous so the band, not the cap, is what varies.
SWEEP_BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0), (25.0, 50.0))
SWEEP_BANDS_NSTATIONS = 8

# Mode "multiband": all bands present simultaneously as separate feature
# groups. One model per horizon; permutation importance per band group reads
# out the optimal MIX (e.g. "3@2km + 2@10km + 2@25km") and which band each
# horizon leans on. (band, n_stations_in_band)
MULTIBAND = (
    ((0.0, 2.0), 3),
    ((2.0, 5.0), 2),
    ((5.0, 10.0), 2),
    ((10.0, 25.0), 2),
    ((25.0, 50.0), 2),
)


# ---------------------------------------------------------------------------
# Feature building (cached per config within a run)
# ---------------------------------------------------------------------------

class FeatureCache:
    """Caches build_dataset (per horizon) and build_features (per config).

    build_features is horizon-independent — the internal lags are fixed and the
    horizon only changes the join offset — so one build is reused across all
    three horizons.
    """

    def __init__(self, window_start: datetime, window_end: datetime, home_id: str):
        self.start = window_start
        self.end = window_end
        self.home_id = home_id
        self._datasets: dict[tuple, tuple[pd.DataFrame, list[str]]] = {}
        self._features: dict[str, pd.DataFrame] = {}

    def dataset(self, target: str, horizon: int) -> tuple[pd.DataFrame, list[str]]:
        key = (target, horizon)
        if key not in self._datasets:
            df, cols = build_dataset(target, horizon, self.home_id)
            df = df[
                (df["valid_time"] >= pd.Timestamp(self.start))
                & (df["valid_time"] < pd.Timestamp(self.end))
            ].reset_index(drop=True)
            self._datasets[key] = (df, cols)
            logger.info("dataset target=%s h=%d -> %d rows", target, horizon, len(df))
        return self._datasets[key]

    def features(self, config: FeatureConfig, tag: str) -> pd.DataFrame:
        if tag not in self._features:
            # Build over the issue-time grid: feature rows are looked up at
            # valid_time - horizon, and the max horizon is 24h, so extend the
            # build window 24h before the dataset window start.
            f_start = pd.Timestamp(self.start) - pd.Timedelta(hours=max(HORIZONS))
            net = build_features(f_start.to_pydatetime(), self.end, config)
            self._features[tag] = net
            logger.info("features tag=%s -> %d hourly rows, %d cols",
                        tag, len(net), net.shape[1])
        return self._features[tag]


def _network_cols(net: pd.DataFrame) -> list[str]:
    """Network feature columns: everything build_features emits except the raw
    wind-direction reference (a bare angle isn't a useful linear feature)."""
    return [c for c in net.columns if c != "wind_ref_deg"]


def _merge_network(
    base: pd.DataFrame, net: pd.DataFrame, horizon: int, suffix: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    """Attach network features to base rows at valid_time - horizon (causal).

    Returns (merged_df, network_feature_cols). NaN network cells (no upwind
    cohort that hour) are filled 0.0 for both model classes so the control and
    treatment share identical rows; a ``net_present`` flag lets the model tell
    "0 = no data" apart from "0 = genuinely flat".
    """
    cols = _network_cols(net)
    net_r = net[cols].copy()
    net_r.columns = [f"{c}{suffix}" for c in cols]
    cols = list(net_r.columns)

    # tz-aware DatetimeIndex (NOT .values, which strips the tz and makes every
    # lookup miss the tz-aware net index -> all-NaN -> dead network features).
    feature_time = pd.DatetimeIndex(base["valid_time"] - pd.Timedelta(hours=horizon))
    merged = base.copy()
    lookup = net_r.reindex(feature_time)
    for c in cols:
        merged[c] = lookup[c].to_numpy()

    present_col = f"net_present{suffix}"
    n_up_col = f"n_upwind{suffix}"
    if n_up_col in merged.columns:
        merged[present_col] = (merged[n_up_col].fillna(0) > 0).astype(float)
        cols.append(present_col)
    merged[cols] = merged[cols].fillna(0.0)
    return merged, cols


def _merge_multiband(
    base: pd.DataFrame, cache: FeatureCache, horizon: int
) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate per-band feature groups so one model sees every band at once."""
    merged = base
    all_cols: list[str] = []
    for (lo, hi), k in MULTIBAND:
        cfg = FeatureConfig(n_stations=k, distance_band_km=(lo, hi))
        net = cache.features(cfg, tag=f"mb_{lo}_{hi}_n{k}")
        suffix = f"_b{lo:g}_{hi:g}"
        merged, cols = _merge_network(merged, net, horizon, suffix=suffix)
        all_cols.extend(cols)
    return merged, all_cols


# ---------------------------------------------------------------------------
# Train + evaluate one (target, horizon, config)
# ---------------------------------------------------------------------------

def _fit_eval(model_name, X_tr, y_tr, X_te, y_te):
    trainer = train_linear if model_name == "linear" else train_xgboost
    model = trainer(X_tr, y_tr)
    pred = model.predict(X_te)
    return pred, {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
    }


def _bootstrap_delta_ci(y_te, pred_base, pred_net, resamples, seed):
    """Bootstrap 95% CI on (MAE_base - MAE_net): >0 means network helps."""
    rng = np.random.RandomState(seed)
    n = len(y_te)
    if n == 0:
        return None
    abs_base = np.abs(y_te - pred_base)
    abs_net = np.abs(y_te - pred_net)
    deltas = np.empty(resamples)
    for i in range(resamples):
        idx = rng.randint(0, n, n)
        deltas[i] = abs_base[idx].mean() - abs_net[idx].mean()
    return {
        "delta_mae_mean": float(deltas.mean()),
        "delta_mae_lo": float(np.percentile(deltas, 2.5)),
        "delta_mae_hi": float(np.percentile(deltas, 97.5)),
        "p_improves": float((deltas > 0).mean()),
    }


def run_config(
    cache: FeatureCache,
    target: str,
    horizon: int,
    label: str,
    config: Optional[FeatureConfig],
    multiband: bool,
    models: tuple[str, ...],
    resamples: int,
) -> list[dict]:
    """Train base vs net for each model class; return one result dict per model."""
    base, base_cols = cache.dataset(target, horizon)
    if multiband:
        merged, net_cols = _merge_multiband(base, cache, horizon)
    else:
        net = cache.features(config, tag=label)
        merged, net_cols = _merge_network(base, net, horizon)

    merged = merged.dropna(subset=base_cols + ["y"]).reset_index(drop=True)
    train_df, test_df = temporal_split(merged)

    y_tr = train_df["y"].to_numpy(float)
    y_te = test_df["y"].to_numpy(float)
    base_te = test_df["openmeteo_baseline"].to_numpy(float)
    om_mae = float(mean_absolute_error(y_te, base_te))
    om_rmse = float(np.sqrt(mean_squared_error(y_te, base_te)))

    Xb_tr = train_df[base_cols].to_numpy(float)
    Xb_te = test_df[base_cols].to_numpy(float)
    Xn_tr = train_df[base_cols + net_cols].to_numpy(float)
    Xn_te = test_df[base_cols + net_cols].to_numpy(float)

    results = []
    for m in models:
        pred_base, mb = _fit_eval(m, Xb_tr, y_tr, Xb_te, y_te)
        pred_net, mn = _fit_eval(m, Xn_tr, y_tr, Xn_te, y_te)
        ci = _bootstrap_delta_ci(y_te, pred_base, pred_net, resamples, BOOTSTRAP_SEED)
        row = {
            "label": label, "target": target, "horizon": horizon, "model": m,
            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "n_net_features": len(net_cols),
            "mae_base": mb["mae"], "rmse_base": mb["rmse"],
            "mae_net": mn["mae"], "rmse_net": mn["rmse"],
            "delta_mae": mb["mae"] - mn["mae"],
            "openmeteo_mae": om_mae, "openmeteo_rmse": om_rmse,
            "skill_base": 1.0 - mb["rmse"] / om_rmse if om_rmse else None,
            "skill_net": 1.0 - mn["rmse"] / om_rmse if om_rmse else None,
            **(ci or {}),
        }
        results.append(row)
        logger.info(
            "  %-22s h=%2d %-7s  base=%.3f net=%.3f  Δ=%+.3f  CI[%+.3f,%+.3f] p=%.2f",
            label, horizon, m, mb["mae"], mn["mae"], row["delta_mae"],
            row.get("delta_mae_lo", float("nan")),
            row.get("delta_mae_hi", float("nan")),
            row.get("p_improves", float("nan")),
        )
    return results


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def _configs_for_sweep(sweep: str):
    """Yield (label, FeatureConfig | None, multiband_flag) for the sweep mode."""
    if sweep in ("n", "all"):
        for n in SWEEP_N:
            yield (f"n={n}_band{SWEEP_N_BAND[1]:g}",
                   FeatureConfig(n_stations=n, distance_band_km=SWEEP_N_BAND),
                   False)
    if sweep in ("bands", "all"):
        for lo, hi in SWEEP_BANDS:
            yield (f"band_{lo:g}-{hi:g}km",
                   FeatureConfig(n_stations=SWEEP_BANDS_NSTATIONS,
                                 distance_band_km=(lo, hi)),
                   False)
    if sweep in ("multiband", "all"):
        yield ("multiband_mix", None, True)


def main():
    p = argparse.ArgumentParser(description="Network-feature ablation sweep")
    p.add_argument("--sweep", choices=("n", "bands", "multiband", "all"),
                   default="all")
    p.add_argument("--target", default="temp_c,rain_mm_1h",
                   help="comma-separated; feature-building is shared across targets")
    p.add_argument("--horizons", default="1,3,24",
                   help="comma-separated subset of 1,3,24")
    p.add_argument("--models", default="linear,xgboost")
    p.add_argument("--window-start", default=DEFAULT_WINDOW_START)
    p.add_argument("--window-end", default=None,
                   help="default: now (UTC)")
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    p.add_argument("--stamp", default=None,
                   help="output dir suffix; default derived from --window-end")
    p.add_argument("--dry-run", action="store_true",
                   help="run only the first config x first horizon, no file output")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    horizons = tuple(int(h) for h in args.horizons.split(","))
    for h in horizons:
        if h not in SUPPORTED_HORIZONS:
            sys.exit(f"horizon {h} not in SUPPORTED_HORIZONS={SUPPORTED_HORIZONS}")
    models = tuple(m.strip() for m in args.models.split(","))
    targets = tuple(t.strip() for t in args.target.split(","))

    window_start = pd.Timestamp(args.window_start, tz="UTC").to_pydatetime()
    if args.window_end:
        window_end = pd.Timestamp(args.window_end, tz="UTC").to_pydatetime()
    else:
        window_end = datetime.now(timezone.utc)

    engine = create_engine(_sync_dsn())
    try:
        home_id = _resolve_home_station(_load_stations(engine))
    finally:
        engine.dispose()
    logger.info("home station=%s window=%s -> %s", home_id,
                window_start.date(), window_end.date())

    cache = FeatureCache(window_start, window_end, home_id)
    configs = list(_configs_for_sweep(args.sweep))

    all_rows: list[dict] = []
    if args.dry_run:
        label, cfg, mb = configs[0]
        logger.info("DRY RUN: %s @ h=%d target=%s", label, horizons[0], targets[0])
        all_rows += run_config(cache, targets[0], horizons[0], label, cfg, mb,
                               models, args.resamples)
        print(json.dumps(all_rows, indent=2))
        return

    for label, cfg, mb in configs:
        for target in targets:
            for h in horizons:
                all_rows += run_config(cache, target, h, label, cfg, mb,
                                       models, args.resamples)

    # ---- persist ----
    stamp = args.stamp or window_end.strftime("%Y%m%dT%H%M")
    out_dir = Path("experiments") / f"sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep": args.sweep, "targets": list(targets), "horizons": list(horizons),
        "models": list(models),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "home_id": home_id, "resamples": args.resamples,
        "sweep_defs": {
            "SWEEP_N": SWEEP_N, "SWEEP_N_BAND": SWEEP_N_BAND,
            "SWEEP_BANDS": SWEEP_BANDS, "MULTIBAND": MULTIBAND,
        },
    }, indent=2, default=str))
    _write_summary(df, out_dir / "summary.md", args)
    logger.info("wrote %d rows -> %s", len(df), out_dir)
    print(f"\nResults: {out_dir}/summary.md")


def _write_summary(df: pd.DataFrame, path: Path, args) -> None:
    lines = [f"# Ablation sweep — {args.sweep} (target={args.target})", ""]
    for h in sorted(df["horizon"].unique()):
        lines.append(f"## +{h}h horizon\n")
        lines.append("| config | model | MAE base | MAE net | Δ MAE | 95% CI | p(↑) | skill net |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sub = df[df["horizon"] == h].sort_values(["label", "model"])
        for _, r in sub.iterrows():
            ci = (f"[{r.get('delta_mae_lo', float('nan')):+.3f}, "
                  f"{r.get('delta_mae_hi', float('nan')):+.3f}]")
            skill = r["skill_net"]
            skill_s = "—" if skill is None else f"{skill:.3f}"
            lines.append(
                f"| {r['label']} | {r['model']} | {r['mae_base']:.3f} | "
                f"{r['mae_net']:.3f} | {r['delta_mae']:+.3f} | {ci} | "
                f"{r.get('p_improves', float('nan')):.2f} | {skill_s} |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
