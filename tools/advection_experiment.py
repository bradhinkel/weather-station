"""Does wind-scaled upwind station selection beat fixed-band cohort averaging?

    python -m tools.advection_experiment --horizons 1 3 6 12 24

Three arms, trained on identical rows with an identical temporal split, so any
difference is the feature set and nothing else:

  base    own station + NWP forecast (what the live site serves today)
  cohort  base + the mean of upwind stations in a FIXED distance band at lag 0
          -- the shape src/features/pipeline.py currently produces
  adv     base + the single station nearest the projected point v*h upwind,
          observed at t -- src/features/advection.py

The hypothesis under test: averaging a band smears the signal, because the station
that matters is the one standing where the incoming air currently is, and that
location moves with the wind. If true, `adv` should beat `cohort`, and the margin
should grow with horizon (v*h spreads the candidates further apart as h rises).

Evaluated on the OWN station only. That is the question the project actually asks,
and it is also where the sample is thinnest (~1.4k rows), so ΔMAE carries bootstrap
CIs and small margins should not be over-read.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.features.advection import (
    ADVECTION_FEATURE_COLS,
    build_advection_features,
    impute_advection,
)
from src.features.bearing import angular_distance
from src.ml import SUPPORTED_HORIZONS
from src.ml.dataset import FEATURE_COLS, _sync_dsn, build_dataset
from src.ml.train import temporal_split, train_linear, train_xgboost
from src.pws.distance import haversine_km

logger = logging.getLogger("advection_experiment")

# Fixed-band control. 5-25 km is the ablation sweep's recommended production band,
# +/-45 deg the upwind tolerance -- i.e. the current design at its own best setting.
COHORT_BAND_KM = (5.0, 25.0)
COHORT_TOLERANCE_DEG = 45.0

COHORT_FEATURE_COLS = ["cohort_temp_c", "cohort_temp_gradient", "cohort_n", "cohort_valid"]

BOOTSTRAP_N = 1000


def _load_geometry(engine) -> tuple[str, float, float, dict, dict]:
    with engine.connect() as conn:
        home = conn.execute(
            text("SELECT station_id, lat, lon FROM stations WHERE is_network = false LIMIT 1")
        ).first()
        rows = conn.execute(
            text("""
                SELECT station_id, lat, lon, distance_km, bearing_deg
                FROM stations
                WHERE is_network = true
                  AND lat IS NOT NULL AND lon IS NOT NULL
                  AND COALESCE(quality_flags->>'retired', 'false') <> 'true'
            """)
        ).fetchall()
    coords = {r.station_id: (r.lat, r.lon) for r in rows}
    geom = {r.station_id: (r.distance_km, r.bearing_deg) for r in rows}
    return home.station_id, home.lat, home.lon, coords, geom


def _load_obs_lookup(engine, start, end) -> dict:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT o.station_id,
                       date_trunc('hour', o.time) AS hour,
                       avg(o.temp_c) AS temp_c
                FROM observations o
                JOIN stations s USING (station_id)
                WHERE s.is_network = true
                  AND o.temp_c IS NOT NULL
                  AND o.time >= :start AND o.time < :end
                GROUP BY 1, 2
            """),
            conn,
            params={"start": start, "end": end},
        )
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return {(r.station_id, r.hour): r.temp_c for r in df.itertuples(index=False)}


def _build_cohort_features(df, horizon_h, geom, obs_lookup) -> pd.DataFrame:
    """Fixed-band upwind cohort mean at lag 0 -- the current pipeline's shape."""
    temps, grads, counts, valids = [], [], [], []
    for row in df.itertuples(index=False):
        theta = float(getattr(row, "f_wind_dir_deg"))
        lag_temp = float(getattr(row, "lag_temp_c"))
        obs_hour = pd.Timestamp(getattr(row, "valid_time")) - pd.Timedelta(hours=horizon_h)

        vals = []
        if np.isfinite(theta):
            for sid, (dist_km, bearing) in geom.items():
                if dist_km is None or bearing is None:
                    continue
                if not (COHORT_BAND_KM[0] <= dist_km <= COHORT_BAND_KM[1]):
                    continue
                if angular_distance(bearing, theta) > COHORT_TOLERANCE_DEG:
                    continue
                t = obs_lookup.get((sid, obs_hour))
                if t is not None and np.isfinite(t):
                    vals.append(t)

        if vals:
            mean = float(np.mean(vals))
            temps.append(mean)
            grads.append(mean - lag_temp if np.isfinite(lag_temp) else np.nan)
            counts.append(float(len(vals)))
            valids.append(1.0)
        else:
            temps.append(np.nan); grads.append(np.nan); counts.append(0.0); valids.append(0.0)

    out = df.copy()
    out["cohort_temp_c"] = temps
    out["cohort_temp_gradient"] = grads
    out["cohort_n"] = counts
    out["cohort_valid"] = valids
    return out


def _impute_cohort(df, fill_from):
    out = df.copy()
    for col in ("cohort_temp_c", "cohort_temp_gradient"):
        mean = fill_from[col].mean()
        if not np.isfinite(mean):
            mean = 0.0
        out[col] = out[col].fillna(mean)
    return out


def _bootstrap_delta(y, pred_a, pred_b, rng) -> tuple[float, float, float]:
    """Bootstrap CI on MAE(a) - MAE(b). Positive = b is better than a."""
    err_a = np.abs(y - pred_a)
    err_b = np.abs(y - pred_b)
    n = len(y)
    deltas = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        deltas[i] = err_a[idx].mean() - err_b[idx].mean()
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _fit_eval(train_df, test_df, cols, y_train, y_test):
    X_tr = train_df[cols].to_numpy(dtype=float)
    X_te = test_df[cols].to_numpy(dtype=float)
    out = {}
    for name, trainer in (("linear", train_linear), ("xgboost", train_xgboost)):
        model = trainer(X_tr, y_train)
        out[name] = model.predict(X_te)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = create_engine(_sync_dsn())
    home_id, home_lat, home_lon, coords, geom = _load_geometry(engine)
    logger.info("Home %s at (%.4f, %.4f); %d network stations", home_id, home_lat, home_lon, len(coords))

    rng = np.random.default_rng(42)
    rows = []

    for h in sorted(args.horizons):
        df, _ = build_dataset("temp_c", h, station_id=home_id)
        if df.empty or len(df) < 200:
            logger.warning("+%dh: only %d own-station rows, skipping", h, len(df))
            continue

        obs_lookup = _load_obs_lookup(
            engine,
            df["valid_time"].min() - pd.Timedelta(hours=h + 2),
            df["valid_time"].max() + pd.Timedelta(hours=2),
        )

        df = build_advection_features(df, h, home_lat, home_lon, coords, obs_lookup, home_id)
        df = _build_cohort_features(df, h, geom, obs_lookup)

        train_df, test_df = temporal_split(df)
        # Impute from TRAIN only -- never let a test-set mean leak backwards.
        train_i = _impute_cohort(impute_advection(train_df, fill_from=train_df), train_df)
        test_i = _impute_cohort(impute_advection(test_df, fill_from=train_df), train_df)

        y_tr = train_df["y"].to_numpy(dtype=float)
        y_te = test_df["y"].to_numpy(dtype=float)
        om = test_df["openmeteo_baseline"].to_numpy(dtype=float)

        arms = {
            "base": FEATURE_COLS,
            "cohort": FEATURE_COLS + COHORT_FEATURE_COLS,
            "adv": FEATURE_COLS + ADVECTION_FEATURE_COLS,
        }
        preds = {k: _fit_eval(train_i, test_i, cols, y_tr, y_te) for k, cols in arms.items()}

        valid_rate = float(test_df["adv_valid"].mean())
        reach = float(test_df["adv_distance_km"].median())
        logger.info(
            "+%dh: %d train / %d test | adv valid %.0f%% | median reach %.1f km | OM MAE %.3f",
            h, len(train_df), len(test_df), 100 * valid_rate, reach, np.mean(np.abs(y_te - om)),
        )

        for model in ("linear", "xgboost"):
            maes = {arm: float(np.mean(np.abs(y_te - preds[arm][model]))) for arm in arms}
            d_adv, lo_a, hi_a = _bootstrap_delta(y_te, preds["base"][model], preds["adv"][model], rng)
            d_coh, lo_c, hi_c = _bootstrap_delta(y_te, preds["base"][model], preds["cohort"][model], rng)
            d_vs, lo_v, hi_v = _bootstrap_delta(y_te, preds["cohort"][model], preds["adv"][model], rng)
            rows.append({
                "horizon": h, "model": model, "n_test": len(y_te),
                "adv_valid_pct": round(100 * valid_rate, 1),
                "median_reach_km": round(reach, 1),
                "om_mae": round(float(np.mean(np.abs(y_te - om))), 4),
                "mae_base": round(maes["base"], 4),
                "mae_cohort": round(maes["cohort"], 4),
                "mae_adv": round(maes["adv"], 4),
                "d_adv_vs_base": round(d_adv, 4), "d_adv_lo": round(lo_a, 4), "d_adv_hi": round(hi_a, 4),
                "d_cohort_vs_base": round(d_coh, 4), "d_coh_lo": round(lo_c, 4), "d_coh_hi": round(hi_c, 4),
                "d_adv_vs_cohort": round(d_vs, 4), "d_vs_lo": round(lo_v, 4), "d_vs_hi": round(hi_v, 4),
            })

    engine.dispose()
    if not rows:
        print("No horizons produced results.")
        return 1

    print()
    print("Own-station temperature. Delta > 0 means the second arm is BETTER (MAE fell).")
    print("CIs are bootstrap 95% on the test split; they exclude 0 only if the effect is real.")
    print()
    hdr = (f"{'h':>4} {'model':>8} {'OM':>6} {'base':>6} {'cohort':>7} {'adv':>6} "
           f"{'adv-base':>20} {'adv-cohort':>20} {'reach':>7} {'valid':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{'+' + str(r['horizon']) + 'h':>4} {r['model']:>8} {r['om_mae']:>6.3f} "
            f"{r['mae_base']:>6.3f} {r['mae_cohort']:>7.3f} {r['mae_adv']:>6.3f} "
            f"{r['d_adv_vs_base']:>+7.3f} [{r['d_adv_lo']:>+6.3f},{r['d_adv_hi']:>+6.3f}] "
            f"{r['d_adv_vs_cohort']:>+7.3f} [{r['d_vs_lo']:>+6.3f},{r['d_vs_hi']:>+6.3f}] "
            f"{r['median_reach_km']:>6.1f}k {r['adv_valid_pct']:>5.0f}%"
        )
    print()
    print("reach = median v*h the advection feature projected. valid = % of test rows where")
    print("wind exceeded the calm threshold AND a station sat within the match radius.")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
