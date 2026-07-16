"""Does quality control make the neighbour network useful?

    python -m tools.network_qc_experiment --horizons 1 3 6 12 24

Three arms on identical own-station rows and an identical temporal split:

  base       own station + NWP (what the live site serves)
  bands_all  base + upwind band means over ALL stations   (status quo: no QC)
  bands_qc   base + upwind band means over qc_status='ok' stations only

The `bands_all` vs `bands_qc` contrast is the point. Nipen et al. (2020) report that
without QC the merged citizen product is only marginally better than raw NWP *and
worse in daytime and summer* — which is exactly this project's test window and exactly
where its network features have measured null-to-harmful. If they are right, QC is the
difference between a useless network and a useful one, and this is the experiment that
shows it.

Feature design follows the physics rather than a fixed cohort:

* **Upwind is per-row.** Stations are selected by bearing against the forecast wind
  direction for the target hour, so "upwind" rotates with the weather instead of being
  frozen at one arc. (The project's recorded assumption of a contiguous SW-S-W arc is
  wrong for this data: the wind is bimodal — SW 27%, N 25%, NE 20% — and W is the
  *rarest* octant at 3.9%.)

* **Bands, not one radius.** Four distance bands enter as separate features so the model
  can learn which reach matters at which lead, rather than having v*h hardcoded by an
  author whose hand-built advection model already failed once. At the SW mean wind of
  3.14 m/s the bands correspond to roughly +1h, +3h, +6h and +9h of travel.

* **Elevation-adjusted before averaging.** The registry spans 0-1174 m (~7.6 C of lapse
  rate). Averaging raw temperatures across a band would let one foothills station drag
  the mean and call it weather.

* **Averaging, deliberately.** A single station carries its own siting error — every CWS
  study finds ~0.5-1.0 C of residual per-station bias surviving QC — so picking one
  station maximises exposure to exactly the noise that dominates here. Averaging kills
  the random component as sqrt(N) while leaving the systematic part, which is why the
  literature saturates at ~4 stations (Nipen) and why this project's own sweep plateaued
  at n=1.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.features.bearing import angular_distance
from src.ml import SUPPORTED_HORIZONS
from src.ml.dataset import FEATURE_COLS, _sync_dsn, build_dataset
from src.ml.train import temporal_split, train_linear, train_xgboost
from src.pws.qc import elevation_adjust

logger = logging.getLogger("network_qc_experiment")

# Bands sized to advection reach at the SW mean wind (3.14 m/s): ~11 km/h of travel.
BANDS_KM = [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0)]
UPWIND_TOLERANCE_DEG = 45.0
BOOTSTRAP_N = 1000


def band_cols() -> list[str]:
    cols: list[str] = []
    for i in range(len(BANDS_KM)):
        cols += [f"band{i}_temp", f"band{i}_grad"]
    cols.append("bands_n_total")
    return cols


def load_geometry(engine, qc_only: bool):
    clause = "AND quality_flags->>'qc_status' = 'ok'" if qc_only else ""
    with engine.connect() as conn:
        home = conn.execute(
            text("SELECT station_id, lat, lon, elevation_m FROM stations WHERE is_network = false LIMIT 1")
        ).first()
        rows = conn.execute(
            text(f"""
                SELECT station_id, elevation_m, distance_km, bearing_deg
                FROM stations
                WHERE is_network = true
                  AND distance_km IS NOT NULL AND bearing_deg IS NOT NULL
                  AND elevation_m IS NOT NULL
                  AND COALESCE(quality_flags->>'retired', 'false') <> 'true'
                  {clause}
            """)
        ).fetchall()
    geom = {r.station_id: (r.distance_km, r.bearing_deg, r.elevation_m) for r in rows}
    return home, geom


def load_obs(engine, start, end) -> dict:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT o.station_id, date_trunc('hour', o.time) AS hour, avg(o.temp_c) AS temp_c
                FROM observations o JOIN stations s USING (station_id)
                WHERE s.is_network = true AND o.temp_c IS NOT NULL
                  AND o.time >= :start AND o.time < :end
                GROUP BY 1, 2
            """),
            conn, params={"start": start, "end": end},
        )
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    out = defaultdict(dict)
    for r in df.itertuples(index=False):
        out[r.hour][r.station_id] = r.temp_c
    return out


def build_band_features(df, horizon_h, geom, obs_by_hour, home_elev) -> pd.DataFrame:
    """Elevation-adjusted upwind band means at t, per row."""
    n_bands = len(BANDS_KM)
    acc = {f"band{i}_{k}": [] for i in range(n_bands) for k in ("temp", "grad")}
    totals = []

    for row in df.itertuples(index=False):
        theta = float(getattr(row, "f_wind_dir_deg"))
        lag_temp = float(getattr(row, "lag_temp_c"))
        hour = pd.Timestamp(getattr(row, "valid_time")) - pd.Timedelta(hours=horizon_h)
        temps_now = obs_by_hour.get(hour, {})

        per_band: list[list[tuple[float, float]]] = [[] for _ in range(n_bands)]
        if np.isfinite(theta) and temps_now:
            for sid, (dist_km, bearing, elev) in geom.items():
                if angular_distance(bearing, theta) > UPWIND_TOLERANCE_DEG:
                    continue
                t = temps_now.get(sid)
                if t is None or not np.isfinite(t):
                    continue
                for bi, (lo, hi) in enumerate(BANDS_KM):
                    if lo <= dist_km < hi:
                        per_band[bi].append((t, elev))
                        break

        total = 0
        for bi in range(n_bands):
            vals = per_band[bi]
            if vals:
                temps = np.array([v[0] for v in vals], dtype=float)
                elevs = np.array([v[1] for v in vals], dtype=float)
                adj = elevation_adjust(temps, elevs, home_elev)
                mean = float(np.mean(adj))
                acc[f"band{bi}_temp"].append(mean)
                acc[f"band{bi}_grad"].append(mean - lag_temp if np.isfinite(lag_temp) else np.nan)
                total += len(vals)
            else:
                acc[f"band{bi}_temp"].append(np.nan)
                acc[f"band{bi}_grad"].append(np.nan)
        totals.append(float(total))

    out = df.copy()
    for k, v in acc.items():
        out[k] = v
    out["bands_n_total"] = totals
    return out


def impute(df, fill_from):
    """Train-mean imputation. Never fill-0: a 0 C band mean is physically absurd and is
    the 2026-06-19 defect that blew Ridge to MAE 20-30."""
    out = df.copy()
    for c in band_cols():
        if c == "bands_n_total":
            continue
        m = fill_from[c].mean()
        out[c] = out[c].fillna(0.0 if not np.isfinite(m) else m)
    return out


def bootstrap_delta(y, pa, pb, rng):
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    n = len(y)
    d = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        d[i] = ea[idx].mean() - eb[idx].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fit_eval(tr, te, cols, ytr):
    Xtr, Xte = tr[cols].to_numpy(dtype=float), te[cols].to_numpy(dtype=float)
    return {
        "linear": train_linear(Xtr, ytr).predict(Xte),
        "xgboost": train_xgboost(Xtr, ytr).predict(Xte),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(SUPPORTED_HORIZONS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = create_engine(_sync_dsn())
    home, geom_all = load_geometry(engine, qc_only=False)
    _, geom_qc = load_geometry(engine, qc_only=True)
    logger.info("home elev %.0fm | %d stations all, %d qc-ok (%.0f%% dropped)",
                home.elevation_m, len(geom_all), len(geom_qc),
                100 * (1 - len(geom_qc) / max(len(geom_all), 1)))

    rng = np.random.default_rng(42)
    rows = []

    for h in sorted(args.horizons):
        df, _ = build_dataset("temp_c", h, station_id=home.station_id)
        if df.empty or len(df) < 200:
            logger.warning("+%dh: %d rows, skipping", h, len(df))
            continue
        df["f_wind_dir_deg"] = (
            np.degrees(np.arctan2(df["wind_dir_sin"], df["wind_dir_cos"])) + 360.0
        ) % 360.0

        obs = load_obs(engine, df["valid_time"].min() - pd.Timedelta(hours=h + 2),
                       df["valid_time"].max() + pd.Timedelta(hours=2))

        d_all = build_band_features(df, h, geom_all, obs, home.elevation_m)
        d_qc = build_band_features(df, h, geom_qc, obs, home.elevation_m)

        tr_a, te_a = temporal_split(d_all)
        tr_q, te_q = temporal_split(d_qc)
        tr_a, te_a = impute(tr_a, tr_a), impute(te_a, tr_a)
        tr_q, te_q = impute(tr_q, tr_q), impute(te_q, tr_q)

        ytr, yte = tr_a["y"].to_numpy(dtype=float), te_a["y"].to_numpy(dtype=float)
        om = te_a["openmeteo_baseline"].to_numpy(dtype=float)

        p_base = fit_eval(tr_a, te_a, FEATURE_COLS, ytr)
        p_all = fit_eval(tr_a, te_a, FEATURE_COLS + band_cols(), ytr)
        p_qc = fit_eval(tr_q, te_q, FEATURE_COLS + band_cols(), ytr)

        logger.info("+%dh: %d/%d rows | upwind stations/row: all %.1f, qc %.1f | OM %.3f",
                    h, len(tr_a), len(te_a), d_all["bands_n_total"].mean(),
                    d_qc["bands_n_total"].mean(), np.mean(np.abs(yte - om)))

        for m in ("linear", "xgboost"):
            mae = {k: float(np.mean(np.abs(yte - p[m]))) for k, p in
                   (("base", p_base), ("all", p_all), ("qc", p_qc))}
            d_a, la, ha = bootstrap_delta(yte, p_base[m], p_all[m], rng)
            d_q, lq, hq = bootstrap_delta(yte, p_base[m], p_qc[m], rng)
            d_v, lv, hv = bootstrap_delta(yte, p_all[m], p_qc[m], rng)
            rows.append({
                "horizon": h, "model": m, "n_test": len(yte),
                "om_mae": round(float(np.mean(np.abs(yte - om))), 4),
                "mae_base": round(mae["base"], 4), "mae_all": round(mae["all"], 4),
                "mae_qc": round(mae["qc"], 4),
                "d_all_vs_base": round(d_a, 4), "all_lo": round(la, 4), "all_hi": round(ha, 4),
                "d_qc_vs_base": round(d_q, 4), "qc_lo": round(lq, 4), "qc_hi": round(hq, 4),
                "d_qc_vs_all": round(d_v, 4), "v_lo": round(lv, 4), "v_hi": round(hv, 4),
            })

    engine.dispose()
    if not rows:
        print("No results.")
        return 1

    print()
    print("Own-station temperature. Delta > 0 = the second arm is BETTER (MAE fell).")
    print()
    hdr = (f"{'h':>4} {'model':>8} {'OM':>6} {'base':>6} {'all':>6} {'qc':>6} "
           f"{'QC vs base':>21} {'QC vs no-QC':>21}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{'+' + str(r['horizon']) + 'h':>4} {r['model']:>8} {r['om_mae']:>6.3f} "
              f"{r['mae_base']:>6.3f} {r['mae_all']:>6.3f} {r['mae_qc']:>6.3f} "
              f"{r['d_qc_vs_base']:>+7.3f} [{r['qc_lo']:>+6.3f},{r['qc_hi']:>+6.3f}] "
              f"{r['d_qc_vs_all']:>+7.3f} [{r['v_lo']:>+6.3f},{r['v_hi']:>+6.3f}]")
    print()
    print("'QC vs no-QC' is the question: does screening the network make it useful?")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
