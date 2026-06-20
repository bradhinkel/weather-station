# weather-station

> **Live:** [weather.bradhinkel.com](https://weather.bradhinkel.com)

Personal weather station case study: an Ecowitt sensor in a Seattle backyard
streams observations to a FastAPI service, which compares the
[Open-Meteo](https://open-meteo.com/) public regional forecast against two
locally-trained ML models (Ridge regression and XGBoost) at three horizons
(+1 h, +3 h, +24 h).

The whole point is to **measure how a backyard microclimate diverges from the
regional forecast** — the biases are the signal, not noise — and to track how
much of that gap a small ML model can recover as data accumulates.

## Live results (Seattle backyard)

> The table below is the **first published window** (retrained 2026-05-07), kept
> as an illustrative snapshot of the expected error-vs-horizon curve. **Current
> numbers are live on [weather.bradhinkel.com](https://weather.bradhinkel.com)**
> — the model retrains weekly and the site's comparison panel reflects the
> latest retrain.

About one month of data, 240–305 paired (forecast, observation) hourly samples
per horizon after a temporal 80/20 split.

| Horizon | Open-Meteo MAE | Linear MAE | XGBoost MAE | Best vs. baseline |
|---------|----------------|------------|-------------|-------------------|
| +1 h    | 2.91 °C        | 1.09 °C    | 1.10 °C     | **−63 %**         |
| +3 h    | 2.91 °C        | 2.09 °C    | 2.21 °C     | **−28 %**         |
| +24 h   | 2.90 °C        | 2.90 °C    | 2.60 °C     | **−10 %**         |

Reading: at +1 h, lagged observations contain a lot of signal that lets even
Ridge regression cut the regional forecast's error by nearly 2/3. By +24 h the
lag features are stale and ML barely edges the public forecast — that's the
expected curve, and seeing it bend is part of the point. The +3 h row sits
exactly in between, as data thinning would predict. A `model_metrics` table
appends a row on every retrain so this curve becomes a time-series.

Rain target is now trainable. The earlier "no rain to train on" blocker was
actually a dataset bug: the builder derived the hourly rain target purely from
`rain_mm_daily_total` deltas, which the network (WU) stations never populate and
which recovered only a fraction of the own station's rain — so `build_dataset`
returned a single non-zero sample. It now prefers the station-reported
`rain_mm_1h` column (both sources populate it), falling back to the daily-total
delta only when that's missing, which surfaces ~11k positive samples per
horizon. Caveat: rain is zero-inflated, so MAE flatters a near-zero predictor —
treat the single-stage regressor here as a pipeline-shakedown, not a skilled
rain forecast. The next step toward real skill is the 2-stage
classifier-then-regressor, judged on rain/no-rain precision/recall.

## Architecture

```
                                    ┌────────────────────────┐
Ecowitt GW2000 ──HTTP──▶  /api/ecowitt│ FastAPI                │
                                    │   • lifespan: init_db,  │
                                    │     scheduler, prefetch │
                                    │   • mtime-cached models │
Open-Meteo ──hourly job──▶  forecasts│   • icons + feels-like  │
                                    └─────────┬──────────────┘
                                              ▼
                              PostgreSQL 16 + TimescaleDB
                              hypertables: observations,
                                           forecasts,
                                           model_metrics
                                              │
                            ┌─────────────────┴─────────────────┐
                            ▼                                   ▼
              /api/predict?target=&horizon=          /api/current, /api/models,
              → Open-Meteo + Ridge + XGBoost           /api/metrics_history
                                              │
                                              ▼
                              Static HTML dashboard at /
                              (vanilla JS, segmented horizon control,
                               weather icons, 3-way comparison)
```

| Component | Stack |
|----------|-------|
| Ingest API | FastAPI + SQLAlchemy + asyncpg |
| Database | PostgreSQL 16 + TimescaleDB 2 |
| Scheduler | APScheduler — forecast pull every hour, weekly retrain |
| ML | scikit-learn (Ridge), XGBoost, joblib bundles |
| Solar / day-night | [pysolar](https://pypi.org/project/pysolar/) |
| Icons | [basmilius/weather-icons](https://github.com/basmilius/weather-icons) static-fill SVG (vendored, MIT) |
| UI | One static HTML page, vanilla JS, fetches the JSON endpoints |
| Production | Ubuntu 24.04 droplet, native Python venv + systemd, nginx + Let's Encrypt |
| Local dev | Docker Compose (`db` + `api` + `grafana`) |

## Targets and horizons

Dataset/training code is parameterised over `(target, horizon)`:

|              | +1 h | +3 h | +24 h |
|--------------|------|------|-------|
| `temp_c`     | trained | trained | trained |
| `rain_mm_1h` | wired in (zero-inflated; needs more data + 2-stage architecture) | same | same |

Models persist as `models/{target}_{horizon}h_{model}.joblib`.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/ecowitt` | Ecowitt sensor webhook (form-encoded) |
| GET  | `/api/current` | Latest observation enriched with `feels_like_c`, `icon_slug`, `cond_label` |
| GET  | `/api/predict?target=…&horizon=…` | 3-way comparison + `weather_code`, `icon_slug`, `cond_label`, `precip_prob_pct` |
| GET  | `/api/models` | Inventory of currently-loadable model bundles |
| GET  | `/api/metrics_history?target=…&horizon=…&model=…` | Time-series of training-run metrics for plotting |
| GET  | `/api/stations/{id}/baseline?days=N` | Forecast bias / MAE summary by lead time |
| GET  | `/health` | Liveness |
| GET  | `/` | Comparison dashboard |

## Running locally

```bash
cp .env.example .env   # set DB_USER, DB_PASSWORD, DB_NAME
docker compose up -d   # postgres + api + grafana
```

API at <http://localhost:8000>; dashboard at `/`; OpenAPI docs at `/docs`.
The `db` service is exposed on `127.0.0.1:5433` so a host venv can run
training scripts:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DB_HOST=127.0.0.1 DB_PORT=5433 \
DB_USER=weather DB_PASSWORD=... DB_NAME=weatherstation \
  python -m src.ml.train --target temp_c --horizon 3
```

Trained `*.joblib` lands in `models/`. The API's predict cache reloads on
mtime change — no service restart required after a retrain.

## Production layout

The droplet runs the whole stack natively (no Docker) to share a host with
two unrelated sites. Conventions:

- App dir: `/opt/weather-station/` (owner `www-data`)
- venv: `/opt/weather-station/venv/`
- env file: `/opt/weather-station/.env`
- systemd: `weather-backend.service` (uvicorn, port 8003) + `weather-retrain.timer`
- nginx: `/etc/nginx/sites-enabled/weather.bradhinkel.com` proxies `→ 127.0.0.1:8003`. The `server_name` includes the droplet IP because Ecowitt firmware sends the resolved IP as the HTTP `Host` header.
- `weather-retrain.timer`: weekly, Sundays 11:00 UTC; appends to `model_metrics`. New `(target, horizon)` combinations need to be bootstrapped manually with `python -m src.ml.train` once before the timer picks them up.

## Neighbor-station sweep (hyperlocal feature selection)

A core part of the process. The path to better hyperlocal accuracy is using
**nearby Weather Underground stations as upwind/advection features** for the
own-station forecast. *Which* neighbors help — how many, and at what distances —
depends on your location, so it's a measurement, not a guess. The sweep harness
finds the best mix for a given site:

```bash
# validate wiring + time one config (~2 min)
python -m tools.ablation_sweep --dry-run
# full sweep: both targets, +1/+3/+24h, n-count + distance-band + multiband modes
python -m tools.ablation_sweep --sweep all
```

It trains **base (own + NWP)** vs **net (base + network)** on identical rows and
the same temporal split per (target, horizon, `FeatureConfig`), so the reported
ΔMAE is purely the network contribution, with bootstrap CIs. Results land in
`experiments/sweep_<stamp>/` (start with `FINDINGS.md`).

**Location-independent** — register your own station, discover the network
around *its* coordinates, and the sweep finds the mix for *your* microclimate.
Full reproduction guide (prerequisites, data accrual, reading the output):
[`experiments/running_the_sweep.md`](experiments/running_the_sweep.md).

> Status: **exploratory / offline.** The latest run (~30-day window) shows
> upwind features help temperature at +1 h/+3 h (Ridge), don't help at +24 h,
> and a small cohort (n≈3–5) in a single mid-range band captures most of it.
> These features are **not yet wired into the served model** — the live site
> currently runs the own-station + NWP model.

## Status & roadmap

| Milestone | When |
|-----------|------|
| Linear regression baseline | ✅ shipped 2026-05-06 |
| XGBoost (early — under-data on purpose) | ✅ shipped 2026-05-06 |
| Public deploy + first weekly retrain | ✅ shipped 2026-05-07 |
| +3 h horizon | ✅ shipped 2026-05-07 |
| Atmospheric UX (icons, feels-like, segmented control) | ✅ shipped 2026-05-07 |
| First "real" results window with ~6 weeks of data | target ~2026-06-15 |
| Random Forest with confidence bands | with the trees milestone |
| LSTM experiment (intentionally under-data) | target ~2026-11-01 |

This is a personal project, intentionally small and slow-paced. The modeling
timeline is deliberately conservative — watch the `model_metrics` time-series
to see how the bias narrows as data accumulates.

The original design hand-off that drove the atmospheric UX changes is preserved
in `Hand-off - Atmospheric UX.md`.
