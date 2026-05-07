# weather-station

Personal weather station case study: an Ecowitt sensor in a Seattle backyard
streams observations to a local FastAPI service, which compares the
[Open-Meteo](https://open-meteo.com/) public regional forecast against two
locally-trained ML models (linear regression and XGBoost) for the same target.

The whole point is to **measure how a backyard microclimate diverges from the
regional forecast** — the biases are the signal, not noise — and to test how
much of that gap a small ML model can recover from short-window data.

## Architecture

```
Ecowitt sensor  ──HTTP POST──▶  FastAPI /api/ecowitt
                                       │
                                       ▼
                               TimescaleDB (observations + forecasts)
                                       ▲
                                       │
              Open-Meteo  ──hourly fetch (APScheduler)──┘
                                       │
                                       ▼
                       /api/predict?target=temp_c&horizon=1
                                       │
                              ┌────────┴─────────┐
                              ▼                  ▼
                     Open-Meteo baseline   joblib model bundles
                                           (linear, xgboost)
                                       │
                                       ▼
                              Static HTML dashboard at /
```

| Component | Stack |
|----------|-------|
| Ingest API | FastAPI + asyncpg |
| Database | PostgreSQL 16 + TimescaleDB hypertables |
| Scheduler | APScheduler (forecast pulls every hour) |
| ML | scikit-learn (Ridge), XGBoost, joblib persistence |
| UI | Single static HTML page, fetches `/api/predict` |
| Deploy | Docker Compose (locally), nginx + systemd (droplet) |

## Targets and horizons

The dataset/training code is parameterised over `(target, horizon)`:

|              | +1 hour | +24 hours |
|--------------|--------|-----------|
| `temp_c`     | trained | trained   |
| `rain_mm_1h` | wired in (zero-inflated; needs more data + 2-stage architecture) | same |

Models persist as `models/{target}_{horizon}h_{model}.joblib`.

## Initial results (Seattle backyard, 2026-04-09 → 2026-05-06)

About one month of data, ~285 paired (forecast, observation) hourly samples
after a temporal 80/20 split.

| Horizon | Open-Meteo MAE | Linear MAE | XGBoost MAE | Best vs. baseline |
|---------|----------------|------------|-------------|-------------------|
| +1 h    | 2.91 °C        | 1.05 °C    | 1.10 °C     | **−64 %**         |
| +24 h   | 2.90 °C        | 2.91 °C    | 2.60 °C     | **−10 %**         |

Reading: at +1 h, lagged observations contain a lot of signal that lets
even Ridge regression cut the regional forecast's error by nearly 2/3. At
+24 h, the lag features are stale and both ML models barely beat the public
forecast. This is the expected story — and exactly the kind of result that
motivates collecting more data before claiming the trees model "works."

## Running locally

```bash
cp .env.example .env   # set DB_USER, DB_PASSWORD, DB_NAME
docker compose up -d   # postgres + api + grafana
```

API is at http://localhost:8000 (dashboard at `/`, API docs at `/docs`).

To train models against a populated database (the `db` service is exposed
on `127.0.0.1:5433` via the compose file):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DB_HOST=127.0.0.1 DB_PORT=5433 \
DB_USER=weather DB_PASSWORD=... DB_NAME=weatherstation \
  python -m src.ml.train --target temp_c --horizon 1
```

The trained `*.joblib` lands in `models/` (a Docker volume, persisted
across rebuilds).

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/ecowitt` | Ecowitt webhook (form-encoded) |
| GET  | `/api/current` | Latest observation |
| GET  | `/api/predict?target=…&horizon=…` | 3-way comparison |
| GET  | `/api/models` | Inventory of trained models |
| GET  | `/api/stations/{id}/baseline?days=N` | Forecast bias / MAE summary |
| GET  | `/health` | Liveness |
| GET  | `/` | Comparison dashboard |

## Status

This is a personal project, intentionally small and slow-paced. The
modeling timeline is deliberately conservative — see commits and the
`/api/predict` metrics to track how the bias narrows as data accumulates.
