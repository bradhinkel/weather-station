# weather-station

> **Live:** [weather.bradhinkel.com](https://weather.bradhinkel.com)

Personal weather station case study: an Ecowitt sensor in a Seattle backyard
streams observations to a FastAPI service, which compares the
[Open-Meteo](https://open-meteo.com/) public regional forecast against three
locally-trained ML models (Ridge, RandomForest, XGBoost) at five horizons
(+1 h, +3 h, +6 h, +12 h, +24 h), plus a two-stage rain model, across a
~320-station Weather Underground neighbour network.

The whole point is to **measure how a backyard microclimate diverges from the
regional forecast** — the biases are the signal, not noise — and to track how
much of that gap a small ML model can recover as data accumulates.

Findings so far are **preliminary and the preliminary part is the point**: ~8
weeks of paired data, a summer-only test window, and five shipped data bugs whose
corrections have been more instructive than any of the results. If you read one
section, read [Errors made](#errors-made).

## Live results

> Current **served-model** metrics (temperature), retrained **2026-07-15** on
> ~290k pooled station-hours, temporal 80/20 split (n_test ≈ 57k). Every number
> below is reproducible from
> [`experiments/model_metrics.csv`](experiments/model_metrics.csv) (stamped with
> the git SHA and row count). Live panel:
> [weather.bradhinkel.com](https://weather.bradhinkel.com).

| Horizon | Open-Meteo MAE | Linear MAE | RandomForest MAE | XGBoost MAE | Best skill vs. baseline |
|---------|----------------|------------|------------------|-------------|-------------------------|
| +1 h    | 1.439 °C       | 0.711      | 0.691            | **0.691**   | **52 %**                |
| +3 h    | 1.456 °C       | 1.223      | 1.316            | **1.133**   | 22 %                    |
| +6 h    | 1.478 °C       | 1.472      | 1.511            | **1.334**   | 10 %                    |
| +12 h   | 1.529 °C       | 1.864      | 1.632            | **1.508**   | **1.4 %**               |
| +24 h   | 1.813 °C       | 1.798      | 1.763            | **1.591**   | 12 %                    |

**Read the +12 h row, not the +1 h row.** The interesting result here is not that
a local model beats a regional forecast an hour out — persistence does that. It is
that the advantage *collapses at +12 h and partially recovers at +24 h*. That
shape is the **diurnal cycle**: the model's strongest local input is an
observation from `t − horizon`, so at +12 h that observation is maximally out of
phase with the target (noon predicting midnight), while at +24 h it is back in
phase. Local data has a **shelf life, and the shelf life is not monotonic**.

Ridge goes **negative-skill at +12 h** (1.864 vs Open-Meteo's 1.529) — actively
worse than doing nothing. Only the trees stay ahead of the baseline across the
whole range, and XGBoost leads at every horizon on ~230k training rows. That
ordering is data-dependent, not a law: on the ~590-row ablation window Ridge beat
XGBoost outright. See [`tools/learning_curve.py`](tools/learning_curve.py), which
measures MAE vs. N rather than asserting that trees win eventually.

**RandomForest is handicapped here and the number should not be read as a verdict
on the model class.** It is capped at 100 trees / depth 10 because the API
mtime-caches every bundle in memory and there are now 5 horizons × 2 targets of
them on a 4 GB droplet that also hosts Postgres; an unconstrained forest is ~100 MB
each, ~1 GB across the set. This is a deployment constraint being reported
honestly, not a fair fight.

### More data is not the lever — variety is

The project's standing assumption was that accuracy would improve as the corpus
grows toward a year. Measured, that is **false for row count**. Holding the calendar
window fixed at 44.2 days and varying only how many rows the model sees
(`--sample random`, +3 h, Open-Meteo baseline 1.456):

| n_train | span | Linear | RandomForest | XGBoost |
|---------|------|--------|--------------|---------|
| 11 527  | 44.2 d | 1.243 | 1.221 | **1.162** |
| 23 055  | 44.2 d | 1.233 | 1.283 | **1.152** |
| 57 639  | 44.2 d | 1.228 | 1.284 | **1.157** |
| 115 279 | 44.2 d | 1.226 | 1.298 | **1.127** |
| 172 919 | 44.2 d | 1.225 | 1.290 | **1.133** |
| 230 559 | 44.2 d | 1.223 | 1.315 | **1.135** |

**The curve is flat.** A 20× increase in training rows buys XGBoost 0.027 °C. The
model plateaued around ~11.5k rows; the corpus holds 230k. Collecting more hours of
the same weather does essentially nothing.

Read the scope carefully. Random sampling holds the window at 44 days, so this
measures *more rows from the same season* — and that is exhausted. What a year would
add is **seasonal variety** (winter, frontal passages, a wet season), which this
cannot test because those months do not exist in the corpus yet. The lever is
different data, not more data.

XGBoost also wins at **every** row count, including 11 527 — there is no
linear→tree crossover in this range. (The one prior data point where Ridge beat
XGBoost was ~590 rows on a different feature set, far below anything here.)
RandomForest *degrading* with more data (1.221 → 1.315) is most likely the depth-10
memory cap: a fixed-depth tree averaging over increasingly heterogeneous data gains
bias. That is an artifact of the sizing decision above, not a property of the model.

Raw data: [`experiments/learning_curve_temp_3h_random.csv`](experiments/learning_curve_temp_3h_random.csv).
The `--sample recent` variant
([`…_temp_3h.csv`](experiments/learning_curve_temp_3h.csv)) shrinks the calendar span
along with n and is **not** a model comparison — at n=11 527 it reports Ridge at MAE
**26.177 °C**, because a ~2.3-day window makes `doy_sin`/`doy_cos` constant,
`StandardScaler` divides by a near-zero std, and Ridge extrapolates onto a test set
weeks away. That number is real output answering a question nobody asked; it is kept
as a worked example of bug class 4, committed by the very tool built to audit the
others.

### The founding question: can the backyard beat the forecast?

The table above is **regional** skill. Training pools all ~320 network stations
(`build_dataset(station_id=None)`), so the model learns the region-average
forecast→observation map and the backyard is <1 % of the rows. Scoring the same
served models on **own-station test rows only** — taking the split from the pooled
frame so those rows were genuinely held out — answers the question the project was
actually started to ask:

| Horizon | Open-Meteo | Linear | RandomForest | XGBoost | best skill |
|---------|-----------|--------|--------------|---------|------------|
| +1 h    | 0.952 °C  | **0.550** | 0.627     | 0.616   | **+42 %**  |
| +3 h    | 0.973 °C  | **0.880** | 1.020     | 0.941   | +9.6 %     |
| +6 h    | 1.030 °C  | 1.158  | 1.212        | **1.050** | **−2 %** |
| +12 h   | 1.141 °C  | 1.619  | 1.342        | **1.150** | **−0.9 %** |
| +24 h   | 1.397 °C  | 1.555  | 1.468        | **1.377** | +1.4 %   |

**The honest answer is: only at +1 h.** Beyond +3 h every served model is at or
below the raw regional forecast in the backyard — negative skill means *worse than
doing nothing*. Ridge at +12 h is 42 % worse than simply believing Open-Meteo.

This is the pooled-vs-own problem with numbers on it for the first time. A model
trained on the region-average map corrects a sheltered backyard *toward the regional
mean*, which is the wrong direction for a microclimate. Regional skill of
52/22/10/1.4/12 % becomes 42/10/−2/−1/1 % in the yard it was built for. **The
regional win does not transfer.**

Note also that **Ridge beats both trees on the microclimate at +1 h** (0.550 vs
0.627/0.616) — the inverse of the pooled ranking, where XGBoost led everywhere. The
trees' pooled advantage comes from fitting the regional signal harder, and that is
precisely the signal a microclimate model does not want.

⚠️ **n_own is only 276–295 rows per horizon** (vs ~57k pooled), and these figures
carry no confidence intervals. The pattern is consistent across all five horizons
and three models, which is why it is reported; a single cell should not be quoted.
Reproduce with:

```bash
python -m tools.own_station_eval --target temp_c
```

Training on the own-station target with neighbour upwind features — rather than
scoring a pooled model against the backyard after the fact — is the active next
step, and the numbers above are the argument for it.

> A previous version of this section claimed the regional average was *easier* to
> forecast than the microclimate (1.68 °C pooled vs ~2.9 °C backyard). That
> comparison was wrong in three ways at once: the 2.9 °C came from **2026-05-07
> with 234 training rows**, the 1.68 °C from a June pooled run, and the two used
> different station populations. Measured properly at true lead times over the
> same window, Open-Meteo forecasts the **own station better** (1.013 °C at +1 h)
> **than the network average** (1.553 °C) — because the pooled baseline is
> inflated by ~320 crowdsourced stations of mixed siting and calibration quality,
> not because the backyard is hard. Note that pooled error barely grows with lead
> (+10 % from +1 h to +24 h) while own-station error grows sharply (+25 %): the
> pooled number is dominated by a constant *sensor-noise floor* that does not care
> about forecast lead. The instrument is part of the measurement.

### Rain

Rain is trained as a two-stage classifier→regressor
([`src/ml/rain_model.py`](src/ml/rain_model.py)): stage 1 predicts P(rain > 0.1 mm)
with `scale_pos_weight` favouring recall, stage 2 predicts amount on wet hours
only. MAE is a *worthless* metric on a zero-inflated target — a predictor that
always says "dry" scores well — so rain is judged on precision/recall/F1.

| Horizon | Precision | Recall | F1 | PR-AUC | wet hours in test |
|---------|-----------|--------|-------|--------|-------------------|
| +1 h    | 0.938     | 0.971  | **0.955** | 0.978 | 2208 / 57 949 (3.8 %) |
| +3 h    | 0.672     | 0.919  | **0.776** | 0.929 | 2173 / 57 807 |
| +6 h    | 0.435     | 0.826  | **0.570** | 0.826 | 2108 / 57 622 |
| +12 h   | 0.116     | 0.746  | **0.201** | 0.624 | 2003 / 57 330 |
| +24 h   | 0.078     | 0.545  | **0.136** | 0.112 | 1891 / 54 298 |

Real chance-of-rain skill exists at **+1 h and +3 h** and is gone by +12 h, where
precision of 0.116 means ~9 of every 10 rain warnings are false alarms. The +1 h
model is largely persistence — it is raining, so it will still be raining. This is
the honest shape of the result and it is what the site now shows.

## Errors made

This project has shipped five data bugs that produced **plausible, publishable,
wrong numbers**. None of them raised an exception. None were caught by a unit test.
Each was caught by noticing that a number was physically impossible. They are
listed here because the corrections are more informative than the results, and
because a reader deserves to know which published figures were later retracted.

| # | Bug | What it produced | The tell |
|---|-----|------------------|----------|
| 1 | `timezone=auto` parsed local-time forecasts into a `TIMESTAMPTZ` column, shifting every `valid_time` by 7 h (2026-05-06) | Open-Meteo looked ~3× worse than reality (5.6 vs 2.8 °C); the first ML model "beat" it by 5× while merely learning the offset | A skill number too good to be true |
| 2 | Rain target derived only from `rain_mm_daily_total` deltas, a column WU stations never populate (2026-06-01) | `build_dataset` returned **1 positive sample in 76k**. The README blamed the weather ("no rain to train on") for a month | A wet-hour rate of ~0 in a Seattle spring |
| 3 | A tz-strip in the ablation feature merge silently zeroed every network column (2026-06-19) | Every sweep config returned **byte-identical** results | Identical-across-configs |
| 4 | fill-0 on physically-offset fields (pressure ~1015 hPa) (2026-06-19) | StandardScaler/Ridge blew up to MAE 20–30 | A magnitude no instrument could produce |
| 5 | The forecast join ignored the horizon (2026-07-15) | Training used a **~1 h-lead forecast at every horizon**, so "+24 h" measured only lag-feature staleness. Caused train/serve skew widening with horizon, and pinned the baseline to a flat **1.68 °C** | Forecast error that does not grow with lead time |

**Bug 5 is the instructive one**, because it survived longest and corrupted the
headline. `nearest_forecast` selected the freshest row with
`forecast_time < valid_time`; `:horizon` only shifted the *lag observation*. For a
historical target hour the freshest forecast is ~1 h old, so the "+24 h model" was
really *predict t from a 1 h-lead forecast for t, plus an observation from t−24 h*.
Serving, meanwhile, asks for `valid_time = now + horizon` and can only ever see a
24 h-lead forecast — so the model was trained to trust a near-nowcast and then met
a much noisier input in production. The flat baseline was the visible symptom of an
invisible train/serve skew.

**What it cost:** every horizon-dependent number this project published before
2026-07-15 was wrong. The retracted figures include the headline **−60 %/−31 %/−24 %**
temperature table, and the rain F1 of **0.89 at +3 h** (true value **0.776**) and
**0.01 at +24 h** (true value **0.136**). Rain F1 at +1 h (0.955 vs 0.96 claimed)
barely moved — the consistency check that confirms the diagnosis, since at +1 h the
training and serving leads coincide and the bug was inert.

**The lesson is not "write more tests."** The test suite covered `bearing`,
`aggregation`, `wind_reference`, `gradient` — pure functions with obvious contracts,
none of which ever had a bug. All five bugs lived in SQL, joins, and ingest, where
correctness is invisible and failure is silent. *The parts that are easy to test are
not the parts that lie to you.* What caught every one of them was a physical
plausibility check applied by hand.

So those checks are now code. [`src/ml/invariants.py`](src/ml/invariants.py) asserts
that baseline error grows with lead time, that no feature is silently constant, that
values sit inside physical bounds, that the forecast lead honours the horizon, and
that the wet-hour rate is climatologically plausible — one predicate per bug above.
[`tools/check_invariants.py`](tools/check_invariants.py) runs them against the live
corpus and exits non-zero, so it can gate a retrain:

```bash
python -m tools.check_invariants --target temp_c
```

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
              → Open-Meteo + Ridge + RF + XGB           /api/metrics_history
                                              │
                                              ▼
                              Static HTML dashboard at /
                              (vanilla JS, segmented horizon control,
                               weather icons, 4-way comparison)
```

| Component | Stack |
|----------|-------|
| Ingest API | FastAPI + SQLAlchemy + asyncpg |
| Database | PostgreSQL 16 + TimescaleDB 2 |
| Scheduler | APScheduler — forecast pull every hour, weekly retrain |
| ML | scikit-learn (Ridge, RandomForest), XGBoost, joblib bundles |
| Solar / day-night | [pysolar](https://pypi.org/project/pysolar/) |
| Icons | [basmilius/weather-icons](https://github.com/basmilius/weather-icons) static-fill SVG (vendored, MIT) |
| UI | One static HTML page, vanilla JS, fetches the JSON endpoints |
| Production | Ubuntu 24.04 droplet, native Python venv + systemd, nginx + Let's Encrypt |
| Local dev | Docker Compose (`db` + `api` + `grafana`) |

## Targets and horizons

Dataset/training code is parameterised over `(target, horizon)`. All ten
combinations are trained and served:

|              | +1 h | +3 h | +6 h | +12 h | +24 h |
|--------------|------|------|------|-------|-------|
| `temp_c`     | trained | trained | trained | trained | trained |
| `rain_mm_1h` | trained | trained | trained | trained | trained |

Each combination fits Ridge, RandomForest and XGBoost; rain additionally fits the
two-stage model. Models persist as `models/{target}_{horizon}h_{model}.joblib`.

A row is `(lag observation at t, forecast issued for t+h, actual at t+h)`. **Both**
the lag observation and the forecast are horizon-lagged: the forecast selected is
the freshest one issued at or before `valid_time - horizon`, which is exactly what
`predict.py` can see when serving. Getting this wrong is bug 5 above.

Horizon coverage is bounded by forecast lead availability — Open-Meteo is pulled
hourly with a 2-day span, so ~48 h of lead exists per target hour, thinning beyond
24 h. 6 h is the primary horizon in
[`experiments/phase7_preregistration.md`](experiments/phase7_preregistration.md).

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/ecowitt` | Ecowitt sensor webhook (form-encoded) |
| GET  | `/api/current` | Latest observation enriched with `feels_like_c`, `icon_slug`, `cond_label` |
| GET  | `/api/predict?target=…&horizon=…` | 4-way comparison (+ `randomforest`, `rain_probability_pct`) + `weather_code`, `icon_slug`, `cond_label`, `precip_prob_pct` |
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

## Reproducing the numbers

`model_metrics` lives only in the production database, so until 2026-07-15 no
figure in this README could be regenerated — or checked — from the repository.
The table is now exported to a committed CSV stamped with the git SHA, row count,
and export time:

```bash
python -m tools.export_metrics --out experiments/model_metrics.csv
```

[`experiments/model_metrics.csv`](experiments/model_metrics.csv) is the full
time-series (one row per retrain × target × horizon × model), so the results table
above is a `groupby(...).tail(1)` away and any later retrain can be diffed against
it. The other analysis entry points:

```bash
python -m tools.check_invariants --target temp_c   # physical-plausibility gate
python -m tools.own_station_eval --target temp_c   # backyard skill, not regional
python -m tools.learning_curve --target temp_c --horizon 3 --out experiments/lc.csv
python -m tools.ablation_sweep --sweep all         # neighbour-station selection
```

Note the split before comparing anything: `temporal_split` is 80/20 **by time**, so
the test set is always the most recent hours, and metrics from different retrains
were computed on different windows. Compare skill-vs-baseline across retrains, not
raw MAE.

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

> Status: **exploratory / offline, and now stale.** The latest run (~30-day
> window) showed upwind features help temperature at +1 h/+3 h (Ridge), don't help
> at +24 h, and that a small cohort (n≈3–5) in a single mid-range band captures
> most of it — station count plateaued at n=1 and the very-near 0–2 km band was the
> *weakest*, both of which cut against the crowdsourcing intuition that motivated
> the network. These features are **not yet wired into the served model**.
>
> ⚠️ **These results predate the forecast-lead join fix** (bug 5 above) and share
> its defect: the harness calls the same `build_dataset`, so every config was
> evaluated against a ~1 h-lead forecast regardless of horizon. The "+24 h harm"
> finding in particular is suspect — it may be measuring lag-feature staleness
> rather than forecast lead. The sweep needs a rerun on the corrected join before
> any of it is cited. The n-plateau and distance-band findings are less exposed
> (they compare configs against each other at a fixed horizon), but have not been
> re-verified.

## Status & roadmap

| Milestone | When |
|-----------|------|
| Linear regression baseline | ✅ shipped 2026-05-06 |
| XGBoost (early — under-data on purpose) | ✅ shipped 2026-05-06 |
| Public deploy + first weekly retrain | ✅ shipped 2026-05-07 |
| +3 h horizon | ✅ shipped 2026-05-07 |
| Atmospheric UX (icons, feels-like, segmented control) | ✅ shipped 2026-05-07 |
| Two-stage rain model (classifier → regressor) | ✅ shipped 2026-07-14 |
| Forecast-lead join fix + invariant checks | ✅ shipped 2026-07-15 |
| Random Forest | ✅ shipped 2026-07-15 (confidence bands still to surface) |
| +6 h / +12 h horizons | ✅ shipped 2026-07-15 |
| Own-station target + neighbour features wired into the served model | next |
| Freeze the Phase 7 pre-registration + lock a holdout | blocked on coverage |
| Wet-season rain results | Oct 2026 – Mar 2026 (calendar, not effort) |
| LSTM experiment (intentionally under-data) | target ~2026-11-01 |

This is a personal project, intentionally small and slow-paced: ~6k lines, ~10
weeks, one 2-vCPU droplet. The modelling timeline is deliberately conservative —
watch the `model_metrics` time-series to see how skill changes as data accumulates.

**Standing prediction, recorded so it can be judged later:** tree models
(XGBoost/RandomForest) will remain the best choice for this problem at ~1 year of
data, and the LSTM will not beat them. The reasoning is that a single-station
bias-correction task has weak sequential structure beyond the lag features already
supplied, and trees need less data, train in seconds, and cost nothing to serve.
This is a *prediction*, not a result — the LSTM has not been built. It is written
down in advance so a negative result stays honest.

The measured half of that claim already holds: XGBoost leads at every row count from
11.5k to 230k on the pooled regional target. The *unmeasured* half is the interesting
one, and the learning curve above narrows it — since accuracy has already plateaued in
row count, "wait for a year of data" only makes sense as "wait for a year of
**seasons**". The LSTM test in November will land on a corpus that is longer but, in
row-count terms, no richer. If a sequence model is going to win anywhere, the honest
place to look is the own-station target (where Ridge currently beats both trees at
+1 h), not the pooled one.

### Known limitations

- **The corpus is ~8 weeks, not 3.5 months.** Observations reach back to
  2026-04-01, but a training row needs a paired forecast and forecasts only survive
  from **2026-05-20** — a `cleanup_job` deleted them at 30 days, silently capping
  the trainable corpus until retention was extended (commit `b1be961`). Earlier
  forecasts are permanently gone.
- **The test set is the driest hours of the year.** A temporal 80/20 split puts the
  most recent ~20 % in test, which right now means late-June/July. Every metric
  above is a summer metric. The backyard recorded **zero wet hours in July**.
- **Rain results are pooled and dry-season.** Own-station and wet-season rain skill
  are untested.
- **The Phase 7 pre-registration is a partial draft, not a lock.** Holdout window,
  git SHA, and episode lists are `<TBD>`; it was never frozen or tagged. The
  discipline is set up; it has not been executed.
- **No feature-importance analysis exists.** No SHAP, no `feature_importances_`.
- **Neighbour/upwind features are not wired into the served model.** The live site
  runs own-station + NWP only; the network sweep is offline and exploratory.

The original design hand-off that drove the atmospheric UX changes is preserved
in `Hand-off - Atmospheric UX.md`.
