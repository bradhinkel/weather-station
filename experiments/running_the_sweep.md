# Running the network-feature ablation sweep

This guide reproduces the Phase 7.4 neighbor-station sweep **at any location**.
Nothing here is hardcoded to the Seattle/Kirkland deployment: the entire
pipeline keys off whatever **own station** you register, discovers neighbors
around *its* lat/lon, and selects the best mix from *that* network. Run it in
Denver and you will get a different — and locally correct — answer, because
your neighbor geometry is different. That divergence is the product, not a bug.

The sweep harness is [`tools/ablation_sweep.py`](../tools/ablation_sweep.py).

---

## What the sweep answers

For each target (`temp_c`, `rain_mm_1h`) and horizon (`+1h`, `+3h`, `+24h`), it
measures how much nearby-station features improve the forecast over the
own-station + NWP baseline, and **which station mix is worth it**:

- **`n` mode** — fix the 0–25 km band, sweep cohort size {1,3,5,10,20}: the
  point past which more stations stop helping (the plateau).
- **`bands` mode** — isolate non-overlapping bands {0–2, 2–5, 5–10, 10–25,
  25–50 km}: the *strength of each distance band per horizon* (e.g. does the
  near band carry +1h while the far band carries +3h?).
- **`multiband` mode** — all bands present at once as separate feature groups:
  per-band importance reads out the **optimal mix** (e.g. "3 @ 2 km + 2 @ 10 km
  + 2 @ 25 km").

Method, in one line: for every (target, horizon, config) it trains **base**
(own + NWP) vs **net** (base + network) on *identical rows and the identical
80/20 temporal split*, so the reported ΔMAE is purely the network contribution,
and a 500-sample bootstrap CI on ΔMAE says whether the gain is real.

**Causality:** a forecast issued at *t* for `valid_time = t+H` may only use
neighbor obs at times ≤ *t*. The harness joins network features at
`valid_time − horizon`, so every neighbor column references a time ≤ issue
time. No leakage.

---

## Prerequisites to reproduce at a new location

You need ~4+ weeks of accumulated data before the sweep is meaningful. Order of
operations from a fresh deployment:

### 1. Register your own station
Point an Ecowitt (or compatible) station at `POST /api/ecowitt`, or insert a row
into `stations` with `is_network = false` and your `lat`/`lon`. Everything
downstream — neighbor discovery, upwind geometry, the holdout — is anchored to
this station's coordinates.

### 2. Get a Weather Underground API key
Create a free PWS-contributor key at
<https://www.wunderground.com/member/api-keys> and put it in `.env`:

```
WU_API_KEY=...
```

### 3. Discover neighbors around *your* location
```bash
python -m src.pws.cli discover --wide      # grid sweep around the own station
python -m src.pws.cli evaluate-quality     # score coverage, set blacklist flags
```
`discover` calls WU's `/v3/location/near` centered on your own-station lat/lon,
so the registered network is whatever exists near you. `evaluate-quality`
blacklists stations below 50% coverage so the sweep only sees usable ones.

### 4. Let data accumulate
The scheduler (`src/scheduler.py`) runs daily WU ingest (00:15 UTC) + own-station
ingest continuously + hourly Open-Meteo forecasts. Watch sufficiency with:
```bash
python -m src.heartbeat --days 30
```
Wait until obs/NWP gap is low and network coverage is high (the Seattle deploy
used ≥ ~95% coverage over a 30-day window). **~4–6 weeks of dense network data**
is the practical floor for a stable 80/20 split.

> Note: the daily WU batch fetches the `hourly/7day` endpoint, so the stored
> network history is **hourly resolution** even though we *poll* once a day —
> sufficient for this offline sweep. Real-time freshness only matters later, for
> live inference with the selected subset.

---

## Running it

Run on the host with DB access (the droplet). The window defaults to the
dense-network period; override `--window-start` to match your own coverage.

```bash
cd /opt/weather-station

# 0. Validate wiring + measure per-config time (one config × +1h). ~2 min.
venv/bin/python -m tools.ablation_sweep --dry-run

# 1. Full sweep: both targets, all 3 horizons, all 3 modes. ~25–35 min.
#    (~2 min per feature-build × 15 distinct configs; model fits are cheap.)
venv/bin/python -m tools.ablation_sweep --sweep all
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--sweep` | `all` | `n` \| `bands` \| `multiband` \| `all` |
| `--target` | `temp_c,rain_mm_1h` | comma-separated; feature-building is shared across targets |
| `--horizons` | `1,3,24` | subset of the supported horizons |
| `--models` | `linear,xgboost` | model classes to compare |
| `--window-start` | `2026-05-12` | **set this to your own network-coverage start** |
| `--window-end` | now (UTC) | end of the evaluation window |
| `--resamples` | `500` | bootstrap resamples for the ΔMAE CI |
| `--stamp` | derived | output dir suffix |
| `--dry-run` | off | one config × first horizon, no files written |

---

## Outputs

Written to `experiments/sweep_<stamp>/`:

- **`summary.md`** — per-horizon tables: MAE base vs net, ΔMAE, 95% CI, p(↑),
  skill vs NWP. Start here.
- **`results.csv`** — every row (target × horizon × config × model) for analysis.
- **`meta.json`** — window, home station id, sweep definitions, git-reproducible
  parameters.

Commit `experiments/sweep_<stamp>/` back to git after a run so the result is
versioned alongside the harness (pre-registration §9).

---

## Reading the result → production decision

1. **`n` mode** gives the plateau — the smallest cohort that captures most of
   the gain.
2. **`bands` / `multiband`** show which distances matter for which horizon.
3. The winning mix becomes the **real-time polling set** for live inference.
   Note the WU free tier (~1,500 calls/day): keep the daily `hourly/7day` batch
   archiving *all* non-blacklisted stations (cheap, builds the training corpus),
   and poll only the selected band-spanning subset live. See the project notes
   on temp-vs-rain band divergence before pruning the live set.

The headline is location-specific: a different deployment, with a different
neighbor network, will land on a different count/band mix — and the sweep is how
you find *yours*.
