# Phase 7 — Crowdsourced PWS Network Layer

**Source:** `Weather_Project_Phase7_Phase8_Plan.docx` (Brad Hinkel, May 2026, v1.0)
**Duration:** 6–8 weeks calendar, in parallel with continued local data collection.
**Strategic goal:** Convert the network-features thesis from a research-backed hypothesis into a quantified, pre-registered empirical result that supports the pitch — including a defensible minimum-useful-network threshold ("you need at least N stations within D km").

---

## Target dates

| Milestone | Target | Notes |
|---|---|---|
| Engineering kickoff (7.1) | **2026-05-12** | Begins now; data accrual in parallel |
| 7.1 complete (network ingestion live + heartbeat reporting) | ~2026-05-26 | Two weeks |
| 7.2 complete (feature pipeline) | ~2026-06-09 | Two weeks |
| 7.3 pre-registration locked | **~2026-07-25** | Gated by data-sufficiency check; slips if rain events or frontal passages are short |
| 7.4 ablations | **Aug 2026** | Run on local dev box |
| 7.5 productization | **Sep 2026** | Winning config promoted back to droplet |

The ~2026-07-25 holdout-lock target assumes ~30 days of own-station data on hand at kickoff and the need for ~104 days total (60 train + 14 val + 30 holdout). The data-sufficiency check (§ below) is the actual gate.

---

## Where this runs

The Phase 7 workload splits across two environments. This isn't decorative — mixing experimental load with production serving on the basic-tier droplet would cost forecast latency.

| Workload | Where | Rationale |
|---|---|---|
| WU / PWSWeather network ingestion (7.1) | **Droplet** | 24/7 uptime requirement; data lives next to own-station observations in TimescaleDB. |
| Feature pipeline jobs (7.2) | **Droplet** | Reads from TimescaleDB; writes derived tables. |
| Data-sufficiency heartbeat (continuous) | **Droplet** | Reports daily on gap rate, sensor drift, station coverage, regime balance. |
| Baseline training + ablation campaign (7.4) | **Local dev box** | Iterative, bursty CPU; cannot share cores with production serving on basic-tier droplet. |
| Data-sufficiency gate check (one-shot at 7.3 lock) | **Local dev box** | Reads from snapshot; refuses to proceed if windows don't clear thresholds. |
| Production model promotion (7.5) | **Droplet** | Same path as today's deployment. |
| Network health monitoring (7.5) | **Droplet** | Lives next to ingestion. |

**Snapshot protocol:** at 7.3 lock time, take a one-shot `pg_dump` of the relevant TimescaleDB hypertables from the droplet, restore locally, and pin the ablation campaign to that frozen snapshot. The snapshot SHA goes into the pre-registration metadata. Re-pulling mid-campaign is forbidden.

---

## Pre-registered experimental questions (Q1–Q8)

These are locked **before** running 7.4 to prevent post-hoc rationalization. All ablations are reported, including null results.

| ID | Question | Sweep / Test |
|----|----------|--------------|
| Q1 | Do upwind station data improve skill vs current XGBoost + NWP baseline? | Network-on vs network-off |
| Q2 | What is the optimal number of upwind stations? | N ∈ {1, 3, 5, 10, 20} |
| Q3 | What is the optimal distance band? | 0–10, 10–25, 25–50, 50–100 km, alone and combined |
| Q4 | Angular tolerance for "upwind"? | ±15°, ±30°, ±45°, ±90° from current wind dir |
| Q5 | Does downwind data add anything? | Downwind-only; upwind+downwind vs upwind-only |
| Q6 | How does network value vary by horizon? | 1h, 3h, 6h, 12h, 24h |
| Q7 | How does value vary by weather regime? | Stable vs frontal-passage stratification |
| Q8 | Is there an asymmetric prevailing-wind effect? | Station-dense vs station-sparse incoming directions |

---

## Sub-phase 7.1 — Data acquisition (1–2 weeks)

**Goal:** Continuous, quality-filtered network observations flowing into TimescaleDB alongside own-station data.

### Tasks
- [ ] **Build the data-sufficiency heartbeat first** (see § below). Doubles as validation of the current "we have ~30 days of own-station data" estimate. Runs on the droplet.
- [ ] Integrate Weather Underground PWS API (primary source).
- [ ] Integrate PWSWeather API (fallback for WU policy / rate-limit risk).
- [ ] Build station discovery: enumerate all PWS within 100 km of home; capture location, sensor types, reporting frequency, historical depth.
- [ ] Persist a station registry table (id, lat/lon, distance, bearing, sensor flags, source).
- [ ] Quality filters: minimum uptime threshold, sensor-drift detection, blacklist for obviously bad stations.
- [ ] Backfill historical network observations where the API permits (critical given the small local dataset).
- [ ] Store network observations in TimescaleDB; align timestamps to own-station cadence.
- [ ] Source-abstraction layer so WU vs PWSWeather is swappable without touching feature code.

### Exit criteria
- ≥ N stations (target: 20+) within 100 km in the registry, with quality flags assigned.
- Live network ingestion running for **≥ 7 consecutive days** with < 5% gap rate.
- Backfill loaded for whatever window the APIs allow; coverage documented.
- Source-abstraction verified by running the same query path against the fallback provider.

### Dependencies / inputs
- API keys for WU and PWSWeather.
- Decision on rate-limit budget and caching policy.

---

## Sub-phase 7.2 — Feature engineering (1–2 weeks)

**Goal:** Directional, distance-aware features derived from the network that the model can actually use.

### Tasks
- [ ] Per-station distance and bearing from home (precompute, store in registry).
- [ ] Directional binning relative to **current** wind direction: upwind / crosswind / downwind, with configurable angular tolerance (parameterized for Q4 sweep).
- [ ] Distance-weighted aggregation features: inverse-distance and Gaussian-kernel variants.
- [ ] Lag features for upwind stations at 1h, 3h, 6h, 12h: temperature, pressure, humidity, wind, rain occurrence.
- [ ] Spatial-gradient features (e.g., "temperature dropping along wind direction").
- [ ] Feature config knobs exposed for the ablation sweeps (N stations, distance bands, angular tolerance, upwind/downwind toggles).

### Exit criteria
- Feature pipeline computes all listed features deterministically from raw network data.
- Unit tests cover directional binning at known wind angles and edge cases (calm wind, missing stations).
- Feature config can be toggled per ablation run without code edits.
- Sanity check: feature distributions look physical (no unit errors, no implausible gradients).

### Dependencies
- 7.1 complete; sufficient backfill to compute lag features on the training window.

---

## Sub-phase 7.3 — Experimental design (1 week)

**Goal:** Pre-registered protocol so 7.4 results are defensible.

### Tasks
- [ ] **Pull TimescaleDB snapshot** from droplet to local dev box; record SHA/timestamp.
- [ ] **Run data-sufficiency gate** against the snapshot for each candidate train/val/holdout window. Must pass before any other 7.3 task proceeds.
- [ ] Establish the **baseline**: current XGBoost using own-station + NWP features only. Freeze the version.
- [ ] Pre-register hypotheses, success criteria, and reporting rules — committed to the repo before any 7.4 run.
- [ ] Define skill-score metrics for temperature, wind, and rain detection (rain = F1 at the relevant horizon).
- [ ] Carve a clean holdout window that includes **≥ 1 frontal passage** and **≥ 1 stable period**.
- [ ] Document the holdout window and lock it; no model selection touches it.
- [ ] Define stratification logic for Q7 (stable vs frontal-passage labeling).

### Exit criteria
- `experiments/phase7_preregistration.md` committed and dated, listing Q1–Q8, metrics, holdout window, and decision rules.
- Frozen baseline model artifact + reproducible eval script.
- Holdout window covers required regimes; verified by labeling and a visual check.

### Dependencies
- 7.2 complete (features exist) — but holdout selection does **not** depend on having run ablations.

---

## Sub-phase 7.4 — Ablation experiments (2–3 weeks)

**Goal:** Answer Q1–Q8 against the locked baseline and holdout.

### Tasks
- [ ] Run Q1 (network-on vs network-off) → headline result.
- [ ] Run Q2 (N-station sweep).
- [ ] Run Q3 (distance-band sweep, alone and combined).
- [ ] Run Q4 (angular-tolerance sweep).
- [ ] Run Q5 (downwind contribution).
- [ ] Run Q6 (skill vs horizon).
- [ ] Run Q7 (regime stratification on holdout).
- [ ] Run Q8 (asymmetric prevailing-wind direction).
- [ ] Compute SHAP / permutation importance for the winning configurations — verify the model is using network features the way the hypothesis predicts, not exploiting incidental correlations.
- [ ] Report **all** runs including null results in a single comparison table.

### Exit criteria
- Skill-score table (per metric × per Q) committed to the repo.
- SHAP/importance plots for the top configurations.
- A documented "winning configuration" with explicit feature selection and config values.
- Holdout result for the winner reported separately from training/validation performance.

### Dependencies
- 7.3 frozen baseline + pre-registration.

---

## Sub-phase 7.5 — Productization (1 week)

**Goal:** Bake the winner into the production service and produce the pitch artifact.

### Tasks
- [ ] Promote the winning network-feature config into the production model pipeline.
- [ ] Add network health monitoring: per-station outage detection, quality drift alerts, graceful missing-data fallback (e.g., degrade to subset of stations or to baseline if too sparse).
- [ ] Production smoke test: forecast service still meets latency/availability SLOs with network features enabled.
- [ ] Write the pitch-ready section: **"How a PWS network improves hyperlocal forecasting"** — anchored to the Q1–Q8 results, with the adoption-threshold answer from Q2/Q3/Q8 stated explicitly.

### Exit criteria
- Production model serving the network-enabled forecast.
- Monitoring dashboard or alert wired for network health.
- Pitch writeup committed; reads as a standalone document for a non-ML reader.
- No regression vs baseline on a rolling production eval window.

---

## Data-sufficiency tool

A single tool with two modes — same metrics, different output.

### Heartbeat mode (continuous, on droplet)

Runs daily. Answers: *"are we getting enough data, and is it good?"*

Reports a rolling-30-day snapshot:
- Gap rate per data source (own-station, each network station, NWP).
- Sensor-drift flags per own-station and per network station.
- Network station coverage % over the window.
- Cumulative own-station hours since last sensor change / fix event.
- Regime tally: frontal-passage episodes detected (heuristic: wind shift > 60° AND pressure fall > 2 hPa in 6h, or similar — finalize during 7.1).
- Rain-positive hour count.
- Stable-period hour count.

Output: dashboard table + alert if any metric trends adversely. Used to give early warning if the 2026-07-25 lock target is at risk.

### Gate mode (one-shot, at 7.3 lock)

Same metrics, applied to each candidate window (train, val, holdout) on the frozen snapshot. **Refuses** to allow 7.3 lock unless all of these hold:

| Check | Threshold |
|---|---|
| Paired forecast/obs hours, holdout, primary horizon | ≥ 500 |
| Paired forecast/obs hours, train, per regime | ≥ 500 |
| Rain-positive hours in holdout | ≥ 50 |
| Frontal-passage episodes in holdout | ≥ 1 |
| Stable-period episodes in holdout | ≥ 1 |
| Frontal-passage episodes in train | ≥ 2 |
| Network station coverage over window | ≥ 70% of station-registry stations active |
| Gap rate per critical source over window | ≤ 5% |

Gate output is a single PASS/FAIL with a per-check breakdown, written into the pre-registration as evidence.

---

## Phase 7 overall success criteria

Drawn directly from the source doc. Phase 7 is "done" when **all** of these hold:

- [ ] **Temperature:** network features produce **≥ 0.05** absolute skill-score improvement at the 6h horizon.
- [ ] **Rain:** network features produce **≥ 0.10** absolute F1 improvement at the 3–6h horizon.
- [ ] **Out-of-sample:** network value persists on the holdout, **including the unseen weather regime**.
- [ ] **Adoption threshold:** a clear, defensible answer to Q2 / Q3 / Q8 — the minimum useful network for the pitch.

If any of these fail, the doc's instruction is to report the null result honestly rather than redesign to hit the bar.

---

## Risks and mitigations (carried from source doc)

| Risk | Mitigation |
|------|-----------|
| WU API rate limits / deprecation | Aggressive caching; source abstraction; PWSWeather fallback path. |
| Station quality variance | Quality filtering + down-weighting noisy stations in aggregation. |
| Sparse stations in prevailing wind direction | Document as real-world constraint; reduced-sample tests where forced. |
| Multicollinearity in features | XGBoost handles it, but monitor feature-importance distribution for a coherent story. |
| Confirmation bias | Pre-registration in 7.3; report all ablations, including nulls. |

---

## Deliverables checklist

- [ ] Skill-score table covering every Q1–Q8 ablation vs baseline (temperature, wind, rain).
- [ ] "What the network buys you" pitch writeup.
- [ ] Production model with network features baked in.
- [ ] Reproducible experiment notebooks tied to the pre-registered hypotheses.

---

## Cross-phase coupling notes

- **Phase 8 ingestion abstraction (8.2)** should land during 7.2/7.3 because the API-driven PWS sources need the same Observation contract Phase 8 introduces. Plan the refactor once, use it twice.
- Phase 7's network features must keep working when WS-2902C becomes the primary station (Phase 8.4 validation).
