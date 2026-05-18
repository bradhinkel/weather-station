# Phase 7 Pre-Registration

> **Status:** PARTIAL DRAFT — sections describing the **feature pipeline as built in 7.1/7.2** are filled in. Fields that depend on the locked-snapshot (holdout window, frontal/stable episode lists, registered git SHA, station count at registration) remain `<TBD>` until coverage stabilizes >= 95% (projected ~2026-06-17) and the snapshot is taken at the start of 7.3. Once those fields are filled and the document is committed + tagged `phase7-prereg-v1`, this file is **frozen** for the duration of the ablation campaign. Any change requires an amendment (see §10).

- **Author:** Brad Hinkel
- **Project:** Hyperlocal Weather Forecasting — Phase 7 (PWS Network Layer)
- **Reference plan:** [`PHASE_7_PLAN.md`](../PHASE_7_PLAN.md)
- **Pre-registered on:** `<TBD: YYYY-MM-DD>`
- **Locked git SHA at registration:** `<TBD>`
- **Holdout window:** `<TBD: start_utc>` → `<TBD: end_utc>`

---

## 1. Purpose

Pre-register hypotheses, metrics, holdout, and decision rules for the Phase 7 ablation campaign **before** running any ablation in 7.4. Goal: prevent post-hoc rationalization and produce evidence a sophisticated buyer can evaluate in a 30-minute technical review.

## 2. Baseline (frozen)

The baseline is the current production XGBoost correction model using **own-station + NWP features only**.

- **Model artifact path:** `<TBD: models/baseline_phase7.joblib>`
- **Training data window:** `<TBD: start_utc>` → `<TBD: end_utc>`
- **Feature list (canonical):** `<TBD: paste feature names>`
- **Hyperparameters:** `<TBD: paste params>`
- **Training script:** `<TBD: path@SHA>`
- **Evaluation script:** `<TBD: path@SHA>`

The baseline is **not** retrained during 7.4. Ablations only change feature inputs and config; the model class and hyperparameters are held constant unless explicitly stated in an ablation spec.

## 3. Data

- **Own-station source:** Ecowitt → TimescaleDB (production). One station, sheltered position — empirically measured wind-direction bias of **-37° CCW** vs the regional network mean (validated 2026-05-18 at wind ≥ 1 m/s, std 7°). Own wind-speed agrees with network within noise (own/network ratio 0.96 median). Per [`PHASE_7_PLAN.md`](../PHASE_7_PLAN.md) the default `FeatureConfig.wind_reference = "network_mean"` to avoid baking this shelter rotation into the upwind/downwind classification.
- **Network sources:** Weather Underground (primary, live), PWSWeather (fallback, planned).
- **Network station count at registration:** `<TBD: locked at registration; as of 2026-05-18 the registry has 246 quality stations within 100km>`.
- **Quality filters applied:**
  - `evaluate-quality` per station against `observations`: a station is non-blacklisted if `coverage_<window>d_pct ≥ 50` over the last 7 days.
  - Network row filter at feature-time: `quality_flags->>'blacklisted' = 'false'` (explicit non-blacklist; unevaluated stations excluded).
  - Excluded-windows registry (`excluded_windows` table) for two known own-station outages: 2026-04-22 → 2026-05-01 (Windows-docker autostart failure, resolved by droplet migration 2026-05-07) and 2026-05-07 (cutover window). Holdout candidate windows must not overlap these.
- **Backfill coverage:** WU `/hourly/7day` endpoint backfills ~7 days per call; daily ingest job at 00:15 UTC continuously extends the live window. **Network-coverage trajectory:** 25% as of 2026-05-18, projected ~95% by 2026-06-17. Pre-registration locks once coverage stabilizes ≥ 95% with < 5% gap over a 30-day window (heartbeat exit criterion from `PHASE_7_PLAN.md` §7.1).

## 4. Holdout

Holdout is **locked** at registration time and never touched by model selection or feature tuning.

- **Window:** `<TBD: start_utc>` → `<TBD: end_utc>`
- **Frontal-passage episodes inside window:** `<TBD: list with timestamps>`
- **Stable-period episodes inside window:** `<TBD: list with timestamps>`
- **Regime labeling method:** `<TBD: rule or dataset reference>`
- **Holdout is excluded from:** training, validation, hyperparameter search, feature selection, threshold tuning.

## 5. Metrics

All metrics evaluated on the locked holdout. Reported with bootstrap 95% CIs (1000 resamples).

| Variable | Metric | Horizons evaluated | Primary horizon |
|----------|--------|-------------------|-----------------|
| Temperature | Skill score vs NWS regional baseline, where `skill = 1 − RMSE_model / RMSE_NWS` | 1h, 3h, 6h, 12h, 24h | **6h** |
| Wind speed | Skill score vs NWS | 1h, 3h, 6h, 12h, 24h | 6h |
| Rain detection | F1 (rain = nonzero accumulation in horizon window) | 1h, 3h, 6h, 12h | **3–6h** |

**Tie-breaking:** if two ablations match within bootstrap CI overlap on the primary metric, prefer the configuration with **fewer network stations** (parsimony / pitch defensibility).

## 6. Hypotheses

Each row states the predicted direction and the threshold that counts as "the hypothesis is supported." Null results are still reported.

| ID | Hypothesis | Predicted direction | "Supported" threshold |
|----|------------|--------------------|-----------------------|
| Q1 | Adding upwind station features improves temperature and rain skill at the 3–6h horizon vs baseline. | + | Δ temp skill @6h ≥ 0.05 **and** Δ rain F1 @3–6h ≥ 0.10 |
| Q2 | There exists an optimal N in {1, 3, 5, 10, 20} beyond which skill plateaus or degrades. | inverted-U or monotone-with-plateau | Identifiable inflection point; not strictly increasing |
| Q3 | Mid-distance bands (10–50 km) carry more value than very-near (0–10 km) or very-far (50–100 km). | mid-band dominance | Best single-band result in 10–25 or 25–50 km |
| Q4 | Tighter angular tolerance (±30°) outperforms loose (±90°) for the upwind signal. | narrower wins | ±30° skill ≥ ±90° skill on primary metrics |
| Q5 | Downwind-only is no better than baseline for short-horizon temp/wind. Upwind + downwind may help cloud/front signals. | downwind alone ≈ baseline; combined may help | Stated per metric |
| Q6 | Network value grows with horizon up to ~6h, then decays. | inverted-U over horizons | Peak at 3h–6h, not at 1h or 24h |
| Q7 | Network value is larger during frontal-passage periods than stable periods. | regime-dependent | Δ on frontal subset > Δ on stable subset |
| Q8 | Skill is higher when wind comes from station-dense directions than station-sparse. | asymmetric | Δ between dense vs sparse direction > 0, statistically distinguishable |

## 7. Ablation matrix

### 7.0 Headline config (the FeatureConfig literal used by A01)

A01 fixes the following `src.features.config.FeatureConfig` values, which serve
as the baseline against which subsequent ablation rows vary one knob at a time.
Sweeps in Q2–Q4 inherit every other field from this row.

| Knob | Value | Notes |
|------|-------|-------|
| `n_stations` | 5 | Q2 sweeps over {1, 3, 5, 10, 20}. |
| `distance_band_km` | (0, 25) | Q3 sweeps. |
| `angular_tolerance_deg` | 30 | Q4 sweeps. |
| `include_downwind` | False | Q5 toggles. |
| `lag_hours` | (1, 3, 6, 12) | Fixed across ablations; Q6 uses these as horizons. |
| `wind_reference` | "network_mean" | Verified shelter; "own" available as Q4-adjacent ablation. |
| `wind_reference_radius_km` | 10 | "Kirkland-local" — 25km variant tested only if 10km mean and 25km mean diverge >20° on the locked snapshot. |
| `wind_reference_min_stations` | 5 | Fallback to "own" below this count. |
| `aggregation_kernel` | "inverse_distance" | Gaussian variant deferred to post-7.4 if Q1 passes. |
| `gaussian_sigma_km` | 5.0 | Unused unless kernel switched. |
| `gradient_near_band_km` | (0, 10) | Spatial-gradient near band. |
| `gradient_far_band_km` | (25, 50) | Spatial-gradient far band. |

### 7.1 Runs

| Run ID | Q | Config delta vs §7.0 | Notes |
|--------|---|----------------------|-------|
| A01 | Q1 | none — exact §7.0 config | **Headline result** |
| A02–A06 | Q2 | `n_stations ∈ {1, 3, 5, 10, 20}` | A04 == A01 |
| A07–A10 | Q3 | `distance_band_km ∈ {(0,10), (10,25), (25,50), (50,100)}` | Single-band alone |
| A11–A13 | Q3 | `distance_band_km ∈ {(0,50), (0,100), (10,100)}` | Combined-band variants |
| A14–A17 | Q4 | `angular_tolerance_deg ∈ {15, 30, 45, 90}` | A15 == A01 |
| A18 | Q5 | downwind-only — implemented by inverting `direction_class` post-binning, see §10 amendment notes | |
| A19 | Q5 | `include_downwind=True` | Upwind + downwind |
| A20 | Q6 | winning §7.1 config evaluated at all 5 horizons | No new training |
| A21 | Q7 | winning config, stratified by regime label (frontal / stable / other) | No new training |
| A22 | Q8 | winning config, stratified by incoming wind sector (8 octants) | No new training |

**Run order:** A01 first (headline), then Q2 → Q3 → Q4 sweeps, then Q5, then Q6–Q8 stratifications on the chosen winner. **Run order does not change** based on intermediate results.

## 8. Decision rules

- The Phase 7 **overall success criteria** (temp ≥0.05 @6h, rain F1 ≥0.10 @3–6h, holdout persistence including unseen regime, defensible answer to Q2/Q3/Q8) are evaluated on the **holdout** using the final winning config from §7.
- If the overall criteria are **not** met, the result is reported as a null finding. The plan in `PHASE_7_PLAN.md` explicitly forbids redesigning to hit the bar.
- A configuration is declared the "winner" only if it beats baseline by the per-metric threshold **with non-overlapping bootstrap 95% CIs** on the primary metric.
- Feature-importance check (SHAP or permutation) on the winner must show network features carrying weight in the direction predicted by the hypothesis. If the model wins but uses network features in an unphysical way (e.g., far-downwind stations dominating short-horizon temperature), the result is flagged as suspect and not promoted to production.

## 9. Reporting requirements

- **All** ablation runs in §7 are reported, including null results.
- Each run records: config, holdout metrics with CIs, training/validation metrics, SHAP/importance summary, runtime, data coverage.
- A single skill-score table is committed at `experiments/phase7_results.md` covering every run.
- A separate `experiments/phase7_writeup.md` is the pitch artifact for 7.5; it must reference run IDs from this pre-registration.
- Raw run logs land under `experiments/runs/<run_id>/`.

## 10. Amendments

If a real-world blocker forces a change (e.g., WU API deprecation mid-campaign, holdout regime turns out to be mislabeled, a station is discovered to be drifting):

1. Append an entry to §11 with date, reason, and what changed.
2. Re-commit and tag.
3. Note the amendment in the final writeup. Do **not** silently revise prior sections.

## 11. Amendment log

| Date | Author | Change | Reason |
|------|--------|--------|--------|
| _(none yet)_ | | | |

---

## Sign-off

Pre-registration is considered locked when:
- All `<TBD>` fields are filled.
- File is committed with the locked git SHA recorded in §0.
- A git tag `phase7-prereg-v1` points at that commit.
- 7.4 ablation runs reference this tag in their run metadata.
