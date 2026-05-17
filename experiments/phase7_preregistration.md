# Phase 7 Pre-Registration

> **Status:** DRAFT — fields marked `<TBD>` are filled before 7.4 begins and the document is committed and tagged. Once committed, this file is **frozen** for the duration of the ablation campaign. Any change requires an amendment (see §10).

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

- **Own-station source:** Ecowitt → TimescaleDB (production).
- **Network sources:** Weather Underground (primary), PWSWeather (fallback).
- **Network station count at registration:** `<TBD: N>` stations within 100 km after quality filtering.
- **Quality filters applied:** `<TBD: uptime ≥X%, drift detector vY, blacklist of M stations>`.
- **Backfill coverage:** `<TBD: window covered, fraction available>`.

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

| Run ID | Q | Config | Notes |
|--------|---|--------|-------|
| A01 | Q1 | upwind + baseline, default N=5, ±30°, 10–50 km | Headline result |
| A02–A06 | Q2 | sweep N ∈ {1, 3, 5, 10, 20} | Hold band and tolerance fixed |
| A07–A10 | Q3 | distance bands {0–10, 10–25, 25–50, 50–100} alone | Hold N and tolerance fixed |
| A11–A13 | Q3 | combined band variants `<TBD: enumerate>` | |
| A14–A17 | Q4 | tolerance ∈ {±15°, ±30°, ±45°, ±90°} | Hold N and band fixed |
| A18–A19 | Q5 | downwind-only; upwind + downwind | |
| A20 | Q6 | winning Q1/Q2/Q3/Q4 config evaluated at all 5 horizons | No new training |
| A21 | Q7 | winning config, stratified by regime | No new training |
| A22 | Q8 | winning config, stratified by incoming wind sector | No new training |

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
