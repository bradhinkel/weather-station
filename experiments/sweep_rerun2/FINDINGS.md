# Ablation sweep findings — rerun2 (exploratory)

**Window:** 2026-05-12 → 2026-06-19 01:15 UTC (clipped to last network obs).
**Split:** 80/20 temporal (~590 train / ~148 test). **Bootstrap:** 500 resamples.
**Status:** exploratory station-selection sweep, NOT the locked pre-registered
campaign. Single ~5-week window, late-spring only. Treat as direction, not proof.

> Three harness bugs were found and fixed before these numbers were trusted:
> (1) a tz-strip that zeroed all network features, (2) the eval window running
> past the (day-lagged) network coverage, (3) fill-0 on offset fields blowing up
> StandardScaler/Ridge. See commits 2ec0da0, e41be55. Earlier `sweep_*` runs were
> invalid and discarded.

## Headline (temperature)

| Horizon | Network effect (Ridge) | Network effect (XGBoost) |
|---------|------------------------|--------------------------|
| **+1h** | **helps** — ΔMAE +0.10 to +0.13 °C, p≈1.00 | neutral→harmful |
| **+3h** | **helps most** — ΔMAE +0.15 to +0.24 °C, p≈1.00 | harmful |
| **+24h**| **hurts** — ΔMAE −0.16 to −0.32 °C | hurts |

Base Ridge MAE: 0.65 (+1h), 1.11 (+3h), 1.14 (+24h). At +3h the network cuts
Ridge MAE by ~18–21% and lifts skill-vs-NWP from ~0 to ~0.15–0.20.

## What the sweep answers

1. **Station count plateaus immediately.** n=1 already captures essentially the
   whole gain (+1h Ridge: n=1 +0.098 vs n=20 +0.120; +3h: n=1 +0.210 vs n=3
   +0.236). A *handful* of stations suffices — excellent for the WU rate limit.
   Parsimony → **n≈3–5**.

2. **Distance: the very-near 0–2 km band is the weak one.** At +1h it does
   nothing (+0.016, n.s.); the 2–5, 5–10, 10–25, 25–50 km bands all help about
   equally (~+0.12). Best single band: **5–10 km** at +1h, **10–25 km** at +3h,
   but margins between mid/far bands are within CI overlap.

3. **No "near-for-+1h, far-for-+3h" split, and no benefit to a band MIX.** The
   pre-run hypotheses (5 km drives +1h / 25 km drives +3h; an explicit
   3@2km+2@10km+2@25km mix) are **not supported** here: bands ≥2 km are roughly
   interchangeable, and `multiband_mix` (+0.10 at +1h) does not beat a single
   well-chosen mid band. The 2 km band being weakest argues against weighting
   the near cluster.

4. **Ridge captures the signal; XGBoost does not (yet).** XGBoost gains nothing
   and is mildly hurt at every horizon — consistent with overfitting ~25–47
   extra columns on a 590-row window. This favors the project's Ridge/RF
   preference for now; revisit XGBoost as data accumulates.

5. **Rain is unreliable, as expected.** Tiny zero-inflated MAEs; the multiband
   rain model even blew up (net ≫ base). Do not act on rain until the 2-stage
   classifier and a full wet season of data. Not a sweep defect — a known
   data/target limitation.

## Production implication

For the live forecast: a **single mid-range band (~5–25 km) with ~3–5 stations**,
feeding a **linear/RF** correction, captures the +1h/+3h benefit. Skip the 0–2 km
cluster and skip network features entirely at +24h (use base-only). That live set
is tiny, so keep the daily `hourly/7day` batch archiving *all* stations for the
training corpus while polling only the chosen few in real time.

## Caveats

- One ~5-week, single-season window; +24h harm and XGBoost no-gain may change
  with more data / seasonality.
- CIs are on the temporal test split, not the locked holdout — this is the
  exploratory sweep, not the pre-registered evaluation.
