# Can a Backyard Beat the Forecast?

**A preliminary report from ten weeks of hyperlocal weather modelling**

Brad Hinkel · July 2026 · Kirkland, Washington
Code and data: [github.com/bradhinkel/weather-station](https://github.com/bradhinkel/weather-station) · Live: [weather.bradhinkel.com](https://weather.bradhinkel.com)

---

## Abstract

I put a weather station in my backyard to answer one question: can hyperlocal data beat
the regional forecast where I actually live? **Yes — by 42% at one hour and 25.4% at
three.** But that answer took ten weeks to reach, and almost none of the difficulty was
in the modelling. The model that shipped for most of that period was *worse than doing
nothing* in my own backyard beyond +6h, for a reason that turned out to be
architectural rather than statistical: trained pooled across 323 stations with **no
station identity among its features**, it was structurally forbidden from representing
"this station runs warm" — the exact thing the project exists to measure. Training on
1,089 of the right rows beats 230,000 of the wrong ones by 15%. Adding elevation-adjusted,
quality-screened upwind neighbour averages takes the backyard from 0.963 °C to 0.718 °C
at +3h. Along the way: a learning curve showing accuracy plateaus at ~11.5k rows against
a corpus of 230k (**more data of the same kind buys nothing**); a quarter of the
crowdsourced network reporting bad data, some of it from indoors; and six shipped defects
that each produced plausible, publishable, wrong numbers. The through-line is
methodological. The model was never the bottleneck. Knowing whether a number meant what
it appeared to mean was the bottleneck — every single time.

---

## Outline

1. [The question](#1-the-question)
2. [The setup](#2-the-setup)
3. [The answer so far](#3-the-answer-so-far)
4. [The architecture forbids the measurement](#4-the-architecture-forbids-the-measurement)
5. [The interaction is the model](#5-the-interaction-is-the-model)
6. [Volume is not the lever; variety is](#6-volume-is-not-the-lever-variety-is)
7. [The crowd does not transfer](#7-the-crowd-does-not-transfer)
8. [The instrument is part of the model](#8-the-instrument-is-part-of-the-model)
9. [What went wrong, and what it misdirected](#9-what-went-wrong-and-what-it-misdirected)
10. [Three standing assertions](#10-three-standing-assertions)
11. [What comes next](#11-what-comes-next)
12. [Reproducibility](#12-reproducibility)
13. [On method: why this was a conversation](#13-on-method-why-this-was-a-conversation)

---

## 1. The question

Regional weather forecasts are produced by numerical weather prediction (NWP) models
solving atmospheric physics on a grid. The grid cell containing my house is kilometres
across. My backyard is not kilometres across. It is sheltered, it sits behind a fence
and under trees, and — from years of living in it — it is **warmer and calmer** than
what the forecast says.

That gap is the premise. If a regional forecast is systematically wrong about my yard
in a consistent direction, then a small model with local sensor data should be able to
correct it. The biases are the signal, not the noise.

The specific question: **can I beat the local forecast using data from my backyard
weather station, plus other nearby personal weather stations?**

There is precedent for the crowdsourced half. Meier et al. (2017) used citizen weather
stations for urban climate research, and a large part of their contribution was
quality-controlling the crowd — a theme this report returns to in §8. But their task
and mine differ in a way that matters: they reconstructed a *spatial field* across a
city from many stations; I am doing *point correction* at one location. Whether the
crowd's value transfers between those two tasks is an empirical question, and §7 gives
an unexpected answer.

This report is preliminary, and the preliminary part is the point. Ten weeks is not a
season. The findings below are offered as direction, not proof — and the report is
explicit about which of them have already been retracted once.

## 2. The setup

An Ecowitt GW2000 in a Kirkland backyard posts observations to a FastAPI service on a
$24/month DigitalOcean droplet (2 vCPU, 4 GB) running Postgres and TimescaleDB
natively. Open-Meteo forecasts are pulled hourly with a 2-day span, so for any target
hour roughly 48 forecasts exist at different lead times. A daily job ingests ~320
Weather Underground neighbour stations discovered by a geometric grid sweep out to
100 km.

A training row is `(observation at t, forecast issued for t+h, actual at t+h)`. Both
the lag observation *and* the forecast are horizon-lagged: the forecast used is the
freshest one issued at or before `valid_time - horizon`, which is exactly what the
serving path can see. Getting this wrong is §9's main story.

Three model classes — Ridge, RandomForest, XGBoost — train on every (target, horizon)
pair, plus a two-stage classifier→regressor for rain. Five horizons: +1, +3, +6, +12,
+24 h. Split is temporal 80/20: the test set is always the most recent hours.

**Scale, stated plainly, because it bears on every conclusion:**

| | |
|---|---|
| Observations | 2026-04-01 → 2026-07-15 |
| **Paired corpus** | **2026-05-20 → 2026-07-15 (~58 days)** — forecasts were being deleted at 30 days until retention was fixed; a row needs both, so earlier pairs are permanently gone |
| Pooled training rows | ~290k across 323 stations |
| **Own-station usable hours** | **1,366** |
| Test window | The most recent ~20% — i.e. **the driest hours of the year** |
| Codebase | ~6k lines, ~40 commits, ten weeks |

Two of those rows do more work than the rest. The corpus is **eight weeks, not the
three months** the observation range suggests. And the own station — the entire point
of the exercise — contributes **1,366 of 290,000 rows, under half a percent**.

## 3. The answer so far

**Yes, at +1 h and +3 h.** But the route to that answer is the report, because the model
that has been serving this project for ten weeks does *not* get there — and understanding
why took every section that follows.

**Where it ends up.** Trained on the own-station target, fed elevation-adjusted
quality-screened upwind neighbour averages (§4, §7):

| +3 h, own station | MAE | skill vs Open-Meteo |
|---|---|---|
| Open-Meteo | 0.963 °C | — |
| **Served model** (pooled, 230k rows) scored on the backyard | 0.941 | **2.3 %** |
| Own-trained (1,089 rows) | 0.798 | 17.1 % |
| **Own-trained + QC'd upwind bands** | **0.718** | **25.4 %** |

**Where it started.** The *served* models — pooled across 323 stations — scored on
own-station test rows only, with the train/test boundary taken from the pooled frame so
those rows were genuinely held out:

| Horizon | Open-Meteo | Ridge | RandomForest | XGBoost | best skill |
|---|---|---|---|---|---|
| +1 h | 0.952 °C | **0.550** | 0.627 | 0.616 | **+42 %** |
| +3 h | 0.973 | **0.880** | 1.020 | 0.941 | +9.6 % |
| +6 h | 1.030 | 1.158 | 1.212 | **1.050** | **−2 %** |
| +12 h | 1.141 | 1.619 | 1.342 | **1.150** | **−0.9 %** |
| +24 h | 1.397 | 1.555 | 1.468 | **1.377** | +1.4 % |

Skill = `1 − MAE_model / MAE_openmeteo`. Negative means **worse than believing the
regional forecast**.

Read those two tables together, because the gap between them is the whole argument. The
served model beats the backyard's forecast **for one hour**; after that it is a wash, and
at +6 h and +12 h it is *actively harmful*. Against the pooled network the same models
score 52 / 22 / 10 / 1.4 / 12 %. **The regional win does not transfer to the yard it was
built for** — and the reason is not data, and not the model class. It is that the model
has no way to know which yard it is standing in (§4).

*Caveat: n_own is 276–295 rows per horizon, without confidence intervals. The pattern is
reported because it is consistent across five horizons and three models; no single cell
should be quoted. The two tables also come from different temporal splits — see §4.*

**Rain**, judged on precision/recall/F1 because MAE on a zero-inflated target rewards a
predictor that always says "dry":

| Horizon | Precision | Recall | F1 |
|---|---|---|---|
| +1 h | 0.938 | 0.971 | **0.955** |
| +3 h | 0.672 | 0.919 | **0.776** |
| +6 h | 0.435 | 0.826 | **0.570** |
| +12 h | 0.116 | 0.746 | **0.201** |
| +24 h | 0.078 | 0.545 | **0.136** |

Real chance-of-rain skill exists at +1 h and +3 h and is gone by +12 h, where a
precision of 0.116 means nine of every ten rain warnings are false. The +1 h model is
largely persistence: it is raining, so it will still be raining. These are pooled,
dry-season numbers — the backyard recorded **zero wet hours in July**.

The rest of this report is about why the +6 h and +12 h cells are negative, and why
that is not the failure it appears to be.

## 4. The architecture forbids the measurement

Here is the model's complete feature list:

```python
FEATURE_COLS = [
    "f_temp_c", "f_humidity_pct", "f_pressure_hpa",
    "f_wind_speed_ms", "wind_dir_sin", "wind_dir_cos",
    "f_precip_mm", "f_weather_code",
    "lag_temp_c", "lag_humidity_pct", "lag_pressure_hpa",
    "lag_wind_speed_ms", "lag_rain_mm_1h",
    "hod_sin", "hod_cos", "doy_sin", "doy_cos",
]
```

**There is no station identity.** No `station_id`, no latitude or longitude, no
elevation, no siting descriptor. The model trains across 323 stations pooled together
with no way to tell them apart, and it is therefore obliged to learn **one universal
correction** that applies to every station in the network at once.

Which means it cannot represent the sentence "this station runs warm." Not *fails to*
— **cannot**. There is no parameter in which that fact could be stored. A model asked
to correct a sheltered backyard and a rooftop gauge in Renton with the same function
will land somewhere between them, which is to say: on the regional mean. That is the
one place a microclimate model must not land.

This explains the measured pattern exactly, and it is worth following the logic:

- **At +1 h it works (+42 %)** because `lag_temp_c` carries the station's *current
  state*. If the yard is warm right now, the lag says so, and the model rides it
  forward. Persistence needs no station identity.
- **At +6 h and beyond it goes negative** because persistence has decayed and what is
  left to exploit is the *systematic* bias — the thing that requires knowing which
  station you are standing in. The model has no way to encode it, so it falls back on
  the region-average map and drags my backyard toward a mean it does not live at.

The skill curve is not measuring how far local data reaches. It is measuring **the
exact point where the architecture runs out of the only local signal it can express**.

The founding hypothesis has therefore never been tested. Ten weeks of work produced a
strong *regional* corrector and scored it against a backyard. The fix is architectural,
not evidentiary: train on the own-station target directly, or give the pooled model a
station identity (an embedding, a per-station offset, or siting covariates) so it can
say what it currently cannot.

**First evidence that this is right, and it is not subtle.** Training on own-station rows
alone — 1,089 of them — against the pooled model's 230,000:

| +3 h, own station | MAE | skill vs Open-Meteo |
|---|---|---|
| Open-Meteo | 0.963 | — |
| Pooled model (230k rows) scored on the backyard | 0.941 | 2.3 % |
| **Own-trained** (1,089 rows, own + NWP) | **0.798** | **17.1 %** |
| **Own-trained + QC'd upwind band means** (§7) | **0.718** | **25.4 %** |

**A model trained on 1,089 of the right rows beats one trained on 230,000 of the wrong
ones, by 15 %.** It was never a data-volume problem. The pooled model had 200× more data
and lost, because it was answering a different question — and no quantity of the wrong
question converges on the right answer.

*Caveat: those two middle rows come from different temporal splits (294 vs 273 test rows;
the pooled evaluation took its boundary from the pooled frame, the own-trained one from
the own frame), so this is indicative rather than controlled. The effect (0.143) dwarfs
the baseline discrepancy between the two splits (0.010), so a split artifact is unlikely,
but a clean head-to-head is owed before this is quoted as settled.*

Stacked together, the two architectural corrections — train on the right target, then
average QC-screened upwind neighbours by distance band — take the backyard from
**0.963 °C (Open-Meteo) to 0.718 °C: 25.4 % skill at +3 h.** That is the founding question,
answered affirmatively, at one horizon, on eight weeks of summer data.

## 5. The interaction is the model

The feature list contains `hod_sin` and `hod_cos`. The model knows the target hour's
phase in the daily cycle. So the diurnal cycle should be handled — and for the tree
models it is:

| Horizon | Ridge | RandomForest | XGBoost |
|---|---|---|---|
| +1 h | 0.711 | 0.691 | 0.691 |
| +3 h | 1.223 | 1.316 | 1.133 |
| +6 h | 1.472 | 1.511 | 1.334 |
| +12 h | **1.864** | 1.632 | 1.508 |
| +24 h | **1.798** | 1.763 | 1.591 |

XGBoost and RandomForest rise **monotonically** with lead time, as a forecast model
should. **Ridge does not.** It peaks at +12 h and then *improves* at +24 h — 1.864 →
1.798 — and does the same on own-station rows (1.619 → 1.555). Same direction, same
magnitude, two different populations.

+12 h is the **anti-diurnal point**: an observation twelve hours old is maximally out
of phase with its target (noon predicting midnight), while a 24-hour-old one is back in
phase. Ridge is the only model that feels it.

The reason is precise and it is not about capacity in the vague sense. Having
`hod_sin` as a feature makes hour-of-day a **main effect**. But the signal that matters
is an **interaction**: a noon reading implies something different about midnight than
it does about 1 pm. Formally the model needs `lag_temp × hod`. Ridge is additive and
cannot express a product of its inputs without an explicit cross term. Trees get it for
free — split on `hod`, then split on `lag_temp`.

This is the paper's cleanest result, and it generalises past weather:

> **Having the variable is not the same as representing the relationship.** The feature
> list looked complete. Every quantity a meteorologist would ask for was present and
> correctly encoded. The model still could not use them, because the structure that
> mattered was multiplicative and the model was additive.

It also converts "use trees, not linear models" from a preference into a **mechanism**.
Trees do not win here because they are fancier. They win because this problem has
interaction structure that an additive model provably cannot represent. That is a
falsifiable claim: adding explicit `lag_temp × hod_sin` cross terms to the Ridge design
matrix should remove the +12 h anomaly. That experiment has not been run.

**A related trap in the same feature list.** `doy_sin`/`doy_cos` are there to encode the
annual cycle. Over a 58-day corpus they encode nothing of the kind — they are a
near-monotonic ramp, "days since 2026-05-20" wearing trigonometry as a costume. The
model cannot learn a year's seasonality from eight weeks; it learns a local trend and
extrapolates it off a cliff. This is not hypothetical: it is the direct cause of a
Ridge model scoring **MAE 26.2 °C** in §6's first attempt. A feature can be correct in
principle, standard practice, and actively harmful at your current sample size, all at
once.

## 6. Volume is not the lever; variety is

The project's standing assumption was that accuracy would improve as the corpus grew
toward a year. Measured, that is **false for row count**.

Holding the calendar window fixed at 44.2 days and varying only how many rows the model
sees (+3 h, Open-Meteo baseline 1.456 °C):

| n_train | span | Ridge | RandomForest | XGBoost |
|---|---|---|---|---|
| 11,527 | 44.2 d | 1.243 | 1.221 | **1.162** |
| 23,055 | 44.2 d | 1.233 | 1.283 | **1.152** |
| 57,639 | 44.2 d | 1.228 | 1.284 | **1.157** |
| 115,279 | 44.2 d | 1.226 | 1.298 | **1.127** |
| 172,919 | 44.2 d | 1.225 | 1.290 | **1.133** |
| 230,559 | 44.2 d | 1.223 | 1.315 | **1.135** |

**The curve is flat.** Twenty times the training rows buys XGBoost 0.027 °C. The model
plateaued around 11.5k rows; the corpus holds 230k. Collecting more hours of Seattle
summer does essentially nothing, and the mechanism is intuitive once stated: summer here
is a persistent, low-variance regime. The model learned it weeks ago. The residual error
is not ignorance. It is irreducible noise plus missing features.

Read the scope carefully, because this is where the result becomes useful rather than
discouraging. Random sampling holds the *season* fixed, so what is exhausted is **more
rows from the same eight weeks**. What a year adds is not rows — it is **seasonal
variety**: winter, frontal passages, storms, a wet season. Nothing here tests that,
because those months do not exist in the corpus yet.

And the plateau applies to the *pooled* model, which §4 established is answering the
wrong question. The model that matters holds **1,366 own-station hours — 8.4× below the
plateau**. At 24 rows per day it reaches ~11.5k in roughly **sixteen months**.

So the two halves of "we need a year of data" separate cleanly:

- For the **pooled regional model**: false. Already saturated by 20×.
- For the **own-station microclimate model**: true, and it is the model that has not
  been built yet.

The 290k pooled rows are ~99% irrelevant to the microclimate question. The corpus that
is genuinely starved is one station × 58 days.

## 7. The crowd does not transfer

An offline ablation sweep compared *own + NWP* features against *own + NWP + network*
on identical rows with bootstrap confidence intervals, across station counts, distance
bands, and upwind-angle tolerances. Two findings cut directly against the intuition that
motivated building a 323-station network:

- **Station count plateaus at n = 1.** One upwind neighbour captures essentially the
  whole gain (+1 h: n=1 gives +0.098 °C, n=20 gives +0.120 °C).
- **The nearest band is the weakest.** The 0–2 km cluster does nothing measurable
  (+0.016, not significant), while 2–5, 5–10, 10–25 and 25–50 km all contribute about
  equally (~+0.12) and are statistically interchangeable.

The pre-run hypotheses — that near stations would drive short horizons and far stations
long ones, and that a deliberate multi-band mix would beat any single band — were **not
supported**.

This is where the Meier comparison earns its place. Crowdsourced PWS networks demonstrably
add value for *spatial field reconstruction*: mapping an urban heat island genuinely
requires many stations, because the quantity being estimated is a surface. **Point
correction is a different problem.** Estimating one location's error needs the upwind air
that is about to arrive, and one station samples that nearly as well as twenty. The
crowd's value is task-dependent, and the transfer is not automatic.

⚠️ **These sweep results predate a defect described in §9 and share it.** The harness
calls the same dataset builder, so every configuration was evaluated against a ~1 h-lead
forecast regardless of horizon. The "network features hurt at +24 h" conclusion is the
most exposed and may be measuring lag staleness rather than forecast lead. The n-plateau
and distance-band findings are less exposed — they compare configurations against each
other at a fixed horizon — but none of it has been re-verified. **The sweep needs a rerun
before any of it is cited as settled.**

### Testing the physics directly: a null result

The 0–2 km finding has a mechanism once you do the arithmetic. Air moves. The parcel
arriving at `t+h` is currently `v · h` upwind, and at the forecast's median wind
(2.26 m/s) that is 8 km at +1 h and 98 km at +12 h. A station 2 km away is **~15 minutes
of advection** from home — it sits in the same air mass and cannot carry new information.
It is not a weak predictor; it is geometrically excluded from being one.

That reasoning implies a better feature than a fixed band: select the station nearest the
projected point `v · h` upwind and read *its* observation at `t`, rather than averaging a
static cohort. `src/features/advection.py` implements it (NWP wind for the velocity — see
§8 for why the network's own anemometers cannot be used), and
`tools/advection_experiment.py` tests three arms — base, base+cohort-mean, base+advection
— on identical own-station rows with bootstrap CIs.

**It does not work, at least not measurably here.**

| Horizon | model | base | cohort | adv | Δ adv vs base | Δ adv vs cohort | valid |
|---|---|---|---|---|---|---|---|
| +1 h | linear | 0.496 | 0.567 | 0.499 | −0.003 [−0.014,+0.009] | +0.066 [+0.006,+0.132] | 64 % |
| +3 h | linear | 0.773 | 0.785 | 0.756 | +0.017 [−0.009,+0.045] | +0.030 [−0.024,+0.092] | 62 % |
| +6 h | linear | 0.915 | 1.006 | 0.900 | +0.016 [−0.020,+0.052] | +0.105 [+0.038,+0.174] | 61 % |
| +6 h | xgboost | 0.992 | 0.971 | 1.023 | **−0.031 [−0.058,−0.004]** | **−0.051 [−0.090,−0.012]** | 61 % |
| +12 h | linear | 1.072 | 1.069 | 1.033 | **+0.039 [+0.019,+0.056]** | +0.038 [−0.014,+0.092] | 36 % |
| +24 h | linear | 1.410 | 1.406 | 1.468 | **−0.058 [−0.072,−0.044]** | −0.060 [−0.101,−0.019] | 10 % |

*(Full grid: [`experiments/advection_vs_cohort.csv`](experiments/advection_vs_cohort.csv).)*

**Against base, the network is null**: one of ten cells is significantly positive, two are
significantly negative. **Against the cohort mean it is mixed** — four significant wins for
dynamic selection, three for band averaging. And at +6 h the two model classes **disagree
on the sign** of the effect, which is the signature of noise rather than signal.

Three reasons this test cannot settle the question, recorded so the next attempt is
better rather than merely repeated:

1. **Coverage runs out.** Median reach is 92 km at +12 h and **193 km at +24 h**, past the
   edge of the 100 km registry, so only **10 % of +24 h rows** have a real upwind station.
   The rest are imputed. That is itself a finding — **a 100 km network caps advection
   features at roughly +6 h at median wind** — but it means the long horizons measure
   imputation, not physics.
2. **Two of the features are wind-speed proxies.** `adv_distance_km` *is* `v · h`, a
   monotonic rescaling of a wind speed the model already has, and `adv_valid` is a
   calm-wind flag. Any gain may be the model learning "is it windy" rather than "what is
   upwind". This is a design error; the clean test uses only `adv_temp_c` and
   `adv_temp_gradient`.
3. **The sample is too small.** 1,089 train / 273 test rows, chasing effects of
   0.02–0.10 °C. The intervals are wide enough to accommodate almost any story.

A prediction was recorded before the run: that advection would help at +3 h and +6 h and
fade by +12 h, with the explicit note that *"if it helps at +12 h but not +3 h, something
is wrong."* +3 h is null; the only significant win is +12 h linear. By its own stated
criterion the result is not to be trusted. It is reported here because a pre-registered
prediction that fails is worth more than one quietly revised afterwards — and because
this is the fourth idea in this report that looked sound and did not survive measurement.

### What actually worked: average the band, don't pick the station

The single-station design above was wrong, and wrong for a reason this report had
already established two sections earlier. Every CWS study finds **0.5–1.0 °C of residual
per-station bias surviving QC** (§8). Selecting *one* station therefore maximises
exposure to precisely the error that dominates this problem. Averaging kills the random
component as √N while leaving the systematic part — which is exactly why the literature
saturates at ~4 stations (Nipen) and why this project's own sweep plateaued at n=1. The
saturation everyone reports is not a disappointment; it is the signature of the random
part being gone.

Rebuilt as **elevation-adjusted upwind band means** — stations selected per row against
the forecast wind bearing, grouped into four distance bands, averaged within each,
adjusted to the home station's elevation before averaging — the network finally pays:

| Horizon | model | Open-Meteo | base | + network (QC'd) | Δ vs base |
|---|---|---|---|---|---|
| +1 h | xgboost | 0.946 | 0.442 | 0.445 | −0.003 [−0.045,+0.038] |
| **+3 h** | **xgboost** | **0.963** | **0.798** | **0.718** | **+0.080 [+0.037,+0.123]** |
| +6 h | xgboost | 1.037 | 0.998 | 0.958 | +0.039 [−0.006,+0.089] |
| +12 h | xgboost | 1.147 | 1.094 | 1.123 | −0.029 [−0.066,+0.005] |

*(Full grid: [`experiments/network_qc_vs_noqc.csv`](experiments/network_qc_vs_noqc.csv).)*

**+3 h is the first clean, significant network win this project has produced**: a 10 %
MAE cut, **25.4 % skill against Open-Meteo, in the backyard**. +1 h is null exactly as
the geometry demands — at 11 km of advection the neighbours are already in the same air
mass. +24 h is discarded: the two model classes disagree in sign there (xgboost +0.052,
linear −0.135), the noise signature again.

### Quality control: necessary, but not the cause

Nipen's result — *without QC the merged product is only marginally better than raw NWP,
and worse in daytime and summer* — predicted that screening would be the difference
between a useless network and a useful one. **It was not.** Un-screened band means score
0.721 at +3 h; QC-screened means score 0.718. QC contributes **+0.002 — nothing.**

The QC pass is still worth having, and the reason is mechanistic rather than
promotional. It helps **Ridge** significantly (+0.042 at +3 h, +0.013 at +1 h) and does
nothing for trees. A tree routes around a bad station by splitting on it; a mean cannot.
Averaging garbage poisons a linear model and merely dilutes a tree's. So QC matters most
for the model class least likely to be shipped — which is worth knowing, and is not what
the literature led this project to expect.

What QC *did* deliver was a measurement of the network's condition, and it is not good:

| | |
|---|---|
| Testable station-hours | 294,516 |
| Hourly readings flagged | **16.3 %** |
| Stations `suspect` or `isolated` | **80 / 322 (24.8 %)** |

**One quarter of the network is bad data**, and it has been feeding every model, sweep
and experiment in this report. The flagged fraction — 16.3 % — lands on Nipen's SCT
removal rate of 16.3 % exactly, from an independently designed pass on a different
network in a different country. That convergence is the best available evidence the
screening is calibrated rather than arbitrary.

The indoor-sensor detector is the part no eyeball would have found: `KWAGRANI70`
correlates **0.19** with its buddy median, `KWASNOHO174` 0.57, `KWASEATT2409` 0.61. Those
stations are climatologically plausible *every single hour* — they pass any outlier test
— they simply do not track the outdoor diurnal cycle, because they are indoors.

Two departures from the published schemes were forced by this network's shape.
CrowdQC+'s **3 km buddy radius** assumes a city: Berlin and Toulouse pack 500–2,000
stations into one. This network is 322 over a 100 km radius — one per 97 km², ~10 km mean
spacing — so a 3 km rule would have isolated nearly everything; k-nearest with a 25 km cap
leaves exactly **one** station isolated. And **elevation adjustment is mandatory here,
not a refinement**: the registry spans **0–1174 m**, some 7.6 °C of legitimate lapse
rate, so an unadjusted comparison against a crowd median would have condemned the Cascade
foothills as broken sensors. Open-Meteo had been returning DEM elevation on every
forecast call since day one, and the code discarded it — the same way it discarded wind
gusts and cloud cover until June.

## 8. The instrument is part of the model

The project's premise is that the backyard's biases are signal. Measuring them produced
a number that inverts the premise's framing.

**Open-Meteo forecasts my station better than it forecasts the network.** At true lead
times over the same window: own-station MAE **1.013 °C** at +1 h, pooled network MAE
**1.553 °C**. The backyard is *easier* to forecast than the average neighbour.

That is not because my yard is unremarkable. It is because the pooled baseline is
inflated by ~320 crowdsourced stations of wildly mixed siting and calibration — gauges
on sunny rooftops, unshielded thermometers, sensors under eaves. The pooled "forecast
error" is substantially **sensor error wearing a forecast's clothing**.

The decomposition makes it visible. Pooled error barely grows with lead time (+10% from
+1 h to +24 h) while own-station error grows sharply (+25%). A forecast error must grow
with lead; a *sensor* error does not care how far ahead you looked. The pooled number is
dominated by a constant noise floor:

> pooled error ≈ station noise (constant in lead) + forecast error (grows with lead)

This matters beyond a baseline correction, because **those same 323 stations are the
network being used as input features**. The crowd is not ground truth. It is ground
truth plus an unknown, station-specific, largely un-modelled error. Meier's group built
a whole QC methodology for exactly this reason.

And it points at the uncomfortable question this project cannot yet answer. The
measured biases are real:

- **Wind direction: −37° CCW** vs the network mean, validated at wind ≥ 1 m/s where the
  spread collapses to a 7° standard deviation. The shelter rotation is highly stable.
  Wind *speed* is not suppressed (own/network ratio 0.96) — the yard is not calmer in
  magnitude, it is **rotated**. My "calmer" intuition may be about turbulence and gust
  character rather than mean speed, which the current sensor package cannot resolve.
- **Temperature: +2.0 °C hotter by day, matched at night** (30 paired days, late
  spring).

That temperature bias is the one to be careful with. A backyard sheltered from wind is
a backyard with **reduced aspiration across the temperature sensor** — the classic
setup for radiative heating error. A poorly-aspirated thermometer in still air reads
high in daylight and correctly at night. That is *precisely* the diurnal signature
observed. So the +2 °C may be microclimate, or instrument, or both, and **the current
instrumentation cannot separate them.** An aspirated or better-shielded reference
sensor alongside the existing one would settle it, and until then the headline "my
backyard is warmer" carries an asterisk.

Note the pleasing convergence with §5, though: the bias is *diurnal* — warm by day,
neutral at night. Capturing it requires exactly the interaction structure §5 identified,
and a station identity §4 showed does not exist. The three findings are the same finding
seen from three directions.

## 9. What went wrong, and what it misdirected

Six data defects have shipped in this project. Each produced **plausible, publishable,
wrong numbers**. None raised an exception. None were caught by a unit test. Every one was
caught by noticing that a number was physically impossible. They are recorded here
because a reader deserves to know which published figures were retracted, and because
several of this report's own earlier claims were casualties.

| # | Defect | What it produced | The tell |
|---|---|---|---|
| 1 | `timezone=auto` parsed local-time forecasts into a UTC column, shifting every `valid_time` by 7 h | Open-Meteo looked 3× worse than reality; the first model "beat" it by 5× while merely learning the offset | A skill number too good to be true |
| 2 | Rain target derived from a column the network never populates | `build_dataset` returned **1 positive sample in 76k**; the README blamed the weather for a month | A wet-hour rate of ~0 in a Seattle spring |
| 3 | A timezone strip silently zeroed every network feature | Every ablation configuration returned **byte-identical** results | Identical-across-configs |
| 4 | fill-0 on pressure (physically ~1015 hPa) | Ridge blew up to MAE 20–30 | A magnitude no instrument could produce |
| 5 | **The forecast join ignored the horizon** | Training used a ~1 h-lead forecast at *every* horizon | Forecast error that did not grow with lead |
| 6 | A learning-curve tool shrank the calendar span along with row count | Ridge at **MAE 26.2 °C** | Same as #4 — an impossible magnitude |

**Defect 5 is the one that misdirected the science.** The training join selected the
freshest forecast issued before the target hour; the horizon parameter only shifted the
*lag observation*. For a historical target hour the freshest forecast is about an hour
old — so the "+24 h model" was really *predict t from a 1 h-lead forecast for t, plus an
observation from t−24 h*. The horizon axis was measuring lag staleness, not forecast
lead. Meanwhile the serving path asks for `now + horizon` and can only ever see a
24 h-lead forecast, so the model was trained to trust a near-nowcast and then handed
something far noisier in production: **train/serve skew that widened with horizon**, and
that the metrics could not see.

What it cost, concretely:

- The headline temperature table (**−60 % / −31 % / −24 %**) was wrong and is retracted.
- Rain F1 at +3 h was reported as **0.89**; the true value is **0.776**. At +24 h it was
  reported as **0.01**; the true value is **0.136**.
- Rain F1 at +1 h barely moved (0.96 → 0.955) — and that is the **consistency check that
  confirms the diagnosis**, because at +1 h the training and serving leads coincide, so
  the defect was inert exactly where theory predicts.
- The Open-Meteo baseline was pinned to a flat **1.68 °C at every horizon** — the
  physical impossibility that exposed the whole thing. Corrected, it behaves: 1.553 →
  1.716 across +1 h → +24 h.
- The §7 sweep inherits the defect and is not yet re-verified.

**Defect 6 deserves its own note**, because it was committed *by the tool built to audit
the other five*, on the same day, while writing this report. The learning curve's first
run took "the most recent N rows," which shrinks the calendar window along with the row
count. At 5% of a 58-day corpus that is a 2.3-day window in which the seasonal features
are constant, the scaler divides by a near-zero standard deviation, and Ridge
extrapolates weeks away. MAE 26.2 °C. The fix was to separate the two variables
(`--sample random` holds the span fixed); the lesson is that knowing about a failure
mode confers no immunity from it.

**The lesson is not "write more tests."** The test suite covered bearing calculations,
aggregation, wind reference, gradients — pure functions with obvious contracts, none of
which ever had a bug. All six defects lived in SQL, joins, and ingest, where correctness
is invisible and failure is silent:

> **The parts that are easy to test are not the parts that lie to you.**

So the physical checks are now code. `src/ml/invariants.py` asserts that baseline error
grows with lead time, that no feature is silently constant, that values sit inside
physical bounds, that the forecast lead honours the horizon, and that the wet-hour rate
is climatologically plausible — one predicate per defect above, each unit-tested against
that defect's actual signature. `tools/check_invariants.py` runs them against the live
corpus and exits non-zero, so it can gate a retrain.

This is the domain-experience argument in its strongest form. The single most valuable
correction in ten weeks — rebuilding rain as *chance of rain first, amount only if
likely* — came from meteorological convention, not from anything the data volunteered.
The single most damaging defect was invisible for weeks because forecast error growing
with lead time is so obvious to a forecaster that nobody thinks to assert it. **Both are
the same lesson: the metric cannot tell you it is answering the wrong question.**

## 10. Three standing assertions

Recorded in advance, with the evidence for and against, so they can be judged later
rather than quietly revised.

### Assertion 1 — the model will improve with a full year of data

**Verdict: supported, with a correction to what "data" means.**

For the pooled regional model this is false: §6 shows a flat curve, saturated by 20×.
But for the **own-station model** — the one that answers the actual question — it is
right, and the numbers are on its side. That model has **1,366 rows against a plateau at
~11.5k**, 8.4× short, reaching it in roughly sixteen months at 24 rows/day. A year is a
good estimate for the model that matters.

The second half is untestable today and probably more important: what a year adds is
**seasonal variety**. Everything measured here is a summer result on a summer test
window. Winter brings frontal passages, storms, and a wet season, which is where local
data has the most to say and where NWP has the most trouble.

A prediction to be scored against: **absolute MAE will get worse in winter**, because
winter temperature is more variable and harder to forecast. **Skill relative to
Open-Meteo may well improve** — but I do not know, and this report will not pretend to.

### Assertion 2 — XGBoost and RandomForest will outperform LSTM and linear models

**Verdict: supported against linear with a mechanism; RandomForest untested fairly; LSTM
properly pre-registered.**

- **Against linear: supported, and §5 supplies a *reason* rather than a benchmark.**
  XGBoost wins at every row count from 11.5k to 230k, and — more convincingly — Ridge is
  the only model that goes non-monotonic at the anti-diurnal point, because it cannot
  express the `lag_temp × hod` interaction that this problem is built out of. That is an
  argument from structure, not from a leaderboard.
- **A second mechanism arrived with the QC work** (§7), and it points the same way.
  Screening the network helps Ridge significantly (+0.042 at +3 h) and does *nothing* for
  XGBoost (+0.002). A tree routes around a bad station by splitting on it; a mean cannot.
  So trees are natively robust to precisely the station noise that dominates this
  problem — which is an argument from structure again, not from a leaderboard, and it is
  the second time in this report that trees win for a reason rather than by a margin.
- **RandomForest: not yet a fair test.** It currently trails, but it is capped at 100
  trees and depth 10 to fit the droplet's memory budget — the API holds every bundle in
  memory on a 4 GB box shared with Postgres, and an unconstrained forest is ~100 MB each
  across ten combinations. Its *degradation* with more data (1.221 → 1.315) is most
  likely that cap: a fixed-depth tree averaging increasingly heterogeneous data gains
  bias. This is a deployment constraint reported honestly, not a verdict on the model
  class. RandomForest was originally preferred for its confidence bands, and those are
  still not surfaced.
- **Against LSTM: unmeasured, and deliberately so.** The LSTM is scheduled for ~November
  2026, intentionally under-data, with the explicit purpose of testing whether deep
  learning is the right tool rather than the default one. **The prediction is that it
  will lose**, on the grounds that a single-station bias-correction task has weak
  sequential structure beyond the lag features already supplied, and that trees need less
  data, train in seconds, and cost nothing to serve. Recording this in advance is what
  makes a negative result honest rather than a rationalisation.

One complication worth carrying rather than hiding: on the **own-station** target at
+1 h, Ridge currently beats both trees (0.550 vs 0.627 / 0.616). The trees' pooled
advantage comes from fitting the regional signal harder — which is exactly the signal a
microclimate model does not want. So the answer to "which model is best" may depend on
*which question is being asked*, not only on how much data there is. That is not a
refutation of the assertion. It is a warning that it will need to be re-tested once §4's
architecture is fixed.

### Assertion 3 — hyperlocal data should still beat the regional model

**Verdict: CONFIRMED at +1 h and +3 h.** Not "plausible", not "untested".

| +3 h, own station | MAE | skill |
|---|---|---|
| Open-Meteo | 0.963 | — |
| Pooled model (230k rows) on the backyard | 0.941 | 2.3 % |
| Own-trained (1,089 rows) | 0.798 | 17.1 % |
| **Own-trained + QC'd upwind band means** | **0.718** | **25.4 %** |

The intuition was right the whole time, and the measurements support the mechanism: a
**stable −37° wind rotation** (std 7°) and a **+2.0 °C daytime warm bias** with nights
matching — real, consistent, directional biases, exactly the shape a correction model can
exploit.

What was wrong was never the hypothesis. It was §4: **the model had no station identity,
so it could not represent a per-station bias at all.** The +42 % at +1 h was the model
exploiting the only local signal it *could* express — persistence, through the lag
feature. The negative skill at +6 h/+12 h was what happens when persistence decays and
there is no channel left for the systematic bias. Give it the right target and the right
neighbours and the signal is there.

Two honest complications remain:

1. **Part of the warm bias may be the instrument** (§8), and the literature makes this
   worse rather than better: Nipen finds +0.5 °C residual warm bias surviving QC, Napoly
   +0.95 K, Fenner 0.5–1.0 K in daytime summer, and Sgoff finds that assimilating
   Netatmo *without* a diurnal-cycle bias correction actively degrades forecasts. A
   wind-sheltered sensor is a poorly-aspirated sensor, and poor aspiration reads high in
   daylight and correct at night — the exact signature measured here. So the *magnitude*
   of "my backyard is warmer" is unverified until a reference sensor exists. Note this
   does not undermine the skill numbers above: an instrumental bias is still learnable and
   still worth correcting. It just is not microclimate.
2. **"Calmer" is not what was measured.** Wind speed matches the network within noise
   (ratio 0.96); direction rotates. The felt calmness is likely turbulence and gust
   structure, which this sensor package cannot resolve. And the network's own wind is
   unusable in absolute terms anyway — median 0.28 m/s against the forecast's 2.26,
   mostly a 10 m-vs-2 m measurement-height artifact shared by every backyard anemometer.

On rain the assertion is appropriately hedged, and the hedge is correct: real skill at
+1 h and +3 h, none by +12 h, every number pooled and from the driest weeks of the year.
The directional signal worth chasing: over 30 paired spring days the regional forecast
**under-called local rain 5×** (21 mm forecast vs 100 mm measured), and watering decisions
would have differed on **~17 % of days**. That is measurable in October, not July.

## 11. What comes next

In leverage order, and shaped by the findings above rather than by the original roadmap:

1. **Ship what §4 and §7 already demonstrated.** The own-trained model plus
   elevation-adjusted, QC-screened upwind band means reaches 25.4 % skill at +3 h, and the
   live site serves neither. This is no longer a research question; it is a deployment
   task. Both architectural routes remain worth testing — a pooled model with per-station
   offsets keeps the data richness, an own-station model keeps the microclimate pure, and
   regional-base-plus-local-correction gets both — but the own-trained result is in hand.
2. **Settle own-vs-pooled with a controlled comparison.** The 0.798-vs-0.941 result in §4
   comes from two different temporal splits. The effect dwarfs the discrepancy, but the
   claim deserves one clean head-to-head before it is load-bearing.
3. **Rerun the neighbour sweep on the corrected join** (§7) before citing any of it.
4. **Test the interaction claim directly** (§5): add explicit `lag_temp × hod` cross
   terms to Ridge. If the +12 h anomaly disappears, the mechanism is confirmed and the
   argument for trees becomes an argument about *representation* rather than about
   families.
5. **Drop or gate `doy_sin`/`doy_cos`** until a full annual cycle exists (§5).
6. **Get a reference temperature sensor** — aspirated or better shielded — to separate
   microclimate from radiative error (§8). Without it, the project's headline claim about
   its own backyard has an asterisk it cannot remove.
7. **Wait for the wet season.** October through March is when rain becomes measurable and
   when local data should matter most. That is calendar, not effort.
8. **Freeze the pre-registration and lock a holdout.** The document exists but remains a
   partial draft — holdout window, git SHA and episode lists are still `<TBD>`, and its
   primary horizon (6 h) was only trained on 2026-07-15. Given that six defects have
   shipped, this discipline is not ceremony; it is the demonstrated remedy.
9. **LSTM, ~November**, as pre-registered — landing on a corpus that is longer but, in
   row-count terms, no richer.

## 12. Reproducibility

Every number in this report comes from
[`experiments/model_metrics.csv`](experiments/model_metrics.csv), a git-SHA-stamped
export of the full training time-series, or from the CSVs beside it. Until 2026-07-15 no
figure the project published could be regenerated from the repository — the metrics
existed only in the production database. That was itself a defect, and on a paper about
whether numbers mean what they claim, an ironic one.

```bash
python -m tools.check_invariants --target temp_c    # physical-plausibility gate
python -m tools.own_station_eval  --target temp_c    # backyard skill, not regional
python -m tools.learning_curve    --target temp_c --horizon 3 --sample random
python -m tools.export_metrics    --out experiments/model_metrics.csv
```

One caution when comparing across retrains: the split is temporal 80/20, so every
retrain's test set is a different window. Compare skill-vs-baseline, not raw MAE.

## 13. On method: why this was a conversation

Most of the analysis in this report, and effectively all of the code, was produced by an
AI agent (Claude Code) working against this repository. That is worth stating plainly,
and it is worth examining, because the way the work went cuts against the prevailing
claim that most software is — and ought to be — written autonomously by AI given a
sufficiently good brief.

### What actually happened

Every substantive correction in this report arrived as a **short question from a
non-specialist**, not from the analysis:

| The question | What it overturned |
|---|---|
| *"Shouldn't the diurnal cycle be addressed within the model — do we include date & time?"* | Killed the report's then-headline finding, that local data has a "non-monotonic shelf life." It was wrong. It had already been committed to the README, to the project's memory, and to this paper's outline as *the most novel result*. It had survived the agent's own review. The true explanation — Ridge cannot represent `lag_temp × hod`, which is why it is the only model that wobbles at +12 h (§5) — is better, and was not found by analysis. |
| *"I would suggest that averaging is good — but averaged by distance."* | Produced **the only significant network win in the project's history** (§7). The agent had built single-station selection, which was precisely backwards given the per-station noise the agent itself had documented two sections earlier. |
| *"Shouldn't it make a difference to track this specific station?"* | Led to the advection work, and surfaced that the network's own anemometers are unusable (0.28 m/s vs the forecast's 2.26) — a 10 m-vs-2 m semantics error sitting in the feature vector. |
| *"How does Meier track individual stations?"* | Corrected the agent's claim that Nipen et al. was data assimilation. It is post-processing — the *same family as this project* — which inverted the framing of where this work sits in the literature. |
| *"Are my points accurate?"* | Started the audit that found defect 5, which invalidated every horizon-dependent number the project had published. |

The person asking is not an ML specialist and does not claim to be. The questions were not
expert challenges. They were the **obvious** questions — the ones the analysis had
reasoned past.

### Why this is structural, not incidental

Notice what every failure in this report has in common. The five shipped defects (§9)
produced *plausible numbers that answered the wrong question*. The three predictions the
agent made and lost in a single day — advection-by-single-station, QC-as-the-cause of the
network win, the shelf-life story — were *plausible reasoning that did not survive
measurement*. **Same species.**

That matters more than it first appears, because plausible-but-wrong is exactly what a
language model produces most fluently. The instrument being used to audit for
plausible-but-wrong is the instrument most prone to generating it. That is not a
capability gap that closes with a better model; it is a structural property of the method,
in the same way that §8's radiative bias is not a calibration problem but a consequence of
where the sensor sits. **The check has to come from outside** — from physics, from
measurement, or from someone asking why the number looks like that.

This is also why the "obviousness" of the questions is the point rather than an
embarrassment. An expert reviewer would have engaged the agent's reasoning on its own
terms. A non-specialist asked whether the clock was in the model — and the reasoning
collapsed, because it had been elaborate and wrong rather than simple and wrong.

### What the 80% claim measures, and what it does not

The claim that most code is and should be written by AI is **not contradicted here**.
Essentially every line in this repository was written by the agent: ~2,000 lines in the
final day alone, 145 unit tests, five analysis tools, a QC subsystem, all deployed and
running. That part worked, completely, and at a speed no human pace matches.

What the claim silently merges is two different activities. **Writing code and deciding
what is true are not the same job.** The 80% statistic measures the first and is entirely
silent on the second. And this report's central finding is that the first was never the
bottleneck:

> The model was never the bottleneck; the measurement was.
> The agent was never the bottleneck; the judgment was.

The division of labour that worked was not "human supervises AI." It was closer to: the
agent supplied mechanical execution, breadth, literature synthesis, and the patience to
audit its own SQL; the human supplied physical intuition, a refusal to accept elaborate
answers, and the domain sense that a backyard is warmer and that noise should be averaged.
Neither half would have produced this report. The agent alone would have shipped the
shelf-life finding — that is not speculation, it *was committed to the repository*. The
human alone would not have found the join defect, built the invariants, or read six papers
in an afternoon.

### The caveat this section is obliged to carry

**This is the weakest evidence in the report and should be read that way.** It is n=1,
unblinded, uncontrolled, and written by one of the two participants — a standard of
evidence the preceding twelve sections would reject on sight. The report demands bootstrap
confidence intervals of a 0.08 °C effect and then makes a process claim from a single
session.

The counterfactual is untested. Nobody ran the autonomous version. "The agent would have
shipped the wrong finding" is a claim about a world that was not observed; what is
documented in git is narrower and less dramatic — three claims made, three questions
asked, three claims withdrawn. It is also possible, as the human involved suggests, that
better-specified goals would have caught some of this, and that the failure is one of
briefing rather than of autonomy. That hypothesis is untested too.

What can be said without overreach is this: on a task whose entire difficulty was
distinguishing true numbers from plausible ones, the pauses were where the value was. Not
the generation. The interruptions.

---

## Conclusion

The question was whether a backyard can beat the forecast. The answer is **yes** — by 42%
at one hour, and by **25.4% at three hours** once the model is trained on the right target
and fed elevation-adjusted, quality-screened, upwind neighbour averages. Both of those
are architectural corrections, not modelling ones. The served model still does neither.

That is the whole report in miniature. The hypothesis was right the entire time; what was
wrong was a join that ignored the horizon, a feature list with no way to name a station,
a fixed cohort that averaged the wrong neighbours, and a quarter of the network quietly
reporting from indoors. None of it was the model.

That is the shape of everything here. The four things this project set out to learn —
that data matters more than technology, that the fashionable model is often the wrong
one, that the regional forecast can be beaten locally, that domain experience is not
optional — are all true. But each turned out to be true in a more specific and less
comfortable way than expected:

- **Data matters** — but not volume. Twenty times the rows bought 0.027 °C, while the
  model that mattered starved on 1,366.
- **The fashionable model is often wrong** — but the interesting reason trees beat linear
  is not capacity, it is that this problem is multiplicative and Ridge is additive.
- **Local can beat regional** — but not through a model architecturally forbidden from
  representing locality.
- **Domain experience matters** — most of all in the form of knowing which numbers are
  impossible. Six defects, every one caught by physics rather than by a test.

The through-line is that the model was never the bottleneck. Every single time, the
bottleneck was the **measurement**: what was joined to what, what the metric rewarded,
what the features could express, what the instrument was actually reading. The
engineering was the easy part and took ten weeks. Learning to distrust its output is the
part that is still going.

That pattern held right down to how this report was produced (§13). The code was never the
constraint; roughly two thousand lines of it appeared in a day. The constraint was knowing
which of the resulting numbers meant anything — and the corrections came from someone
asking whether the clock was in the model, and whether it might be better to average.

Which is the argument for calling this preliminary and meaning it. A version of this
report written six weeks ago would have claimed a 60% win over the regional forecast, a
skilled 24-hour rain model, and a validated crowdsourcing strategy. All three were
artifacts. A version written this morning would have claimed a non-monotonic shelf life
for local data. That was an artifact too, and it survived until someone asked an obvious
question about it.

The most valuable thing here is not the results table. It is a precise, dated account of
why the earlier results tables were wrong — and a set of assertions written down in
advance, so that the next time it is wrong, that will be visible too.

---

*Preliminary. ~8 weeks of paired data, summer-only test window, one station, one
season, one city. Sweep results pending re-verification. Instrument bias unresolved.
Judge accordingly.*
