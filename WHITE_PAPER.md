# Can a Backyard Beat the Forecast?

**A preliminary report from ten weeks of hyperlocal weather modelling**

Brad Hinkel · July 2026 · Kirkland, Washington
Code and data: [github.com/bradhinkel/weather-station](https://github.com/bradhinkel/weather-station) · Live: [weather.bradhinkel.com](https://weather.bradhinkel.com)

---

## Abstract

I put a weather station in my backyard to answer one question: can hyperlocal data
beat the regional forecast where I actually live? Ten weeks in, the honest answer is
*not yet, and for an instructive reason*. Against my own station, the current model
beats Open-Meteo by 42% one hour out, ties it at three, and is **worse than doing
nothing** beyond six. But this is not a data problem and not a model problem — it is
an **architecture** problem. The model is trained pooled across 323 stations with no
station identity among its features, so it cannot represent "this station runs warm."
It is structurally forbidden from learning the very thing the project was built to
measure. Separately, a learning curve shows accuracy plateaus at ~11.5k training rows
against a corpus of 230k: **more data of the same kind buys nothing**, while the
own-station model that matters holds only 1,366 rows and is 8.4× *below* that plateau.
The headline result is therefore methodological: the hard part of this project has
never been the model. It has been knowing whether a number means what it appears to
mean.

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

Against the pooled network, the model looks strong. Against my actual backyard, which
is the question, it does not.

**Own-station skill** — the served models scored on own-station test rows only, with
the train/test boundary taken from the pooled frame so those rows were genuinely held
out:

| Horizon | Open-Meteo | Ridge | RandomForest | XGBoost | best skill |
|---|---|---|---|---|---|
| +1 h | 0.952 °C | **0.550** | 0.627 | 0.616 | **+42 %** |
| +3 h | 0.973 | **0.880** | 1.020 | 0.941 | +9.6 % |
| +6 h | 1.030 | 1.158 | 1.212 | **1.050** | **−2 %** |
| +12 h | 1.141 | 1.619 | 1.342 | **1.150** | **−0.9 %** |
| +24 h | 1.397 | 1.555 | 1.468 | **1.377** | +1.4 % |

Skill = `1 − MAE_model / MAE_openmeteo`. Negative means **worse than believing the
regional forecast**.

**The backyard beats the forecast for one hour.** After that it is a wash, and at +6 h
and +12 h the model is actively harmful. For comparison, the same models against the
pooled network score 52 / 22 / 10 / 1.4 / 12 %. The regional win does not transfer to
the yard it was built for.

*Caveat: n_own is 276–295 rows per horizon, without confidence intervals. The pattern
is reported because it is consistent across five horizons and three models; no single
cell should be quoted.*

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

**Verdict: untested, and the current evidence against it is an artifact.**

The intuition is that the backyard is warmer and calmer than the regional forecast says.
The measurements support the underlying claim: a **stable −37° wind rotation** (std 7°)
and a **+2.0 °C daytime warm bias** with nights matching. Those are real, consistent,
directional biases — precisely the shape a correction model should be able to exploit.

The reason the numbers do not yet show it is §4: **the model has no station identity, so
it cannot represent a per-station bias at all.** It is not that hyperlocal data failed to
beat the regional model. It is that the experiment testing that hypothesis has not been
run. The +42 % at +1 h is the model exploiting the only local signal it *can* express —
persistence through the lag feature. The negative skill at +6 h and +12 h is what happens
when persistence decays and there is no channel left for the systematic bias.

Two honest complications:

1. **Part of the warm bias may be the instrument** (§8). A wind-sheltered sensor is a
   poorly-aspirated sensor, and poor aspiration reads high in daylight and correct at
   night — the exact diurnal signature observed. This does not refute the assertion; it
   means the assertion's *magnitude* is unverified until there is a reference sensor.
   And note that if part of the bias is instrumental, it is still learnable and still
   worth correcting — it is just not "microclimate."
2. **"Calmer" is not what was measured.** Wind speed matches the network within noise
   (ratio 0.96); it is direction that rotates. The felt calmness is likely turbulence and
   gust structure, which this sensor package does not resolve.

On rain, the assertion is appropriately hedged, and the hedge is correct. There is real
skill at +1 h and +3 h and none by +12 h, but every rain number here is pooled and from
the driest weeks of the year, with the backyard recording zero wet hours in July. The
one directional signal worth chasing: over 30 paired spring days the regional forecast
**under-called local rain 5×** (21 mm forecast vs 100 mm measured), and watering
decisions would have differed on **~17 % of days**. If hyperlocal rain has value, that is
where it lives — and it will be measurable in October, not July.

## 11. What comes next

In leverage order, and shaped by the findings above rather than by the original roadmap:

1. **Give the model a station identity, or train on the own-station target.** §4 says
   this is the single blocking issue: everything else is optimisation on top of a model
   that cannot express the hypothesis. Both routes are worth testing — a pooled model
   with per-station offsets keeps the data richness; an own-station model keeps the
   microclimate pure. The regional-base-plus-local-correction shape gets both.
2. **Rerun the neighbour sweep on the corrected join** (§7) before citing any of it.
3. **Test the interaction claim directly** (§5): add explicit `lag_temp × hod` cross
   terms to Ridge. If the +12 h anomaly disappears, the mechanism is confirmed and the
   argument for trees becomes an argument about *representation* rather than about
   families.
4. **Drop or gate `doy_sin`/`doy_cos`** until a full annual cycle exists (§5).
5. **Get a reference temperature sensor** — aspirated or better shielded — to separate
   microclimate from radiative error (§8). Without it, the project's headline claim about
   its own backyard has an asterisk it cannot remove.
6. **Wait for the wet season.** October through March is when rain becomes measurable and
   when local data should matter most. That is calendar, not effort.
7. **Freeze the pre-registration and lock a holdout.** The document exists but remains a
   partial draft — holdout window, git SHA and episode lists are still `<TBD>`, and its
   primary horizon (6 h) was only trained on 2026-07-15. Given that six defects have
   shipped, this discipline is not ceremony; it is the demonstrated remedy.
8. **LSTM, ~November**, as pre-registered — landing on a corpus that is longer but, in
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

---

## Conclusion

The question was whether a backyard can beat the forecast. Ten weeks in, it can, for one
hour, by 42%. Beyond that the honest answer is that **the experiment has not been run**,
because the model was built without any way to know which backyard it was standing in.

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

Which is the argument for calling this preliminary and meaning it. A version of this
report written six weeks ago would have claimed a 60% win over the regional forecast, a
skilled 24-hour rain model, and a validated crowdsourcing strategy. All three were
artifacts. The most valuable thing here is not the results table. It is a precise,
dated account of why the earlier results table was wrong — and a set of assertions
written down in advance, so that the next time it is wrong, that will be visible too.

---

*Preliminary. ~8 weeks of paired data, summer-only test window, one station, one
season, one city. Sweep results pending re-verification. Instrument bias unresolved.
Judge accordingly.*
