# Hand-off — Atmospheric UX & Weather Icons

**For:** Claude Code, working in the `weather-station` repo
**From:** Design (this artifact mirrors `Weather Station UX.html` → "A · Atmospheric")
**Live site:** weather.bradhinkel.com

This doc covers three things:

1. UI changes for the home screen (simplified 1h / 3h / 24h forecast)
2. The derivation logic for the **weather icon** (so the UI can pick one) and an icon-set source you can fetch
3. The derivation logic for **"Feels like"** temperature

Each section ends with a checklist Claude Code can work down.

---

## 1 · Forecast UX simplification

### Old behavior
A 24-cell horizontal hourly strip showing temp + precipitation % per hour.

### New behavior
A **single forecast card** with three horizon tabs — `Next 1h` / `Next 3h` / `Next 24h` — and only two readouts per horizon: **Temperature** and **Rain**.

```
┌─────────────────────────────────────────┐
│  [ Next 1h ] [ Next 3h ] [ Next 24h ]   │  ← segmented control (1 active)
├─────────────────────────────────────────┤
│  ☂  Light rain likely     Conf 82%      │  ← icon + plain-language summary
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ TEMPERATURE │  │ RAIN          │    │
│  │   18°       │  │   55%         │    │  ← big numbers, tabular
│  │ Range 16–20°│  │ Accum 0.4 mm  │    │
│  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────┘
```

### Data shape (for the API)

Replace the existing `/api/stations/{id}/forecast` shape with three rolled-up records:

```jsonc
{
  "horizons": [
    {
      "key": "1h",
      "valid_from": "2026-05-07T14:00:00Z",
      "valid_to":   "2026-05-07T15:00:00Z",
      "temp_c":      { "mid": 18, "lo": 17, "hi": 19 },
      "rain":        { "pop": 25, "accum_mm": 0.0 },
      "weather_code": 2,           // WMO code, see §2
      "cond_label":  "Partly cloudy",
      "confidence":  91
    },
    { "key": "3h", ... },
    { "key": "24h", ... }
  ]
}
```

**Roll-up rules** (server-side, when building the response):

| Field          | 1h horizon                        | 3h / 24h horizons                          |
| -------------- | --------------------------------- | ------------------------------------------ |
| `temp_c.mid`   | corrected forecast for next hour  | mean of corrected forecasts in the window  |
| `temp_c.lo/hi` | mid ± lower/upper quantile width  | min/max of forecasts in the window         |
| `rain.pop`     | next-hour PoP                     | `1 - Π(1 - hourly_pop_i)` (compound)       |
| `rain.accum_mm`| next-hour mm                      | sum of hourly mm in the window             |
| `weather_code` | next-hour code                    | most-severe code in the window (rain > cloud > clear) |
| `confidence`   | model's reported skill at that lead time |                                  |

### Frontend checklist

- [ ] Add a `useState('3h')` segmented control above the existing forecast card.
- [ ] Remove the 24-cell hourly row.
- [ ] Render a single condition row (icon + `cond_label` + `confidence`%).
- [ ] Render two cards side-by-side: **Temperature** (`mid`° + range) and **Rain** (`pop`% + `accum_mm` mm).
- [ ] When the user taps a horizon, swap the card body (no full re-fetch — all three horizons should arrive in the same payload).
- [ ] Default selected horizon = `3h`.

A working reference implementation is in `direction-atmospheric.jsx` in the design project — copy the `.atmo__forecast`, `.atmo__horizons`, `.atmo__horizon-*` styles and the `Atmospheric` component's forecast section.

---

## 2 · Weather icon derivation

The station does not see the sky — but Open-Meteo returns a **WMO weather code** that is already discretized exactly the way we want. Use that as the primary key and override with local sensors when they disagree.

### 2.1 · Source of truth: Open-Meteo `weather_code`

Open-Meteo includes `weather_code` in `/v1/forecast` (we already pull it in `src/openmeteo.py`). The codes are the standard **WMO 4677** set:

| Code(s) | Meaning                | Our icon slug         |
| ------- | ---------------------- | --------------------- |
| 0       | Clear sky              | `clear-day`           |
| 1       | Mainly clear           | `partly-cloudy-day`   |
| 2       | Partly cloudy          | `partly-cloudy-day`   |
| 3       | Overcast               | `overcast`            |
| 45, 48  | Fog / depositing rime  | `fog`                 |
| 51, 53, 55 | Drizzle             | `drizzle`             |
| 56, 57  | Freezing drizzle       | `sleet`               |
| 61, 63, 65 | Rain                | `rain`                |
| 66, 67  | Freezing rain          | `sleet`               |
| 71, 73, 75 | Snow                | `snow`                |
| 77      | Snow grains            | `snow`                |
| 80, 81, 82 | Rain showers        | `rain`                |
| 85, 86  | Snow showers           | `snow`                |
| 95      | Thunderstorm           | `thunderstorms`       |
| 96, 99  | Thunderstorm + hail    | `thunderstorms-rain`  |

Reference: [Open-Meteo docs — WMO Weather codes](https://open-meteo.com/en/docs#weather_variable_documentation).

### 2.2 · Local-sensor overrides

After the Open-Meteo code is mapped, run two checks against your own station's most recent observation:

**Override A — actual rain right now:**
```
if rain_mm_1h >= 0.1 and code is "clear" or "partly":
    weather_code = 61   # treat as light rain
```

**Override B — clear-sky solar shortfall (catches missed cloud cover):**
```
expected = clear_sky_solar(lat, lon, t)        # see formula below
ratio    = solar_wm2 / expected
if hour is daytime and code in {0, 1, 2} and ratio < 0.4:
    weather_code = 3        # demote to overcast
```

`clear_sky_solar(lat, lon, t)` is the standard solar-elevation formula:
```
1361 * max(0, sin(solar_elevation(lat, lon, t)))   # W/m²
```
Use `pysolar` (PyPI: `pysolar`) — `pysolar.solar.get_altitude(lat, lon, t)` gives elevation in degrees. Convert to radians, take `sin`, multiply by 1361.

### 2.3 · Day vs night

Append `-day` or `-night` to the icon slug based on solar elevation:
```
is_day = solar_elevation(lat, lon, t) > 0
suffix = "-day" if is_day else "-night"
```
`clear` → `clear-day` / `clear-night`. `overcast`, `rain`, `fog`, `snow`, `thunderstorms` are the same day/night.

### 2.4 · Icon set source

Use **`basmilius/weather-icons`** on GitHub — MIT licensed, ~200 icons, animated + static SVG, day/night variants, hand-tuned and used by Home Assistant.

- **Repo:** https://github.com/basmilius/weather-icons
- **License:** MIT
- **What we want:** the `production/fill/svg-static/` folder (static SVG, full color, no animation) — keeps render cost low and avoids accessibility issues.

**Filenames are predictable** — they match our slugs almost 1:1:
```
clear-day.svg              partly-cloudy-day.svg
clear-night.svg            partly-cloudy-night.svg
overcast.svg               rain.svg
fog.svg                    drizzle.svg
snow.svg                   sleet.svg
thunderstorms.svg          thunderstorms-rain.svg
```

**Two ways for Claude Code to fetch them:**

```bash
# Option A — vendor the whole static-fill set into the repo (recommended)
mkdir -p web/public/icons/weather
curl -L https://github.com/basmilius/weather-icons/archive/refs/heads/dev.tar.gz \
  | tar -xz --strip-components=4 -C web/public/icons/weather \
    weather-icons-dev/production/fill/svg-static
```

```bash
# Option B — fetch individual files at runtime / build time
# Raw URL pattern:
https://raw.githubusercontent.com/basmilius/weather-icons/dev/production/fill/svg-static/{slug}.svg
# Example:
curl -O https://raw.githubusercontent.com/basmilius/weather-icons/dev/production/fill/svg-static/partly-cloudy-day.svg
```

Vendor them (Option A) — it's ~200 KB total and removes a runtime dependency on GitHub.

### 2.5 · Putting it together — Python helper

Create `src/icons.py`:

```python
from datetime import datetime
from pysolar.solar import get_altitude
from math import sin, radians

# WMO weather code → base icon slug
WMO_TO_SLUG = {
    0: "clear", 1: "partly-cloudy", 2: "partly-cloudy", 3: "overcast",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle",
    56: "sleet", 57: "sleet",
    61: "rain", 63: "rain", 65: "rain",
    66: "sleet", 67: "sleet",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    80: "rain", 81: "rain", 82: "rain",
    85: "snow", 86: "snow",
    95: "thunderstorms", 96: "thunderstorms-rain", 99: "thunderstorms-rain",
}
DAY_NIGHT_SLUGS = {"clear", "partly-cloudy"}   # everything else is day-agnostic

def is_day(lat: float, lon: float, t: datetime) -> bool:
    return get_altitude(lat, lon, t) > 0

def clear_sky_solar(lat: float, lon: float, t: datetime) -> float:
    """Watts per m² assuming a perfectly clear sky. Floor 0 at night."""
    elev = get_altitude(lat, lon, t)
    return 1361 * max(0.0, sin(radians(elev)))

def pick_icon_slug(
    *, weather_code: int, lat: float, lon: float, t: datetime,
    rain_mm_1h: float | None = None,
    solar_wm2: float | None = None,
) -> str:
    base = WMO_TO_SLUG.get(weather_code, "overcast")

    # Local override A: actual rain in the gauge wins over a "clear" code
    if rain_mm_1h is not None and rain_mm_1h >= 0.1 and base in {"clear", "partly-cloudy"}:
        base = "rain"

    # Local override B: clear-sky solar shortfall during the day → demote
    day = is_day(lat, lon, t)
    if day and solar_wm2 is not None and base in {"clear", "partly-cloudy"}:
        expected = clear_sky_solar(lat, lon, t)
        if expected > 50 and (solar_wm2 / expected) < 0.4:
            base = "overcast"

    if base in DAY_NIGHT_SLUGS:
        return f"{base}-{'day' if day else 'night'}"
    return base
```

Use `pick_icon_slug(...)` server-side and put the slug on every forecast row. The frontend just does:

```jsx
<img src={`/icons/weather/${slug}.svg`} alt={cond_label} />
```

### Backend checklist

- [ ] `pip install pysolar` and add it to `requirements.txt`.
- [ ] Vendor `basmilius/weather-icons` static-fill SVGs into `web/public/icons/weather/`.
- [ ] Add `src/icons.py` with `WMO_TO_SLUG`, `is_day`, `clear_sky_solar`, `pick_icon_slug`.
- [ ] In `src/openmeteo.py`, persist `weather_code` (already in the schema).
- [ ] When building the `/forecast` response, call `pick_icon_slug` for each horizon and add `weather_code` + `cond_label` to the payload.
- [ ] Map condition labels in a small dict (`{"clear-day": "Clear", "partly-cloudy-day": "Partly cloudy", ...}`) — keep them friendly, not technical.

---

## 3 · "Feels like" temperature

Use the **Australian BoM apparent-temperature** formula as the unified default — it handles cold, mild, and warm regimes smoothly and only needs `temp_c`, `humidity_pct`, and `wind_speed_ms`. No piecewise switching.

### Formula

```
e  = (humidity_pct / 100) * 6.105 * exp(17.27 * T / (237.7 + T))
AT = T + 0.33 * e − 0.70 * wind_speed_ms − 4.00
```

Where:
- `T` = air temperature in °C
- `humidity_pct` = 0–100
- `wind_speed_ms` = 10-meter wind speed in m/s
- `e` = water-vapor pressure (hPa)
- `AT` = apparent temperature in °C

Reference: [Australian Bureau of Meteorology — apparent temperature](http://www.bom.gov.au/info/thermal_stress/#atapproximation).

### Optional — match Open-Meteo's `apparent_temperature`

Open-Meteo also returns an `apparent_temperature` variable derived from the same family of formulas. To stay consistent with their published forecast, you can:
1. Use **our local formula** (above) on observed data — for the live "feels like" reading.
2. Use **Open-Meteo's `apparent_temperature`** field on forecast data — for forecast cards.

Both will be within ~0.5 °C of each other in practice. Document this choice in the API response so dashboards know which is which.

### Helper

Add to `src/units.py` (create if missing):

```python
from math import exp

def apparent_temperature(temp_c: float, humidity_pct: float, wind_ms: float) -> float:
    """Australian BoM apparent-temperature (°C). Unified across hot/cold regimes."""
    e = (humidity_pct / 100.0) * 6.105 * exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * e - 0.70 * wind_ms - 4.00
```

Call it on every observation insert and persist to a `feels_like_c` column on `observations` (add a migration).

### Backend checklist

- [ ] Add `feels_like_c FLOAT` column to `observations` (Alembic migration).
- [ ] In `src/api.py` `POST /api/ecowitt`, compute `feels_like_c = apparent_temperature(...)` before insert.
- [ ] Surface `feels_like_c` in `GET /api/stations/{id}/current`.
- [ ] In the forecast endpoint, return Open-Meteo's `apparent_temperature` on each horizon as `feels_like_c`.

### Frontend checklist

- [ ] Replace the hard-coded `Feels {current.feels}°` with `current.feels_like_c` from `/current`.
- [ ] If `|feels - temp| < 1°`, hide the "Feels" line — it's not informative.

---

## 4 · Files to read in this design project

| File                          | Purpose                                            |
| ----------------------------- | -------------------------------------------------- |
| `direction-atmospheric.jsx`   | Reference implementation of the new forecast card  |
| `data.js` → `horizons` array  | Exact data shape the new UI consumes               |
| `model-components.jsx`        | The persistent comparison strip (already shipped)  |

When in doubt, run the design HTML side-by-side and diff visually.

---

## 5 · Out of scope for this hand-off

These came up in the design conversation but are NOT part of this change:

- 7-day outlook (the `days` array exists in mock data but is not surfaced)
- Map view / network of community stations (Phase 6)
- Icon animations (we explicitly chose static SVG to keep CPU low on mobile)

Anything not on a checklist above can wait.
