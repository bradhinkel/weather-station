"""Physical-plausibility bounds shared by the training-data clip
(:mod:`src.ml.dataset`) and the station value-quality scorer
(:mod:`src.pws.registry`).

A reading outside these bounds is a sensor fault, not weather. The world
1-hour rainfall record is ~305 mm; a backyard tipping bucket reporting more
than ``RAIN_MM_1H_MAX`` in one hour is jammed or miscalibrated — the concrete
case that motivated this module was WU station KWARENTO432 (Renton) stuck at
896.37 mm/h with a 2690 mm/hr rate. Bounds are deliberately generous so that
real extremes survive; the goal is to catch garbage, not to trim the tail.

Keeping the numbers in one place means the clip that scrubs a single bad row at
train time and the scorer that retires a persistently-bad station agree on what
"impossible" means.
"""

from __future__ import annotations

# Rain — a trailing-hour accumulation (mm) and an instantaneous rate (mm/hr).
RAIN_MM_1H_MAX: float = 100.0
RAIN_RATE_MM_HR_MAX: float = 500.0

# Air temperature (°C) — comfortably outside any Earth-surface reading.
TEMP_C_MIN: float = -50.0
TEMP_C_MAX: float = 60.0

# Sea-level / station pressure (hPa) — just outside the observed world records
# (870 hPa in a typhoon eye, 1085 hPa in a Siberian high).
PRESSURE_HPA_MIN: float = 870.0
PRESSURE_HPA_MAX: float = 1085.0
