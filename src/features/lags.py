"""Lag lookup helpers — Phase 7.2.

Operates on a tidy hourly observation frame with at least these columns:
  station_id, time_hour (UTC, hour-floored Timestamp), <field columns>

The DB loader in :mod:`src.features.pipeline` produces this shape; tests can
build it by hand. No DB dependency here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def slice_at_lag(
    obs_hourly: pd.DataFrame,
    target_time: datetime,
    lag_hours: int,
    station_ids: Iterable[str],
) -> pd.DataFrame:
    """Return the subset of ``obs_hourly`` matching (target_time - lag_hours)
    on hour granularity, for the given station IDs.

    Missing stations or hours produce zero-row results; the caller decides
    how to propagate that (typically NaN feature output).
    """
    target = pd.Timestamp(target_time)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    lag_hour = (target - pd.Timedelta(hours=lag_hours)).floor("h")

    station_set = set(station_ids)
    if not station_set:
        return obs_hourly.iloc[0:0]

    return obs_hourly[
        (obs_hourly["station_id"].isin(station_set))
        & (obs_hourly["time_hour"] == lag_hour)
    ]
