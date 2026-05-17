"""Excluded-window registry — Phase 7.1.

Persists time windows that should be excluded from train/val/holdout splits:
known outages, sensor anomalies, calibration periods. The 7.2 feature pipeline
and 7.3 gate check both consume this so that bad data doesn't get silently
mixed in.

Conventions
-----------
- ``start_time`` is inclusive, ``end_time`` is exclusive (half-open interval).
- All times stored in UTC.
- ``source`` tags how the window was added: ``manual`` for human entry,
  ``heartbeat`` for auto-detected by a future watchdog. The default is
  ``manual``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import ExcludedWindow, engine

logger = logging.getLogger(__name__)


async def add_window(
    station_id: str,
    start: datetime,
    end: datetime,
    reason: str,
    source: str = "manual",
) -> int:
    """Insert one excluded window. Returns the new row's id."""
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = ExcludedWindow(
            station_id=station_id,
            start_time=start,
            end_time=end,
            reason=reason,
            source=source,
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_windows(station_id: Optional[str] = None) -> list[ExcludedWindow]:
    """Return all windows, optionally filtered by station, oldest first."""
    stmt = select(ExcludedWindow)
    if station_id is not None:
        stmt = stmt.where(ExcludedWindow.station_id == station_id)
    stmt = stmt.order_by(ExcludedWindow.start_time)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return list((await session.execute(stmt)).scalars().all())


def is_excluded(t: datetime, windows: Iterable[ExcludedWindow]) -> bool:
    """True if ``t`` falls inside any window. Half-open: [start, end)."""
    return any(w.start_time <= t < w.end_time for w in windows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_dt(s: str) -> datetime:
    """Parse ISO 8601; assume UTC if no tz suffix."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def _cli_list(args) -> None:
    rows = await list_windows(args.station_id)
    if not rows:
        print("(no excluded windows)")
        return
    header = f"{'id':>4}  {'station_id':<34}  {'start (UTC)':<19}  {'end (UTC)':<19}  {'source':<10}  reason"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.id:>4}  {r.station_id:<34}  "
            f"{r.start_time.strftime('%Y-%m-%d %H:%M:%S'):<19}  "
            f"{r.end_time.strftime('%Y-%m-%d %H:%M:%S'):<19}  "
            f"{r.source:<10}  {r.reason}"
        )


async def _cli_add(args) -> None:
    new_id = await add_window(
        station_id=args.station_id,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        reason=args.reason,
        source=args.source,
    )
    print(f"added excluded window id={new_id}")


def main():
    parser = argparse.ArgumentParser(description="Excluded-window registry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list all windows (optionally filtered)")
    p_list.add_argument("--station-id", default=None)
    p_list.set_defaults(func=_cli_list)

    p_add = sub.add_parser("add", help="add a new window")
    p_add.add_argument("--station-id", required=True)
    p_add.add_argument("--start", required=True, help="ISO 8601, UTC if no tz")
    p_add.add_argument("--end",   required=True, help="ISO 8601, UTC if no tz")
    p_add.add_argument("--reason", required=True)
    p_add.add_argument("--source", default="manual")
    p_add.set_defaults(func=_cli_add)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
