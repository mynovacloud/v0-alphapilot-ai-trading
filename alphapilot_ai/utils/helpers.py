"""Misc helper utilities."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_naive() -> datetime:
    """
    Return a *naive* UTC datetime suitable for comparisons against SQLAlchemy
    DateTime columns that were declared without `timezone=True`.

    All `Column(DateTime)` columns in this project store naive UTC values
    (psycopg/SQLAlchemy strip the tzinfo on insert), so any query filter or
    sort key that pairs an aware `utcnow()` with one of those columns blows
    up with "can't compare offset-naive and offset-aware datetimes" on
    PostgreSQL. Use this helper for the FILTER side of those comparisons.
    """
    return datetime.utcnow()


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC). Handles naive datetimes from DB."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def ensure_naive_utc(dt: datetime | None) -> datetime | None:
    """Inverse of `ensure_utc` — strip tzinfo so the value matches naive DB columns."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def time_since_minutes(dt: datetime | None) -> float:
    """Calculate minutes since a given datetime. Handles timezone-naive datetimes."""
    if dt is None:
        return 0.0
    dt_utc = ensure_utc(dt)
    return (utcnow() - dt_utc).total_seconds() / 60


def fmt_money(value: float | int | None, currency: str = "$") -> str:
    if value is None:
        return f"{currency}0.00"
    sign = "-" if value < 0 else ""
    return f"{sign}{currency}{abs(value):,.2f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "0.00%"
    return f"{value * 100:.{digits}f}%" if abs(value) <= 1.5 else f"{value:.{digits}f}%"


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def coerce(value: Any, type_: type, default: Any = None) -> Any:
    try:
        return type_(value)
    except (TypeError, ValueError):
        return default
