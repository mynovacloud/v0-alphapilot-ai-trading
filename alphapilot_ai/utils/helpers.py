"""Misc helper utilities."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
