"""Helpers for P&L computations over arbitrary trade DataFrames."""
from __future__ import annotations

import pandas as pd


def realized_pnl(df: pd.DataFrame) -> float:
    if df.empty or "realized_pnl" not in df.columns:
        return 0.0
    return float(df["realized_pnl"].sum())


def unrealized_pnl(df: pd.DataFrame) -> float:
    if df.empty or "unrealized_pnl" not in df.columns:
        return 0.0
    return float(df["unrealized_pnl"].sum())


def win_rate(df: pd.DataFrame) -> float:
    if df.empty or "realized_pnl" not in df.columns:
        return 0.0
    closed = df[df["status"] == "closed"] if "status" in df.columns else df
    if closed.empty:
        return 0.0
    return float((closed["realized_pnl"] > 0).mean())
