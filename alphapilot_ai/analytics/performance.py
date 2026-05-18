"""Higher-level performance analytics: profit factor, Sharpe placeholder, streaks."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.portfolio import get_all_trades_df


def performance_metrics() -> dict[str, Any]:
    df = get_all_trades_df()
    if df.empty:
        return {
            "profit_factor": 0.0,
            "sharpe_placeholder": 0.0,
            "max_drawdown": 0.0,
            "avg_rr": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "biggest_win": 0.0,
            "biggest_loss": 0.0,
            "avg_trade_duration_hours": 0.0,
        }
    closed = df[df["status"] == "closed"].copy()
    if closed.empty:
        return {
            "profit_factor": 0.0,
            "sharpe_placeholder": 0.0,
            "max_drawdown": 0.0,
            "avg_rr": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "biggest_win": 0.0,
            "biggest_loss": 0.0,
            "avg_trade_duration_hours": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_closed": 0,
        }

    pnl = closed["realized_pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Sharpe placeholder: mean / std (no risk-free rate)
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0

    sorted_closed = closed.sort_values("closed_at")
    equity = sorted_closed["realized_pnl"].cumsum().values
    peak = np.maximum.accumulate(equity)
    denom = np.maximum(np.abs(peak), 1)
    max_dd = float(((peak - equity) / denom).max()) if len(equity) else 0.0

    avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0
    avg_loss = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
    avg_rr = float(avg_win / avg_loss) if avg_loss > 0 else 0.0

    # Streaks
    streak = max_w = max_l = 0
    last = None
    for v in pnl:
        outcome = "W" if v > 0 else "L"
        if outcome == last:
            streak += 1
        else:
            streak = 1
            last = outcome
        if outcome == "W":
            max_w = max(max_w, streak)
        else:
            max_l = max(max_l, streak)

    durations = []
    for _, row in sorted_closed.iterrows():
        if pd.notna(row["opened_at"]) and pd.notna(row["closed_at"]):
            durations.append(
                (row["closed_at"].replace(tzinfo=None) - row["opened_at"].replace(tzinfo=None)).total_seconds() / 3600
            )
    avg_duration = float(np.mean(durations)) if durations else 0.0

    return {
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        "sharpe_placeholder": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "avg_rr": round(avg_rr, 3),
        "max_consecutive_wins": int(max_w),
        "max_consecutive_losses": int(max_l),
        "biggest_win": round(float(pnl.max()), 2),
        "biggest_loss": round(float(pnl.min()), 2),
        "avg_trade_duration_hours": round(avg_duration, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "total_closed": int(len(closed)),
    }
