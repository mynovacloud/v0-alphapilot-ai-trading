"""Mock backtester for strategies."""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from database.db import session_scope
from database.models import BacktestResult, Strategy


def _recommendation(win_rate: float, drawdown: float, total_pnl: float) -> str:
    if total_pnl < 0 or drawdown > 0.4:
        return "avoid"
    if win_rate > 0.6 and drawdown < 0.2:
        return "safe"
    if win_rate > 0.5:
        return "moderate"
    return "risky"


def run_backtest(strategy_id: int, n_trades: int = 200) -> dict[str, Any]:
    """Generate mock backtest results and persist them."""
    rng = np.random.default_rng()

    with session_scope() as s:
        strat = s.get(Strategy, strategy_id)
        if not strat:
            return {"ok": False, "reason": "Strategy not found"}

        # Bias outcomes by strategy risk profile
        win_bias = {
            "Conservative": 0.58,
            "Moderate": 0.52,
            "Aggressive": 0.46,
            "Degenerate": 0.42,
        }.get(strat.risk_level, 0.5)

        outcomes = rng.binomial(1, win_bias, size=n_trades)
        wins_pnl = rng.normal(loc=80, scale=40, size=int(outcomes.sum())).clip(min=1)
        loss_pnl = -rng.normal(loc=70, scale=35, size=int((1 - outcomes).sum())).clip(min=1)
        all_pnl = np.concatenate([wins_pnl, loss_pnl])
        rng.shuffle(all_pnl)

        equity = np.cumsum(all_pnl)
        peak = np.maximum.accumulate(equity)
        drawdown = float(((peak - equity) / np.maximum(peak, 1)).max())

        win_rate = float(outcomes.mean())
        total_pnl = float(all_pnl.sum())
        avg_win = float(wins_pnl.mean()) if len(wins_pnl) else 0.0
        avg_loss = float(loss_pnl.mean()) if len(loss_pnl) else 0.0
        risk_score = round(min(1.0, drawdown + (1 - win_rate) * 0.5), 3)
        recommendation = _recommendation(win_rate, drawdown, total_pnl)

        result = BacktestResult(
            strategy_id=strategy_id,
            total_trades=n_trades,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            drawdown=round(drawdown, 4),
            risk_score=risk_score,
            recommendation=recommendation,
        )
        s.add(result)
        s.flush()

        return {
            "ok": True,
            "strategy_id": strategy_id,
            "total_trades": n_trades,
            "win_rate": round(win_rate, 3),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "drawdown": round(drawdown, 4),
            "risk_score": risk_score,
            "recommendation": recommendation,
            "equity_curve": equity.tolist(),
        }
