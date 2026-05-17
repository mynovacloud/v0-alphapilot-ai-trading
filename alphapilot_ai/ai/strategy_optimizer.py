"""
Strategy optimizer.

Suggests adjustments to strategy parameters based on observed performance.
This is intentionally rule-based today; a future ML model can replace
`recommend_adjustments`.
"""
from __future__ import annotations

from typing import Any


class StrategyOptimizer:
    def recommend_adjustments(self, stats: dict[str, Any]) -> dict[str, Any]:
        win_rate = float(stats.get("win_rate", 0.5))
        drawdown = float(stats.get("drawdown", 0.0))
        avg_win = float(stats.get("avg_win", 0.0))
        avg_loss = float(stats.get("avg_loss", 0.0))

        suggestions: dict[str, Any] = {}

        if drawdown > 0.3:
            suggestions["max_position_size"] = "decrease 25%"
        if win_rate < 0.45:
            suggestions["min_confidence"] = "increase to 0.7"
        if abs(avg_loss) > avg_win * 1.5 and avg_win > 0:
            suggestions["stop_loss_pct"] = "tighten to 3%"
        if win_rate > 0.6 and drawdown < 0.15:
            suggestions["max_position_size"] = "consider increasing 10%"

        if not suggestions:
            suggestions["status"] = "no adjustments needed"
        return suggestions
