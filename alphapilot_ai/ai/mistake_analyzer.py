"""
Mistake analyzer.

Examines closed trades and produces 'lessons' that get stored in
AILearningMemory.
"""
from __future__ import annotations

from typing import Any


class MistakeAnalyzer:
    def analyze_trade(self, trade: dict[str, Any]) -> list[str]:
        lessons: list[str] = []
        pnl = trade.get("realized_pnl", 0.0)
        confidence = trade.get("confidence", 0.5)
        slippage = trade.get("slippage", 0.0)
        liquidity = trade.get("liquidity", 0.5)
        duration_hours = trade.get("duration_hours", 0)

        if pnl < 0 and confidence > 0.75:
            lessons.append(
                "High-confidence trade lost money — recalibrate confidence weighting downward."
            )
        if pnl < 0 and liquidity < 0.4:
            lessons.append("Avoid low-liquidity markets after slippage-driven losses.")
        if slippage > 5:
            lessons.append("Excessive slippage detected — reduce position size in this market.")
        if pnl > 0 and duration_hours < 1:
            lessons.append(
                "Profitable scalp — consider tightening exits to lock gains faster."
            )
        if pnl < 0 and duration_hours > 48:
            lessons.append("Held losing trade too long — enforce stop-loss discipline.")
        return lessons

    def analyze_session(self, trades: list[dict[str, Any]]) -> list[str]:
        lessons: list[str] = []
        if not trades:
            return lessons
        wins = [t for t in trades if t.get("realized_pnl", 0) > 0]
        losses = [t for t in trades if t.get("realized_pnl", 0) < 0]
        win_rate = len(wins) / len(trades) if trades else 0

        if win_rate < 0.4:
            lessons.append(
                f"Session win rate {win_rate:.0%} — strategy may be miscalibrated; reduce size."
            )
        if len(losses) >= 3:
            consec = 0
            max_consec = 0
            for t in trades:
                if t.get("realized_pnl", 0) < 0:
                    consec += 1
                    max_consec = max(max_consec, consec)
                else:
                    consec = 0
            if max_consec >= 3:
                lessons.append(
                    f"Detected {max_consec} consecutive losses — apply cooldown rule."
                )
        if len(trades) > 30:
            lessons.append("High trade frequency — watch for overtrading.")

        for t in trades:
            lessons.extend(self.analyze_trade(t))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for l in lessons:
            if l not in seen:
                unique.append(l)
                seen.add(l)
        return unique
