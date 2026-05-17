"""
Decision engine.

Rule-based + stochastic decision making. The architecture is intentionally
simple so a real ML model (scikit-learn / PyTorch) can replace `score_signal`
later without changing callers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Decision:
    side: str  # BUY / SELL / HOLD
    confidence: float
    reasoning: str


class DecisionEngine:
    def score_signal(self, snapshot: dict[str, Any], strategy_type: str = "Momentum") -> float:
        """
        Convert a market snapshot into a 0..1 confidence score.
        Replace this with a real ML inference call later.
        """
        liq = float(snapshot.get("liquidity", 0.5))
        vol = float(snapshot.get("volatility", 0.3))
        edge = float(snapshot.get("ai_probability", 0.5)) - float(
            snapshot.get("market_probability", 0.5)
        )

        # Strategy-specific weighting
        if strategy_type == "Momentum":
            base = 0.5 + 0.5 * vol
        elif strategy_type == "Mean Reversion":
            base = 0.7 - 0.4 * vol
        elif strategy_type == "Probability Edge":
            base = 0.5 + 1.5 * abs(edge)
        elif strategy_type == "Volatility Breakout":
            base = 0.4 + 0.6 * vol
        else:
            base = 0.55

        # Liquidity dampening
        score = base * (0.5 + 0.5 * liq)
        # Add small stochastic noise
        score += random.uniform(-0.05, 0.05)
        return max(0.05, min(0.99, score))

    def decide(self, snapshot: dict[str, Any], strategy_type: str = "Momentum") -> Decision:
        confidence = self.score_signal(snapshot, strategy_type)
        edge = float(snapshot.get("ai_probability", 0.5)) - float(
            snapshot.get("market_probability", 0.5)
        )

        if confidence < 0.45:
            return Decision("HOLD", confidence, "Confidence too low to act.")

        if strategy_type == "Probability Edge":
            side = "BUY" if edge > 0 else "SELL"
            return Decision(side, confidence, f"Probability edge {edge:+.1%}")

        if strategy_type == "Mean Reversion":
            side = "SELL" if random.random() > 0.5 else "BUY"
            return Decision(side, confidence, "Mean reversion entry signal.")

        # Momentum / Trend / default: bias to BUY when volatility is high
        side = "BUY" if random.random() > 0.4 else "SELL"
        return Decision(side, confidence, f"{strategy_type} signal.")
