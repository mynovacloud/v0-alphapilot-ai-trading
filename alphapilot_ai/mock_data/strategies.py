"""Default strategy templates."""
from __future__ import annotations

DEFAULT_STRATEGIES = [
    {
        "name": "Momentum BTC",
        "strategy_type": "Momentum",
        "market_type": "Crypto",
        "description": "Ride trends in BTC using volatility-weighted entries.",
        "max_position_size": 2000.0,
        "min_confidence": 0.6,
        "risk_level": "Moderate",
    },
    {
        "name": "Mean Reversion SPY",
        "strategy_type": "Mean Reversion",
        "market_type": "Stocks",
        "description": "Fade extremes on SPY when RSI > 70 or < 30.",
        "max_position_size": 3000.0,
        "min_confidence": 0.65,
        "risk_level": "Conservative",
    },
    {
        "name": "Probability Edge",
        "strategy_type": "Probability Edge",
        "market_type": "Prediction Markets",
        "description": "Take positions where AI estimated probability differs from market price.",
        "max_position_size": 1000.0,
        "min_confidence": 0.55,
        "risk_level": "Moderate",
    },
]
