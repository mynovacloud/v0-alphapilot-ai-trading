"""Mock market data generators for crypto, stocks, and prediction markets."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "LINK-USD"]
STOCK_SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "SPY"]
PREDICTION_SYMBOLS = [
    "ELECTION-2028",
    "FED-RATE-CUT-Q1",
    "OSCAR-BEST-PIC",
    "SUPERBOWL-WIN",
    "BTC-100K-EOY",
]


def generate_price_series(
    days: int = 90, start_price: float = 100.0, volatility: float = 0.02
) -> pd.DataFrame:
    """Geometric Brownian motion-style mock price series."""
    rng = np.random.default_rng()
    returns = rng.normal(loc=0.0005, scale=volatility, size=days)
    prices = start_price * np.cumprod(1 + returns)
    dates = [datetime.utcnow() - timedelta(days=days - i) for i in range(days)]
    return pd.DataFrame({"date": dates, "price": prices})


def crypto_snapshot(symbol: str | None = None) -> dict[str, Any]:
    sym = symbol or random.choice(CRYPTO_SYMBOLS)
    price = round(random.uniform(0.5, 75_000), 2)
    return {
        "platform": "Crypto Exchange",
        "symbol": sym,
        "market_type": "Crypto",
        "current_price": price,
        "fair_value": round(price * random.uniform(0.9, 1.1), 2),
        "volume": round(random.uniform(1e6, 5e9), 2),
        "volatility": round(random.uniform(0.1, 1.2), 3),
        "liquidity": round(random.uniform(0.3, 0.99), 2),
    }


def stock_snapshot(symbol: str | None = None) -> dict[str, Any]:
    sym = symbol or random.choice(STOCK_SYMBOLS)
    price = round(random.uniform(20, 600), 2)
    return {
        "platform": "Equity Broker",
        "symbol": sym,
        "market_type": "Stocks",
        "current_price": price,
        "fair_value": round(price * random.uniform(0.85, 1.15), 2),
        "volume": round(random.uniform(1e5, 5e7), 2),
        "volatility": round(random.uniform(0.05, 0.6), 3),
        "liquidity": round(random.uniform(0.4, 0.99), 2),
        "trend": random.choice(["up", "down", "sideways"]),
    }


def prediction_snapshot(symbol: str | None = None) -> dict[str, Any]:
    sym = symbol or random.choice(PREDICTION_SYMBOLS)
    market_prob = round(random.uniform(0.05, 0.95), 3)
    ai_prob = round(min(0.99, max(0.01, market_prob + random.uniform(-0.18, 0.18))), 3)
    return {
        "platform": random.choice(["Polymarket", "Kalshi"]),
        "symbol": sym,
        "market_type": "Prediction Markets",
        "market_question": f"Will event '{sym}' resolve YES?",
        "current_price": market_prob,
        "fair_value": ai_prob,
        "market_probability": market_prob,
        "ai_probability": ai_prob,
        "volume": round(random.uniform(5_000, 1_000_000), 2),
        "liquidity": round(random.uniform(0.2, 0.95), 2),
        "volatility": round(random.uniform(0.05, 0.5), 3),
        "time_remaining_days": random.randint(1, 180),
    }


def random_snapshot() -> dict[str, Any]:
    return random.choice([crypto_snapshot, stock_snapshot, prediction_snapshot])()
