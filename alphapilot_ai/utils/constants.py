"""Application-wide constants."""
from __future__ import annotations

SUPPORTED_PLATFORMS: list[str] = [
    "Polymarket",
    "Kalshi",
    "Webull",
    "Crypto.com",
    "Robinhood",
    "E*TRADE",
    "Coinbase",
    "Binance",
    "Kraken",
    "Fidelity",
    "Interactive Brokers",
    "Custom API",
]

MARKET_TYPES: list[str] = [
    "Crypto",
    "Stocks",
    "Options",
    "Prediction Markets",
    "Forex",
    "Futures",
    "Custom",
]

STRATEGY_TYPES: list[str] = [
    "Momentum",
    "Mean Reversion",
    "Arbitrage Scanner",
    "Market Making",
    "Volatility Breakout",
    "News Reaction",
    "Trend Following",
    "Probability Edge",
    "Custom AI",
]

RISK_LEVELS: list[str] = ["Conservative", "Moderate", "Aggressive", "Degenerate"]

SUGGESTED_ACTIONS: list[str] = [
    "Watch",
    "Paper Trade",
    "Avoid",
    "Strong Opportunity",
    "Needs More Data",
    "High Risk",
    "Low Liquidity",
]

# Default starting paper balance for new wallets.
DEFAULT_PAPER_BALANCE: float = 10_000.0
