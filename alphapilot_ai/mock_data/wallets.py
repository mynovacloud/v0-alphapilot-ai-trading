"""Mock wallet templates (used by Add Wallet flow when seeding extra demo data)."""
from __future__ import annotations

WALLET_TEMPLATES = [
    {"platform": "Polymarket", "name": "Polymarket Predictions", "balance": 5_000.0, "risk": "Moderate"},
    {"platform": "Kalshi", "name": "Kalshi Events", "balance": 3_000.0, "risk": "Conservative"},
    {"platform": "Coinbase", "name": "Coinbase Spot", "balance": 10_000.0, "risk": "Moderate"},
    {"platform": "Binance", "name": "Binance Futures", "balance": 7_500.0, "risk": "Aggressive"},
    {"platform": "Webull", "name": "Webull Equities", "balance": 12_000.0, "risk": "Conservative"},
]
