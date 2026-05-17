"""Polymarket connector (mocked).

To replace with real API later:
- POST to Polymarket REST/CLOB endpoints
- Handle wallet signing for orders
- Map markets <-> internal symbols
"""
from __future__ import annotations

import random
from typing import Any

from connectors.base_connector import BaseConnector


class PolymarketConnector(BaseConnector):
    platform = "Polymarket"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        prob = round(random.uniform(0.1, 0.9), 3)
        return {
            "platform": self.platform,
            "symbol": symbol,
            "market_question": f"Will event '{symbol}' resolve YES?",
            "market_probability": prob,
            "ai_probability": round(min(0.99, max(0.01, prob + random.uniform(-0.15, 0.15))), 3),
            "volume": round(random.uniform(10_000, 1_000_000), 2),
            "liquidity": round(random.uniform(0.2, 0.95), 2),
            "mock": True,
        }
