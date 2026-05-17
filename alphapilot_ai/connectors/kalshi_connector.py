"""Kalshi connector (mocked). See base_connector for the contract."""
from __future__ import annotations

import random
from typing import Any

from connectors.base_connector import BaseConnector


class KalshiConnector(BaseConnector):
    platform = "Kalshi"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        prob = round(random.uniform(0.05, 0.95), 3)
        return {
            "platform": self.platform,
            "symbol": symbol,
            "market_question": f"Will '{symbol}' happen?",
            "market_probability": prob,
            "ai_probability": round(min(0.99, max(0.01, prob + random.uniform(-0.1, 0.1))), 3),
            "volume": round(random.uniform(5_000, 250_000), 2),
            "liquidity": round(random.uniform(0.2, 0.9), 2),
            "mock": True,
        }
