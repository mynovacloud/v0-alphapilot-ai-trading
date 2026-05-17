"""Generic equity broker connector template (mocked) used for stock-like platforms."""
from __future__ import annotations

import random
from typing import Any

from connectors.base_connector import BaseConnector


class _EquityConnector(BaseConnector):
    """Shared mock implementation for equity brokers."""

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        price = round(random.uniform(20, 600), 2)
        return {
            "platform": self.platform,
            "symbol": symbol,
            "current_price": price,
            "trend": random.choice(["up", "down", "sideways"]),
            "volume": round(random.uniform(100_000, 50_000_000), 2),
            "volatility": round(random.uniform(0.05, 0.6), 3),
            "mock": True,
        }


class WebullConnector(_EquityConnector):
    platform = "Webull"


class RobinhoodConnector(_EquityConnector):
    platform = "Robinhood"


class ETradeConnector(_EquityConnector):
    platform = "E*TRADE"


class FidelityConnector(_EquityConnector):
    platform = "Fidelity"


class IBKRConnector(_EquityConnector):
    platform = "Interactive Brokers"
