"""Custom connector slot for any user-defined future platform."""
from __future__ import annotations

import random
from typing import Any

from connectors.base_connector import BaseConnector


class CustomConnector(BaseConnector):
    platform = "Custom API"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "symbol": symbol,
            "current_price": round(random.uniform(1, 1000), 2),
            "mock": True,
        }
