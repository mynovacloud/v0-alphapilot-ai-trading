"""Crypto exchange connectors (mocked)."""
from __future__ import annotations

import random
from typing import Any

from connectors.base_connector import BaseConnector


class _CryptoConnector(BaseConnector):
    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        price = round(random.uniform(0.1, 75_000), 2)
        return {
            "platform": self.platform,
            "symbol": symbol,
            "current_price": price,
            "bid": round(price * 0.999, 2),
            "ask": round(price * 1.001, 2),
            "volume_24h": round(random.uniform(1e6, 5e9), 2),
            "volatility": round(random.uniform(0.1, 1.2), 3),
            "mock": True,
        }


class CryptocomConnector(_CryptoConnector):
    platform = "Crypto.com"


class CoinbaseConnector(_CryptoConnector):
    platform = "Coinbase"


class BinanceConnector(_CryptoConnector):
    platform = "Binance"


class KrakenConnector(_CryptoConnector):
    platform = "Kraken"
