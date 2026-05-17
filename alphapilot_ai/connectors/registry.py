"""Connector registry: maps platform name -> connector class."""
from __future__ import annotations

from typing import Type

from connectors.base_connector import BaseConnector
from connectors.coinbase_perp_connector import CoinbasePerpConnector
from connectors.crypto_connectors import (
    BinanceConnector,
    CoinbaseConnector,
    CryptocomConnector,
    KrakenConnector,
)
from connectors.custom_connector import CustomConnector
from connectors.equity_connectors import (
    ETradeConnector,
    FidelityConnector,
    IBKRConnector,
    RobinhoodConnector,
    WebullConnector,
)
from connectors.kalshi_connector import KalshiConnector
from connectors.polymarket_connector import PolymarketConnector

CONNECTOR_REGISTRY: dict[str, Type[BaseConnector]] = {
    "Coinbase": CoinbaseConnector,
    "Coinbase Perp": CoinbasePerpConnector,
    "Binance": BinanceConnector,
    "Kraken": KrakenConnector,
    "Crypto.com": CryptocomConnector,
    "Polymarket": PolymarketConnector,
    "Kalshi": KalshiConnector,
    "Webull": WebullConnector,
    "Robinhood": RobinhoodConnector,
    "E*TRADE": ETradeConnector,
    "Fidelity": FidelityConnector,
    "Interactive Brokers": IBKRConnector,
    "Custom API": CustomConnector,
}

# Platforms with REAL authenticated read-only API support today.
# When the user adds a wallet on one of these and provides API keys, we
# actually validate the keys and can pull live balances.
REAL_AUTH_PLATFORMS: set[str] = {"Coinbase", "Coinbase Perp", "Binance", "Kraken"}

# Platforms that support perpetual futures (leverage, shorts).
PERP_PLATFORMS: set[str] = {"Coinbase Perp"}


def get_connector(platform: str, **kwargs) -> BaseConnector:
    cls = CONNECTOR_REGISTRY.get(platform, CustomConnector)
    return cls(**kwargs)
