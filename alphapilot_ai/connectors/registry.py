"""Connector registry: maps platform name -> connector class."""
from __future__ import annotations

from typing import Type

from connectors.base_connector import BaseConnector
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
    "Polymarket": PolymarketConnector,
    "Kalshi": KalshiConnector,
    "Webull": WebullConnector,
    "Crypto.com": CryptocomConnector,
    "Robinhood": RobinhoodConnector,
    "E*TRADE": ETradeConnector,
    "Coinbase": CoinbaseConnector,
    "Binance": BinanceConnector,
    "Kraken": KrakenConnector,
    "Fidelity": FidelityConnector,
    "Interactive Brokers": IBKRConnector,
    "Custom API": CustomConnector,
}


def get_connector(platform: str, **kwargs) -> BaseConnector:
    cls = CONNECTOR_REGISTRY.get(platform, CustomConnector)
    return cls(**kwargs)
