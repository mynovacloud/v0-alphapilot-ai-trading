"""
Base connector defining the contract every platform connector must implement.

All real-API integrations should:
1. Subclass `BaseConnector`
2. Override `connect`, `validate_credentials`, `fetch_*`, `place_paper_trade`
3. **NEVER** override `place_live_trade` without explicit safety review --
   it is locked by default at the framework level.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseConnector(ABC):
    platform: str = "base"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        account_id: str = "",
        sandbox: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.account_id = account_id
        self.sandbox = sandbox
        self._connected = False

    # --- Lifecycle -------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """Mock connection. Replace with real API auth call."""
        logger.info("[%s] Mock connect (sandbox=%s)", self.platform, self.sandbox)
        self._connected = True
        return {"ok": True, "platform": self.platform, "sandbox": self.sandbox, "mock": True}

    def disconnect(self) -> None:
        self._connected = False

    def get_connection_status(self) -> str:
        return "connected (mock)" if self._connected else "disconnected"

    def validate_credentials(self) -> dict[str, Any]:
        """Mocked validation. Real implementations should hit a real auth endpoint."""
        ok = bool(self.api_key) or self.sandbox  # in sandbox we accept blank
        return {"valid": ok, "mock": True, "note": "Replace with real validation later."}

    # --- Data fetchers (mocked) -----------------------------------------

    @abstractmethod
    def fetch_market_data(self, symbol: str) -> dict[str, Any]: ...

    def fetch_balance(self) -> dict[str, Any]:
        return {"cash": round(random.uniform(1_000, 25_000), 2), "currency": "USD", "mock": True}

    def fetch_positions(self) -> list[dict[str, Any]]:
        return []

    def fetch_trade_history(self) -> list[dict[str, Any]]:
        return []

    def sync_account(self) -> dict[str, Any]:
        return {
            "balance": self.fetch_balance(),
            "positions": self.fetch_positions(),
            "history_count": len(self.fetch_trade_history()),
            "mock": True,
        }

    # --- Trading --------------------------------------------------------

    def place_paper_trade(self, **kwargs: Any) -> dict[str, Any]:
        """
        Default implementation: tells the caller to use the central paper engine
        in `trading/paper_trading_engine.py`. Connectors generally don't simulate
        fills themselves; the engine does, so accounting stays consistent.
        """
        return {"ok": True, "delegated": "paper_trading_engine", **kwargs}

    def place_live_trade(self, **kwargs: Any) -> dict[str, Any]:
        """
        LOCKED. Live trading is disabled by default for safety.

        Enabling requires:
        - LIVE_TRADING_ENABLED=true in .env
        - User confirmation in Settings
        - A real implementation in the subclass
        """
        if not settings.live_trading_enabled:
            raise PermissionError(
                "Live trading is locked by default. "
                "Set LIVE_TRADING_ENABLED=true and review risk controls before enabling."
            )
        raise NotImplementedError("Live trading not implemented for this connector.")
