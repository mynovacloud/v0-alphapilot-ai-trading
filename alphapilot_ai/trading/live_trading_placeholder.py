"""
Live trading placeholder. LOCKED BY DEFAULT.

DO NOT implement real order routing here without:
- Reviewing the entire risk manager
- Adding manual approval flow in Settings
- Encrypting credentials
- Implementing per-broker confirmation
"""
from __future__ import annotations

from config.settings import settings


def place_live_trade(*args, **kwargs):
    if not settings.live_trading_enabled:
        raise PermissionError(
            "Live trading is locked by default. "
            "Set LIVE_TRADING_ENABLED=true and review risk controls before enabling."
        )
    raise NotImplementedError(
        "Live trading is not implemented. Implement per-connector real order routing first."
    )
