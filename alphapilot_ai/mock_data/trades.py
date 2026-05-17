"""Mock trade generator helpers (used by tests / demos)."""
from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from utils.helpers import utcnow


def random_trade(wallet_id: int, market_type: str = "Crypto") -> dict[str, Any]:
    side = random.choice(["BUY", "SELL"])
    entry = round(random.uniform(20, 400), 2)
    qty = round(random.uniform(0.5, 5), 2)
    is_closed = random.random() > 0.3
    pnl = round(random.uniform(-200, 350), 2) if is_closed else 0.0
    opened = utcnow() - timedelta(days=random.randint(0, 20))
    return {
        "wallet_id": wallet_id,
        "symbol": random.choice(["BTC-USD", "ETH-USD", "AAPL", "TSLA", "SPY"]),
        "market_type": market_type,
        "side": side,
        "qty": qty,
        "entry_price": entry,
        "exit_price": round(entry + pnl / max(qty, 0.01), 2) if is_closed else None,
        "fees": round(qty * entry * 0.001, 2),
        "slippage": round(random.uniform(0, 1.2), 2),
        "realized_pnl": pnl,
        "unrealized_pnl": 0.0 if is_closed else round(random.uniform(-50, 80), 2),
        "confidence": round(random.uniform(0.4, 0.9), 2),
        "status": "closed" if is_closed else "open",
        "opened_at": opened,
        "closed_at": opened + timedelta(hours=random.randint(1, 60)) if is_closed else None,
    }
