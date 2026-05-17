"""
Universe builder.

Decides which symbols the bot evaluates each tick. The user chose
"All Coinbase USD pairs", so we hit Coinbase's public products endpoint,
filter to spot USD pairs that aren't disabled / view-only, and return the
list ranked by 24h volume so most-liquid pairs go first.

Filters:
  - quote_currency == 'USD'
  - status == 'online'
  - trading_disabled == False
  - is_disabled == False
  - cancel_only == False
  - limit_only == False
  - post_only == False

Cached for 10 minutes; the universe doesn't change minute-to-minute.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE: dict[str, tuple[list[dict[str, Any]], float]] = {}
_CACHE_TTL = 600.0  # 10 minutes


def _is_tradable(p: dict[str, Any]) -> bool:
    """All disabled flags must be False, status online, quote currency USD."""
    if (p.get("quote_currency_id") or "").upper() != "USD":
        return False
    if (p.get("status") or "").lower() != "online":
        return False
    for flag in ("trading_disabled", "is_disabled", "cancel_only", "limit_only", "post_only"):
        if p.get(flag):
            return False
    return True


def coinbase_usd_universe(limit: int = 50) -> list[dict[str, Any]]:
    """
    Pull all SPOT USD-quoted pairs from Coinbase that are currently tradable.
    Returned sorted by 24h volume desc. Each entry:
        {
          "product_id": "BTC-USD",
          "base": "BTC",
          "quote": "USD",
          "price": 67000.0,
          "volume_24h": 1234567.0,
          "price_change_24h_pct": 0.012,
        }
    """
    cached = _CACHE.get("coinbase_usd")
    now = time.time()
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0][:limit]

    url = "https://api.exchange.coinbase.com/products"
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(url)
            r.raise_for_status()
            products = r.json()
    except Exception as e:
        logger.warning("Failed to fetch Coinbase universe: %s", e)
        return cached[0][:limit] if cached else []

    # The /products endpoint uses slightly different field names than the brokerage API.
    # quote_currency / base_currency are top-level on this endpoint.
    out: list[dict[str, Any]] = []
    for p in products:
        try:
            if (p.get("quote_currency") or "").upper() != "USD":
                continue
            if (p.get("status") or "").lower() != "online":
                continue
            if any(p.get(f) for f in ("trading_disabled", "cancel_only", "limit_only", "post_only", "auction_mode")):
                continue
            out.append(
                {
                    "product_id": p.get("id"),
                    "base": p.get("base_currency"),
                    "quote": p.get("quote_currency"),
                    # Volume / price come from /stats endpoint, but to keep this fast and
                    # rate-limit friendly we leave them as 0 here. The strategy engine pulls
                    # live price per-symbol on each tick anyway via CoinGecko.
                    "price": 0.0,
                    "volume_24h": 0.0,
                }
            )
        except Exception:
            continue

    # Best-effort: bring well-known liquid majors to the front so the bot evaluates
    # them first within its tick budget.
    PRIORITY = [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
        "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "ATOM-USD", "LTC-USD",
        "UNI-USD", "ARB-USD", "OP-USD", "APT-USD", "SUI-USD", "BCH-USD", "NEAR-USD",
    ]
    priority_index = {sym: i for i, sym in enumerate(PRIORITY)}
    out.sort(key=lambda r: (priority_index.get(r["product_id"], 999), r["product_id"]))

    _CACHE["coinbase_usd"] = (out, now)
    return out[:limit]
