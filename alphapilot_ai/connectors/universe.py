"""
Universe builder.

Decides which symbols the bot evaluates each tick. We hit Coinbase's
public products endpoint, filter to spot USD pairs that are currently
tradable, and — by default — restrict the result to a curated set of
liquid majors.

Why curated-liquid by default
-----------------------------
Coinbase lists 300+ USD pairs, most of them thin micro-caps. The bot
has no edge there: fees + slippage swamp any small move, and the learned
playbook is full of rules to that effect ("micro-price assets", "fees +
slippage consumed the whole loss"). Worse, the old builder appended
every non-priority symbol ALPHABETICALLY, so a limit of 200 pulled in
`00-USD`, `A8-USD`, `BOBBOB-USD`, `DOOD-USD` — junk the bot then traded
and lost on. Restricting to the curated list keeps the bot on names
where a technical signal can actually resolve into a fill at a fair
price. Pass `liquid_only=False` to get the full tradable list back.

Filters (always applied):
  - quote currency == 'USD'
  - status == 'online'
  - not trading_disabled / cancel_only / limit_only / post_only / auction
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE: dict[str, tuple[list[dict[str, Any]], float]] = {}
_CACHE_TTL = 600.0  # 10 minutes

# Curated liquid universe. This is the bot's tradable set when
# `liquid_only` is True (the default). Ordered loosely by tier so the
# most-liquid names are evaluated first within a tick's budget. A symbol
# that has since been delisted simply drops out — it won't survive the
# intersection with Coinbase's live `online` product list.
_LIQUID_UNIVERSE: tuple[str, ...] = (
    # Top tier — highest liquidity
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
    # Layer 1s
    "AVAX-USD", "DOT-USD", "ATOM-USD", "NEAR-USD", "APT-USD", "SUI-USD",
    "TON-USD", "TRX-USD", "HBAR-USD", "ALGO-USD", "ICP-USD",
    # Layer 2s & scaling
    "ARB-USD", "OP-USD", "IMX-USD", "STRK-USD",
    # DeFi
    "LINK-USD", "UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "LDO-USD",
    "SNX-USD", "COMP-USD", "SUSHI-USD", "1INCH-USD", "BAL-USD", "YFI-USD",
    # Memecoins (high volatility)
    "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD",
    # Infrastructure
    "FIL-USD", "AR-USD", "RENDER-USD", "GRT-USD",
    # Gaming & metaverse
    "AXS-USD", "SAND-USD", "MANA-USD", "GALA-USD",
    # Majors / store-of-value
    "BCH-USD", "LTC-USD", "ETC-USD", "XLM-USD", "VET-USD",
    # AI
    "FET-USD", "TAO-USD",
    # Misc high-volume
    "INJ-USD", "SEI-USD", "TIA-USD", "PYTH-USD", "JTO-USD", "JUP-USD",
)
_PRIORITY_INDEX: dict[str, int] = {sym: i for i, sym in enumerate(_LIQUID_UNIVERSE)}


def _rank(rows: list[dict[str, Any]], limit: int, liquid_only: bool) -> list[dict[str, Any]]:
    """Filter to the liquid set (if requested), order by priority, cap at limit."""
    if liquid_only:
        rows = [r for r in rows if r["product_id"] in _PRIORITY_INDEX]
    rows = sorted(rows, key=lambda r: (_PRIORITY_INDEX.get(r["product_id"], 999), r["product_id"]))
    return rows[:limit]


def coinbase_usd_universe(limit: int = 50, *, liquid_only: bool = True) -> list[dict[str, Any]]:
    """
    Pull SPOT USD-quoted pairs from Coinbase that are currently tradable.

    With `liquid_only=True` (default) the result is restricted to the
    curated `_LIQUID_UNIVERSE`; `limit` then acts as a cap, not a target.
    Each entry:
        {
          "product_id": "BTC-USD",
          "base": "BTC",
          "quote": "USD",
          "price": 0.0,        # live price is fetched per-symbol each tick
          "volume_24h": 0.0,
        }
    """
    cached = _CACHE.get("coinbase_usd")
    now = time.time()
    if cached and (now - cached[1]) < _CACHE_TTL:
        return _rank(cached[0], limit, liquid_only)

    url = "https://api.exchange.coinbase.com/products"
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(url)
            r.raise_for_status()
            products = r.json()
    except Exception as e:
        logger.warning("Failed to fetch Coinbase universe: %s", e)
        return _rank(cached[0], limit, liquid_only) if cached else []

    # The /products endpoint uses base_currency / quote_currency at top level.
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
                    "price": 0.0,
                    "volume_24h": 0.0,
                }
            )
        except Exception:
            continue

    _CACHE["coinbase_usd"] = (out, now)
    ranked = _rank(out, limit, liquid_only)
    logger.info(
        "Universe built: %d tradable USD pairs, %d after liquid filter (liquid_only=%s)",
        len(out), len(ranked), liquid_only,
    )
    return ranked
