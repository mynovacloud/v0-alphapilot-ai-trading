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

# =====================================================================
# Phase A of the signal overhaul: hyperfocus on a handful of majors.
#
# Trading 50+ symbols means the per-symbol learning loop never accumulates
# enough samples to detect a real pattern. With 8 majors, every symbol
# gets thousands of fingerprints per week and per-symbol calibration
# starts being statistically meaningful. The broader curated list is
# preserved below for the day we want to expand or A/B test a wider
# universe — flip _LIQUID_UNIVERSE to _BROAD_UNIVERSE to revert.
# =====================================================================
_FOCUSED_UNIVERSE: tuple[str, ...] = (
    "BTC-USD",   # the macro anchor
    "ETH-USD",   # second anchor; clean structure
    "SOL-USD",   # high-volatility L1 with deep volume
    "LINK-USD",  # DeFi staple, well-respected levels
    "AVAX-USD",  # L1 with clean technicals
    "AAVE-USD",  # DeFi leader, mid-cap behavior
    "ARB-USD",   # L2 with real volume
    "INJ-USD",   # ecosystem token with active liquidity
)

# Broader curated list (preserved for fallback / future expansion).
# Ordered loosely by tier so the most-liquid names are evaluated first
# within a tick's budget. A symbol that has since been delisted simply
# drops out — it won't survive the intersection with Coinbase's live
# `online` product list.
_BROAD_UNIVERSE: tuple[str, ...] = (
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

# Backward-compatible alias for the focused default. The live bot
# resolves the *active* universe via get_active_universe() instead,
# which reads from settings and falls back to _FOCUSED_UNIVERSE.
_LIQUID_UNIVERSE: tuple[str, ...] = _FOCUSED_UNIVERSE


def _is_valid_symbol_shape(sym: str) -> bool:
    """A token is a valid Coinbase USD pair shape iff it looks like
    BASE-USD where BASE is alphanumeric and at most 12 characters."""
    if not sym.endswith("-USD"):
        return False
    base = sym[:-4]
    return bool(base) and len(base) <= 12 and all(c.isalnum() for c in base)


def parse_symbols(raw: str) -> tuple[str, ...]:
    """Parse a free-form symbol list into canonical Coinbase product IDs.

    Accepts comma, space, semicolon, or newline-separated input. Each
    token gets:
      - whitespace trimmed, uppercased
      - leading '$' stripped (some users habitually type "$BTC")
      - '-USD' auto-suffixed if no dash is present ("BTC" -> "BTC-USD")
      - shape-validated (BASE alphanumeric, length <= 12, "-USD" suffix)
      - deduped (first occurrence wins, order preserved)

    Invalid tokens are silently dropped. Empty/all-invalid input returns
    an empty tuple, which callers treat as 'fall back to the default'.
    """
    if not raw:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    text = raw.replace(",", " ").replace(";", " ").replace("\n", " ").replace("\t", " ")
    for token in text.split():
        sym = token.strip().upper()
        if not sym:
            continue
        if sym.startswith("$"):
            sym = sym[1:]
        if "-" not in sym:
            sym = f"{sym}-USD"
        if not _is_valid_symbol_shape(sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return tuple(out)


def get_active_universe() -> tuple[str, ...]:
    """Resolve the symbols the bot should trade RIGHT NOW.

    Reads `bot_focused_symbols` from settings on every call (no caching
    here — the upstream candle-products cache is plenty), so a Settings
    edit propagates to the live bot on the very next tick. Returns the
    hardcoded _FOCUSED_UNIVERSE when the setting is empty or unparseable.
    """
    try:
        from config.bot_config import get as cfg_get   # local import: avoid cycle at module load
        raw = cfg_get("bot_focused_symbols") or ""
        parsed = parse_symbols(raw)
        if parsed:
            return parsed
    except Exception:
        pass
    return _FOCUSED_UNIVERSE


def _rank(rows: list[dict[str, Any]], limit: int, liquid_only: bool) -> list[dict[str, Any]]:
    """Filter + order the candidate universe.

    When `liquid_only` is True, restrict to the operator's *active*
    focused list (settings-driven). Order by the operator's preferred
    order within that list. When False, return every tradable product
    but still sort by the broader curated list's priority so familiar
    majors appear first.
    """
    if liquid_only:
        active = get_active_universe()
        active_set = frozenset(active)
        priority = {s: i for i, s in enumerate(active)}
        rows = [r for r in rows if r["product_id"] in active_set]
    else:
        priority = {s: i for i, s in enumerate(_BROAD_UNIVERSE)}
    rows = sorted(rows, key=lambda r: (priority.get(r["product_id"], 999), r["product_id"]))
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
