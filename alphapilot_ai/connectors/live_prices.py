"""
Live market price helper.

Uses CoinGecko's free public API as a universal source of truth for crypto
prices, so the paper-trading engine can simulate fills against REAL market
prices without requiring any exchange API key.

No API key required. Cached briefly in-memory to avoid rate limits.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

# Maps a user-friendly symbol to a CoinGecko coin id.
# Add more symbols here as needed.
SYMBOL_TO_COINGECKO: dict[str, str] = {
    "BTC": "bitcoin", "BTC-USD": "bitcoin", "BTC/USD": "bitcoin", "BTCUSD": "bitcoin", "BTCUSDT": "bitcoin",
    "ETH": "ethereum", "ETH-USD": "ethereum", "ETH/USD": "ethereum", "ETHUSD": "ethereum", "ETHUSDT": "ethereum",
    "SOL": "solana", "SOL-USD": "solana", "SOL/USD": "solana", "SOLUSD": "solana", "SOLUSDT": "solana",
    "DOGE": "dogecoin", "DOGE-USD": "dogecoin", "DOGEUSDT": "dogecoin",
    "XRP": "ripple", "XRP-USD": "ripple", "XRPUSDT": "ripple",
    "ADA": "cardano", "ADA-USD": "cardano",
    "AVAX": "avalanche-2", "AVAX-USD": "avalanche-2",
    "LINK": "chainlink", "LINK-USD": "chainlink",
    "DOT": "polkadot", "DOT-USD": "polkadot",
    "MATIC": "matic-network", "MATIC-USD": "matic-network",
    "LTC": "litecoin", "LTC-USD": "litecoin",
    "SHIB": "shiba-inu",
    "BNB": "binancecoin", "BNB-USD": "binancecoin", "BNBUSDT": "binancecoin",
    "ATOM": "cosmos",
    "UNI": "uniswap",
    "TRX": "tron",
    "ARB": "arbitrum",
    "OP": "optimism",
    "APT": "aptos",
    "SUI": "sui",
    "PEPE": "pepe",
}

_CACHE: dict[str, tuple[float, float]] = {}  # coin_id -> (price, ts)
_CACHE_TTL = 15.0  # seconds


def _normalize(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "")


def coingecko_id(symbol: str) -> str | None:
    """Convert a user-typed symbol like 'btc-usd' into a CoinGecko id like 'bitcoin'."""
    return SYMBOL_TO_COINGECKO.get(_normalize(symbol))


def get_price(symbol: str) -> dict[str, Any]:
    """
    Fetch the live USD price for a crypto symbol.

    Strategy:
      1. If symbol is mapped in SYMBOL_TO_COINGECKO, use CoinGecko (primary).
      2. Otherwise, try Coinbase's public ticker endpoint at
         https://api.exchange.coinbase.com/products/{SYMBOL}/ticker
         which works for ANY tradable Coinbase product (BTC-USD, BONK-USD, etc.)
         and requires no API key.

    Returns:
        {"ok": True, "symbol": "BTC-USD", "price": 67421.10, "source": "...", "live": True}
    or, on failure, an error payload with ok=False.
    """
    sym = _normalize(symbol)
    coin_id = coingecko_id(sym)

    now = time.time()

    # ---- Path 1: CoinGecko (preferred for mapped majors) ----
    if coin_id:
        cached = _CACHE.get(coin_id)
        if cached and (now - cached[1]) < _CACHE_TTL:
            return {"ok": True, "symbol": sym, "price": cached[0], "source": "coingecko (cached)", "live": True}

        url = "https://api.coingecko.com/api/v3/simple/price"
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(url, params={"ids": coin_id, "vs_currencies": "usd"})
                r.raise_for_status()
                data = r.json()
            price = float(data[coin_id]["usd"])
            _CACHE[coin_id] = (price, now)
            return {"ok": True, "symbol": sym, "price": price, "source": "coingecko", "live": True}
        except Exception as e:
            logger.warning("CoinGecko fetch failed for %s, falling back to Coinbase: %s", sym, e)
            # fall through to Coinbase

    # ---- Path 2: Coinbase public ticker (works for any Coinbase USD product) ----
    cb_key = f"cb:{sym}"
    cached = _CACHE.get(cb_key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return {"ok": True, "symbol": sym, "price": cached[0], "source": "coinbase (cached)", "live": True}

    # Coinbase product IDs use BASE-USD form. If user passed BTCUSD, normalize.
    product_id = sym if "-" in sym else (
        f"{sym[:-3]}-USD" if sym.endswith("USD") and len(sym) > 3 else sym
    )
    url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(url)
            r.raise_for_status()
            data = r.json()
        price = float(data["price"])
        _CACHE[cb_key] = (price, now)
        return {"ok": True, "symbol": sym, "price": price, "source": "coinbase", "live": True}
    except Exception as e:
        logger.warning("Coinbase ticker fetch failed for %s: %s", sym, e)
        return {"ok": False, "symbol": sym, "error": f"Live price fetch failed: {e}"}


def known_symbols() -> list[str]:
    """Return the list of symbols we currently know how to price."""
    seen = set()
    out: list[str] = []
    for k in SYMBOL_TO_COINGECKO.keys():
        if "-" in k and k not in seen:
            out.append(k)
            seen.add(k)
    return sorted(out)
