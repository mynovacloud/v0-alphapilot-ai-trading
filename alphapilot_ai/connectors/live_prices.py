"""
Live market price helper.

Uses CoinGecko's free public API as a universal source of truth for crypto
prices, so the paper-trading engine can simulate fills against REAL market
prices without requiring any exchange API key.

No API key required. Cached briefly in-memory to avoid rate limits.
"""
from __future__ import annotations

import time
import threading
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

# Maps a user-friendly symbol to a CoinGecko coin id.
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

# Global cache with longer TTL for performance
_CACHE: dict[str, tuple[float, float]] = {}  # key -> (price, timestamp)
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 5.0  # seconds - short for trading accuracy
_BATCH_CACHE_TTL = 10.0  # seconds for batch operations

# Reusable HTTP client for connection pooling
_HTTP_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client() -> httpx.Client:
    """Get or create a reusable HTTP client with connection pooling."""
    global _HTTP_CLIENT
    with _CLIENT_LOCK:
        if _HTTP_CLIENT is None:
            _HTTP_CLIENT = httpx.Client(
                timeout=5.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return _HTTP_CLIENT


def _normalize(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "")


def coingecko_id(symbol: str) -> str | None:
    """Convert a user-typed symbol like 'btc-usd' into a CoinGecko id like 'bitcoin'."""
    return SYMBOL_TO_COINGECKO.get(_normalize(symbol))


def get_price(symbol: str, use_cache: bool = True) -> dict[str, Any]:
    """
    Fetch the live USD price for a crypto symbol.
    
    Args:
        symbol: The crypto symbol (e.g., 'BTC-USD', 'ETH')
        use_cache: If True, return cached price if fresh enough
    """
    sym = _normalize(symbol)
    coin_id = coingecko_id(sym)
    now = time.time()
    client = _get_client()

    # Check cache first
    cache_key = coin_id or f"cb:{sym}"
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached and (now - cached[1]) < _CACHE_TTL:
                return {"ok": True, "symbol": sym, "price": cached[0], "source": "cache", "live": True}

    # ---- Path 1: CoinGecko (preferred for mapped majors) ----
    if coin_id:
        url = "https://api.coingecko.com/api/v3/simple/price"
        try:
            r = client.get(url, params={"ids": coin_id, "vs_currencies": "usd"})
            r.raise_for_status()
            data = r.json()
            price = float(data[coin_id]["usd"])
            with _CACHE_LOCK:
                _CACHE[cache_key] = (price, now)
            return {"ok": True, "symbol": sym, "price": price, "source": "coingecko", "live": True}
        except Exception as e:
            logger.warning("CoinGecko fetch failed for %s, falling back to Coinbase: %s", sym, e)

    # ---- Path 2: Coinbase public ticker ----
    product_id = sym if "-" in sym else (
        f"{sym[:-3]}-USD" if sym.endswith("USD") and len(sym) > 3 else f"{sym}-USD"
    )
    url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
    try:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
        price = float(data["price"])
        with _CACHE_LOCK:
            _CACHE[f"cb:{sym}"] = (price, now)
        return {"ok": True, "symbol": sym, "price": price, "source": "coinbase", "live": True}
    except Exception as e:
        # Return stale cache if available
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key) or _CACHE.get(f"cb:{sym}")
            if cached:
                return {"ok": True, "symbol": sym, "price": cached[0], "source": "cache (stale)", "live": False}
        logger.warning("Price fetch failed for %s: %s", sym, e)
        
        # Log to activity log for debug console
        try:
            from database.db import session_scope
            from database.models import ActivityLog
            with session_scope() as s:
                s.add(ActivityLog(
                    category="api",
                    level="warn",
                    message=f"[live_prices] Price fetch failed for {sym}: {str(e)[:200]}",
                ))
        except Exception:
            pass
        
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


def get_prices_batch(symbols: list[str]) -> dict[str, float]:
    """
    Fetch prices for multiple symbols in as few API calls as possible.
    Returns a dict of symbol -> price for successful fetches.
    
    This is MUCH faster than calling get_price() in a loop.
    """
    now = time.time()
    client = _get_client()
    result: dict[str, float] = {}
    
    # Separate into CoinGecko-mapped and Coinbase-only symbols
    coingecko_symbols: dict[str, str] = {}  # normalized_sym -> coin_id
    coinbase_symbols: list[str] = []
    
    for sym in symbols:
        normalized = _normalize(sym)
        
        # Check cache first
        cache_key = coingecko_id(normalized) or f"cb:{normalized}"
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached and (now - cached[1]) < _BATCH_CACHE_TTL:
                result[normalized] = cached[0]
                continue
        
        coin_id = coingecko_id(normalized)
        if coin_id:
            coingecko_symbols[normalized] = coin_id
        else:
            coinbase_symbols.append(normalized)
    
    # Batch fetch from CoinGecko (up to 250 coins per call)
    if coingecko_symbols:
        unique_ids = list(set(coingecko_symbols.values()))
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            r = client.get(url, params={"ids": ",".join(unique_ids), "vs_currencies": "usd"})
            r.raise_for_status()
            data = r.json()
            
            for sym, coin_id in coingecko_symbols.items():
                if coin_id in data and "usd" in data[coin_id]:
                    price = float(data[coin_id]["usd"])
                    result[sym] = price
                    with _CACHE_LOCK:
                        _CACHE[coin_id] = (price, now)
        except Exception as e:
            logger.warning("CoinGecko batch fetch failed: %s", e)
            # Fall back to individual Coinbase fetches
            coinbase_symbols.extend(coingecko_symbols.keys())
    
    # Fetch remaining from Coinbase (no batch API, but use connection pooling)
    for sym in coinbase_symbols:
        if sym in result:
            continue
        product_id = sym if "-" in sym else f"{sym}-USD"
        url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
        try:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
            price = float(data["price"])
            result[sym] = price
            with _CACHE_LOCK:
                _CACHE[f"cb:{sym}"] = (price, now)
        except Exception as e:
            logger.debug("Coinbase fetch failed for %s: %s", sym, e)
            # Try cached value as fallback
            with _CACHE_LOCK:
                cached = _CACHE.get(f"cb:{sym}")
                if cached:
                    result[sym] = cached[0]
    
    return result


def prefetch_universe(symbols: list[str]) -> None:
    """
    Pre-fetch prices for a list of symbols in the background.
    Call this at the start of a tick to warm the cache.
    """
    try:
        get_prices_batch(symbols)
    except Exception as e:
        logger.warning("Prefetch failed: %s", e)
