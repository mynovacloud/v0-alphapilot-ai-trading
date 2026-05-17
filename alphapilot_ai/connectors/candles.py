"""
Historical candles helper.

Pulls OHLCV bars from Coinbase's public exchange API. No API key needed
for public market data. Used by the strategy engine to compute real
indicators (EMA crossovers, Z-scores, ATR, volatility) instead of the
synthetic placeholders the bot used in v1.

Endpoint:
    GET https://api.exchange.coinbase.com/products/{product_id}/candles
        ?granularity={seconds}&start={iso}&end={iso}

Granularity must be one of: 60, 300, 900, 3600, 21600, 86400 (Coinbase rule).
Response is a list of [time, low, high, open, close, volume] in DESCENDING
time order. We normalize it to ascending and to a list of dicts.

Cached briefly per (product, granularity) to keep the bot tick fast and
stay well below Coinbase's public-data rate limits.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}

_CACHE: dict[tuple[str, int], tuple[list[dict[str, Any]], float]] = {}
_CACHE_TTL = 30.0  # seconds — short, but enough to amortize a tick


def get_candles(
    product_id: str,
    granularity: int = 300,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Return up to `limit` recent candles, oldest -> newest.

    Each candle:
        {"time": <unix int>, "open": float, "high": float, "low": float,
         "close": float, "volume": float}

    Returns [] on any failure (the strategy engine treats empty data as "no signal").
    """
    if granularity not in ALLOWED_GRANULARITIES:
        # Snap to closest allowed bucket
        granularity = min(ALLOWED_GRANULARITIES, key=lambda g: abs(g - granularity))

    key = (product_id.upper(), granularity)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0][-limit:]

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(url, params={"granularity": granularity})
            r.raise_for_status()
            raw = r.json()
    except Exception as e:
        logger.warning("Candles fetch failed for %s g=%s: %s", product_id, granularity, e)
        return cached[0][-limit:] if cached else []

    # Coinbase returns newest-first. Sort ascending.
    raw_sorted = sorted(raw, key=lambda row: row[0])
    out: list[dict[str, Any]] = []
    for row in raw_sorted:
        if not isinstance(row, list) or len(row) < 6:
            continue
        out.append(
            {
                "time": int(row[0]),
                "low": float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    _CACHE[key] = (out, now)
    return out[-limit:]
