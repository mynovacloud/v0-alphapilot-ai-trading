"""
Fear & Greed Index Integration
==============================
Fetches the Crypto Fear & Greed Index from alternative.me.

The index ranges from 0-100:
- 0-24: Extreme Fear (buying opportunity)
- 25-44: Fear
- 45-55: Neutral
- 56-74: Greed
- 75-100: Extreme Greed (selling opportunity)

The index is calculated from:
- Volatility (25%)
- Market momentum/volume (25%)
- Social media (15%)
- Surveys (15%)
- Bitcoin dominance (10%)
- Google Trends (10%)
"""

from __future__ import annotations
import time
import httpx
from dataclasses import dataclass
from typing import Optional, Literal


# Alternative.me Fear & Greed API (free, no key required)
FEAR_GREED_API = "https://api.alternative.me/fng/"

# Cache
_cache: dict = {}
_cache_ttl = 1800  # 30 minutes (index updates daily)


@dataclass
class FearGreedData:
    """Fear & Greed Index data."""
    value: int  # 0-100
    value_classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: int
    
    # Derived signals
    sentiment: Literal["EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"]
    signal: Literal["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
    confidence_adjustment: float  # -0.15 to +0.15
    
    # Historical context (if available)
    yesterday_value: Optional[int]
    last_week_value: Optional[int]
    last_month_value: Optional[int]
    
    # Trend
    trend: Literal["IMPROVING", "STABLE", "WORSENING"]
    
    # Summary for Claude
    summary: str


def _get_cached(key: str) -> Optional[FearGreedData]:
    """Get cached data if still valid."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return data
    return None


def _set_cache(key: str, data: FearGreedData):
    """Cache data."""
    _cache[key] = (data, time.time())


def get_fear_greed() -> Optional[FearGreedData]:
    """
    Get the current Crypto Fear & Greed Index.
    
    Returns FearGreedData with current value, classification, and trading signals.
    """
    cache_key = "fear_greed"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        # Fetch current and historical data
        params = {"limit": 31}  # Get last 31 days
        
        with httpx.Client(timeout=10) as client:
            resp = client.get(FEAR_GREED_API, params=params)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            if not data.get("data"):
                return None
            
            entries = data["data"]
            
            if not entries:
                return None
            
            # Current value
            current = entries[0]
            value = int(current.get("value", 50))
            classification = current.get("value_classification", "Neutral")
            timestamp = int(current.get("timestamp", time.time()))
            
            # Historical values
            yesterday_value = int(entries[1]["value"]) if len(entries) > 1 else None
            last_week_value = int(entries[7]["value"]) if len(entries) > 7 else None
            last_month_value = int(entries[30]["value"]) if len(entries) > 30 else None
            
            # Determine sentiment
            if value <= 24:
                sentiment = "EXTREME_FEAR"
                signal = "STRONG_BUY"  # Contrarian: extreme fear = buy
                confidence_adj = 0.15
            elif value <= 44:
                sentiment = "FEAR"
                signal = "BUY"
                confidence_adj = 0.08
            elif value <= 55:
                sentiment = "NEUTRAL"
                signal = "NEUTRAL"
                confidence_adj = 0.0
            elif value <= 74:
                sentiment = "GREED"
                signal = "SELL"
                confidence_adj = -0.05
            else:
                sentiment = "EXTREME_GREED"
                signal = "STRONG_SELL"  # Contrarian: extreme greed = sell
                confidence_adj = -0.12
            
            # Determine trend
            if yesterday_value is not None:
                if value > yesterday_value + 5:
                    trend = "IMPROVING"  # Getting less fearful / more greedy
                elif value < yesterday_value - 5:
                    trend = "WORSENING"  # Getting more fearful
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"
            
            # Generate summary
            summary_parts = [
                f"Fear & Greed Index: {value}/100 ({classification})",
            ]
            
            if yesterday_value:
                change = value - yesterday_value
                summary_parts.append(f"24h change: {change:+d}")
            
            if last_week_value:
                change = value - last_week_value
                summary_parts.append(f"7d change: {change:+d}")
            
            summary_parts.append(f"Signal: {signal}")
            
            summary = " | ".join(summary_parts)
            
            result = FearGreedData(
                value=value,
                value_classification=classification,
                timestamp=timestamp,
                sentiment=sentiment,
                signal=signal,
                confidence_adjustment=confidence_adj,
                yesterday_value=yesterday_value,
                last_week_value=last_week_value,
                last_month_value=last_month_value,
                trend=trend,
                summary=summary,
            )
            
            _set_cache(cache_key, result)
            return result
            
    except Exception as e:
        return None


def get_fear_greed_signal() -> dict:
    """
    Quick function to get Fear & Greed signal for trading decisions.
    
    Returns dict suitable for Claude decision context.
    """
    fg = get_fear_greed()
    
    if not fg:
        return {
            "available": False,
            "value": None,
            "sentiment": "UNKNOWN",
            "signal": "NEUTRAL",
            "confidence_adjustment": 0.0,
            "summary": "Fear & Greed Index unavailable",
        }
    
    return {
        "available": True,
        "value": fg.value,
        "classification": fg.value_classification,
        "sentiment": fg.sentiment,
        "signal": fg.signal,
        "confidence_adjustment": fg.confidence_adjustment,
        "yesterday": fg.yesterday_value,
        "last_week": fg.last_week_value,
        "last_month": fg.last_month_value,
        "trend": fg.trend,
        "summary": fg.summary,
        "interpretation": _get_interpretation(fg),
    }


def _get_interpretation(fg: FearGreedData) -> str:
    """
    Get a detailed interpretation of the Fear & Greed Index.
    """
    if fg.sentiment == "EXTREME_FEAR":
        return (
            "Market is in EXTREME FEAR. Historically, this is a strong buying opportunity. "
            "Most retail traders are panic selling. Consider accumulating quality assets. "
            "However, extreme fear can persist - use proper risk management."
        )
    elif fg.sentiment == "FEAR":
        return (
            "Market sentiment is fearful. This often precedes recoveries. "
            "Good time to look for entry opportunities on pullbacks. "
            "Don't go all-in, but accumulation makes sense."
        )
    elif fg.sentiment == "NEUTRAL":
        return (
            "Market sentiment is balanced. No strong directional bias from sentiment. "
            "Focus on technical analysis and fundamentals for trade decisions. "
            "Normal position sizing appropriate."
        )
    elif fg.sentiment == "GREED":
        return (
            "Market is getting greedy. Caution advised for new long positions. "
            "Consider taking partial profits on winning positions. "
            "Tighten stop losses."
        )
    else:  # EXTREME_GREED
        return (
            "Market is in EXTREME GREED. This often precedes corrections. "
            "Strongly consider taking profits and reducing exposure. "
            "New longs are high risk. Wait for pullback or reversal confirmation."
        )


def should_trade_based_on_fear_greed() -> tuple[bool, str]:
    """
    Quick check if current Fear & Greed suggests favorable conditions for trading.
    
    Returns:
        (should_trade, reason)
    """
    fg = get_fear_greed()
    
    if not fg:
        return True, "Fear & Greed data unavailable - proceeding with technicals only"
    
    if fg.sentiment == "EXTREME_GREED":
        return False, f"Extreme Greed ({fg.value}/100) - high risk for new longs"
    
    if fg.sentiment == "EXTREME_FEAR":
        return True, f"Extreme Fear ({fg.value}/100) - contrarian buying opportunity"
    
    return True, f"Fear & Greed at {fg.value}/100 ({fg.value_classification}) - normal conditions"
