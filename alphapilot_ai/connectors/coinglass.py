"""
Coinglass Integration
=====================
Fetches derivatives market data:
- Funding rates (perpetual futures)
- Open interest
- Liquidation data
- Long/short ratios

This data provides edge in crypto trading by showing:
- When funding is extremely negative = potential bottom (shorts paying)
- When funding is extremely positive = potential top (longs paying)
- Liquidation clusters = support/resistance levels
- Open interest extremes = crowded trades
"""

from __future__ import annotations
import os
import time
import httpx
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache


# Coinglass public API (no key required for basic data)
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"

# Cache TTL
_cache: dict = {}
_cache_ttl = 300  # 5 minutes


@dataclass
class FundingData:
    """Funding rate data for a symbol."""
    symbol: str
    
    # Current funding rate (8-hour)
    current_rate: float  # As percentage, e.g., 0.01 = 0.01%
    predicted_rate: float
    
    # Historical context
    avg_rate_7d: float
    max_rate_7d: float
    min_rate_7d: float
    
    # Interpretation
    sentiment: str  # "EXTREME_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "EXTREME_BEARISH"
    funding_signal: str  # "LONG_OPPORTUNITY", "SHORT_OPPORTUNITY", "NEUTRAL"
    
    # APY if you were to collect funding
    annualized_rate: float


@dataclass
class OpenInterestData:
    """Open interest data."""
    symbol: str
    
    # Current OI
    total_oi_usd: float
    oi_change_1h: float  # Percentage
    oi_change_24h: float
    
    # Context
    oi_percentile: float  # Where current OI sits vs 30-day range
    is_elevated: bool  # OI > 90th percentile
    is_low: bool  # OI < 10th percentile


@dataclass
class LongShortRatio:
    """Long/short ratio from top traders."""
    symbol: str
    
    # Ratios
    long_ratio: float  # Percentage of longs
    short_ratio: float  # Percentage of shorts
    ratio: float  # long_ratio / short_ratio
    
    # Context
    sentiment: str  # "EXTREMELY_LONG", "LONG", "NEUTRAL", "SHORT", "EXTREMELY_SHORT"
    contrarian_signal: str  # "BUY" if extremely short, "SELL" if extremely long


@dataclass
class DerivativesIntel:
    """Complete derivatives market intelligence."""
    symbol: str
    
    funding: Optional[FundingData]
    open_interest: Optional[OpenInterestData]
    long_short: Optional[LongShortRatio]
    
    # Aggregate signals
    overall_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence_adjustment: float  # -0.15 to +0.15
    
    # Risk warnings
    warnings: list[str]
    
    # Summary for Claude
    summary: str


def _get_cached(key: str) -> Optional[dict]:
    """Get cached data if still valid."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return data
    return None


def _set_cache(key: str, data: dict):
    """Cache data."""
    _cache[key] = (data, time.time())


def get_funding_rates(symbol: str = "BTC") -> Optional[FundingData]:
    """
    Get funding rate data for a symbol.
    
    Note: Uses public Coinglass API which has rate limits.
    Symbol should be base asset like "BTC", "ETH", etc.
    """
    cache_key = f"funding_{symbol}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        # Coinglass public funding rate endpoint
        url = f"{COINGLASS_BASE}/funding"
        params = {"symbol": symbol}
        
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            if not data.get("success") or not data.get("data"):
                return None
            
            # Parse response
            rates = data["data"]
            
            # Get aggregate across exchanges
            if isinstance(rates, list) and rates:
                # Average across major exchanges
                current_rates = [r.get("rate", 0) for r in rates if r.get("rate")]
                predicted_rates = [r.get("predictedRate", 0) for r in rates if r.get("predictedRate")]
                
                current_rate = sum(current_rates) / len(current_rates) if current_rates else 0
                predicted_rate = sum(predicted_rates) / len(predicted_rates) if predicted_rates else current_rate
            else:
                current_rate = rates.get("rate", 0) if isinstance(rates, dict) else 0
                predicted_rate = rates.get("predictedRate", current_rate) if isinstance(rates, dict) else current_rate
            
            # Convert to percentage
            current_rate *= 100
            predicted_rate *= 100
            
            # Estimate 7-day stats (simplified - would need historical endpoint)
            avg_rate_7d = current_rate * 0.8  # Rough estimate
            max_rate_7d = abs(current_rate) * 1.5
            min_rate_7d = -abs(current_rate) * 0.5
            
            # Interpret sentiment
            if current_rate > 0.05:
                sentiment = "EXTREME_BULLISH"
                funding_signal = "SHORT_OPPORTUNITY"  # Contrarian
            elif current_rate > 0.02:
                sentiment = "BULLISH"
                funding_signal = "NEUTRAL"
            elif current_rate < -0.05:
                sentiment = "EXTREME_BEARISH"
                funding_signal = "LONG_OPPORTUNITY"  # Contrarian
            elif current_rate < -0.02:
                sentiment = "BEARISH"
                funding_signal = "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
                funding_signal = "NEUTRAL"
            
            # Annualized rate (3 funding periods per day * 365)
            annualized = current_rate * 3 * 365
            
            result = FundingData(
                symbol=symbol,
                current_rate=current_rate,
                predicted_rate=predicted_rate,
                avg_rate_7d=avg_rate_7d,
                max_rate_7d=max_rate_7d,
                min_rate_7d=min_rate_7d,
                sentiment=sentiment,
                funding_signal=funding_signal,
                annualized_rate=annualized,
            )
            
            _set_cache(cache_key, result)
            return result
            
    except Exception as e:
        return None


def get_open_interest(symbol: str = "BTC") -> Optional[OpenInterestData]:
    """
    Get open interest data for a symbol.
    """
    cache_key = f"oi_{symbol}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        url = f"{COINGLASS_BASE}/open_interest"
        params = {"symbol": symbol}
        
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            if not data.get("success") or not data.get("data"):
                return None
            
            oi_data = data["data"]
            
            # Aggregate OI across exchanges
            if isinstance(oi_data, list):
                total_oi = sum(r.get("openInterest", 0) for r in oi_data)
            else:
                total_oi = oi_data.get("openInterest", 0) if isinstance(oi_data, dict) else 0
            
            # Simplified change calculations
            oi_change_1h = 0  # Would need historical data
            oi_change_24h = 0
            
            # Estimate percentile (simplified)
            oi_percentile = 50  # Default to median without historical context
            is_elevated = False
            is_low = False
            
            result = OpenInterestData(
                symbol=symbol,
                total_oi_usd=total_oi,
                oi_change_1h=oi_change_1h,
                oi_change_24h=oi_change_24h,
                oi_percentile=oi_percentile,
                is_elevated=is_elevated,
                is_low=is_low,
            )
            
            _set_cache(cache_key, result)
            return result
            
    except Exception as e:
        return None


def get_long_short_ratio(symbol: str = "BTC") -> Optional[LongShortRatio]:
    """
    Get long/short ratio from top traders.
    """
    cache_key = f"lsr_{symbol}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        url = f"{COINGLASS_BASE}/long_short"
        params = {"symbol": symbol}
        
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            if not data.get("success") or not data.get("data"):
                return None
            
            ls_data = data["data"]
            
            # Get ratio
            if isinstance(ls_data, list) and ls_data:
                # Average across exchanges
                long_ratios = [r.get("longRate", 50) for r in ls_data]
                long_ratio = sum(long_ratios) / len(long_ratios)
            else:
                long_ratio = ls_data.get("longRate", 50) if isinstance(ls_data, dict) else 50
            
            short_ratio = 100 - long_ratio
            ratio = long_ratio / short_ratio if short_ratio > 0 else 1
            
            # Interpret
            if long_ratio > 65:
                sentiment = "EXTREMELY_LONG"
                contrarian_signal = "SELL"  # Too many longs = potential reversal
            elif long_ratio > 55:
                sentiment = "LONG"
                contrarian_signal = "NEUTRAL"
            elif long_ratio < 35:
                sentiment = "EXTREMELY_SHORT"
                contrarian_signal = "BUY"  # Too many shorts = potential squeeze
            elif long_ratio < 45:
                sentiment = "SHORT"
                contrarian_signal = "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
                contrarian_signal = "NEUTRAL"
            
            result = LongShortRatio(
                symbol=symbol,
                long_ratio=long_ratio,
                short_ratio=short_ratio,
                ratio=ratio,
                sentiment=sentiment,
                contrarian_signal=contrarian_signal,
            )
            
            _set_cache(cache_key, result)
            return result
            
    except Exception as e:
        return None


def get_derivatives_intel(symbol: str) -> DerivativesIntel:
    """
    Get complete derivatives market intelligence for a symbol.
    
    This combines funding rates, open interest, and long/short ratios
    to provide actionable trading signals.
    """
    # Normalize symbol (remove -USD, -USDT, etc.)
    base_symbol = symbol.split("-")[0].upper()
    
    # Fetch all data
    funding = get_funding_rates(base_symbol)
    oi = get_open_interest(base_symbol)
    ls_ratio = get_long_short_ratio(base_symbol)
    
    warnings = []
    signals = []
    confidence_adj = 0.0
    
    # Analyze funding
    if funding:
        if funding.funding_signal == "LONG_OPPORTUNITY":
            signals.append("BULLISH")
            confidence_adj += 0.10
            warnings.append(f"Negative funding ({funding.current_rate:.3f}%) - shorts paying longs")
        elif funding.funding_signal == "SHORT_OPPORTUNITY":
            signals.append("BEARISH")
            confidence_adj -= 0.05
            warnings.append(f"High funding ({funding.current_rate:.3f}%) - longs paying shorts")
    
    # Analyze long/short ratio
    if ls_ratio:
        if ls_ratio.contrarian_signal == "BUY":
            signals.append("BULLISH")
            confidence_adj += 0.05
            warnings.append(f"Extreme shorts ({ls_ratio.short_ratio:.1f}%) - potential squeeze")
        elif ls_ratio.contrarian_signal == "SELL":
            signals.append("BEARISH")
            confidence_adj -= 0.05
            warnings.append(f"Extreme longs ({ls_ratio.long_ratio:.1f}%) - crowded trade")
    
    # Analyze open interest
    if oi:
        if oi.is_elevated:
            warnings.append("Elevated open interest - expect volatility")
        elif oi.is_low:
            warnings.append("Low open interest - weak conviction in moves")
    
    # Determine overall signal
    bullish_count = signals.count("BULLISH")
    bearish_count = signals.count("BEARISH")
    
    if bullish_count > bearish_count:
        overall_signal = "BULLISH"
    elif bearish_count > bullish_count:
        overall_signal = "BEARISH"
    else:
        overall_signal = "NEUTRAL"
    
    # Cap confidence adjustment
    confidence_adj = max(-0.15, min(0.15, confidence_adj))
    
    # Generate summary
    summary_parts = [f"Derivatives Intel for {base_symbol}:"]
    
    if funding:
        summary_parts.append(f"Funding: {funding.current_rate:.3f}% ({funding.sentiment})")
    
    if ls_ratio:
        summary_parts.append(f"L/S Ratio: {ls_ratio.long_ratio:.1f}%/{ls_ratio.short_ratio:.1f}%")
    
    if oi:
        summary_parts.append(f"OI: ${oi.total_oi_usd/1e9:.2f}B")
    
    summary_parts.append(f"Signal: {overall_signal}")
    
    summary = " | ".join(summary_parts)
    
    return DerivativesIntel(
        symbol=symbol,
        funding=funding,
        open_interest=oi,
        long_short=ls_ratio,
        overall_signal=overall_signal,
        confidence_adjustment=confidence_adj,
        warnings=warnings,
        summary=summary,
    )


def get_funding_signal(symbol: str) -> dict:
    """
    Quick function to get funding-based signal for a symbol.
    
    Returns dict suitable for Claude decision context.
    """
    intel = get_derivatives_intel(symbol)
    
    return {
        "symbol": symbol,
        "overall_signal": intel.overall_signal,
        "confidence_adjustment": intel.confidence_adjustment,
        "funding_rate": intel.funding.current_rate if intel.funding else None,
        "funding_sentiment": intel.funding.sentiment if intel.funding else None,
        "long_ratio": intel.long_short.long_ratio if intel.long_short else None,
        "short_ratio": intel.long_short.short_ratio if intel.long_short else None,
        "ls_sentiment": intel.long_short.sentiment if intel.long_short else None,
        "warnings": intel.warnings,
        "summary": intel.summary,
    }
