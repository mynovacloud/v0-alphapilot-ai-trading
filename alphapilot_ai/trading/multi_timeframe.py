"""
Multi-Timeframe Analysis (MTF)
==============================
Analyzes price action across multiple timeframes to confirm signals.

Principle:
- Higher timeframe sets the trend direction (bias)
- Lower timeframe provides entry timing
- Alignment across timeframes = higher probability trade

Timeframe Hierarchy:
- 1D (daily) -> Weekly trend context
- 4H -> Daily trend context  
- 1H -> Intraday trend
- 15m -> Entry timing
- 5m -> Scalp timing
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

from connectors.candles import get_candles
from trading.indicators import ema, rsi, macd, bollinger_bands, atr


class TrendBias(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe."""
    timeframe: str  # "5m", "15m", "1h", "4h", "1d"
    granularity: int  # seconds
    
    # Trend
    trend: Literal["UP", "DOWN", "FLAT"]
    trend_strength: float  # 0-100
    
    # Key levels
    current_price: float
    ema_20: float
    ema_50: float
    ema_200: float
    
    # Momentum
    rsi: float
    macd_histogram: float
    
    # Volatility
    atr_pct: float
    bb_position: float  # 0-1 where in bands
    
    # Signals
    ema_alignment: bool  # Price > EMA20 > EMA50
    momentum_bullish: bool
    momentum_bearish: bool


@dataclass
class MTFAnalysis:
    """Complete multi-timeframe analysis."""
    symbol: str
    
    # Individual timeframes
    tf_5m: Optional[TimeframeAnalysis]
    tf_15m: Optional[TimeframeAnalysis]
    tf_1h: Optional[TimeframeAnalysis]
    tf_4h: Optional[TimeframeAnalysis]
    tf_1d: Optional[TimeframeAnalysis]
    
    # Aggregate signals
    overall_bias: TrendBias
    alignment_score: float  # 0-1, how aligned are timeframes
    
    # Trade recommendations
    entry_timing: str  # "NOW", "WAIT_PULLBACK", "WAIT_BREAKOUT", "NO_TRADE"
    confidence_boost: float  # -0.2 to +0.2 adjustment to base confidence
    
    # Specific conditions
    higher_tf_support: bool  # Higher TFs support the trade
    divergence_warning: bool  # Lower TF diverging from higher
    
    # Context for Claude
    summary: str
    details: dict


def _analyze_timeframe(
    symbol: str,
    granularity: int,
    timeframe_name: str,
) -> Optional[TimeframeAnalysis]:
    """
    Analyze a single timeframe.
    """
    # Determine how many candles we need
    candle_counts = {
        60: 300,     # 5m worth of 1m candles
        300: 250,    # ~20h of 5m candles
        900: 200,    # ~50h of 15m candles  
        3600: 200,   # ~8 days of 1h candles
        14400: 150,  # ~25 days of 4h candles
        86400: 100,  # ~100 days of 1d candles
    }
    
    limit = candle_counts.get(granularity, 200)
    
    candle_data = get_candles(symbol, granularity=granularity, limit=limit)
    if not candle_data or not candle_data.get("candles"):
        return None
    
    candles = candle_data["candles"]
    if len(candles) < 50:
        return None
    
    # Convert to arrays
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    
    # Calculate indicators
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_200_arr = ema(close, 200) if len(close) >= 200 else np.full_like(close, np.nan)
    
    rsi_14 = rsi(close, 14)
    macd_result = macd(close)
    atr_14 = atr(high, low, close, 14)
    bb = bollinger_bands(close, 20, 2.0)
    
    # Get latest values
    i = len(close) - 1
    current_price = close[i]
    ema_20_val = ema_20[i]
    ema_50_val = ema_50[i]
    ema_200_val = ema_200_arr[i] if not np.isnan(ema_200_arr[i]) else ema_50_val
    rsi_val = rsi_14[i]
    macd_hist = macd_result.histogram[i]
    atr_val = atr_14[i]
    
    # Determine trend
    if current_price > ema_20_val > ema_50_val:
        trend = "UP"
    elif current_price < ema_20_val < ema_50_val:
        trend = "DOWN"
    else:
        trend = "FLAT"
    
    # Trend strength from price vs EMAs
    price_vs_ema20 = (current_price - ema_20_val) / ema_20_val * 100
    price_vs_ema50 = (current_price - ema_50_val) / ema_50_val * 100
    trend_strength = min(100, abs(price_vs_ema20) * 10 + abs(price_vs_ema50) * 5)
    
    # EMA alignment
    ema_alignment = current_price > ema_20_val > ema_50_val
    
    # Momentum
    momentum_bullish = rsi_val > 50 and macd_hist > 0
    momentum_bearish = rsi_val < 50 and macd_hist < 0
    
    # Bollinger position
    bb_range = bb.upper[i] - bb.lower[i]
    bb_position = (current_price - bb.lower[i]) / bb_range if bb_range > 0 else 0.5
    
    # ATR as percentage
    atr_pct = (atr_val / current_price * 100) if current_price > 0 else 0
    
    return TimeframeAnalysis(
        timeframe=timeframe_name,
        granularity=granularity,
        trend=trend,
        trend_strength=trend_strength,
        current_price=current_price,
        ema_20=ema_20_val,
        ema_50=ema_50_val,
        ema_200=ema_200_val,
        rsi=rsi_val,
        macd_histogram=macd_hist,
        atr_pct=atr_pct,
        bb_position=bb_position,
        ema_alignment=ema_alignment,
        momentum_bullish=momentum_bullish,
        momentum_bearish=momentum_bearish,
    )


def analyze_multi_timeframe(symbol: str) -> Optional[MTFAnalysis]:
    """
    Perform multi-timeframe analysis on a symbol.
    
    Fetches data from multiple timeframes and determines:
    1. Overall trend bias
    2. Alignment across timeframes
    3. Entry timing recommendation
    """
    # Analyze each timeframe
    tf_5m = _analyze_timeframe(symbol, 300, "5m")
    tf_15m = _analyze_timeframe(symbol, 900, "15m")
    tf_1h = _analyze_timeframe(symbol, 3600, "1h")
    tf_4h = _analyze_timeframe(symbol, 14400, "4h")
    tf_1d = _analyze_timeframe(symbol, 86400, "1d")
    
    # Need at least 15m and 1h
    if not tf_15m or not tf_1h:
        return None
    
    # Calculate alignment score
    timeframes = [tf for tf in [tf_5m, tf_15m, tf_1h, tf_4h, tf_1d] if tf]
    
    bullish_count = sum(1 for tf in timeframes if tf.trend == "UP")
    bearish_count = sum(1 for tf in timeframes if tf.trend == "DOWN")
    total_tfs = len(timeframes)
    
    # Alignment = how many agree with the majority
    majority_count = max(bullish_count, bearish_count)
    alignment_score = majority_count / total_tfs if total_tfs > 0 else 0
    
    # Determine overall bias (weighted by timeframe importance)
    # Higher timeframes have more weight
    weights = {"5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}
    
    bullish_weight = sum(
        weights.get(tf.timeframe, 1) for tf in timeframes 
        if tf.trend == "UP"
    )
    bearish_weight = sum(
        weights.get(tf.timeframe, 1) for tf in timeframes 
        if tf.trend == "DOWN"
    )
    total_weight = bullish_weight + bearish_weight
    
    if total_weight == 0:
        overall_bias = TrendBias.NEUTRAL
    elif bullish_weight > bearish_weight * 1.5:
        overall_bias = TrendBias.STRONG_BULLISH if alignment_score > 0.7 else TrendBias.BULLISH
    elif bearish_weight > bullish_weight * 1.5:
        overall_bias = TrendBias.STRONG_BEARISH if alignment_score > 0.7 else TrendBias.BEARISH
    else:
        overall_bias = TrendBias.NEUTRAL
    
    # Check higher timeframe support
    higher_tf_support = False
    if tf_1h and tf_4h:
        if tf_1h.trend == tf_4h.trend and tf_1h.trend != "FLAT":
            higher_tf_support = True
    elif tf_1h and tf_1d:
        if tf_1h.trend == tf_1d.trend and tf_1h.trend != "FLAT":
            higher_tf_support = True
    
    # Check for divergence (lower TF going opposite to higher)
    divergence_warning = False
    if tf_15m and tf_4h:
        if tf_15m.trend == "UP" and tf_4h.trend == "DOWN":
            divergence_warning = True
        elif tf_15m.trend == "DOWN" and tf_4h.trend == "UP":
            divergence_warning = True
    
    # Entry timing
    if alignment_score >= 0.8 and not divergence_warning:
        entry_timing = "NOW"
    elif alignment_score >= 0.6 and higher_tf_support:
        # Wait for pullback to key level
        entry_timing = "WAIT_PULLBACK"
    elif alignment_score < 0.4:
        entry_timing = "NO_TRADE"
    else:
        entry_timing = "WAIT_BREAKOUT"
    
    # Confidence adjustment
    if alignment_score >= 0.8 and higher_tf_support:
        confidence_boost = 0.15
    elif alignment_score >= 0.6 and higher_tf_support:
        confidence_boost = 0.10
    elif alignment_score >= 0.6:
        confidence_boost = 0.05
    elif divergence_warning:
        confidence_boost = -0.10
    else:
        confidence_boost = 0.0
    
    # Generate summary
    summary_parts = []
    
    if overall_bias in [TrendBias.STRONG_BULLISH, TrendBias.BULLISH]:
        summary_parts.append(f"MTF Analysis: BULLISH bias ({alignment_score*100:.0f}% alignment)")
    elif overall_bias in [TrendBias.STRONG_BEARISH, TrendBias.BEARISH]:
        summary_parts.append(f"MTF Analysis: BEARISH bias ({alignment_score*100:.0f}% alignment)")
    else:
        summary_parts.append(f"MTF Analysis: NEUTRAL/MIXED ({alignment_score*100:.0f}% alignment)")
    
    if higher_tf_support:
        summary_parts.append("Higher TFs confirm direction.")
    if divergence_warning:
        summary_parts.append("WARNING: Lower TF diverging from trend.")
    
    summary_parts.append(f"Entry timing: {entry_timing}")
    
    summary = " ".join(summary_parts)
    
    # Details for Claude
    details = {
        "timeframes": {
            tf.timeframe: {
                "trend": tf.trend,
                "trend_strength": round(tf.trend_strength, 1),
                "rsi": round(tf.rsi, 1),
                "macd_hist": round(tf.macd_histogram, 6),
                "ema_alignment": tf.ema_alignment,
                "bb_position": round(tf.bb_position, 2),
            }
            for tf in timeframes
        },
        "bullish_timeframes": bullish_count,
        "bearish_timeframes": bearish_count,
        "neutral_timeframes": total_tfs - bullish_count - bearish_count,
        "weighted_bullish": bullish_weight,
        "weighted_bearish": bearish_weight,
    }
    
    return MTFAnalysis(
        symbol=symbol,
        tf_5m=tf_5m,
        tf_15m=tf_15m,
        tf_1h=tf_1h,
        tf_4h=tf_4h,
        tf_1d=tf_1d,
        overall_bias=overall_bias,
        alignment_score=alignment_score,
        entry_timing=entry_timing,
        confidence_boost=confidence_boost,
        higher_tf_support=higher_tf_support,
        divergence_warning=divergence_warning,
        summary=summary,
        details=details,
    )


def get_mtf_signal_boost(symbol: str) -> dict:
    """
    Quick function to get MTF-based confidence adjustment.
    
    Returns:
        {
            "boost": float (-0.2 to 0.2),
            "bias": str,
            "entry_timing": str,
            "summary": str
        }
    """
    analysis = analyze_multi_timeframe(symbol)
    
    if not analysis:
        return {
            "boost": 0.0,
            "bias": "UNKNOWN",
            "entry_timing": "UNKNOWN",
            "summary": "MTF analysis unavailable",
        }
    
    return {
        "boost": analysis.confidence_boost,
        "bias": analysis.overall_bias.value,
        "entry_timing": analysis.entry_timing,
        "alignment": analysis.alignment_score,
        "higher_tf_support": analysis.higher_tf_support,
        "divergence_warning": analysis.divergence_warning,
        "summary": analysis.summary,
        "details": analysis.details,
    }
