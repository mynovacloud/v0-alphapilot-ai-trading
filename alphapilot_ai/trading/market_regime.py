"""
Market Regime Detection
=======================
Identifies the current market state to adapt strategy selection.

Regimes:
- TRENDING_UP: Strong bullish trend, use momentum strategies
- TRENDING_DOWN: Strong bearish trend, use momentum/short strategies  
- RANGING: Sideways market, use mean reversion strategies
- VOLATILE: High volatility, use breakout strategies or reduce size
- ACCUMULATION: Low volatility after downtrend, potential bottom
- DISTRIBUTION: Low volatility after uptrend, potential top
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeAnalysis:
    """Complete market regime analysis."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification
    
    # Trend metrics
    trend_direction: Literal["UP", "DOWN", "FLAT"]
    trend_strength: float  # 0-100 (ADX-like)
    
    # Volatility metrics
    volatility_percentile: float  # Current vol vs historical (0-100)
    is_expanding: bool  # Volatility expanding or contracting
    
    # Range metrics
    range_bound: bool  # Is price stuck in a range?
    range_high: float
    range_low: float
    
    # Recommended actions
    recommended_strategy: str
    position_size_multiplier: float  # 0.5-1.5 based on regime
    
    # Raw metrics for Claude
    metrics: dict


def detect_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 50
) -> Optional[RegimeAnalysis]:
    """
    Detect the current market regime using multiple techniques.
    
    Uses:
    1. ADX for trend strength
    2. Bollinger Bandwidth for volatility
    3. Linear regression slope for trend direction
    4. Price range analysis for ranging detection
    5. Volume patterns for accumulation/distribution
    """
    if len(close) < lookback:
        return None
    
    # Use recent data
    h = high[-lookback:]
    l = low[-lookback:]
    c = close[-lookback:]
    v = volume[-lookback:]
    
    # 1. Trend Direction via Linear Regression
    x = np.arange(lookback)
    slope, intercept = np.polyfit(x, c, 1)
    slope_pct = (slope / c[0]) * 100 * lookback  # Normalized slope
    
    if slope_pct > 5:
        trend_direction = "UP"
    elif slope_pct < -5:
        trend_direction = "DOWN"
    else:
        trend_direction = "FLAT"
    
    # 2. Trend Strength via simplified ADX approximation
    # Using directional movement
    plus_dm = np.maximum(np.diff(h), 0)
    minus_dm = np.maximum(-np.diff(l), 0)
    
    # Zero out when opposite is larger
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1])
        )
    )
    
    atr = np.mean(tr[-14:])
    plus_di = 100 * np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-14:]) / atr if atr > 0 else 0
    
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
    trend_strength = dx  # Simplified ADX
    
    # 3. Volatility Analysis
    returns = np.diff(np.log(c))
    current_vol = np.std(returns[-14:]) * np.sqrt(252)  # Annualized
    historical_vol = np.std(returns) * np.sqrt(252)
    
    # Volatility percentile (how current compares to history)
    rolling_vols = []
    for i in range(14, len(returns)):
        rv = np.std(returns[i-14:i]) * np.sqrt(252)
        rolling_vols.append(rv)
    
    if rolling_vols:
        vol_percentile = sum(1 for rv in rolling_vols if rv < current_vol) / len(rolling_vols) * 100
    else:
        vol_percentile = 50
    
    # Volatility expanding or contracting
    recent_vol = np.std(returns[-7:]) if len(returns) >= 7 else current_vol
    older_vol = np.std(returns[-14:-7]) if len(returns) >= 14 else current_vol
    is_expanding = recent_vol > older_vol * 1.1
    
    # 4. Range Detection
    recent_high = np.max(h[-20:])
    recent_low = np.min(l[-20:])
    range_size = (recent_high - recent_low) / recent_low * 100
    
    # Check if price has been bouncing within range
    touches_high = np.sum(h[-20:] > recent_high * 0.99)
    touches_low = np.sum(l[-20:] < recent_low * 1.01)
    range_bound = touches_high >= 2 and touches_low >= 2 and range_size < 10
    
    # 5. Volume Analysis for Accumulation/Distribution
    # Rising price + rising volume = healthy trend
    # Rising price + falling volume = distribution
    # Falling price + falling volume = accumulation
    price_change = (c[-1] - c[0]) / c[0] * 100
    vol_change = (np.mean(v[-10:]) - np.mean(v[:10])) / np.mean(v[:10]) * 100 if np.mean(v[:10]) > 0 else 0
    
    # 6. Determine Regime
    regime = MarketRegime.UNKNOWN
    confidence = 0.5
    recommended_strategy = "Momentum"
    size_multiplier = 1.0
    
    # High volatility overrides other regimes
    if vol_percentile > 80:
        regime = MarketRegime.VOLATILE
        confidence = min(0.9, vol_percentile / 100)
        recommended_strategy = "Volatility Breakout"
        size_multiplier = 0.6  # Reduce size in high vol
    
    # Strong trend
    elif trend_strength > 25:
        if trend_direction == "UP":
            regime = MarketRegime.TRENDING_UP
            recommended_strategy = "Momentum"
            size_multiplier = 1.2
        else:
            regime = MarketRegime.TRENDING_DOWN
            recommended_strategy = "Momentum"
            size_multiplier = 1.0
        confidence = min(0.9, trend_strength / 50)
    
    # Range bound market
    elif range_bound or trend_strength < 15:
        # Check for accumulation/distribution
        if price_change < -5 and vol_change < -20:
            regime = MarketRegime.ACCUMULATION
            recommended_strategy = "Mean Reversion"
            size_multiplier = 0.8
            confidence = 0.6
        elif price_change > 5 and vol_change < -20:
            regime = MarketRegime.DISTRIBUTION
            recommended_strategy = "Mean Reversion"
            size_multiplier = 0.7
            confidence = 0.6
        else:
            regime = MarketRegime.RANGING
            recommended_strategy = "Mean Reversion"
            size_multiplier = 0.9
            confidence = 0.7
    
    else:
        # Weak/moderate trend
        if trend_direction == "UP":
            regime = MarketRegime.TRENDING_UP
        elif trend_direction == "DOWN":
            regime = MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING
        recommended_strategy = "Momentum"
        confidence = 0.5
        size_multiplier = 0.9
    
    return RegimeAnalysis(
        regime=regime,
        confidence=confidence,
        trend_direction=trend_direction,
        trend_strength=trend_strength,
        volatility_percentile=vol_percentile,
        is_expanding=is_expanding,
        range_bound=range_bound,
        range_high=recent_high,
        range_low=recent_low,
        recommended_strategy=recommended_strategy,
        position_size_multiplier=size_multiplier,
        metrics={
            "slope_pct": round(slope_pct, 2),
            "adx_approx": round(trend_strength, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "current_volatility": round(current_vol * 100, 2),
            "vol_percentile": round(vol_percentile, 1),
            "range_size_pct": round(range_size, 2),
            "price_change_pct": round(price_change, 2),
            "volume_change_pct": round(vol_change, 2),
        }
    )


def get_regime_trading_rules(regime: MarketRegime) -> dict:
    """
    Get specific trading rules for each regime.
    """
    rules = {
        MarketRegime.TRENDING_UP: {
            "bias": "LONG",
            "entry_on_pullback": True,
            "use_trailing_stop": True,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 4.0,
            "dca_on_dip": True,
            "max_position_pct": 0.30,
        },
        MarketRegime.TRENDING_DOWN: {
            "bias": "SHORT",
            "entry_on_pullback": True,
            "use_trailing_stop": True,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 3.0,
            "dca_on_dip": False,
            "max_position_pct": 0.20,
        },
        MarketRegime.RANGING: {
            "bias": "NEUTRAL",
            "entry_on_pullback": False,
            "use_trailing_stop": False,
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 2.0,
            "dca_on_dip": False,
            "max_position_pct": 0.15,
            "buy_at_support": True,
            "sell_at_resistance": True,
        },
        MarketRegime.VOLATILE: {
            "bias": "NEUTRAL",
            "entry_on_pullback": False,
            "use_trailing_stop": True,
            "stop_loss_atr_mult": 3.0,
            "take_profit_atr_mult": 5.0,
            "dca_on_dip": False,
            "max_position_pct": 0.10,
            "wait_for_confirmation": True,
        },
        MarketRegime.ACCUMULATION: {
            "bias": "LONG",
            "entry_on_pullback": False,
            "use_trailing_stop": False,
            "stop_loss_atr_mult": 2.5,
            "take_profit_atr_mult": 4.0,
            "dca_on_dip": True,
            "max_position_pct": 0.25,
            "scale_in_slowly": True,
        },
        MarketRegime.DISTRIBUTION: {
            "bias": "SHORT",
            "entry_on_pullback": False,
            "use_trailing_stop": False,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 3.0,
            "dca_on_dip": False,
            "max_position_pct": 0.15,
            "reduce_exposure": True,
        },
    }
    
    return rules.get(regime, rules[MarketRegime.RANGING])
