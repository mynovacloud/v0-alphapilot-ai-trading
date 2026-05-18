"""
Advanced Market Regime Detection
================================
Identifies the current market state to adapt strategy selection using
multiple detection methods and confidence scoring.

Regimes:
- TRENDING_UP: Strong bullish trend, use momentum strategies
- TRENDING_DOWN: Strong bearish trend, use momentum/short strategies  
- RANGING: Sideways market, use mean reversion strategies
- VOLATILE: High volatility, use breakout strategies or reduce size
- ACCUMULATION: Low volatility after downtrend, potential bottom
- DISTRIBUTION: Low volatility after uptrend, potential top
- BREAKOUT: Price breaking out of established range
- CONSOLIDATION: Narrowing volatility, potential breakout imminent

Detection Methods:
1. ADX-based trend strength
2. Bollinger Band width for volatility
3. Linear regression for trend direction
4. Volume profile analysis
5. Price range compression/expansion
6. Hidden Markov Model-inspired state transitions
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Tuple
from enum import Enum
from datetime import datetime
from collections import deque


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    BREAKOUT = "BREAKOUT"
    CONSOLIDATION = "CONSOLIDATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeAnalysis:
    """Complete market regime analysis with multiple detection methods."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification
    
    # Trend metrics
    trend_direction: Literal["UP", "DOWN", "FLAT"]
    trend_strength: float  # 0-100 (ADX-like)
    
    # Volatility metrics
    volatility_percentile: float  # Current vol vs historical (0-100)
    is_expanding: bool  # Volatility expanding or contracting
    volatility_state: Literal["LOW", "NORMAL", "HIGH", "EXTREME"]
    
    # Range metrics
    range_bound: bool  # Is price stuck in a range?
    range_high: float
    range_low: float
    range_position: float  # 0-1 where in range (0=bottom, 1=top)
    
    # Momentum metrics
    momentum_score: float  # -100 to 100
    momentum_divergence: bool  # Price/momentum divergence detected
    
    # Volume metrics
    volume_trend: Literal["INCREASING", "DECREASING", "STABLE"]
    volume_confirmation: bool  # Volume confirms price move
    
    # Recommended actions
    recommended_strategy: str
    position_size_multiplier: float  # 0.5-1.5 based on regime
    stop_loss_multiplier: float  # ATR multiplier for stops
    take_profit_multiplier: float  # ATR multiplier for targets
    
    # Previous regime for transition detection
    previous_regime: Optional[MarketRegime] = None
    regime_duration_bars: int = 0
    
    # Raw metrics for Claude
    metrics: Dict = field(default_factory=dict)
    
    # Confidence breakdown
    detection_methods: Dict[str, Dict] = field(default_factory=dict)


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


# =============================================================================
# Advanced Regime Detection System
# =============================================================================

class RegimeHistory:
    """Track regime history for transition analysis."""
    
    def __init__(self, max_history: int = 100):
        self._history: deque = deque(maxlen=max_history)
        self._current_regime: Optional[MarketRegime] = None
        self._regime_start_idx: int = 0
    
    def update(self, regime: MarketRegime, bar_index: int) -> Tuple[Optional[MarketRegime], int]:
        """
        Update regime history and return previous regime + duration.
        
        Returns:
            Tuple of (previous_regime, bars_in_previous_regime)
        """
        previous = self._current_regime
        duration = bar_index - self._regime_start_idx if self._current_regime else 0
        
        if regime != self._current_regime:
            self._history.append({
                "regime": self._current_regime,
                "duration": duration,
                "end_idx": bar_index,
            })
            self._current_regime = regime
            self._regime_start_idx = bar_index
        
        return previous, duration
    
    def get_transition_probability(self, from_regime: MarketRegime, to_regime: MarketRegime) -> float:
        """Calculate historical transition probability between regimes."""
        transitions_from = 0
        transitions_to = 0
        
        for i in range(len(self._history) - 1):
            if self._history[i]["regime"] == from_regime:
                transitions_from += 1
                if self._history[i + 1]["regime"] == to_regime:
                    transitions_to += 1
        
        if transitions_from == 0:
            return 0.0
        return transitions_to / transitions_from
    
    def get_average_duration(self, regime: MarketRegime) -> float:
        """Get average duration for a specific regime."""
        durations = [h["duration"] for h in self._history if h["regime"] == regime]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)


class AdvancedRegimeDetector:
    """
    Advanced market regime detector using multiple methods:
    
    1. Trend Analysis (ADX, Linear Regression, Moving Average alignment)
    2. Volatility Analysis (ATR percentile, Bollinger Width, VIX-like measure)
    3. Volume Profile (OBV trend, Volume/Price divergence)
    4. Pattern Recognition (Range detection, Breakout detection)
    5. Momentum Analysis (RSI regime, MACD histogram pattern)
    """
    
    def __init__(self):
        self._regime_history = RegimeHistory()
        self._bar_count = 0
    
    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        lookback: int = 50
    ) -> Optional[RegimeAnalysis]:
        """
        Comprehensive regime detection using multiple methods.
        
        Each method votes on the regime with a confidence score.
        Final regime is determined by weighted consensus.
        """
        if len(close) < lookback:
            return None
        
        self._bar_count += 1
        
        # Prepare data
        h = high[-lookback:]
        l = low[-lookback:]
        c = close[-lookback:]
        v = volume[-lookback:]
        
        # Run all detection methods
        methods = {}
        
        # Method 1: Trend Analysis
        trend_result = self._analyze_trend(h, l, c)
        methods["trend"] = trend_result
        
        # Method 2: Volatility Analysis
        vol_result = self._analyze_volatility(h, l, c)
        methods["volatility"] = vol_result
        
        # Method 3: Volume Analysis
        vol_price_result = self._analyze_volume(c, v)
        methods["volume"] = vol_price_result
        
        # Method 4: Range/Pattern Analysis
        pattern_result = self._analyze_patterns(h, l, c)
        methods["pattern"] = pattern_result
        
        # Method 5: Momentum Analysis
        momentum_result = self._analyze_momentum(c)
        methods["momentum"] = momentum_result
        
        # Combine methods into final regime
        regime, confidence = self._combine_methods(methods)
        
        # Get regime-specific recommendations
        recommendations = self._get_recommendations(regime, methods)
        
        # Update history
        previous_regime, duration = self._regime_history.update(regime, self._bar_count)
        
        # Calculate range position
        recent_high = np.max(h[-20:])
        recent_low = np.min(l[-20:])
        range_size = recent_high - recent_low
        range_position = (c[-1] - recent_low) / range_size if range_size > 0 else 0.5
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_direction=trend_result["direction"],
            trend_strength=trend_result["strength"],
            volatility_percentile=vol_result["percentile"],
            is_expanding=vol_result["expanding"],
            volatility_state=vol_result["state"],
            range_bound=pattern_result["range_bound"],
            range_high=recent_high,
            range_low=recent_low,
            range_position=range_position,
            momentum_score=momentum_result["score"],
            momentum_divergence=momentum_result["divergence"],
            volume_trend=vol_price_result["trend"],
            volume_confirmation=vol_price_result["confirms_price"],
            recommended_strategy=recommendations["strategy"],
            position_size_multiplier=recommendations["size_mult"],
            stop_loss_multiplier=recommendations["sl_mult"],
            take_profit_multiplier=recommendations["tp_mult"],
            previous_regime=previous_regime,
            regime_duration_bars=duration,
            metrics=self._compile_metrics(methods),
            detection_methods=methods,
        )
    
    def _analyze_trend(self, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict:
        """Trend analysis using ADX, linear regression, and MA alignment."""
        # Linear regression for direction
        x = np.arange(len(c))
        slope, _ = np.polyfit(x, c, 1)
        slope_pct = (slope / c[0]) * 100 * len(c)
        
        if slope_pct > 5:
            direction = "UP"
        elif slope_pct < -5:
            direction = "DOWN"
        else:
            direction = "FLAT"
        
        # ADX calculation
        plus_dm = np.maximum(np.diff(h), 0)
        minus_dm = np.maximum(-np.diff(l), 0)
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
        
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
        )
        
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        plus_di = 100 * np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-14:]) / atr if atr > 0 else 0
        
        di_sum = plus_di + minus_di
        adx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        
        # MA alignment (short > medium > long for uptrend)
        sma10 = np.mean(c[-10:]) if len(c) >= 10 else c[-1]
        sma20 = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
        sma50 = np.mean(c[-50:]) if len(c) >= 50 else c[-1]
        
        ma_aligned_up = sma10 > sma20 > sma50
        ma_aligned_down = sma10 < sma20 < sma50
        
        # Determine trend regime vote
        if adx > 30 and direction == "UP":
            vote = MarketRegime.TRENDING_UP
            vote_conf = min(0.9, adx / 50)
        elif adx > 30 and direction == "DOWN":
            vote = MarketRegime.TRENDING_DOWN
            vote_conf = min(0.9, adx / 50)
        elif adx < 20:
            vote = MarketRegime.RANGING
            vote_conf = min(0.8, (25 - adx) / 25)
        else:
            vote = MarketRegime.UNKNOWN
            vote_conf = 0.5
        
        return {
            "direction": direction,
            "strength": adx,
            "slope_pct": slope_pct,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "ma_aligned_up": ma_aligned_up,
            "ma_aligned_down": ma_aligned_down,
            "vote": vote,
            "confidence": vote_conf,
        }
    
    def _analyze_volatility(self, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict:
        """Volatility analysis for regime detection."""
        returns = np.diff(np.log(c))
        
        # Current volatility (annualized)
        current_vol = np.std(returns[-14:]) * np.sqrt(252) if len(returns) >= 14 else 0
        historical_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Volatility percentile
        rolling_vols = []
        for i in range(14, len(returns)):
            rv = np.std(returns[i-14:i]) * np.sqrt(252)
            rolling_vols.append(rv)
        
        if rolling_vols:
            vol_percentile = sum(1 for rv in rolling_vols if rv < current_vol) / len(rolling_vols) * 100
        else:
            vol_percentile = 50
        
        # Volatility state
        if vol_percentile > 85:
            state = "EXTREME"
        elif vol_percentile > 65:
            state = "HIGH"
        elif vol_percentile < 25:
            state = "LOW"
        else:
            state = "NORMAL"
        
        # Volatility expanding or contracting
        recent_vol = np.std(returns[-7:]) if len(returns) >= 7 else current_vol
        older_vol = np.std(returns[-14:-7]) if len(returns) >= 14 else current_vol
        expanding = recent_vol > older_vol * 1.1
        
        # Bollinger Band width
        sma20 = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
        std20 = np.std(c[-20:]) if len(c) >= 20 else 0
        bb_width = (4 * std20 / sma20 * 100) if sma20 > 0 else 0
        
        # Vote based on volatility
        if vol_percentile > 80:
            vote = MarketRegime.VOLATILE
            vote_conf = min(0.9, vol_percentile / 100)
        elif vol_percentile < 20 and not expanding:
            vote = MarketRegime.CONSOLIDATION
            vote_conf = min(0.8, (25 - vol_percentile) / 25)
        else:
            vote = MarketRegime.UNKNOWN
            vote_conf = 0.3
        
        return {
            "current": current_vol,
            "historical": historical_vol,
            "percentile": vol_percentile,
            "state": state,
            "expanding": expanding,
            "bb_width": bb_width,
            "vote": vote,
            "confidence": vote_conf,
        }
    
    def _analyze_volume(self, c: np.ndarray, v: np.ndarray) -> Dict:
        """Volume analysis for confirmation and accumulation/distribution."""
        if len(v) < 20:
            return {
                "trend": "STABLE",
                "confirms_price": True,
                "obv_trend": "FLAT",
                "vote": MarketRegime.UNKNOWN,
                "confidence": 0.3,
            }
        
        # Volume trend
        recent_vol = np.mean(v[-10:])
        older_vol = np.mean(v[-20:-10])
        
        if recent_vol > older_vol * 1.3:
            vol_trend = "INCREASING"
        elif recent_vol < older_vol * 0.7:
            vol_trend = "DECREASING"
        else:
            vol_trend = "STABLE"
        
        # OBV (On-Balance Volume) trend
        obv = [0.0]
        for i in range(1, len(c)):
            if c[i] > c[i-1]:
                obv.append(obv[-1] + v[i])
            elif c[i] < c[i-1]:
                obv.append(obv[-1] - v[i])
            else:
                obv.append(obv[-1])
        
        obv = np.array(obv)
        obv_slope = (obv[-1] - obv[-10]) / 10 if len(obv) >= 10 else 0
        
        if obv_slope > 0:
            obv_trend = "UP"
        elif obv_slope < 0:
            obv_trend = "DOWN"
        else:
            obv_trend = "FLAT"
        
        # Price/Volume confirmation
        price_up = c[-1] > c[-10] if len(c) >= 10 else True
        confirms = (price_up and obv_trend == "UP") or (not price_up and obv_trend == "DOWN")
        
        # Volume pattern for accumulation/distribution
        price_change = (c[-1] - c[0]) / c[0] * 100
        vol_change = (np.mean(v[-10:]) - np.mean(v[:10])) / (np.mean(v[:10]) + 0.001) * 100
        
        if price_change < -5 and vol_change < -20:
            vote = MarketRegime.ACCUMULATION
            vote_conf = 0.7
        elif price_change > 5 and vol_change < -20:
            vote = MarketRegime.DISTRIBUTION
            vote_conf = 0.7
        else:
            vote = MarketRegime.UNKNOWN
            vote_conf = 0.3
        
        return {
            "trend": vol_trend,
            "confirms_price": confirms,
            "obv_trend": obv_trend,
            "price_change_pct": price_change,
            "vol_change_pct": vol_change,
            "vote": vote,
            "confidence": vote_conf,
        }
    
    def _analyze_patterns(self, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict:
        """Pattern analysis for range, breakout, and consolidation detection."""
        # Range detection
        recent_high = np.max(h[-20:])
        recent_low = np.min(l[-20:])
        range_size = (recent_high - recent_low) / recent_low * 100
        
        # Count touches of support/resistance
        touches_high = np.sum(h[-20:] > recent_high * 0.99)
        touches_low = np.sum(l[-20:] < recent_low * 1.01)
        range_bound = touches_high >= 2 and touches_low >= 2 and range_size < 10
        
        # Breakout detection
        prev_high = np.max(h[-30:-5]) if len(h) >= 30 else recent_high
        prev_low = np.min(l[-30:-5]) if len(l) >= 30 else recent_low
        
        breakout_up = c[-1] > prev_high * 1.01
        breakout_down = c[-1] < prev_low * 0.99
        
        # Consolidation (narrowing range)
        if len(h) >= 30:
            early_range = (np.max(h[-30:-15]) - np.min(l[-30:-15])) / np.min(l[-30:-15]) * 100
            late_range = (np.max(h[-15:]) - np.min(l[-15:])) / np.min(l[-15:]) * 100
            narrowing = late_range < early_range * 0.7
        else:
            narrowing = False
        
        # Determine vote
        if breakout_up:
            vote = MarketRegime.BREAKOUT
            vote_conf = 0.8
        elif breakout_down:
            vote = MarketRegime.BREAKOUT
            vote_conf = 0.8
        elif narrowing:
            vote = MarketRegime.CONSOLIDATION
            vote_conf = 0.7
        elif range_bound:
            vote = MarketRegime.RANGING
            vote_conf = 0.75
        else:
            vote = MarketRegime.UNKNOWN
            vote_conf = 0.3
        
        return {
            "range_bound": range_bound,
            "range_size_pct": range_size,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "narrowing": narrowing,
            "touches_high": touches_high,
            "touches_low": touches_low,
            "vote": vote,
            "confidence": vote_conf,
        }
    
    def _analyze_momentum(self, c: np.ndarray) -> Dict:
        """Momentum analysis using RSI-like and rate of change."""
        if len(c) < 20:
            return {
                "score": 0,
                "divergence": False,
                "rsi": 50,
                "vote": MarketRegime.UNKNOWN,
                "confidence": 0.3,
            }
        
        # RSI calculation
        gains = []
        losses = []
        for i in range(1, len(c)):
            change = c[i] - c[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Momentum score (-100 to 100)
        momentum_score = (rsi - 50) * 2
        
        # Check for divergence (price vs momentum)
        price_higher = c[-1] > c[-14] if len(c) >= 14 else False
        recent_rsi = rsi
        older_gains = np.mean(gains[-28:-14]) if len(gains) >= 28 else avg_gain
        older_losses = np.mean(losses[-28:-14]) if len(losses) >= 28 else avg_loss
        
        if older_losses == 0:
            older_rsi = 100
        else:
            older_rs = older_gains / older_losses
            older_rsi = 100 - (100 / (1 + older_rs))
        
        # Bearish divergence: price higher but RSI lower
        # Bullish divergence: price lower but RSI higher
        bearish_div = price_higher and recent_rsi < older_rsi - 5
        bullish_div = not price_higher and recent_rsi > older_rsi + 5
        divergence = bearish_div or bullish_div
        
        # Vote based on RSI extremes
        if rsi > 70:
            vote = MarketRegime.DISTRIBUTION
            vote_conf = min(0.7, (rsi - 70) / 30)
        elif rsi < 30:
            vote = MarketRegime.ACCUMULATION
            vote_conf = min(0.7, (30 - rsi) / 30)
        else:
            vote = MarketRegime.UNKNOWN
            vote_conf = 0.3
        
        return {
            "score": momentum_score,
            "divergence": divergence,
            "bearish_divergence": bearish_div,
            "bullish_divergence": bullish_div,
            "rsi": rsi,
            "vote": vote,
            "confidence": vote_conf,
        }
    
    def _combine_methods(self, methods: Dict) -> Tuple[MarketRegime, float]:
        """Combine all detection method votes into final regime."""
        # Weight each method
        weights = {
            "trend": 0.30,
            "volatility": 0.25,
            "pattern": 0.20,
            "volume": 0.15,
            "momentum": 0.10,
        }
        
        # Collect votes
        votes: Dict[MarketRegime, float] = {}
        
        for method_name, result in methods.items():
            vote = result.get("vote", MarketRegime.UNKNOWN)
            conf = result.get("confidence", 0.5)
            weight = weights.get(method_name, 0.1)
            
            if vote not in votes:
                votes[vote] = 0
            votes[vote] += conf * weight
        
        # Remove UNKNOWN from consideration if other votes exist
        if len(votes) > 1 and MarketRegime.UNKNOWN in votes:
            del votes[MarketRegime.UNKNOWN]
        
        # Get highest voted regime
        if not votes:
            return MarketRegime.UNKNOWN, 0.5
        
        best_regime = max(votes.keys(), key=lambda k: votes[k])
        total_weight = sum(votes.values())
        confidence = votes[best_regime] / total_weight if total_weight > 0 else 0.5
        
        # Override rules for specific conditions
        vol_result = methods.get("volatility", {})
        if vol_result.get("state") == "EXTREME":
            best_regime = MarketRegime.VOLATILE
            confidence = max(confidence, 0.8)
        
        pattern_result = methods.get("pattern", {})
        if pattern_result.get("breakout_up") or pattern_result.get("breakout_down"):
            best_regime = MarketRegime.BREAKOUT
            confidence = max(confidence, 0.75)
        
        return best_regime, min(0.95, confidence)
    
    def _get_recommendations(self, regime: MarketRegime, methods: Dict) -> Dict:
        """Get trading recommendations based on regime."""
        recommendations = {
            MarketRegime.TRENDING_UP: {
                "strategy": "Momentum",
                "size_mult": 1.2,
                "sl_mult": 2.0,
                "tp_mult": 4.0,
            },
            MarketRegime.TRENDING_DOWN: {
                "strategy": "Momentum",
                "size_mult": 1.0,
                "sl_mult": 2.0,
                "tp_mult": 3.0,
            },
            MarketRegime.RANGING: {
                "strategy": "Mean Reversion",
                "size_mult": 0.9,
                "sl_mult": 1.5,
                "tp_mult": 2.0,
            },
            MarketRegime.VOLATILE: {
                "strategy": "Scalping",
                "size_mult": 0.5,
                "sl_mult": 3.0,
                "tp_mult": 5.0,
            },
            MarketRegime.ACCUMULATION: {
                "strategy": "Mean Reversion",
                "size_mult": 0.8,
                "sl_mult": 2.5,
                "tp_mult": 4.0,
            },
            MarketRegime.DISTRIBUTION: {
                "strategy": "Trend Following",
                "size_mult": 0.7,
                "sl_mult": 2.0,
                "tp_mult": 3.0,
            },
            MarketRegime.BREAKOUT: {
                "strategy": "Breakout",
                "size_mult": 1.1,
                "sl_mult": 1.5,
                "tp_mult": 3.0,
            },
            MarketRegime.CONSOLIDATION: {
                "strategy": "Breakout",
                "size_mult": 0.7,
                "sl_mult": 1.5,
                "tp_mult": 2.5,
            },
        }
        
        base = recommendations.get(regime, {
            "strategy": "Momentum",
            "size_mult": 1.0,
            "sl_mult": 2.0,
            "tp_mult": 3.0,
        })
        
        # Adjust for volume confirmation
        vol_result = methods.get("volume", {})
        if not vol_result.get("confirms_price", True):
            base["size_mult"] *= 0.8
        
        return base
    
    def _compile_metrics(self, methods: Dict) -> Dict:
        """Compile all metrics for Claude context."""
        metrics = {}
        
        for method_name, result in methods.items():
            for key, value in result.items():
                if key not in ("vote", "confidence"):
                    if isinstance(value, (int, float)):
                        metrics[f"{method_name}_{key}"] = round(value, 2) if isinstance(value, float) else value
                    else:
                        metrics[f"{method_name}_{key}"] = value
        
        return metrics


# Singleton instance
_advanced_detector: Optional[AdvancedRegimeDetector] = None


def get_advanced_detector() -> AdvancedRegimeDetector:
    """Get the singleton advanced regime detector."""
    global _advanced_detector
    if _advanced_detector is None:
        _advanced_detector = AdvancedRegimeDetector()
    return _advanced_detector


def detect_regime_advanced(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 50
) -> Optional[RegimeAnalysis]:
    """
    Convenience function for advanced regime detection.
    
    This is the recommended entry point for regime detection.
    """
    detector = get_advanced_detector()
    return detector.detect(high, low, close, volume, lookback)
