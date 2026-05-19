"""
Advanced Signal Engine
======================

A sophisticated multi-factor signal generation system that combines:
1. Multiple technical indicators with dynamic weighting
2. Market regime detection for context-aware signals
3. Volume analysis for confirmation
4. Multi-timeframe confluence
5. Machine learning-inspired pattern recognition

This engine determines:
- WHICH crypto to trade (signal strength ranking)
- WHETHER to trade (quality filters)
- WHAT DIRECTION (BUY/SELL based on consensus)

The engine is designed to minimize losing trades by:
- Requiring multiple confirming indicators
- Detecting adverse market conditions
- Filtering out low-quality setups
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from connectors.candles import get_candles
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalQuality(str, Enum):
    """Quality grade of a trading signal."""
    A_PLUS = "A+"    # Highest conviction - multiple strong confirmations
    A = "A"          # Strong signal - multiple confirmations
    B = "B"          # Good signal - decent confirmations
    C = "C"          # Marginal signal - weak confirmations
    F = "F"          # No trade - conflicting or weak


@dataclass
class AdvancedSignal:
    """Rich signal with full context for decision making."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    quality: SignalQuality
    
    # Component scores (each 0-100)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    pattern_score: float = 0.0
    
    # Risk metrics
    suggested_stop_pct: float = 0.05
    suggested_target_pct: float = 0.10
    risk_reward_ratio: float = 2.0
    
    # Context
    regime: str = "UNKNOWN"
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Raw indicator values for transparency
    indicators: Dict[str, float] = field(default_factory=dict)


class AdvancedSignalEngine:
    """
    Multi-factor signal generation engine.
    
    Philosophy:
    - Quality over quantity: Fewer, higher-conviction trades
    - Multiple confirmations required: No single indicator can trigger a trade
    - Context-aware: Different strategies for different market conditions
    - Risk-first: Always know the stop and target before entering
    """
    
    def __init__(self):
        # Indicator periods (tuned for crypto's 24/7 market)
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 55  # Longer-term trend
        self.rsi_period = 14
        self.atr_period = 14
        self.bb_period = 20
        self.volume_ma_period = 20
        
        # Signal thresholds
        self.min_volume_ratio = 0.8  # Min volume vs average
        self.min_confirmations = 3   # Minimum confirming factors
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def analyze(self, symbol: str, candles: List[Dict]) -> AdvancedSignal:
        """
        Perform comprehensive analysis on a symbol.
        
        Returns an AdvancedSignal with action, confidence, and full context.
        """
        if not candles or len(candles) < 60:
            return self._no_signal(symbol, "Insufficient data")
        
        # Extract price data
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c.get("volume", 0)) for c in candles]
        
        current_price = closes[-1]
        
        # Calculate all indicators
        indicators = self._calculate_indicators(closes, highs, lows, volumes)
        
        # Score each dimension
        trend_score = self._score_trend(indicators, closes)
        momentum_score = self._score_momentum(indicators)
        volatility_score = self._score_volatility(indicators)
        volume_score = self._score_volume(indicators, volumes)
        pattern_score = self._score_patterns(closes, highs, lows)
        
        # Determine market regime
        regime = self._detect_regime(indicators, closes)
        
        # Calculate directional bias
        bullish_factors = []
        bearish_factors = []
        
        # Trend factors
        if indicators["ema_fast"] > indicators["ema_slow"]:
            bullish_factors.append("EMA bullish cross")
        else:
            bearish_factors.append("EMA bearish cross")
            
        if current_price > indicators["ema_trend"]:
            bullish_factors.append("Above long-term trend")
        else:
            bearish_factors.append("Below long-term trend")
        
        # Momentum factors
        if indicators["rsi"] < self.rsi_oversold:
            bullish_factors.append(f"RSI oversold ({indicators['rsi']:.0f})")
        elif indicators["rsi"] > self.rsi_overbought:
            bearish_factors.append(f"RSI overbought ({indicators['rsi']:.0f})")
        
        if indicators["macd_histogram"] > 0:
            bullish_factors.append("MACD bullish")
        else:
            bearish_factors.append("MACD bearish")
            
        if indicators["macd_histogram"] > indicators.get("macd_histogram_prev", 0):
            bullish_factors.append("MACD momentum increasing")
        else:
            bearish_factors.append("MACD momentum decreasing")
        
        # Price action factors
        if current_price > indicators["bb_upper"] * 0.98:
            bearish_factors.append("Near upper Bollinger")
        elif current_price < indicators["bb_lower"] * 1.02:
            bullish_factors.append("Near lower Bollinger")
            
        # Volume confirmation
        if indicators["volume_ratio"] > 1.2:
            if len(bullish_factors) > len(bearish_factors):
                bullish_factors.append("Volume confirming")
            elif len(bearish_factors) > len(bullish_factors):
                bearish_factors.append("Volume confirming")
        
        # Trend strength
        if indicators["adx"] > 25:
            if indicators["plus_di"] > indicators["minus_di"]:
                bullish_factors.append(f"Strong uptrend (ADX={indicators['adx']:.0f})")
            else:
                bearish_factors.append(f"Strong downtrend (ADX={indicators['adx']:.0f})")
        
        # Calculate final direction and confidence
        bull_count = len(bullish_factors)
        bear_count = len(bearish_factors)
        total_factors = bull_count + bear_count
        
        if total_factors == 0:
            return self._no_signal(symbol, "No clear factors")
        
        # Determine action
        if bull_count >= self.min_confirmations and bull_count > bear_count + 1:
            action = "BUY"
            alignment = bull_count / total_factors
            key_factors = bullish_factors
        elif bear_count >= self.min_confirmations and bear_count > bull_count + 1:
            action = "SELL"
            alignment = bear_count / total_factors
            key_factors = bearish_factors
        else:
            return self._no_signal(symbol, f"Conflicting signals: {bull_count} bull vs {bear_count} bear")
        
        # Calculate confidence (0-1)
        # Weighted average of component scores, adjusted by factor alignment
        raw_confidence = (
            trend_score * 0.25 +
            momentum_score * 0.25 +
            volatility_score * 0.15 +
            volume_score * 0.20 +
            pattern_score * 0.15
        ) / 100
        
        # Boost confidence for strong alignment, penalize for conflict
        confidence = raw_confidence * (0.7 + alignment * 0.3)
        confidence = min(0.95, max(0.30, confidence))
        
        # Quality grade
        if confidence >= 0.75 and bull_count >= 5:
            quality = SignalQuality.A_PLUS
        elif confidence >= 0.65 and bull_count >= 4:
            quality = SignalQuality.A
        elif confidence >= 0.55 and bull_count >= 3:
            quality = SignalQuality.B
        else:
            quality = SignalQuality.C
        
        # Calculate dynamic stops based on ATR
        atr = indicators["atr"]
        atr_pct = atr / current_price
        
        # Wider stops in volatile markets, tighter in calm markets
        stop_multiplier = 2.0 if regime in ["VOLATILE", "TRENDING_UP", "TRENDING_DOWN"] else 1.5
        target_multiplier = stop_multiplier * 2.0  # Always maintain 2:1 R:R minimum
        
        suggested_stop = min(0.10, max(0.03, atr_pct * stop_multiplier))
        suggested_target = min(0.20, max(0.06, atr_pct * target_multiplier))
        
        # Build warnings
        warnings = []
        if volume_score < 40:
            warnings.append("Low volume - signal may be weak")
        if indicators["atr_percentile"] > 80:
            warnings.append("High volatility - use wider stops")
        if regime == "RANGING" and action == "BUY" and current_price > indicators["sma_20"]:
            warnings.append("Buying high in ranging market")
        
        return AdvancedSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            quality=quality,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            volume_score=volume_score,
            pattern_score=pattern_score,
            suggested_stop_pct=suggested_stop,
            suggested_target_pct=suggested_target,
            risk_reward_ratio=suggested_target / suggested_stop if suggested_stop > 0 else 2.0,
            regime=regime,
            reasoning=f"{action} signal with {bull_count if action == 'BUY' else bear_count} confirming factors",
            key_factors=key_factors[:5],  # Top 5 factors
            warnings=warnings,
            indicators=indicators,
        )
    
    def _calculate_indicators(self, closes: List[float], highs: List[float], 
                              lows: List[float], volumes: List[float]) -> Dict[str, float]:
        """Calculate all technical indicators."""
        n = len(closes)
        
        # EMAs
        ema_fast = self._ema(closes, self.ema_fast)[-1]
        ema_slow = self._ema(closes, self.ema_slow)[-1]
        ema_trend = self._ema(closes, self.ema_trend)[-1]
        
        # SMA
        sma_20 = sum(closes[-20:]) / 20 if n >= 20 else closes[-1]
        
        # RSI
        rsi = self._rsi(closes, self.rsi_period)
        
        # MACD
        macd_line = self._ema(closes, 12)[-1] - self._ema(closes, 26)[-1]
        signal_line = self._ema([self._ema(closes, 12)[i] - self._ema(closes, 26)[i] 
                                 for i in range(n)], 9)[-1]
        macd_histogram = macd_line - signal_line
        macd_histogram_prev = 0
        if n > 1:
            macd_prev = self._ema(closes[:-1], 12)[-1] - self._ema(closes[:-1], 26)[-1]
            signal_prev = self._ema([self._ema(closes[:-1], 12)[i] - self._ema(closes[:-1], 26)[i] 
                                     for i in range(n-1)], 9)[-1]
            macd_histogram_prev = macd_prev - signal_prev
        
        # ATR
        atr = self._atr(closes, highs, lows, self.atr_period)
        atr_history = [self._atr(closes[:i+1], highs[:i+1], lows[:i+1], self.atr_period) 
                       for i in range(max(20, self.atr_period), n)]
        atr_percentile = self._percentile_rank(atr, atr_history) if atr_history else 50
        
        # Bollinger Bands
        bb_sma = sma_20
        bb_std = self._std(closes[-20:]) if n >= 20 else 0
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_sma if bb_sma > 0 else 0
        
        # ADX and DI
        adx, plus_di, minus_di = self._adx(closes, highs, lows, 14)
        
        # Volume
        avg_volume = sum(volumes[-self.volume_ma_period:]) / self.volume_ma_period if volumes else 0
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "sma_20": sma_20,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_histogram,
            "macd_histogram_prev": macd_histogram_prev,
            "atr": atr,
            "atr_percentile": atr_percentile,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "volume_ratio": volume_ratio,
        }
    
    def _score_trend(self, ind: Dict, closes: List[float]) -> float:
        """Score trend strength and alignment (0-100)."""
        score = 50  # Neutral
        
        # EMA alignment
        if ind["ema_fast"] > ind["ema_slow"] > ind["ema_trend"]:
            score += 20  # Perfect bullish alignment
        elif ind["ema_fast"] < ind["ema_slow"] < ind["ema_trend"]:
            score += 20  # Perfect bearish alignment (still tradeable)
        elif ind["ema_fast"] > ind["ema_slow"]:
            score += 10
        
        # Price vs trend
        current = closes[-1]
        if current > ind["ema_trend"]:
            score += 10
        
        # ADX for trend strength
        if ind["adx"] > 40:
            score += 15
        elif ind["adx"] > 25:
            score += 10
        elif ind["adx"] < 15:
            score -= 10
        
        return min(100, max(0, score))
    
    def _score_momentum(self, ind: Dict) -> float:
        """Score momentum indicators (0-100)."""
        score = 50
        
        # RSI
        if 40 <= ind["rsi"] <= 60:
            score += 10  # Neutral zone, room to move
        elif ind["rsi"] < 30 or ind["rsi"] > 70:
            score += 15  # Extremes can be good entry points
        
        # MACD
        if ind["macd_histogram"] > 0 and ind["macd_histogram"] > ind["macd_histogram_prev"]:
            score += 20  # Bullish and increasing
        elif ind["macd_histogram"] < 0 and ind["macd_histogram"] < ind["macd_histogram_prev"]:
            score += 20  # Bearish and increasing (for shorts)
        elif abs(ind["macd_histogram"]) > abs(ind["macd_histogram_prev"]):
            score += 10  # Momentum building
        
        # DI crossover
        di_diff = abs(ind["plus_di"] - ind["minus_di"])
        if di_diff > 20:
            score += 10
        
        return min(100, max(0, score))
    
    def _score_volatility(self, ind: Dict) -> float:
        """Score volatility conditions (0-100). Higher = better for trading."""
        score = 50
        
        # ATR percentile - moderate volatility is ideal
        if 30 <= ind["atr_percentile"] <= 70:
            score += 20  # Ideal volatility
        elif ind["atr_percentile"] < 20:
            score -= 10  # Too quiet
        elif ind["atr_percentile"] > 85:
            score -= 10  # Too volatile
        
        # Bollinger width
        if 0.03 <= ind["bb_width"] <= 0.08:
            score += 15  # Healthy volatility
        elif ind["bb_width"] < 0.02:
            score -= 5  # Squeeze, may breakout
        
        return min(100, max(0, score))
    
    def _score_volume(self, ind: Dict, volumes: List[float]) -> float:
        """Score volume confirmation (0-100)."""
        score = 50
        
        # Current volume vs average
        if ind["volume_ratio"] > 1.5:
            score += 25  # High volume confirmation
        elif ind["volume_ratio"] > 1.0:
            score += 15
        elif ind["volume_ratio"] < 0.5:
            score -= 20  # Low volume warning
        
        # Volume trend (last 5 candles)
        if len(volumes) >= 5:
            recent = volumes[-5:]
            if recent[-1] > recent[0]:
                score += 10  # Increasing volume
        
        return min(100, max(0, score))
    
    def _score_patterns(self, closes: List[float], highs: List[float], lows: List[float]) -> float:
        """Score price patterns (0-100)."""
        score = 50
        
        if len(closes) < 10:
            return score
        
        # Higher highs / Higher lows (uptrend)
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        
        hh = all(recent_highs[i] >= recent_highs[i-1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] >= recent_lows[i-1] for i in range(1, len(recent_lows)))
        ll = all(recent_lows[i] <= recent_lows[i-1] for i in range(1, len(recent_lows)))
        lh = all(recent_highs[i] <= recent_highs[i-1] for i in range(1, len(recent_highs)))
        
        if hh and hl:
            score += 20  # Clear uptrend structure
        elif ll and lh:
            score += 20  # Clear downtrend structure
        
        # Support/resistance touches
        price_range = max(highs[-20:]) - min(lows[-20:])
        if price_range > 0:
            current = closes[-1]
            range_position = (current - min(lows[-20:])) / price_range
            
            if range_position < 0.2 or range_position > 0.8:
                score += 10  # Near range extreme
        
        return min(100, max(0, score))
    
    def _detect_regime(self, ind: Dict, closes: List[float]) -> str:
        """Detect current market regime."""
        adx = ind["adx"]
        atr_pct = ind["atr_percentile"]
        
        if adx > 30:
            if ind["plus_di"] > ind["minus_di"]:
                return "TRENDING_UP"
            else:
                return "TRENDING_DOWN"
        elif atr_pct > 75:
            return "VOLATILE"
        elif atr_pct < 25 and adx < 20:
            return "CONSOLIDATION"
        else:
            return "RANGING"
    
    def _no_signal(self, symbol: str, reason: str) -> AdvancedSignal:
        """Return a HOLD signal."""
        return AdvancedSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            quality=SignalQuality.F,
            reasoning=reason,
        )
    
    # --- Helper functions ---
    
    def _ema(self, values: List[float], period: int) -> List[float]:
        """Exponential moving average."""
        if not values or period <= 1:
            return list(values)
        k = 2.0 / (period + 1.0)
        out = []
        prev = values[0]
        for v in values:
            prev = v * k + prev * (1.0 - k)
            out.append(prev)
        return out
    
    def _rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, closes: List[float], highs: List[float], lows: List[float], period: int) -> float:
        """Average True Range."""
        if len(closes) < period + 1:
            return 0.0
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        return sum(trs[-period:]) / period if trs else 0.0
    
    def _adx(self, closes: List[float], highs: List[float], lows: List[float], period: int) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, -DI."""
        if len(closes) < period + 1:
            return 20.0, 25.0, 25.0  # Default neutral values
        
        plus_dm, minus_dm, tr = [], [], []
        
        for i in range(1, len(closes)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        atr = sum(tr[-period:]) / period if tr else 1.0
        plus_di = 100 * sum(plus_dm[-period:]) / period / atr if atr > 0 else 25
        minus_di = 100 * sum(minus_dm[-period:]) / period / atr if atr > 0 else 25
        
        dx_list = []
        for i in range(period, len(plus_dm)):
            p = sum(plus_dm[i-period+1:i+1]) / period
            m = sum(minus_dm[i-period+1:i+1]) / period
            if p + m > 0:
                dx_list.append(100 * abs(p - m) / (p + m))
        
        adx = sum(dx_list[-period:]) / period if dx_list else 20.0
        
        return adx, plus_di, minus_di
    
    def _std(self, values: List[float]) -> float:
        """Standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _percentile_rank(self, value: float, history: List[float]) -> float:
        """Calculate percentile rank of value within history."""
        if not history:
            return 50.0
        below = sum(1 for v in history if v < value)
        return 100 * below / len(history)


# Global instance
_signal_engine: Optional[AdvancedSignalEngine] = None


def get_signal_engine() -> AdvancedSignalEngine:
    """Get or create the global signal engine instance."""
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = AdvancedSignalEngine()
    return _signal_engine


def analyze_symbol(symbol: str) -> AdvancedSignal:
    """Convenience function to analyze a symbol."""
    engine = get_signal_engine()
    candles = get_candles(symbol, granularity="FIFTEEN_MINUTE", limit=100)
    return engine.analyze(symbol, candles if candles else [])
