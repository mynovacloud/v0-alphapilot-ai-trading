"""
Market Intelligence Module - Makes the bot smarter than average humans.

This module adds the "trader's intuition" that humans have:
1. Cross-asset correlation (BTC leads alts, macro awareness)
2. Sentiment analysis (fear/greed, funding rates)  
3. Liquidity analysis (spread, volume profile)
4. Smart entry timing (wait for pullbacks, support levels)
5. News/event awareness (earnings, upgrades, macro)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.helpers import utcnow

logger = get_logger(__name__)


# =============================================================================
# MARKET INTELLIGENCE DATA STRUCTURES
# =============================================================================

@dataclass
class MarketContext:
    """Overall market context that affects all trades."""
    btc_trend: str = "neutral"  # bullish, bearish, neutral
    btc_strength: float = 0.0   # -1 to 1
    market_fear_greed: float = 50.0  # 0-100 (0=extreme fear, 100=extreme greed)
    total_market_momentum: float = 0.0  # -1 to 1
    correlation_regime: str = "normal"  # normal, high_correlation, divergence
    volatility_regime: str = "normal"  # low, normal, high, extreme
    recommended_exposure: float = 1.0  # 0-1, how much of normal size to use
    avoid_longs: bool = False
    avoid_shorts: bool = False
    reasoning: str = ""


@dataclass  
class LiquidityAnalysis:
    """Liquidity metrics for a specific symbol."""
    spread_pct: float = 0.0  # bid-ask spread as percentage
    volume_24h: float = 0.0
    volume_trend: str = "stable"  # increasing, stable, decreasing
    liquidity_score: float = 1.0  # 0-1, higher = more liquid
    is_thin: bool = False  # True if liquidity is dangerously thin
    slippage_estimate: float = 0.0  # Estimated slippage for typical order


@dataclass
class EntryTiming:
    """Smart entry timing analysis."""
    should_wait: bool = False
    wait_reason: str = ""
    suggested_entry: Optional[float] = None
    distance_from_support: float = 0.0  # % above nearest support
    distance_from_resistance: float = 0.0  # % below nearest resistance
    pullback_opportunity: bool = False
    entry_quality: str = "fair"  # excellent, good, fair, poor
    

@dataclass
class SymbolIntelligence:
    """Complete intelligence for a trading decision."""
    symbol: str
    market_context: MarketContext
    liquidity: LiquidityAnalysis
    entry_timing: EntryTiming
    correlation_to_btc: float = 0.0  # -1 to 1
    sector: str = "unknown"  # defi, layer1, meme, etc.
    sentiment_score: float = 0.5  # 0-1
    should_trade: bool = True
    skip_reason: str = ""
    confidence_adjustment: float = 1.0  # Multiplier for signal confidence
    size_adjustment: float = 1.0  # Multiplier for position size


# =============================================================================
# CROSS-ASSET CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """
    Analyzes how assets move together.
    Key insight: When BTC dumps, alts dump harder. When BTC pumps, alts may lag.
    """
    
    # Known correlations (would be dynamically calculated in production)
    SECTOR_CORRELATIONS = {
        "layer1": {"btc_correlation": 0.85, "eth_correlation": 0.90},
        "defi": {"btc_correlation": 0.75, "eth_correlation": 0.95},
        "meme": {"btc_correlation": 0.60, "eth_correlation": 0.70},
        "layer2": {"btc_correlation": 0.80, "eth_correlation": 0.92},
        "gaming": {"btc_correlation": 0.65, "eth_correlation": 0.75},
        "ai": {"btc_correlation": 0.70, "eth_correlation": 0.80},
    }
    
    SYMBOL_SECTORS = {
        "BTC-USD": "layer1", "ETH-USD": "layer1", "SOL-USD": "layer1",
        "ADA-USD": "layer1", "AVAX-USD": "layer1", "DOT-USD": "layer1",
        "LINK-USD": "defi", "UNI-USD": "defi", "AAVE-USD": "defi",
        "DOGE-USD": "meme", "SHIB-USD": "meme", "PEPE-USD": "meme",
        "ARB-USD": "layer2", "OP-USD": "layer2", "MATIC-USD": "layer2",
        "IMX-USD": "gaming", "GALA-USD": "gaming", "AXS-USD": "gaming",
        "FET-USD": "ai", "RNDR-USD": "ai", "AGIX-USD": "ai",
    }
    
    def __init__(self):
        self._btc_prices: list[tuple[float, float]] = []  # (timestamp, price)
        self._eth_prices: list[tuple[float, float]] = []
        self._last_btc_trend: str = "neutral"
        self._last_btc_change: float = 0.0
    
    def update_btc_price(self, price: float):
        """Track BTC price for trend analysis."""
        now = time.time()
        self._btc_prices.append((now, price))
        # Keep last 100 prices
        if len(self._btc_prices) > 100:
            self._btc_prices = self._btc_prices[-100:]
        self._calculate_btc_trend()
    
    def update_eth_price(self, price: float):
        """Track ETH price."""
        now = time.time()
        self._eth_prices.append((now, price))
        if len(self._eth_prices) > 100:
            self._eth_prices = self._eth_prices[-100:]
    
    def _calculate_btc_trend(self):
        """Calculate BTC trend from recent prices."""
        if len(self._btc_prices) < 10:
            return
        
        recent = [p for _, p in self._btc_prices[-10:]]
        older = [p for _, p in self._btc_prices[-20:-10]] if len(self._btc_prices) >= 20 else recent
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        self._last_btc_change = change
        
        if change > 0.02:
            self._last_btc_trend = "bullish"
        elif change < -0.02:
            self._last_btc_trend = "bearish"
        else:
            self._last_btc_trend = "neutral"
    
    def get_sector(self, symbol: str) -> str:
        """Get the sector for a symbol."""
        return self.SYMBOL_SECTORS.get(symbol, "unknown")
    
    def get_btc_correlation(self, symbol: str) -> float:
        """Get estimated correlation to BTC."""
        sector = self.get_sector(symbol)
        if sector in self.SECTOR_CORRELATIONS:
            return self.SECTOR_CORRELATIONS[sector]["btc_correlation"]
        return 0.75  # Default moderate correlation
    
    def should_avoid_alt_longs(self) -> tuple[bool, str]:
        """
        Key insight: Don't buy alts when BTC is dumping.
        Alts typically dump 1.5-2x harder than BTC.
        """
        if self._last_btc_trend == "bearish" and self._last_btc_change < -0.03:
            return True, f"BTC dumping ({self._last_btc_change:.1%}), alts will dump harder"
        return False, ""
    
    def should_avoid_alt_shorts(self) -> tuple[bool, str]:
        """Don't short alts when BTC is pumping hard."""
        if self._last_btc_trend == "bullish" and self._last_btc_change > 0.05:
            return True, f"BTC pumping ({self._last_btc_change:.1%}), alts may follow"
        return False, ""
    
    def get_btc_context(self) -> tuple[str, float]:
        """Get current BTC trend and strength."""
        return self._last_btc_trend, self._last_btc_change


# =============================================================================
# LIQUIDITY ANALYZER
# =============================================================================

class LiquidityAnalyzer:
    """
    Analyzes liquidity to avoid getting stuck in illiquid positions.
    """
    
    # Minimum volume thresholds (24h volume in USD)
    MIN_VOLUME_TIER1 = 100_000_000  # $100M+ = very liquid
    MIN_VOLUME_TIER2 = 10_000_000   # $10M+ = liquid
    MIN_VOLUME_TIER3 = 1_000_000    # $1M+ = acceptable
    MIN_VOLUME_DANGER = 500_000     # Below $500K = danger zone
    
    def analyze(self, symbol: str, volume_24h: float, bid: float, ask: float) -> LiquidityAnalysis:
        """Analyze liquidity for a symbol."""
        spread_pct = (ask - bid) / bid if bid > 0 else 0
        
        # Determine liquidity score
        if volume_24h >= self.MIN_VOLUME_TIER1:
            liquidity_score = 1.0
            volume_trend = "strong"
        elif volume_24h >= self.MIN_VOLUME_TIER2:
            liquidity_score = 0.8
            volume_trend = "stable"
        elif volume_24h >= self.MIN_VOLUME_TIER3:
            liquidity_score = 0.6
            volume_trend = "moderate"
        elif volume_24h >= self.MIN_VOLUME_DANGER:
            liquidity_score = 0.3
            volume_trend = "weak"
        else:
            liquidity_score = 0.1
            volume_trend = "dangerous"
        
        # Penalize wide spreads
        if spread_pct > 0.005:  # 0.5% spread
            liquidity_score *= 0.7
        elif spread_pct > 0.002:  # 0.2% spread
            liquidity_score *= 0.9
        
        is_thin = liquidity_score < 0.3 or spread_pct > 0.01
        
        # Estimate slippage (rough approximation)
        slippage_estimate = spread_pct / 2 + (1 - liquidity_score) * 0.005
        
        return LiquidityAnalysis(
            spread_pct=spread_pct,
            volume_24h=volume_24h,
            volume_trend=volume_trend,
            liquidity_score=liquidity_score,
            is_thin=is_thin,
            slippage_estimate=slippage_estimate,
        )


# =============================================================================
# SMART ENTRY TIMING
# =============================================================================

class EntryTimingAnalyzer:
    """
    Determines optimal entry timing instead of buying immediately.
    Key insight: Waiting for pullbacks improves average entry price.
    """
    
    def analyze(
        self,
        symbol: str,
        current_price: float,
        prices: list[float],  # Recent price history
        side: str,  # BUY or SELL
    ) -> EntryTiming:
        """Analyze entry timing for a potential trade."""
        if len(prices) < 20:
            return EntryTiming(entry_quality="fair")
        
        # Calculate support/resistance from recent prices
        recent_low = min(prices[-20:])
        recent_high = max(prices[-20:])
        price_range = recent_high - recent_low
        
        if price_range == 0:
            return EntryTiming(entry_quality="fair")
        
        # Distance from support/resistance
        dist_from_support = (current_price - recent_low) / price_range
        dist_from_resistance = (recent_high - current_price) / price_range
        
        # For BUY: prefer entries near support
        # For SELL: prefer entries near resistance
        result = EntryTiming(
            distance_from_support=dist_from_support,
            distance_from_resistance=dist_from_resistance,
        )
        
        if side == "BUY":
            if dist_from_support < 0.2:
                result.entry_quality = "excellent"
                result.suggested_entry = current_price
            elif dist_from_support < 0.4:
                result.entry_quality = "good"
                result.suggested_entry = current_price
            elif dist_from_support > 0.8:
                # Price near highs, wait for pullback
                result.should_wait = True
                result.wait_reason = f"Price near resistance ({dist_from_resistance:.0%} away), wait for pullback"
                result.entry_quality = "poor"
                result.suggested_entry = recent_low + price_range * 0.5
                result.pullback_opportunity = True
            else:
                result.entry_quality = "fair"
                result.suggested_entry = current_price
        
        elif side == "SELL":
            if dist_from_resistance < 0.2:
                result.entry_quality = "excellent"
                result.suggested_entry = current_price
            elif dist_from_resistance < 0.4:
                result.entry_quality = "good"
                result.suggested_entry = current_price
            elif dist_from_resistance > 0.8:
                result.should_wait = True
                result.wait_reason = f"Price near support ({dist_from_support:.0%} away), wait for bounce"
                result.entry_quality = "poor"
                result.suggested_entry = recent_high - price_range * 0.5
                result.pullback_opportunity = True
            else:
                result.entry_quality = "fair"
                result.suggested_entry = current_price
        
        return result


# =============================================================================
# SENTIMENT ANALYZER (Simplified - would connect to APIs in production)
# =============================================================================

class SentimentAnalyzer:
    """
    Tracks market sentiment from various sources.
    In production, this would connect to:
    - Fear & Greed Index API
    - Funding rates from exchanges
    - Social sentiment APIs
    """
    
    def __init__(self):
        self._fear_greed: float = 50.0
        self._funding_rates: dict[str, float] = {}
        self._last_update: float = 0
    
    def get_fear_greed(self) -> float:
        """Get current fear/greed index (0-100)."""
        # In production, fetch from alternative.me API
        # For now, estimate from recent BTC volatility
        return self._fear_greed
    
    def set_fear_greed(self, value: float):
        """Update fear/greed (would be called by a background job)."""
        self._fear_greed = max(0, min(100, value))
        self._last_update = time.time()
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for a symbol."""
        return self._funding_rates.get(symbol, 0.0)
    
    def set_funding_rate(self, symbol: str, rate: float):
        """Update funding rate."""
        self._funding_rates[symbol] = rate
    
    def interpret_sentiment(self) -> tuple[str, float]:
        """
        Interpret overall sentiment.
        Returns (sentiment_label, confidence_adjustment)
        """
        fg = self._fear_greed
        
        if fg < 20:
            # Extreme fear = contrarian buy opportunity
            return "extreme_fear", 1.2
        elif fg < 35:
            return "fear", 1.1
        elif fg < 65:
            return "neutral", 1.0
        elif fg < 80:
            return "greed", 0.9
        else:
            # Extreme greed = be cautious
            return "extreme_greed", 0.7


# =============================================================================
# MAIN MARKET INTELLIGENCE CLASS
# =============================================================================

class MarketIntelligence:
    """
    Central hub for all market intelligence.
    Makes the bot "smarter" by considering factors humans naturally consider.
    """
    
    def __init__(self):
        self.correlation = CorrelationAnalyzer()
        self.liquidity = LiquidityAnalyzer()
        self.entry_timing = EntryTimingAnalyzer()
        self.sentiment = SentimentAnalyzer()
        self._price_cache: dict[str, list[float]] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update price cache for a symbol."""
        if symbol not in self._price_cache:
            self._price_cache[symbol] = []
        self._price_cache[symbol].append(price)
        # Keep last 100 prices
        if len(self._price_cache[symbol]) > 100:
            self._price_cache[symbol] = self._price_cache[symbol][-100:]
        
        # Update BTC/ETH trackers
        if symbol == "BTC-USD":
            self.correlation.update_btc_price(price)
        elif symbol == "ETH-USD":
            self.correlation.update_eth_price(price)
    
    def analyze_trade_opportunity(
        self,
        symbol: str,
        current_price: float,
        side: str,
        volume_24h: float = 0,
        bid: float = 0,
        ask: float = 0,
    ) -> SymbolIntelligence:
        """
        Complete intelligence analysis for a trading opportunity.
        This is what makes the bot smarter than average.
        """
        result = SymbolIntelligence(
            symbol=symbol,
            market_context=MarketContext(),
            liquidity=LiquidityAnalysis(),
            entry_timing=EntryTiming(),
        )
        
        # 1. Cross-asset correlation check
        btc_trend, btc_change = self.correlation.get_btc_context()
        result.market_context.btc_trend = btc_trend
        result.market_context.btc_strength = btc_change
        result.correlation_to_btc = self.correlation.get_btc_correlation(symbol)
        result.sector = self.correlation.get_sector(symbol)
        
        # Check if we should avoid this trade due to BTC movement
        if side == "BUY" and symbol != "BTC-USD":
            avoid, reason = self.correlation.should_avoid_alt_longs()
            if avoid:
                result.should_trade = False
                result.skip_reason = reason
                result.confidence_adjustment = 0.5
        elif side == "SELL" and symbol != "BTC-USD":
            avoid, reason = self.correlation.should_avoid_alt_shorts()
            if avoid:
                result.should_trade = False
                result.skip_reason = reason
                result.confidence_adjustment = 0.5
        
        # 2. Liquidity analysis
        if bid > 0 and ask > 0:
            result.liquidity = self.liquidity.analyze(symbol, volume_24h, bid, ask)
            if result.liquidity.is_thin:
                result.should_trade = False
                result.skip_reason = f"Liquidity too thin (score={result.liquidity.liquidity_score:.2f})"
                result.size_adjustment = 0.5
        
        # 3. Entry timing
        prices = self._price_cache.get(symbol, [])
        if prices:
            result.entry_timing = self.entry_timing.analyze(symbol, current_price, prices, side)
            
            # Adjust confidence based on entry quality
            quality_adjustments = {
                "excellent": 1.15,
                "good": 1.05,
                "fair": 1.0,
                "poor": 0.8,
            }
            result.confidence_adjustment *= quality_adjustments.get(result.entry_timing.entry_quality, 1.0)
            
            # If entry timing suggests waiting, reduce size but don't skip
            if result.entry_timing.should_wait:
                result.size_adjustment *= 0.5
                result.market_context.reasoning += f" Entry timing: {result.entry_timing.wait_reason}."
        
        # 4. Sentiment adjustment
        sentiment_label, sentiment_adj = self.sentiment.interpret_sentiment()
        result.sentiment_score = self.sentiment.get_fear_greed() / 100
        result.confidence_adjustment *= sentiment_adj
        
        # Contrarian logic: extreme fear = increase buy confidence
        if sentiment_label == "extreme_fear" and side == "BUY":
            result.confidence_adjustment *= 1.1
            result.market_context.reasoning += " Extreme fear detected - contrarian buy opportunity."
        elif sentiment_label == "extreme_greed" and side == "BUY":
            result.size_adjustment *= 0.7
            result.market_context.reasoning += " Extreme greed detected - reducing position size."
        
        # 5. Calculate recommended exposure
        result.market_context.recommended_exposure = min(1.0, result.confidence_adjustment * result.size_adjustment)
        
        return result
    
    def get_market_context(self) -> MarketContext:
        """Get overall market context."""
        btc_trend, btc_change = self.correlation.get_btc_context()
        sentiment_label, _ = self.sentiment.interpret_sentiment()
        
        ctx = MarketContext(
            btc_trend=btc_trend,
            btc_strength=btc_change,
            market_fear_greed=self.sentiment.get_fear_greed(),
        )
        
        # Determine if we should avoid longs/shorts
        if btc_trend == "bearish" and btc_change < -0.05:
            ctx.avoid_longs = True
            ctx.recommended_exposure = 0.5
            ctx.reasoning = "BTC in strong downtrend - reduce long exposure"
        elif btc_trend == "bullish" and btc_change > 0.05:
            ctx.avoid_shorts = True
            ctx.reasoning = "BTC in strong uptrend - avoid shorts"
        
        if sentiment_label == "extreme_greed":
            ctx.recommended_exposure *= 0.7
            ctx.reasoning += " Market in extreme greed - be cautious."
        elif sentiment_label == "extreme_fear":
            ctx.reasoning += " Market in extreme fear - contrarian opportunities exist."
        
        return ctx


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_market_intelligence: Optional[MarketIntelligence] = None


def get_market_intelligence() -> MarketIntelligence:
    """Get or create the MarketIntelligence singleton."""
    global _market_intelligence
    if _market_intelligence is None:
        _market_intelligence = MarketIntelligence()
    return _market_intelligence
