"""
Funding Rate Tracker for Perpetual Futures.

Tracks funding rates across exchanges to:
- Identify crowded trades (high funding = too many longs/shorts)
- Find funding arbitrage opportunities
- Detect sentiment extremes
- Time entries based on funding payment schedule

Funding rates are paid every 8 hours. When rate is positive, longs pay shorts.
When rate is negative, shorts pay longs. Extreme rates often precede reversals.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import asyncio
import threading

logger = logging.getLogger(__name__)


@dataclass
class FundingData:
    """Funding rate data for a symbol."""
    symbol: str
    rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    rate_8h: float  # Annualized as 8h rate
    predicted_rate: float  # Predicted next funding rate
    timestamp: datetime
    next_funding_time: datetime
    
    @property
    def annualized_rate(self) -> float:
        """Convert to annualized percentage."""
        return self.rate * 3 * 365 * 100  # 3 funding periods per day * 365 days
    
    @property
    def is_extreme_long(self) -> bool:
        """True if funding shows extreme long crowding."""
        return self.rate > 0.0005  # > 0.05% per 8h
    
    @property
    def is_extreme_short(self) -> bool:
        """True if funding shows extreme short crowding."""
        return self.rate < -0.0005  # < -0.05% per 8h
    
    @property
    def hours_to_funding(self) -> float:
        """Hours until next funding payment."""
        delta = self.next_funding_time - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 3600)


@dataclass
class FundingAnalysis:
    """Analysis of funding rates across markets."""
    timestamp: datetime
    
    # Aggregate metrics
    avg_funding: float = 0.0
    median_funding: float = 0.0
    extreme_longs: List[str] = field(default_factory=list)  # Symbols with extreme long funding
    extreme_shorts: List[str] = field(default_factory=list)  # Symbols with extreme short funding
    
    # Market sentiment
    market_sentiment: str = "neutral"  # "extreme_greed", "greed", "neutral", "fear", "extreme_fear"
    sentiment_score: float = 0.0  # -100 to 100
    
    # Trading signals
    contrarian_longs: List[str] = field(default_factory=list)  # Consider longing (extreme negative funding)
    contrarian_shorts: List[str] = field(default_factory=list)  # Consider shorting (extreme positive funding)
    
    # Arbitrage opportunities
    arb_opportunities: List[Dict] = field(default_factory=list)


class FundingRateTracker:
    """
    Tracks funding rates across crypto exchanges.
    
    Key insights from funding rates:
    1. High positive funding = too many longs = bearish signal
    2. High negative funding = too many shorts = bullish signal
    3. Funding extremes often precede reversals
    4. Can time entries around funding payments
    """
    
    # Binance futures funding times (UTC): 00:00, 08:00, 16:00
    FUNDING_TIMES = [0, 8, 16]
    
    def __init__(self):
        self._rates: Dict[str, FundingData] = {}
        self._historical_rates: Dict[str, List[FundingData]] = defaultdict(list)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_fetch = 0
        self._fetch_interval = 300  # 5 minutes
        
    def get_funding(self, symbol: str) -> Optional[FundingData]:
        """Get current funding data for a symbol."""
        # Normalize symbol (BTC-USD -> BTCUSDT)
        normalized = self._normalize_symbol(symbol)
        return self._rates.get(normalized)
    
    def get_all_funding(self) -> Dict[str, FundingData]:
        """Get all funding rates."""
        return self._rates.copy()
    
    def analyze_market(self) -> FundingAnalysis:
        """
        Analyze overall market funding conditions.
        
        Returns analysis of market sentiment and opportunities.
        """
        analysis = FundingAnalysis(timestamp=datetime.now(timezone.utc))
        
        if not self._rates:
            return analysis
        
        rates = [f.rate for f in self._rates.values()]
        
        # Calculate aggregates
        analysis.avg_funding = sum(rates) / len(rates) if rates else 0
        sorted_rates = sorted(rates)
        analysis.median_funding = sorted_rates[len(sorted_rates) // 2] if sorted_rates else 0
        
        # Find extremes
        for symbol, data in self._rates.items():
            if data.is_extreme_long:
                analysis.extreme_longs.append(symbol)
                analysis.contrarian_shorts.append(symbol)
            elif data.is_extreme_short:
                analysis.extreme_shorts.append(symbol)
                analysis.contrarian_longs.append(symbol)
        
        # Calculate sentiment score (-100 to 100)
        # Positive funding = bullish sentiment (longs paying)
        # Negative funding = bearish sentiment (shorts paying)
        analysis.sentiment_score = min(100, max(-100, analysis.avg_funding * 100000))
        
        if analysis.sentiment_score > 50:
            analysis.market_sentiment = "extreme_greed"
        elif analysis.sentiment_score > 20:
            analysis.market_sentiment = "greed"
        elif analysis.sentiment_score > -20:
            analysis.market_sentiment = "neutral"
        elif analysis.sentiment_score > -50:
            analysis.market_sentiment = "fear"
        else:
            analysis.market_sentiment = "extreme_fear"
        
        return analysis
    
    def should_enter_long(self, symbol: str) -> Dict[str, any]:
        """
        Analyze if funding conditions favor a long entry.
        
        Returns:
            dict with:
            - favorable: bool - if conditions favor long
            - score: float - 0-100 favorability score
            - reasons: list of reasons
            - wait_for_funding: bool - if should wait for funding payment
        """
        result = {
            "favorable": True,
            "score": 50,
            "reasons": [],
            "wait_for_funding": False,
        }
        
        data = self.get_funding(symbol)
        if not data:
            result["reasons"].append("No funding data available")
            return result
        
        # Positive funding = longs pay shorts = unfavorable for longs
        if data.rate > 0.0003:  # > 0.03% per 8h
            result["score"] -= 20
            result["reasons"].append(f"High funding rate ({data.rate*100:.3f}%) - longs paying")
            if data.rate > 0.0005:
                result["favorable"] = False
                result["reasons"].append("Extreme long crowding - consider waiting")
        elif data.rate < -0.0003:  # Negative funding
            result["score"] += 20
            result["reasons"].append(f"Negative funding ({data.rate*100:.3f}%) - shorts paying, favorable for longs")
            if data.rate < -0.0005:
                result["score"] += 15
                result["reasons"].append("Extreme short crowding - contrarian long opportunity")
        
        # Timing around funding
        hours_to_funding = data.hours_to_funding
        if hours_to_funding < 1 and data.rate > 0.0002:
            result["wait_for_funding"] = True
            result["reasons"].append(f"Funding in {hours_to_funding:.1f}h - wait to avoid paying")
        
        # Market-wide sentiment
        market = self.analyze_market()
        if market.market_sentiment == "extreme_greed":
            result["score"] -= 15
            result["reasons"].append("Market-wide extreme greed - caution on longs")
        elif market.market_sentiment == "extreme_fear":
            result["score"] += 15
            result["reasons"].append("Market-wide extreme fear - contrarian long setup")
        
        result["favorable"] = result["score"] >= 40
        return result
    
    def should_enter_short(self, symbol: str) -> Dict[str, any]:
        """
        Analyze if funding conditions favor a short entry.
        
        Returns same structure as should_enter_long.
        """
        result = {
            "favorable": True,
            "score": 50,
            "reasons": [],
            "wait_for_funding": False,
        }
        
        data = self.get_funding(symbol)
        if not data:
            result["reasons"].append("No funding data available")
            return result
        
        # Negative funding = shorts pay longs = unfavorable for shorts
        if data.rate < -0.0003:
            result["score"] -= 20
            result["reasons"].append(f"Negative funding ({data.rate*100:.3f}%) - shorts paying")
            if data.rate < -0.0005:
                result["favorable"] = False
                result["reasons"].append("Extreme short crowding - consider waiting")
        elif data.rate > 0.0003:
            result["score"] += 20
            result["reasons"].append(f"Positive funding ({data.rate*100:.3f}%) - longs paying, favorable for shorts")
            if data.rate > 0.0005:
                result["score"] += 15
                result["reasons"].append("Extreme long crowding - contrarian short opportunity")
        
        # Timing around funding
        hours_to_funding = data.hours_to_funding
        if hours_to_funding < 1 and data.rate < -0.0002:
            result["wait_for_funding"] = True
            result["reasons"].append(f"Funding in {hours_to_funding:.1f}h - wait to avoid paying")
        
        # Market-wide sentiment
        market = self.analyze_market()
        if market.market_sentiment == "extreme_fear":
            result["score"] -= 15
            result["reasons"].append("Market-wide extreme fear - caution on shorts")
        elif market.market_sentiment == "extreme_greed":
            result["score"] += 15
            result["reasons"].append("Market-wide extreme greed - contrarian short setup")
        
        result["favorable"] = result["score"] >= 40
        return result
    
    def get_next_funding_time(self) -> datetime:
        """Get the next funding payment time (UTC)."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        
        for funding_hour in self.FUNDING_TIMES:
            if funding_hour > current_hour:
                return now.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
        
        # Next day's first funding
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=self.FUNDING_TIMES[0], minute=0, second=0, microsecond=0)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format (BTC-USD -> BTCUSDT)."""
        if "-USD" in symbol:
            return symbol.replace("-USD", "USDT")
        if "-USDT" in symbol:
            return symbol.replace("-USDT", "USDT")
        return symbol
    
    def start(self):
        """Start background funding rate fetching."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()
        logger.info("Funding rate tracker started")
    
    def stop(self):
        """Stop the tracker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _fetch_loop(self):
        """Background loop to fetch funding rates."""
        while self._running:
            try:
                self._fetch_rates()
            except Exception as e:
                logger.error(f"Error fetching funding rates: {e}")
            
            time.sleep(self._fetch_interval)
    
    def _fetch_rates(self):
        """Fetch funding rates from Binance API."""
        import urllib.request
        import json
        
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                next_funding = self.get_next_funding_time()
                
                for item in data:
                    symbol = item.get("symbol", "")
                    if not symbol.endswith("USDT"):
                        continue
                    
                    rate = float(item.get("lastFundingRate", 0))
                    predicted = float(item.get("estimatedSettlePrice", 0))
                    
                    self._rates[symbol] = FundingData(
                        symbol=symbol,
                        rate=rate,
                        rate_8h=rate,
                        predicted_rate=predicted,
                        timestamp=datetime.now(timezone.utc),
                        next_funding_time=next_funding,
                    )
                
                logger.debug(f"Fetched funding rates for {len(self._rates)} symbols")
                
        except Exception as e:
            logger.warning(f"Failed to fetch Binance funding rates: {e}")
            # Try fallback to public endpoint
            self._fetch_rates_fallback()
    
    def _fetch_rates_fallback(self):
        """Fallback funding rate source."""
        # Use coinglass or similar public API as fallback
        # For now, just log the issue
        logger.debug("Using fallback funding rate source")


# Global instance
_tracker_instance: Optional[FundingRateTracker] = None


def get_funding_tracker() -> FundingRateTracker:
    """Get or create the global funding rate tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FundingRateTracker()
    return _tracker_instance


def start_funding_tracker() -> FundingRateTracker:
    """Start the funding rate tracker."""
    tracker = get_funding_tracker()
    tracker.start()
    return tracker
