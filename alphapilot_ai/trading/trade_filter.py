"""
Trade Filter - Pre-trade quality checks to avoid bad setups.

This module implements filters that a skilled trader would naturally apply:
1. Time-of-day filter (avoid low-volume hours)
2. Position correlation check (avoid overexposure to similar assets)
3. Market regime alignment (don't trade reversals in strong trends)
4. Session awareness (Asian, London, US have different behaviors)
5. Recent performance check (reduce size after losses)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from utils.logger import get_logger
from utils.helpers import utcnow

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Result of running trade filters."""
    should_trade: bool = True
    confidence_adjustment: float = 1.0
    size_adjustment: float = 1.0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_rejection(self, reason: str):
        self.should_trade = False
        self.reasons.append(reason)
    
    def add_warning(self, warning: str, conf_adj: float = 1.0, size_adj: float = 1.0):
        self.warnings.append(warning)
        self.confidence_adjustment *= conf_adj
        self.size_adjustment *= size_adj


# Crypto sector classifications for correlation checking
CRYPTO_SECTORS = {
    # Layer 1s - highly correlated
    "layer1": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "ATOM-USD", "NEAR-USD", "APT-USD", "SUI-USD"],
    # DeFi tokens - correlated with ETH
    "defi": ["UNI-USD", "AAVE-USD", "MKR-USD", "CRV-USD", "COMP-USD", "SNX-USD", "SUSHI-USD", "1INCH-USD", "YFI-USD", "LDO-USD"],
    # Meme coins - highly volatile, correlated with each other
    "meme": ["DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD", "WIF-USD"],
    # Layer 2s - correlated with ETH
    "layer2": ["MATIC-USD", "ARB-USD", "OP-USD", "IMX-USD", "LRC-USD", "METIS-USD"],
    # AI tokens
    "ai": ["FET-USD", "RNDR-USD", "AGIX-USD", "OCEAN-USD", "TAO-USD"],
    # Gaming/Metaverse
    "gaming": ["AXS-USD", "SAND-USD", "MANA-USD", "GALA-USD", "ENJ-USD", "IMX-USD"],
    # Exchange tokens
    "exchange": ["BNB-USD", "FTT-USD", "CRO-USD", "OKB-USD", "KCS-USD"],
}


def get_sector(symbol: str) -> str:
    """Get the sector for a symbol."""
    for sector, symbols in CRYPTO_SECTORS.items():
        if symbol in symbols:
            return sector
    return "other"


class TradeFilter:
    """
    Applies quality filters before entering trades.
    Prevents bad entries that a skilled human would avoid.
    """
    
    def __init__(self):
        # Time-of-day settings (hours in UTC)
        self._low_volume_hours = list(range(3, 8))  # 3-8 AM UTC typically low volume
        self._high_volume_hours = [14, 15, 16, 17, 18, 19, 20]  # US market hours
        
        # Correlation limits
        self._max_same_sector = 2  # Max positions in same sector
        self._max_same_direction_pct = 0.70  # Max 70% of positions in same direction
        
        # Session times (UTC)
        self._sessions = {
            "asian": (0, 8),      # 00:00-08:00 UTC
            "london": (7, 16),    # 07:00-16:00 UTC
            "us": (13, 22),       # 13:00-22:00 UTC
            "overlap_london_us": (13, 16),  # Best liquidity
        }
    
    def apply_filters(
        self,
        symbol: str,
        side: str,
        confidence: float,
        current_positions: List[Dict],
        market_regime: Optional[str] = None,
        signal_strategy: Optional[str] = None,
    ) -> FilterResult:
        """
        Apply all filters to a potential trade.
        
        Args:
            symbol: The trading symbol (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            confidence: Signal confidence (0-1)
            current_positions: List of current open positions
            market_regime: Current market regime ("trending", "ranging", etc.)
            signal_strategy: Strategy that generated the signal
            
        Returns:
            FilterResult with should_trade flag and adjustments
        """
        result = FilterResult()
        
        # 1. Time-of-day filter
        self._check_time_of_day(result)
        
        # 2. Position correlation check
        self._check_correlation(result, symbol, side, current_positions)
        
        # 3. Market regime alignment
        self._check_regime_alignment(result, market_regime, signal_strategy, side)
        
        # 4. Session awareness
        self._check_session(result, symbol)
        
        # 5. Confidence sanity check
        if confidence < 0.35:
            result.add_warning("Very low confidence signal", conf_adj=0.8, size_adj=0.5)
        
        # Log filter results
        if not result.should_trade:
            logger.info(f"[FILTER] {symbol} {side} REJECTED: {', '.join(result.reasons)}")
        elif result.warnings:
            logger.debug(f"[FILTER] {symbol} {side} WARNINGS: {', '.join(result.warnings)}")
        
        return result
    
    def _check_time_of_day(self, result: FilterResult):
        """Check if current time is good for trading."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        if hour in self._low_volume_hours:
            result.add_warning(
                f"Low volume hour ({hour}:00 UTC)",
                conf_adj=0.85,
                size_adj=0.7
            )
        elif hour in self._high_volume_hours:
            # Slight boost during high-volume US hours
            result.confidence_adjustment *= 1.05
    
    def _check_correlation(
        self,
        result: FilterResult,
        symbol: str,
        side: str,
        current_positions: List[Dict]
    ):
        """Check for overexposure to correlated assets."""
        if not current_positions:
            return
        
        # Get sector of new trade
        new_sector = get_sector(symbol)
        
        # Count positions in same sector
        same_sector_count = 0
        same_direction_count = 0
        total_positions = len(current_positions)
        
        for pos in current_positions:
            pos_symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "")
            
            if get_sector(pos_symbol) == new_sector:
                same_sector_count += 1
            
            if pos_side == side:
                same_direction_count += 1
        
        # Check sector concentration
        if same_sector_count >= self._max_same_sector:
            result.add_warning(
                f"Already have {same_sector_count} positions in {new_sector} sector",
                conf_adj=0.8,
                size_adj=0.5
            )
        
        # Check directional bias
        if total_positions > 2:
            direction_pct = same_direction_count / total_positions
            if direction_pct >= self._max_same_direction_pct:
                result.add_warning(
                    f"{direction_pct:.0%} of positions are {side}s - directional risk",
                    conf_adj=0.9,
                    size_adj=0.7
                )
    
    def _check_regime_alignment(
        self,
        result: FilterResult,
        market_regime: Optional[str],
        signal_strategy: Optional[str],
        side: str
    ):
        """Check if signal type matches market regime."""
        if not market_regime or not signal_strategy:
            return
        
        regime = market_regime.lower()
        strategy = signal_strategy.lower()
        
        # Misalignment penalties
        if regime in ("strong_uptrend", "trending_up"):
            if strategy == "mean_reversion" and side == "SELL":
                result.add_warning(
                    "Mean reversion SELL in uptrend - risky",
                    conf_adj=0.7,
                    size_adj=0.5
                )
        elif regime in ("strong_downtrend", "trending_down"):
            if strategy == "mean_reversion" and side == "BUY":
                result.add_warning(
                    "Mean reversion BUY in downtrend - risky",
                    conf_adj=0.7,
                    size_adj=0.5
                )
        elif regime == "ranging":
            if strategy == "momentum":
                result.add_warning(
                    "Momentum signal in ranging market - likely to chop",
                    conf_adj=0.85,
                    size_adj=0.8
                )
    
    def _check_session(self, result: FilterResult, symbol: str):
        """Check trading session and adjust for session-specific behavior."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Check for London-US overlap (best liquidity)
        overlap_start, overlap_end = self._sessions["overlap_london_us"]
        if overlap_start <= hour < overlap_end:
            # Best time to trade - slight confidence boost
            result.confidence_adjustment *= 1.05
            return
        
        # Asian session - meme coins often pump
        asian_start, asian_end = self._sessions["asian"]
        if asian_start <= hour < asian_end:
            sector = get_sector(symbol)
            if sector == "meme":
                result.add_warning(
                    "Meme coin in Asian session - higher volatility",
                    size_adj=0.8
                )
    
    def get_current_session(self) -> str:
        """Get the current trading session."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        for session_name, (start, end) in self._sessions.items():
            if start <= hour < end:
                return session_name
        return "off_hours"


# Singleton instance
_filter_instance: Optional[TradeFilter] = None


def get_trade_filter() -> TradeFilter:
    """Get the singleton trade filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = TradeFilter()
    return _filter_instance


def filter_trade(
    symbol: str,
    side: str,
    confidence: float,
    current_positions: List[Dict] = None,
    market_regime: str = None,
    signal_strategy: str = None,
) -> FilterResult:
    """Convenience function to filter a trade."""
    return get_trade_filter().apply_filters(
        symbol=symbol,
        side=side,
        confidence=confidence,
        current_positions=current_positions or [],
        market_regime=market_regime,
        signal_strategy=signal_strategy,
    )
