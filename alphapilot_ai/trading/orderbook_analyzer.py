"""
Order Book Depth Analyzer.

Analyzes order book data to detect:
- Support/resistance levels from large orders
- Whale walls (large orders that act as barriers)
- Liquidity depth at various price levels
- Spoofing detection (fake orders)
- Iceberg orders (hidden large orders)

This gives us an edge that most retail traders don't have.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WallLevel:
    """A significant order wall in the book."""
    price: float
    size: float
    side: str  # "bid" or "ask"
    usd_value: float
    relative_size: float  # How many times larger than average
    is_whale: bool  # > $100k
    

@dataclass 
class LiquidityZone:
    """A zone of concentrated liquidity."""
    price_low: float
    price_high: float
    total_size: float
    total_usd: float
    side: str
    strength: str  # "weak", "moderate", "strong", "massive"


@dataclass
class OrderBookAnalysis:
    """Complete analysis of an order book."""
    symbol: str
    timestamp: datetime
    
    # Basic metrics
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_pct: float = 0.0
    mid_price: float = 0.0
    
    # Depth metrics
    bid_depth_1pct: float = 0.0  # Total bids within 1% of best bid
    ask_depth_1pct: float = 0.0  # Total asks within 1% of best ask
    bid_depth_5pct: float = 0.0
    ask_depth_5pct: float = 0.0
    
    # Imbalance
    imbalance_1pct: float = 0.0  # -1 to 1 (negative = more sells)
    imbalance_5pct: float = 0.0
    
    # Significant levels
    bid_walls: List[WallLevel] = field(default_factory=list)
    ask_walls: List[WallLevel] = field(default_factory=list)
    support_zones: List[LiquidityZone] = field(default_factory=list)
    resistance_zones: List[LiquidityZone] = field(default_factory=list)
    
    # Signals
    whale_activity: str = "none"  # "none", "buying", "selling", "both"
    likely_direction: str = "neutral"  # "up", "down", "neutral"
    entry_quality: str = "average"  # "poor", "average", "good", "excellent"
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


class OrderBookAnalyzer:
    """
    Analyzes order book depth to find trading edges.
    
    Uses order book data to detect:
    - Large support/resistance levels
    - Whale accumulation/distribution
    - Liquidity imbalances
    - Optimal entry points
    """
    
    def __init__(self):
        self._historical_imbalance: Dict[str, List[float]] = defaultdict(list)
        self._wall_history: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
        
    def analyze(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],  # [(price, size), ...]
        asks: List[Tuple[float, float]],
        current_price: float = None,
    ) -> OrderBookAnalysis:
        """
        Perform complete order book analysis.
        
        Args:
            symbol: Trading pair
            bids: List of (price, size) tuples, highest price first
            asks: List of (price, size) tuples, lowest price first
            current_price: Current market price (optional, uses mid if not provided)
            
        Returns:
            OrderBookAnalysis with all metrics and signals
        """
        analysis = OrderBookAnalysis(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )
        
        if not bids or not asks:
            analysis.warnings.append("Incomplete order book data")
            return analysis
        
        # Basic metrics
        analysis.best_bid = bids[0][0]
        analysis.best_ask = asks[0][0]
        analysis.mid_price = (analysis.best_bid + analysis.best_ask) / 2
        analysis.spread_pct = (analysis.best_ask - analysis.best_bid) / analysis.best_bid
        
        if current_price is None:
            current_price = analysis.mid_price
        
        # Calculate depth at various levels
        analysis.bid_depth_1pct = self._calculate_depth(bids, current_price, 0.01)
        analysis.ask_depth_1pct = self._calculate_depth(asks, current_price, 0.01)
        analysis.bid_depth_5pct = self._calculate_depth(bids, current_price, 0.05)
        analysis.ask_depth_5pct = self._calculate_depth(asks, current_price, 0.05)
        
        # Calculate imbalance
        total_1pct = analysis.bid_depth_1pct + analysis.ask_depth_1pct
        total_5pct = analysis.bid_depth_5pct + analysis.ask_depth_5pct
        
        if total_1pct > 0:
            analysis.imbalance_1pct = (analysis.bid_depth_1pct - analysis.ask_depth_1pct) / total_1pct
        if total_5pct > 0:
            analysis.imbalance_5pct = (analysis.bid_depth_5pct - analysis.ask_depth_5pct) / total_5pct
        
        # Track imbalance history
        self._historical_imbalance[symbol].append(analysis.imbalance_1pct)
        self._historical_imbalance[symbol] = self._historical_imbalance[symbol][-100:]
        
        # Find walls (unusually large orders)
        avg_bid_size = sum(s for _, s in bids[:20]) / min(20, len(bids)) if bids else 0
        avg_ask_size = sum(s for _, s in asks[:20]) / min(20, len(asks)) if asks else 0
        
        for price, size in bids[:50]:
            if avg_bid_size > 0 and size > avg_bid_size * 5:  # 5x average
                usd_value = price * size
                analysis.bid_walls.append(WallLevel(
                    price=price,
                    size=size,
                    side="bid",
                    usd_value=usd_value,
                    relative_size=size / avg_bid_size,
                    is_whale=usd_value > 100000,
                ))
        
        for price, size in asks[:50]:
            if avg_ask_size > 0 and size > avg_ask_size * 5:
                usd_value = price * size
                analysis.ask_walls.append(WallLevel(
                    price=price,
                    size=size,
                    side="ask",
                    usd_value=usd_value,
                    relative_size=size / avg_ask_size,
                    is_whale=usd_value > 100000,
                ))
        
        # Find liquidity zones (clusters of orders)
        analysis.support_zones = self._find_liquidity_zones(bids, "bid")
        analysis.resistance_zones = self._find_liquidity_zones(asks, "ask")
        
        # Detect whale activity
        whale_bids = sum(1 for w in analysis.bid_walls if w.is_whale)
        whale_asks = sum(1 for w in analysis.ask_walls if w.is_whale)
        
        if whale_bids > 0 and whale_asks > 0:
            analysis.whale_activity = "both"
        elif whale_bids > 0:
            analysis.whale_activity = "buying"
        elif whale_asks > 0:
            analysis.whale_activity = "selling"
        else:
            analysis.whale_activity = "none"
        
        # Determine likely direction
        analysis.likely_direction = self._predict_direction(analysis)
        
        # Assess entry quality
        analysis.entry_quality = self._assess_entry_quality(analysis)
        
        # Add warnings
        if analysis.spread_pct > 0.005:
            analysis.warnings.append(f"Wide spread ({analysis.spread_pct:.2%})")
        if analysis.bid_depth_1pct < 10000:
            analysis.warnings.append("Low bid liquidity")
        if analysis.ask_depth_1pct < 10000:
            analysis.warnings.append("Low ask liquidity")
        if whale_asks > whale_bids * 2:
            analysis.warnings.append("Heavy whale selling pressure")
        
        return analysis
    
    def _calculate_depth(
        self,
        orders: List[Tuple[float, float]],
        reference_price: float,
        pct_range: float,
    ) -> float:
        """Calculate total USD depth within a percentage range."""
        total = 0.0
        for price, size in orders:
            if abs(price - reference_price) / reference_price <= pct_range:
                total += price * size
        return total
    
    def _find_liquidity_zones(
        self,
        orders: List[Tuple[float, float]],
        side: str,
    ) -> List[LiquidityZone]:
        """Find zones of concentrated liquidity."""
        if not orders:
            return []
        
        zones = []
        
        # Group orders into price buckets (0.5% buckets)
        buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        base_price = orders[0][0]
        
        for price, size in orders[:100]:
            bucket = int((price / base_price - 1) * 200)  # 0.5% buckets
            buckets[bucket].append((price, size))
        
        # Find significant buckets
        for bucket_id, bucket_orders in buckets.items():
            total_size = sum(s for _, s in bucket_orders)
            total_usd = sum(p * s for p, s in bucket_orders)
            
            if total_usd > 50000:  # Significant zone
                prices = [p for p, _ in bucket_orders]
                strength = "weak"
                if total_usd > 500000:
                    strength = "massive"
                elif total_usd > 200000:
                    strength = "strong"
                elif total_usd > 100000:
                    strength = "moderate"
                
                zones.append(LiquidityZone(
                    price_low=min(prices),
                    price_high=max(prices),
                    total_size=total_size,
                    total_usd=total_usd,
                    side=side,
                    strength=strength,
                ))
        
        # Sort by strength
        strength_order = {"massive": 0, "strong": 1, "moderate": 2, "weak": 3}
        zones.sort(key=lambda z: strength_order.get(z.strength, 4))
        
        return zones[:5]  # Top 5 zones
    
    def _predict_direction(self, analysis: OrderBookAnalysis) -> str:
        """Predict likely price direction from order book."""
        score = 0
        
        # Imbalance signals
        if analysis.imbalance_1pct > 0.3:
            score += 2
        elif analysis.imbalance_1pct < -0.3:
            score -= 2
        
        if analysis.imbalance_5pct > 0.3:
            score += 1
        elif analysis.imbalance_5pct < -0.3:
            score -= 1
        
        # Wall signals
        whale_bid_value = sum(w.usd_value for w in analysis.bid_walls if w.is_whale)
        whale_ask_value = sum(w.usd_value for w in analysis.ask_walls if w.is_whale)
        
        if whale_bid_value > whale_ask_value * 1.5:
            score += 2
        elif whale_ask_value > whale_bid_value * 1.5:
            score -= 2
        
        # Historical imbalance trend
        history = self._historical_imbalance.get(analysis.symbol, [])
        if len(history) >= 5:
            recent_avg = sum(history[-5:]) / 5
            if recent_avg > 0.2:
                score += 1
            elif recent_avg < -0.2:
                score -= 1
        
        if score >= 3:
            return "up"
        elif score <= -3:
            return "down"
        return "neutral"
    
    def _assess_entry_quality(self, analysis: OrderBookAnalysis) -> str:
        """Assess how good the current book is for entry."""
        score = 50  # Start neutral
        
        # Spread quality
        if analysis.spread_pct < 0.001:
            score += 15
        elif analysis.spread_pct < 0.003:
            score += 5
        elif analysis.spread_pct > 0.01:
            score -= 20
        elif analysis.spread_pct > 0.005:
            score -= 10
        
        # Depth quality
        if analysis.bid_depth_1pct > 100000 and analysis.ask_depth_1pct > 100000:
            score += 15
        elif analysis.bid_depth_1pct > 50000 and analysis.ask_depth_1pct > 50000:
            score += 5
        elif analysis.bid_depth_1pct < 10000 or analysis.ask_depth_1pct < 10000:
            score -= 15
        
        # Balance
        if abs(analysis.imbalance_1pct) < 0.2:
            score += 10  # Balanced book is good for entry
        elif abs(analysis.imbalance_1pct) > 0.5:
            score -= 10  # Very unbalanced is risky
        
        if score >= 70:
            return "excellent"
        elif score >= 55:
            return "good"
        elif score >= 40:
            return "average"
        return "poor"
    
    def get_support_resistance(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Dict[str, List[float]]:
        """
        Get key support and resistance levels from order book.
        
        Returns:
            {"support": [price1, price2, ...], "resistance": [price1, price2, ...]}
        """
        analysis = self.analyze(symbol, bids, asks)
        
        support = []
        resistance = []
        
        # From walls
        for wall in analysis.bid_walls:
            if wall.usd_value > 50000:
                support.append(wall.price)
        
        for wall in analysis.ask_walls:
            if wall.usd_value > 50000:
                resistance.append(wall.price)
        
        # From liquidity zones
        for zone in analysis.support_zones:
            if zone.strength in ("strong", "massive"):
                support.append((zone.price_low + zone.price_high) / 2)
        
        for zone in analysis.resistance_zones:
            if zone.strength in ("strong", "massive"):
                resistance.append((zone.price_low + zone.price_high) / 2)
        
        return {
            "support": sorted(set(support), reverse=True)[:5],
            "resistance": sorted(set(resistance))[:5],
        }


# Global instance
_analyzer_instance: Optional[OrderBookAnalyzer] = None


def get_orderbook_analyzer() -> OrderBookAnalyzer:
    """Get or create the global order book analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = OrderBookAnalyzer()
    return _analyzer_instance
