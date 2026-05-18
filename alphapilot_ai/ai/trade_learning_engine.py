"""
Advanced Trade Learning Engine
==============================

This module implements pattern recognition and machine learning from historical trades.
It learns from past successes and failures to improve future trade decisions.

Key Features:
1. Feature extraction from trade context (price patterns, indicators, timing)
2. Trade outcome classification (successful patterns vs failures)
3. Pattern similarity matching for new trade opportunities
4. Confidence calibration based on historical accuracy
5. Adaptive learning that weights recent trades more heavily

The engine maintains a "trade memory" that stores feature vectors from past trades
along with their outcomes, enabling pattern-based predictions for new trades.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict, Tuple
from collections import defaultdict

from database.db import session_scope
from database.models import (
    PaperTrade,
    ClaudeDecision,
    TradeReflection,
    AILearningMemory,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeFeatures:
    """Feature vector extracted from a trade context."""
    # Technical indicators at entry
    rsi: float = 50.0
    macd_histogram: float = 0.0
    bollinger_percent_b: float = 0.5
    adx: float = 20.0
    relative_volume: float = 1.0
    
    # Price action features
    price_vs_sma20: float = 0.0  # % above/below SMA20
    price_vs_sma50: float = 0.0
    recent_return_6bar: float = 0.0
    recent_return_24bar: float = 0.0
    volatility_percentile: float = 50.0
    
    # Market regime
    regime: str = "UNKNOWN"
    trend_strength: float = 0.0
    
    # Timing features
    hour_of_day: int = 12
    day_of_week: int = 0
    
    # Signal features
    signal_confidence: float = 0.5
    strategy_type: str = "Momentum"
    side: str = "BUY"
    
    # Context features
    open_positions_count: int = 0
    recent_win_rate: float = 0.5
    consecutive_losses: int = 0
    
    def to_vector(self) -> List[float]:
        """Convert to numeric vector for similarity calculations."""
        regime_map = {
            "TRENDING_UP": 1.0, "TRENDING_DOWN": -1.0, "RANGING": 0.0,
            "VOLATILE": 0.5, "ACCUMULATION": 0.3, "DISTRIBUTION": -0.3,
            "UNKNOWN": 0.0
        }
        side_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        
        return [
            (self.rsi - 50) / 50,  # Normalize RSI to [-1, 1]
            self.macd_histogram * 10,  # Scale MACD
            (self.bollinger_percent_b - 0.5) * 2,  # Normalize BB %B
            (self.adx - 25) / 25,  # Normalize ADX
            (self.relative_volume - 1) / 2,  # Normalize volume
            self.price_vs_sma20 / 5,  # Scale price deviation
            self.price_vs_sma50 / 10,
            self.recent_return_6bar * 100,
            self.recent_return_24bar * 50,
            (self.volatility_percentile - 50) / 50,
            regime_map.get(self.regime, 0.0),
            self.trend_strength / 50,
            (self.hour_of_day - 12) / 12,
            (self.day_of_week - 3) / 3,
            (self.signal_confidence - 0.5) * 2,
            side_map.get(self.side, 0.0),
            min(self.open_positions_count / 5, 1.0),
            (self.recent_win_rate - 0.5) * 2,
            min(self.consecutive_losses / 3, 1.0) * -1,
        ]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeFeatures":
        """Create features from a dictionary."""
        return cls(
            rsi=float(data.get("rsi", 50)),
            macd_histogram=float(data.get("macd_histogram", 0)),
            bollinger_percent_b=float(data.get("bollinger_percent_b", 0.5)),
            adx=float(data.get("adx", 20)),
            relative_volume=float(data.get("relative_volume", 1)),
            price_vs_sma20=float(data.get("price_vs_sma20", 0)),
            price_vs_sma50=float(data.get("price_vs_sma50", 0)),
            recent_return_6bar=float(data.get("recent_return_6bar", 0)),
            recent_return_24bar=float(data.get("recent_return_24bar", 0)),
            volatility_percentile=float(data.get("volatility_percentile", 50)),
            regime=str(data.get("regime", "UNKNOWN")),
            trend_strength=float(data.get("trend_strength", 0)),
            hour_of_day=int(data.get("hour_of_day", 12)),
            day_of_week=int(data.get("day_of_week", 0)),
            signal_confidence=float(data.get("signal_confidence", 0.5)),
            strategy_type=str(data.get("strategy_type", "Momentum")),
            side=str(data.get("side", "BUY")),
            open_positions_count=int(data.get("open_positions_count", 0)),
            recent_win_rate=float(data.get("recent_win_rate", 0.5)),
            consecutive_losses=int(data.get("consecutive_losses", 0)),
        )


@dataclass
class TradePattern:
    """A learned pattern from historical trades."""
    pattern_id: int
    features: TradeFeatures
    outcome: str  # "WIN", "LOSS", "BREAKEVEN"
    pnl_pct: float
    trade_id: int
    symbol: str
    timestamp: datetime
    weight: float = 1.0  # Decay weight based on recency
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "features": self.features.__dict__,
            "outcome": self.outcome,
            "pnl_pct": self.pnl_pct,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "weight": self.weight,
        }


class TradeLearningEngine:
    """
    Pattern-based learning engine for trade decisions.
    
    Uses k-nearest neighbors approach with exponential time decay
    to find similar historical patterns and predict outcomes.
    """
    
    def __init__(self, decay_half_life_days: float = 30.0):
        """
        Initialize the learning engine.
        
        Args:
            decay_half_life_days: Half-life for exponential time decay weighting.
                                  Older patterns have less influence.
        """
        self.decay_half_life = decay_half_life_days
        self._pattern_cache: List[TradePattern] = []
        self._cache_loaded = False
        
        # Performance tracking by context
        self._performance_by_regime: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0}
        )
        self._performance_by_strategy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0}
        )
        self._performance_by_hour: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0}
        )
    
    def load_patterns_from_db(self, limit: int = 1000) -> int:
        """
        Load historical trade patterns from database.
        
        Returns:
            Number of patterns loaded.
        """
        patterns = []
        now = utcnow()
        
        with session_scope() as s:
            # Get closed trades with their decisions
            trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.status == "closed")
                .filter(PaperTrade.realized_pnl.isnot(None))
                .order_by(PaperTrade.closed_at.desc())
                .limit(limit)
                .all()
            )
            
            for i, trade in enumerate(trades):
                # Extract features from trade context
                features = self._extract_features_from_trade(trade, s)
                
                # Determine outcome
                pnl = float(trade.realized_pnl or 0)
                entry = float(trade.entry_price or 1)
                pnl_pct = (pnl / (entry * float(trade.qty or 1))) * 100
                
                if pnl > 0:
                    outcome = "WIN"
                elif pnl < 0:
                    outcome = "LOSS"
                else:
                    outcome = "BREAKEVEN"
                
                # Calculate time decay weight
                closed_at = trade.closed_at or trade.opened_at or now
                days_ago = (now - closed_at).total_seconds() / 86400
                weight = math.exp(-math.log(2) * days_ago / self.decay_half_life)
                
                pattern = TradePattern(
                    pattern_id=i,
                    features=features,
                    outcome=outcome,
                    pnl_pct=pnl_pct,
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    timestamp=closed_at,
                    weight=weight,
                )
                patterns.append(pattern)
                
                # Update performance tracking
                self._update_performance_tracking(pattern)
        
        self._pattern_cache = patterns
        self._cache_loaded = True
        logger.info(f"[LEARNING_ENGINE] Loaded {len(patterns)} trade patterns")
        return len(patterns)
    
    def _extract_features_from_trade(self, trade: PaperTrade, session) -> TradeFeatures:
        """Extract features from a trade and its associated decision."""
        features = TradeFeatures()
        
        # Try to find the associated ClaudeDecision
        decision = (
            session.query(ClaudeDecision)
            .filter(
                ClaudeDecision.wallet_id == trade.wallet_id,
                ClaudeDecision.symbol == trade.symbol,
                ClaudeDecision.created_at <= trade.opened_at,
            )
            .order_by(ClaudeDecision.created_at.desc())
            .first()
        )
        
        if decision and decision.context_snapshot:
            try:
                ctx = json.loads(decision.context_snapshot) if isinstance(
                    decision.context_snapshot, str
                ) else decision.context_snapshot
                
                # Extract technical indicators
                indicators = ctx.get("technical_signal", {}).get("indicators", {})
                features.rsi = float(indicators.get("rsi", 50))
                features.macd_histogram = float(indicators.get("macd_histogram", 0))
                features.bollinger_percent_b = float(indicators.get("bollinger_percent_b", 0.5))
                features.adx = float(indicators.get("adx", 20))
                features.relative_volume = float(indicators.get("relative_volume", 1))
                
                # Price action
                features.recent_return_6bar = float(indicators.get("return_lb", 0))
                
                # Market regime
                regime_data = ctx.get("market_regime", {})
                features.regime = str(regime_data.get("regime", "UNKNOWN"))
                features.trend_strength = float(regime_data.get("trend_strength", 0))
                
                # Signal info
                signal = ctx.get("technical_signal", {})
                features.signal_confidence = float(signal.get("confidence", 0.5))
                features.strategy_type = str(signal.get("strategy", "Momentum"))
                features.side = str(signal.get("side", "BUY"))
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        
        # Timing features
        opened_at = trade.opened_at or utcnow()
        features.hour_of_day = opened_at.hour
        features.day_of_week = opened_at.weekday()
        
        # Get recent performance context
        recent_trades = (
            session.query(PaperTrade)
            .filter(
                PaperTrade.wallet_id == trade.wallet_id,
                PaperTrade.status == "closed",
                PaperTrade.closed_at < trade.opened_at,
            )
            .order_by(PaperTrade.closed_at.desc())
            .limit(10)
            .all()
        )
        
        if recent_trades:
            wins = sum(1 for t in recent_trades if float(t.realized_pnl or 0) > 0)
            features.recent_win_rate = wins / len(recent_trades)
            
            # Count consecutive losses
            consec = 0
            for t in recent_trades:
                if float(t.realized_pnl or 0) < 0:
                    consec += 1
                else:
                    break
            features.consecutive_losses = consec
        
        # Count open positions at time of trade
        open_count = (
            session.query(PaperTrade)
            .filter(
                PaperTrade.wallet_id == trade.wallet_id,
                PaperTrade.status == "open",
                PaperTrade.opened_at < trade.opened_at,
            )
            .count()
        )
        features.open_positions_count = open_count
        
        return features
    
    def _update_performance_tracking(self, pattern: TradePattern):
        """Update performance statistics by context."""
        outcome_win = 1 if pattern.outcome == "WIN" else 0
        outcome_loss = 1 if pattern.outcome == "LOSS" else 0
        
        # By regime
        regime = pattern.features.regime
        self._performance_by_regime[regime]["wins"] += outcome_win
        self._performance_by_regime[regime]["losses"] += outcome_loss
        self._performance_by_regime[regime]["total_pnl"] += pattern.pnl_pct
        
        # By strategy
        strategy = pattern.features.strategy_type
        self._performance_by_strategy[strategy]["wins"] += outcome_win
        self._performance_by_strategy[strategy]["losses"] += outcome_loss
        self._performance_by_strategy[strategy]["total_pnl"] += pattern.pnl_pct
        
        # By hour
        hour = pattern.features.hour_of_day
        self._performance_by_hour[hour]["wins"] += outcome_win
        self._performance_by_hour[hour]["losses"] += outcome_loss
        self._performance_by_hour[hour]["total_pnl"] += pattern.pnl_pct
    
    def find_similar_patterns(
        self,
        features: TradeFeatures,
        k: int = 10,
        min_weight: float = 0.1,
    ) -> List[Tuple[TradePattern, float]]:
        """
        Find k most similar historical patterns.
        
        Args:
            features: Current trade features to match.
            k: Number of similar patterns to return.
            min_weight: Minimum time decay weight to consider.
        
        Returns:
            List of (pattern, similarity_score) tuples, sorted by similarity.
        """
        if not self._cache_loaded:
            self.load_patterns_from_db()
        
        if not self._pattern_cache:
            return []
        
        query_vector = features.to_vector()
        similarities = []
        
        for pattern in self._pattern_cache:
            if pattern.weight < min_weight:
                continue
            
            pattern_vector = pattern.features.to_vector()
            
            # Cosine similarity with weight adjustment
            similarity = self._cosine_similarity(query_vector, pattern_vector)
            weighted_similarity = similarity * pattern.weight
            
            similarities.append((pattern, weighted_similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(v1) != len(v2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def predict_outcome(
        self,
        features: TradeFeatures,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Predict trade outcome based on similar historical patterns.
        
        Returns:
            Dict with predicted outcome, confidence, and supporting evidence.
        """
        similar = self.find_similar_patterns(features, k=k)
        
        if not similar:
            return {
                "predicted_outcome": "UNKNOWN",
                "confidence": 0.0,
                "win_probability": 0.5,
                "expected_pnl_pct": 0.0,
                "similar_patterns": 0,
                "evidence": [],
            }
        
        # Weighted voting
        win_weight = 0.0
        loss_weight = 0.0
        total_weight = 0.0
        weighted_pnl = 0.0
        
        evidence = []
        
        for pattern, similarity in similar:
            weight = similarity * pattern.weight
            total_weight += weight
            weighted_pnl += pattern.pnl_pct * weight
            
            if pattern.outcome == "WIN":
                win_weight += weight
            elif pattern.outcome == "LOSS":
                loss_weight += weight
            
            evidence.append({
                "trade_id": pattern.trade_id,
                "symbol": pattern.symbol,
                "outcome": pattern.outcome,
                "pnl_pct": round(pattern.pnl_pct, 2),
                "similarity": round(similarity, 3),
                "age_days": (utcnow() - pattern.timestamp).days if pattern.timestamp else 0,
            })
        
        if total_weight == 0:
            win_prob = 0.5
        else:
            win_prob = win_weight / total_weight
        
        expected_pnl = weighted_pnl / total_weight if total_weight > 0 else 0
        
        # Determine prediction
        if win_prob > 0.6:
            predicted = "WIN"
            confidence = (win_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        elif win_prob < 0.4:
            predicted = "LOSS"
            confidence = (0.5 - win_prob) * 2
        else:
            predicted = "UNCERTAIN"
            confidence = 1 - abs(win_prob - 0.5) * 2
        
        return {
            "predicted_outcome": predicted,
            "confidence": round(confidence, 3),
            "win_probability": round(win_prob, 3),
            "expected_pnl_pct": round(expected_pnl, 2),
            "similar_patterns": len(similar),
            "evidence": evidence[:5],  # Top 5 most similar
        }
    
    def get_confidence_adjustment(
        self,
        features: TradeFeatures,
        base_confidence: float,
    ) -> Tuple[float, List[str]]:
        """
        Adjust confidence based on learned patterns.
        
        Returns:
            Tuple of (adjusted_confidence, list of reasons for adjustment).
        """
        reasons = []
        adjustment = 0.0
        
        prediction = self.predict_outcome(features)
        
        # Adjust based on pattern prediction
        if prediction["similar_patterns"] >= 5:
            if prediction["predicted_outcome"] == "WIN" and prediction["confidence"] > 0.3:
                adj = min(0.15, prediction["confidence"] * 0.2)
                adjustment += adj
                reasons.append(f"Similar patterns historically winning (+{adj:.2%})")
            elif prediction["predicted_outcome"] == "LOSS" and prediction["confidence"] > 0.3:
                adj = min(0.15, prediction["confidence"] * 0.2)
                adjustment -= adj
                reasons.append(f"Similar patterns historically losing (-{adj:.2%})")
        
        # Adjust based on context performance
        regime = features.regime
        if regime in self._performance_by_regime:
            stats = self._performance_by_regime[regime]
            total = stats["wins"] + stats["losses"]
            if total >= 10:
                regime_wr = stats["wins"] / total
                if regime_wr > 0.6:
                    adj = (regime_wr - 0.5) * 0.1
                    adjustment += adj
                    reasons.append(f"Strong performance in {regime} regime (+{adj:.2%})")
                elif regime_wr < 0.4:
                    adj = (0.5 - regime_wr) * 0.1
                    adjustment -= adj
                    reasons.append(f"Weak performance in {regime} regime (-{adj:.2%})")
        
        # Adjust based on strategy performance
        strategy = features.strategy_type
        if strategy in self._performance_by_strategy:
            stats = self._performance_by_strategy[strategy]
            total = stats["wins"] + stats["losses"]
            if total >= 10:
                strat_wr = stats["wins"] / total
                if strat_wr > 0.6:
                    adj = (strat_wr - 0.5) * 0.08
                    adjustment += adj
                    reasons.append(f"Strong {strategy} strategy performance (+{adj:.2%})")
                elif strat_wr < 0.4:
                    adj = (0.5 - strat_wr) * 0.08
                    adjustment -= adj
                    reasons.append(f"Weak {strategy} strategy performance (-{adj:.2%})")
        
        # Adjust based on time-of-day performance
        hour = features.hour_of_day
        if hour in self._performance_by_hour:
            stats = self._performance_by_hour[hour]
            total = stats["wins"] + stats["losses"]
            if total >= 5:
                hour_wr = stats["wins"] / total
                if hour_wr > 0.65:
                    adj = 0.05
                    adjustment += adj
                    reasons.append(f"Good historical performance at {hour}:00 UTC (+{adj:.2%})")
                elif hour_wr < 0.35:
                    adj = 0.05
                    adjustment -= adj
                    reasons.append(f"Poor historical performance at {hour}:00 UTC (-{adj:.2%})")
        
        # Consecutive losses penalty
        if features.consecutive_losses >= 3:
            adj = min(0.1, features.consecutive_losses * 0.03)
            adjustment -= adj
            reasons.append(f"{features.consecutive_losses} consecutive losses (-{adj:.2%})")
        
        # Apply adjustment with bounds
        adjusted = max(0.0, min(1.0, base_confidence + adjustment))
        
        return adjusted, reasons
    
    def get_best_strategy_for_regime(self, regime: str) -> Tuple[str, float]:
        """
        Get the best performing strategy for a given market regime.
        
        Returns:
            Tuple of (strategy_name, win_rate).
        """
        best_strategy = "Momentum"
        best_wr = 0.5
        
        # Filter patterns by regime
        regime_patterns = [p for p in self._pattern_cache if p.features.regime == regime]
        
        if not regime_patterns:
            return best_strategy, best_wr
        
        # Calculate win rate by strategy within this regime
        strategy_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})
        
        for pattern in regime_patterns:
            strat = pattern.features.strategy_type
            strategy_stats[strat]["total"] += 1
            if pattern.outcome == "WIN":
                strategy_stats[strat]["wins"] += 1
        
        for strat, stats in strategy_stats.items():
            if stats["total"] >= 5:
                wr = stats["wins"] / stats["total"]
                if wr > best_wr:
                    best_wr = wr
                    best_strategy = strat
        
        return best_strategy, best_wr
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of what the engine has learned."""
        if not self._cache_loaded:
            self.load_patterns_from_db()
        
        total_patterns = len(self._pattern_cache)
        wins = sum(1 for p in self._pattern_cache if p.outcome == "WIN")
        losses = sum(1 for p in self._pattern_cache if p.outcome == "LOSS")
        
        # Best performing contexts
        best_regime = max(
            self._performance_by_regime.items(),
            key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"] + 0.001),
            default=("UNKNOWN", {"wins": 0, "losses": 0, "total_pnl": 0})
        )
        
        best_strategy = max(
            self._performance_by_strategy.items(),
            key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"] + 0.001),
            default=("Momentum", {"wins": 0, "losses": 0, "total_pnl": 0})
        )
        
        best_hour = max(
            self._performance_by_hour.items(),
            key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"] + 0.001),
            default=(12, {"wins": 0, "losses": 0, "total_pnl": 0})
        )
        
        return {
            "total_patterns": total_patterns,
            "overall_win_rate": wins / total_patterns if total_patterns > 0 else 0,
            "wins": wins,
            "losses": losses,
            "best_regime": {
                "name": best_regime[0],
                "stats": best_regime[1],
            },
            "best_strategy": {
                "name": best_strategy[0],
                "stats": best_strategy[1],
            },
            "best_hour_utc": {
                "hour": best_hour[0],
                "stats": best_hour[1],
            },
            "performance_by_regime": dict(self._performance_by_regime),
            "performance_by_strategy": dict(self._performance_by_strategy),
        }


# Singleton instance
_learning_engine: Optional[TradeLearningEngine] = None


def get_learning_engine() -> TradeLearningEngine:
    """Get the singleton learning engine instance."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = TradeLearningEngine()
    return _learning_engine


def refresh_learning_engine(limit: int = 1000) -> int:
    """Refresh the learning engine with latest trade data."""
    engine = get_learning_engine()
    return engine.load_patterns_from_db(limit=limit)
