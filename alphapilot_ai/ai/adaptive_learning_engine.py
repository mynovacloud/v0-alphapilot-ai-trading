"""
Adaptive Learning Engine
========================

This is the advanced machine learning core of AlphaPilot that learns from:
1. Historical trade outcomes - What worked and what didn't
2. Market patterns - Recognizes repeating setups across different conditions
3. Strategy performance - Dynamically adjusts strategy weights
4. Entry/exit timing - Learns optimal holding periods and exit conditions
5. Trade feature similarity - kNN-based pattern matching from past trades
6. Market regime transitions - Adapts to changing market conditions

The engine maintains several "memory banks":
- Pattern Memory: Stores recognized market patterns with success rates
- Strategy Memory: Performance metrics per strategy per market regime
- Timing Memory: Optimal entry/exit patterns
- Mistake Memory: Common errors to avoid
- Trade Memory: Feature vectors from historical trades for similarity matching

All learning is incremental and persisted to the database so knowledge
accumulates over time without requiring retraining.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Literal, List, Dict, Tuple

from database.db import session_scope
from database.models import (
    AILearningMemory,
    ClaudeDecision,
    PaperTrade,
    TradeReflection,
    ActivityLog,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid circular dependencies
def _get_trade_learning_engine():
    from ai.trade_learning_engine import get_learning_engine
    return get_learning_engine()

def _get_advanced_regime_detector():
    from trading.market_regime import get_advanced_detector
    return get_advanced_detector()

def _get_strategy_selector():
    from trading.adaptive_strategy_selector import get_strategy_selector
    return get_strategy_selector()


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PatternSignature:
    """A recognized market pattern that can predict future moves."""
    name: str
    conditions: dict[str, Any]  # RSI range, volume state, regime, etc.
    success_rate: float  # 0-1 historical success
    sample_count: int
    avg_return: float  # Average return when pattern triggered
    avg_hold_time: float  # Average holding time in minutes
    confidence_boost: float  # How much to boost signal confidence
    last_seen: datetime
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "conditions": self.conditions,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "avg_return": self.avg_return,
            "avg_hold_time": self.avg_hold_time,
            "confidence_boost": self.confidence_boost,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PatternSignature":
        return cls(
            name=d.get("name", "unknown"),
            conditions=d.get("conditions", {}),
            success_rate=d.get("success_rate", 0.5),
            sample_count=d.get("sample_count", 0),
            avg_return=d.get("avg_return", 0),
            avg_hold_time=d.get("avg_hold_time", 0),
            confidence_boost=d.get("confidence_boost", 0),
            last_seen=datetime.fromisoformat(d["last_seen"]) if d.get("last_seen") else utcnow(),
        )


@dataclass
class StrategyPerformance:
    """Performance tracking for a strategy in a specific regime."""
    strategy_name: str
    regime: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.5
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    weight: float = 1.0  # Dynamic weight for this strategy
    last_updated: datetime = field(default_factory=utcnow)
    
    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "regime": self.regime,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": self.total_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "weight": self.weight,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrategyPerformance":
        return cls(
            strategy_name=d.get("strategy_name", "unknown"),
            regime=d.get("regime", "UNKNOWN"),
            trades=d.get("trades", 0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            total_pnl=d.get("total_pnl", 0),
            avg_win=d.get("avg_win", 0),
            avg_loss=d.get("avg_loss", 0),
            win_rate=d.get("win_rate", 0.5),
            profit_factor=d.get("profit_factor", 1.0),
            sharpe_ratio=d.get("sharpe_ratio", 0),
            weight=d.get("weight", 1.0),
            last_updated=datetime.fromisoformat(d["last_updated"]) if d.get("last_updated") else utcnow(),
        )


@dataclass
class TimingPattern:
    """Learned optimal timing for entries and exits."""
    pattern_type: Literal["entry", "exit"]
    conditions: dict[str, Any]
    optimal_action: str  # "immediate", "wait_pullback", "scale_in", etc.
    avg_improvement: float  # % improvement vs baseline
    sample_count: int
    confidence: float


@dataclass
class AdaptiveRecommendation:
    """Output from the adaptive learning engine."""
    # Core recommendation
    recommended_action: Literal["BUY", "SELL", "HOLD", "WAIT"]
    confidence_adjustment: float  # -0.3 to +0.3 to apply to base confidence
    size_multiplier: float  # 0.5 to 1.5
    
    # Strategy recommendation
    preferred_strategy: str
    strategy_weight: float
    
    # Pattern matches
    matched_patterns: list[PatternSignature]
    pattern_confidence_boost: float
    
    # Timing
    entry_timing: str  # "immediate", "wait_pullback", "scale_in"
    suggested_hold_time: float  # minutes
    
    # Risk adjustments
    stop_loss_multiplier: float
    take_profit_multiplier: float
    
    # Learning context
    similar_past_trades: int
    historical_success_rate: float
    
    # Explanations for Claude
    reasoning: list[str]
    warnings: list[str]
    
    # --- Fields with defaults must come after fields without defaults ---
    # Trade Learning Engine insights
    ml_prediction: Optional[Dict[str, Any]] = None  # From trade_learning_engine
    ml_confidence_adjustment: float = 0.0
    ml_reasoning: List[str] = field(default_factory=list)
    
    # Advanced regime analysis
    regime_analysis: Optional[Dict[str, Any]] = None
    regime_recommended_strategy: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "recommended_action": self.recommended_action,
            "confidence_adjustment": self.confidence_adjustment,
            "size_multiplier": self.size_multiplier,
            "preferred_strategy": self.preferred_strategy,
            "strategy_weight": self.strategy_weight,
            "matched_patterns": [p.to_dict() for p in self.matched_patterns],
            "pattern_confidence_boost": self.pattern_confidence_boost,
            "ml_prediction": self.ml_prediction,
            "ml_confidence_adjustment": self.ml_confidence_adjustment,
            "ml_reasoning": self.ml_reasoning,
            "entry_timing": self.entry_timing,
            "suggested_hold_time": self.suggested_hold_time,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "take_profit_multiplier": self.take_profit_multiplier,
            "similar_past_trades": self.similar_past_trades,
            "historical_success_rate": self.historical_success_rate,
            "regime_analysis": self.regime_analysis,
            "regime_recommended_strategy": self.regime_recommended_strategy,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
        }


# ============================================================================
# Pattern Recognition
# ============================================================================

class PatternRecognizer:
    """
    Recognizes market patterns by analyzing indicator states and correlating
    with historical trade outcomes.
    """
    
    # Predefined pattern templates that we look for
    PATTERN_TEMPLATES = {
        "oversold_bounce": {
            "description": "RSI oversold with volume spike - potential reversal",
            "conditions": {
                "rsi_range": (0, 30),
                "volume_ratio_min": 1.3,
                "trend_not": "STRONG_DOWN",
            },
            "expected_direction": "BUY",
        },
        "overbought_reversal": {
            "description": "RSI overbought with declining volume - potential top",
            "conditions": {
                "rsi_range": (70, 100),
                "volume_ratio_max": 0.8,
            },
            "expected_direction": "SELL",
        },
        "breakout_volume": {
            "description": "Price breakout with strong volume confirmation",
            "conditions": {
                "bb_percent_b_min": 1.0,
                "volume_ratio_min": 1.5,
                "adx_min": 20,
            },
            "expected_direction": "BUY",
        },
        "breakdown_volume": {
            "description": "Price breakdown with volume confirmation",
            "conditions": {
                "bb_percent_b_max": 0.0,
                "volume_ratio_min": 1.5,
                "adx_min": 20,
            },
            "expected_direction": "SELL",
        },
        "momentum_continuation": {
            "description": "Strong trend with aligned indicators",
            "conditions": {
                "adx_min": 25,
                "macd_histogram_positive": True,
                "trend": "UP",
            },
            "expected_direction": "BUY",
        },
        "mean_reversion_setup": {
            "description": "Extended from mean in ranging market",
            "conditions": {
                "regime": "RANGING",
                "bb_percent_b_range": (-0.1, 0.1),
            },
            "expected_direction": "BUY",
        },
        "fear_capitulation": {
            "description": "Extreme fear with oversold conditions",
            "conditions": {
                "fear_greed_max": 25,
                "rsi_range": (0, 35),
            },
            "expected_direction": "BUY",
        },
        "greed_exhaustion": {
            "description": "Extreme greed with overbought conditions", 
            "conditions": {
                "fear_greed_min": 75,
                "rsi_range": (65, 100),
            },
            "expected_direction": "SELL",
        },
        "divergence_bullish": {
            "description": "Price making lower lows but RSI making higher lows",
            "conditions": {
                "divergence_type": "bullish",
            },
            "expected_direction": "BUY",
        },
        "divergence_bearish": {
            "description": "Price making higher highs but RSI making lower highs",
            "conditions": {
                "divergence_type": "bearish",
            },
            "expected_direction": "SELL",
        },
        "mtf_alignment_bullish": {
            "description": "All timeframes aligned bullish",
            "conditions": {
                "mtf_alignment_min": 0.8,
                "mtf_bias": "BULLISH",
            },
            "expected_direction": "BUY",
        },
        "mtf_alignment_bearish": {
            "description": "All timeframes aligned bearish",
            "conditions": {
                "mtf_alignment_min": 0.8,
                "mtf_bias": "BEARISH",
            },
            "expected_direction": "SELL",
        },
    }
    
    def __init__(self):
        self.learned_patterns: dict[str, PatternSignature] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from database."""
        with session_scope() as s:
            rows = s.query(AILearningMemory).filter(
                AILearningMemory.category == "pattern"
            ).all()
            for row in rows:
                try:
                    data = json.loads(row.content) if row.content else {}
                    if "name" in data:
                        self.learned_patterns[data["name"]] = PatternSignature.from_dict(data)
                except Exception:
                    continue
    
    def _save_pattern(self, pattern: PatternSignature):
        """Save or update a pattern in the database."""
        with session_scope() as s:
            existing = s.query(AILearningMemory).filter(
                AILearningMemory.category == "pattern",
                AILearningMemory.content.like(f'%"name": "{pattern.name}"%')
            ).first()
            
            if existing:
                existing.content = json.dumps(pattern.to_dict())
                existing.weight = pattern.success_rate * pattern.sample_count / 10  # Higher weight for proven patterns
            else:
                s.add(AILearningMemory(
                    category="pattern",
                    content=json.dumps(pattern.to_dict()),
                    weight=pattern.success_rate,
                ))
    
    def match_patterns(self, market_state: dict) -> list[PatternSignature]:
        """
        Find patterns that match the current market state.
        
        Args:
            market_state: Dict containing current indicator values:
                - rsi: RSI value
                - macd_histogram: MACD histogram value
                - bb_percent_b: Bollinger Band %B
                - adx: ADX value
                - volume_ratio: Current volume / average volume
                - regime: Market regime string
                - fear_greed: Fear & Greed index value
                - mtf_alignment: Multi-timeframe alignment score
                - mtf_bias: Overall MTF bias
                - trend: Trend direction
        
        Returns:
            List of matched PatternSignature objects
        """
        matched = []
        
        for template_name, template in self.PATTERN_TEMPLATES.items():
            if self._check_conditions(market_state, template["conditions"]):
                # Check if we have learned data about this pattern
                learned = self.learned_patterns.get(template_name)
                
                if learned and learned.sample_count >= 5:
                    matched.append(learned)
                else:
                    # Create a new pattern signature with default values
                    matched.append(PatternSignature(
                        name=template_name,
                        conditions=template["conditions"],
                        success_rate=0.5,  # Unknown, assume 50/50
                        sample_count=0,
                        avg_return=0,
                        avg_hold_time=60,  # Default 1 hour
                        confidence_boost=0.05,  # Small boost for pattern match
                        last_seen=utcnow(),
                    ))
        
        return matched
    
    def _check_conditions(self, state: dict, conditions: dict) -> bool:
        """Check if market state matches pattern conditions."""
        for key, value in conditions.items():
            if key == "rsi_range":
                rsi = state.get("rsi", 50)
                if not (value[0] <= rsi <= value[1]):
                    return False
            
            elif key == "volume_ratio_min":
                if state.get("volume_ratio", 1.0) < value:
                    return False
            
            elif key == "volume_ratio_max":
                if state.get("volume_ratio", 1.0) > value:
                    return False
            
            elif key == "bb_percent_b_min":
                if state.get("bb_percent_b", 0.5) < value:
                    return False
            
            elif key == "bb_percent_b_max":
                if state.get("bb_percent_b", 0.5) > value:
                    return False
            
            elif key == "bb_percent_b_range":
                bb = state.get("bb_percent_b", 0.5)
                if not (value[0] <= bb <= value[1]):
                    return False
            
            elif key == "adx_min":
                if state.get("adx", 0) < value:
                    return False
            
            elif key == "macd_histogram_positive":
                if value and state.get("macd_histogram", 0) <= 0:
                    return False
                if not value and state.get("macd_histogram", 0) > 0:
                    return False
            
            elif key == "regime":
                if state.get("regime", "").upper() != value.upper():
                    return False
            
            elif key == "trend":
                if state.get("trend", "").upper() != value.upper():
                    return False
            
            elif key == "trend_not":
                if state.get("trend", "").upper() == value.upper():
                    return False
            
            elif key == "fear_greed_min":
                if state.get("fear_greed", 50) < value:
                    return False
            
            elif key == "fear_greed_max":
                if state.get("fear_greed", 50) > value:
                    return False
            
            elif key == "mtf_alignment_min":
                if state.get("mtf_alignment", 0) < value:
                    return False
            
            elif key == "mtf_bias":
                if state.get("mtf_bias", "").upper() != value.upper():
                    return False
        
        return True
    
    def update_pattern_from_trade(self, pattern_name: str, trade_outcome: dict):
        """
        Update a pattern's statistics based on a trade outcome.
        
        Args:
            pattern_name: Name of the pattern
            trade_outcome: Dict with 'success' (bool), 'pnl', 'hold_time_min'
        """
        pattern = self.learned_patterns.get(pattern_name)
        
        if pattern is None:
            template = self.PATTERN_TEMPLATES.get(pattern_name, {})
            pattern = PatternSignature(
                name=pattern_name,
                conditions=template.get("conditions", {}),
                success_rate=0.5,
                sample_count=0,
                avg_return=0,
                avg_hold_time=60,
                confidence_boost=0,
                last_seen=utcnow(),
            )
        
        # Incremental update
        n = pattern.sample_count
        success = 1.0 if trade_outcome.get("success", False) else 0.0
        pnl = trade_outcome.get("pnl", 0)
        hold_time = trade_outcome.get("hold_time_min", 60)
        
        # Update running averages
        pattern.sample_count = n + 1
        pattern.success_rate = (pattern.success_rate * n + success) / (n + 1)
        pattern.avg_return = (pattern.avg_return * n + pnl) / (n + 1)
        pattern.avg_hold_time = (pattern.avg_hold_time * n + hold_time) / (n + 1)
        pattern.last_seen = utcnow()
        
        # Calculate confidence boost based on success rate
        # Patterns with > 60% success get positive boost, < 40% get negative
        if pattern.sample_count >= 10:
            pattern.confidence_boost = (pattern.success_rate - 0.5) * 0.3
        else:
            pattern.confidence_boost = (pattern.success_rate - 0.5) * 0.1
        
        self.learned_patterns[pattern_name] = pattern
        self._save_pattern(pattern)


# ============================================================================
# Strategy Performance Tracker
# ============================================================================

class StrategyPerformanceTracker:
    """
    Tracks performance of each strategy across different market regimes
    and dynamically adjusts strategy weights.
    """
    
    def __init__(self):
        self.performance: dict[str, dict[str, StrategyPerformance]] = defaultdict(dict)
        self._load_performance()
    
    def _load_performance(self):
        """Load performance data from database."""
        with session_scope() as s:
            rows = s.query(AILearningMemory).filter(
                AILearningMemory.category == "strategy_performance"
            ).all()
            for row in rows:
                try:
                    data = json.loads(row.content) if row.content else {}
                    perf = StrategyPerformance.from_dict(data)
                    self.performance[perf.strategy_name][perf.regime] = perf
                except Exception:
                    continue
    
    def _save_performance(self, perf: StrategyPerformance):
        """Save performance data to database."""
        key = f"{perf.strategy_name}_{perf.regime}"
        with session_scope() as s:
            existing = s.query(AILearningMemory).filter(
                AILearningMemory.category == "strategy_performance",
                AILearningMemory.content.like(f'%"strategy_name": "{perf.strategy_name}"%'),
                AILearningMemory.content.like(f'%"regime": "{perf.regime}"%'),
            ).first()
            
            if existing:
                existing.content = json.dumps(perf.to_dict())
                existing.weight = perf.weight
            else:
                s.add(AILearningMemory(
                    category="strategy_performance",
                    content=json.dumps(perf.to_dict()),
                    weight=perf.weight,
                ))
    
    def record_trade(
        self,
        strategy_name: str,
        regime: str,
        pnl: float,
        is_win: bool,
    ):
        """Record a trade outcome for a strategy."""
        if strategy_name not in self.performance:
            self.performance[strategy_name] = {}
        
        if regime not in self.performance[strategy_name]:
            self.performance[strategy_name][regime] = StrategyPerformance(
                strategy_name=strategy_name,
                regime=regime,
            )
        
        perf = self.performance[strategy_name][regime]
        
        # Update stats
        perf.trades += 1
        perf.total_pnl += pnl
        
        if is_win:
            perf.wins += 1
            # Update average win (running average)
            perf.avg_win = (perf.avg_win * (perf.wins - 1) + pnl) / perf.wins
        else:
            perf.losses += 1
            # Update average loss (running average)
            perf.avg_loss = (perf.avg_loss * (perf.losses - 1) + abs(pnl)) / perf.losses
        
        # Recalculate metrics
        perf.win_rate = perf.wins / perf.trades if perf.trades > 0 else 0.5
        
        gross_wins = perf.avg_win * perf.wins
        gross_losses = perf.avg_loss * perf.losses
        perf.profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
            float('inf') if gross_wins > 0 else 1.0
        )
        
        # Update weight based on performance
        # Weight is based on: win_rate * profit_factor * sqrt(trades)
        # This rewards both winning percentage and consistency
        perf.weight = self._calculate_weight(perf)
        perf.last_updated = utcnow()
        
        self._save_performance(perf)
    
    def _calculate_weight(self, perf: StrategyPerformance) -> float:
        """Calculate dynamic weight for a strategy."""
        if perf.trades < 5:
            return 1.0  # Not enough data, use neutral weight
        
        # Base weight from win rate (0.5 = 1.0, 0.6 = 1.2, 0.4 = 0.8)
        win_rate_factor = 0.5 + perf.win_rate
        
        # Profit factor contribution (clamped 0.5-2.0)
        pf = min(2.0, perf.profit_factor) if perf.profit_factor != float('inf') else 2.0
        pf_factor = 0.5 + pf / 4  # 0.5-1.0 range
        
        # Sample size confidence (more trades = more confidence)
        sample_factor = min(1.0, math.sqrt(perf.trades / 50))
        
        # Combined weight (0.25 to 2.0 range)
        weight = win_rate_factor * pf_factor * (0.5 + 0.5 * sample_factor)
        return max(0.25, min(2.0, weight))
    
    def get_strategy_weight(self, strategy_name: str, regime: str) -> float:
        """Get the current weight for a strategy in a regime."""
        if strategy_name in self.performance and regime in self.performance[strategy_name]:
            return self.performance[strategy_name][regime].weight
        return 1.0  # Default weight
    
    def get_best_strategy(self, regime: str) -> tuple[str, float]:
        """Get the best performing strategy for a regime."""
        best_strategy = "Momentum"
        best_weight = 1.0
        
        for strategy_name, regimes in self.performance.items():
            if regime in regimes:
                perf = regimes[regime]
                if perf.weight > best_weight and perf.trades >= 10:
                    best_strategy = strategy_name
                    best_weight = perf.weight
        
        return best_strategy, best_weight
    
    def get_performance_summary(self) -> dict:
        """Get a summary of all strategy performance."""
        summary = {}
        for strategy, regimes in self.performance.items():
            summary[strategy] = {}
            for regime, perf in regimes.items():
                summary[strategy][regime] = {
                    "trades": perf.trades,
                    "win_rate": round(perf.win_rate * 100, 1),
                    "profit_factor": round(perf.profit_factor, 2) if perf.profit_factor != float('inf') else "inf",
                    "total_pnl": round(perf.total_pnl, 2),
                    "weight": round(perf.weight, 2),
                }
        return summary


# ============================================================================
# Main Adaptive Learning Engine
# ============================================================================

class AdaptiveLearningEngine:
    """
    The main learning engine that combines pattern recognition, strategy
    tracking, historical analysis, and ML-based predictions to provide 
    adaptive trading recommendations.
    
    Now integrates:
    - Trade Learning Engine: kNN-based similarity matching
    - Advanced Regime Detector: Multi-method regime classification  
    - Adaptive Strategy Selector: UCB-based strategy selection
    """
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.strategy_tracker = StrategyPerformanceTracker()
        self._mistake_patterns: list[dict] = []
        self._trade_learning_engine = None
        self._regime_detector = None
        self._strategy_selector = None
        self._load_mistakes()
    
    def _get_trade_learner(self):
        """Lazy load trade learning engine."""
        if self._trade_learning_engine is None:
            try:
                self._trade_learning_engine = _get_trade_learning_engine()
                self._trade_learning_engine.load_patterns_from_db()
            except Exception as e:
                logger.warning(f"Could not load trade learning engine: {e}")
        return self._trade_learning_engine
    
    def _get_regime_detector(self):
        """Lazy load regime detector."""
        if self._regime_detector is None:
            try:
                self._regime_detector = _get_advanced_regime_detector()
            except Exception as e:
                logger.warning(f"Could not load regime detector: {e}")
        return self._regime_detector
    
    def _get_strat_selector(self):
        """Lazy load strategy selector."""
        if self._strategy_selector is None:
            try:
                self._strategy_selector = _get_strategy_selector()
                self._strategy_selector.load_performance_data()
            except Exception as e:
                logger.warning(f"Could not load strategy selector: {e}")
        return self._strategy_selector
    
    def _load_mistakes(self):
        """Load common mistake patterns from the database."""
        with session_scope() as s:
            rows = s.query(AILearningMemory).filter(
                AILearningMemory.category == "mistake"
            ).order_by(AILearningMemory.weight.desc()).limit(20).all()
            
            self._mistake_patterns = []
            for row in rows:
                try:
                    data = json.loads(row.content) if row.content else {}
                    self._mistake_patterns.append(data)
                except Exception:
                    self._mistake_patterns.append({"content": row.content})
    
    def analyze(
        self,
        signal_direction: str,  # BUY, SELL, HOLD
        signal_confidence: float,
        strategy_name: str,
        market_state: dict,
        symbol: str,
        wallet_id: Optional[int] = None,
    ) -> AdaptiveRecommendation:
        """
        Analyze a trading signal with historical learning context.
        
        Args:
            signal_direction: The raw signal direction
            signal_confidence: The base confidence (0-1)
            strategy_name: Which strategy generated this signal
            market_state: Current market indicators
            symbol: Trading symbol
            wallet_id: Optional wallet for personalized learning
        
        Returns:
            AdaptiveRecommendation with adjustments and context
        """
        reasoning = []
        warnings = []
        
        # 1. Pattern Recognition
        matched_patterns = self.pattern_recognizer.match_patterns(market_state)
        pattern_boost = 0.0
        
        for pattern in matched_patterns:
            if pattern.sample_count >= 10:
                pattern_boost += pattern.confidence_boost
                reasoning.append(
                    f"Pattern '{pattern.name}' matched: {pattern.success_rate*100:.0f}% success rate "
                    f"over {pattern.sample_count} trades, avg return {pattern.avg_return:.2f}%"
                )
            elif pattern.sample_count > 0:
                pattern_boost += pattern.confidence_boost * 0.5  # Less weight for new patterns
                reasoning.append(
                    f"Pattern '{pattern.name}' matched (limited data: {pattern.sample_count} trades)"
                )
        
        # Cap pattern boost
        pattern_boost = max(-0.15, min(0.15, pattern_boost))
        
        # 2. Strategy Performance
        regime = market_state.get("regime", "UNKNOWN")
        strategy_weight = self.strategy_tracker.get_strategy_weight(strategy_name, regime)
        best_strategy, best_weight = self.strategy_tracker.get_best_strategy(regime)
        
        if strategy_weight < 0.7:
            warnings.append(
                f"Strategy '{strategy_name}' has poor performance in {regime} regime "
                f"(weight: {strategy_weight:.2f}). Consider '{best_strategy}' instead."
            )
        elif strategy_weight > 1.3:
            reasoning.append(
                f"Strategy '{strategy_name}' performs well in {regime} regime "
                f"(weight: {strategy_weight:.2f})"
            )
        
        # 3. Historical Similar Trades
        similar_trades = self._find_similar_trades(
            symbol=symbol,
            direction=signal_direction,
            regime=regime,
            strategy=strategy_name,
        )
        
        historical_success = 0.5
        if similar_trades["count"] >= 5:
            historical_success = similar_trades["success_rate"]
            reasoning.append(
                f"Found {similar_trades['count']} similar historical trades: "
                f"{historical_success*100:.0f}% success rate, avg P&L ${similar_trades['avg_pnl']:.2f}"
            )
        
        # 4. Check for Mistake Patterns
        for mistake in self._mistake_patterns:
            if self._matches_mistake(market_state, signal_direction, mistake):
                warnings.append(f"Potential mistake pattern: {mistake.get('content', 'Unknown pattern')}")
        
        # =====================================================================
        # 5. ML-BASED INSIGHTS FROM TRADE LEARNING ENGINE
        # =====================================================================
        ml_prediction = None
        ml_confidence_adj = 0.0
        ml_reasoning = []
        
        try:
            trade_learner = self._get_trade_learner()
            if trade_learner:
                # Build features from market state
                from ai.trade_learning_engine import TradeFeatures
                
                features = TradeFeatures(
                    rsi=float(market_state.get("rsi", 50)),
                    macd_histogram=float(market_state.get("macd_histogram", 0)),
                    bollinger_percent_b=float(market_state.get("bb_percent_b", 0.5)),
                    adx=float(market_state.get("adx", 20)),
                    relative_volume=float(market_state.get("volume_ratio", 1)),
                    volatility_percentile=float(market_state.get("volatility_percentile", 50)),
                    regime=regime,
                    trend_strength=float(market_state.get("trend_strength", 0)),
                    hour_of_day=datetime.utcnow().hour,
                    day_of_week=datetime.utcnow().weekday(),
                    signal_confidence=signal_confidence,
                    strategy_type=strategy_name,
                    side=signal_direction,
                    recent_win_rate=historical_success,
                )
                
                # Get ML prediction
                ml_prediction = trade_learner.predict_outcome(features)
                
                if ml_prediction.get("similar_patterns", 0) >= 5:
                    win_prob = ml_prediction.get("win_probability", 0.5)
                    predicted_outcome = ml_prediction.get("predicted_outcome", "UNCERTAIN")
                    confidence = ml_prediction.get("confidence", 0)
                    
                    # Apply ML confidence adjustment
                    if predicted_outcome == "WIN" and confidence > 0.3:
                        ml_adj = min(0.12, confidence * 0.15)
                        ml_confidence_adj += ml_adj
                        ml_reasoning.append(
                            f"ML predicts WIN ({win_prob:.0%} probability) based on "
                            f"{ml_prediction['similar_patterns']} similar patterns"
                        )
                    elif predicted_outcome == "LOSS" and confidence > 0.3:
                        ml_adj = min(0.12, confidence * 0.15)
                        ml_confidence_adj -= ml_adj
                        ml_reasoning.append(
                            f"ML predicts LOSS ({1-win_prob:.0%} probability) - caution advised"
                        )
                        warnings.append(f"ML model predicts unfavorable outcome")
                    
                    expected_pnl = ml_prediction.get("expected_pnl_pct", 0)
                    if expected_pnl != 0:
                        ml_reasoning.append(f"Expected PnL: {expected_pnl:+.2f}%")
                
                # Get confidence adjustment from learning engine
                adjusted_conf, adj_reasons = trade_learner.get_confidence_adjustment(
                    features, signal_confidence
                )
                for reason in adj_reasons:
                    ml_reasoning.append(reason)
                
                # Add additional ML adjustment
                if adj_reasons:
                    extra_adj = adjusted_conf - signal_confidence
                    ml_confidence_adj += extra_adj * 0.5  # Partial weight
        except Exception as e:
            logger.debug(f"ML prediction unavailable: {e}")
        
        # =====================================================================
        # 6. ADAPTIVE STRATEGY SELECTION
        # =====================================================================
        regime_recommended_strategy = None
        regime_analysis_dict = None
        
        try:
            strat_selector = self._get_strat_selector()
            if strat_selector:
                # Get strategy selection based on current regime
                selection = strat_selector.select_strategy(symbol=symbol)
                
                if selection.confidence > 0.6:
                    regime_recommended_strategy = selection.selected_strategy.value
                    
                    if regime_recommended_strategy != strategy_name:
                        if selection.confidence > 0.75:
                            warnings.append(
                                f"Strategy selector recommends '{regime_recommended_strategy}' "
                                f"over '{strategy_name}' with {selection.confidence:.0%} confidence"
                            )
                        else:
                            reasoning.append(
                                f"Alternative strategy '{regime_recommended_strategy}' "
                                f"may perform better in current conditions"
                            )
        except Exception as e:
            logger.debug(f"Strategy selector unavailable: {e}")
        
        # =====================================================================
        # 7. Calculate Final Adjustments
        # =====================================================================
        # Base confidence adjustment from patterns
        confidence_adj = pattern_boost
        
        # Adjust based on strategy weight
        if strategy_weight != 1.0:
            confidence_adj += (strategy_weight - 1.0) * 0.1
        
        # Adjust based on historical similar trades
        if similar_trades["count"] >= 10:
            confidence_adj += (historical_success - 0.5) * 0.15
        
        # Add ML-based adjustment
        confidence_adj += ml_confidence_adj
        
        # Clamp total adjustment
        confidence_adj = max(-0.35, min(0.35, confidence_adj))
        
        # =====================================================================
        # 8. Size and Timing Recommendations
        # =====================================================================
        size_mult = 1.0
        
        # ML prediction affects sizing
        if ml_prediction and ml_prediction.get("predicted_outcome") == "WIN":
            size_mult *= 1.1
        elif ml_prediction and ml_prediction.get("predicted_outcome") == "LOSS":
            size_mult *= 0.8
        
        if strategy_weight > 1.2 and len(matched_patterns) > 0:
            size_mult = min(1.3, size_mult * strategy_weight)
            reasoning.append(f"Increased position size to {size_mult:.1f}x due to favorable conditions")
        elif strategy_weight < 0.8 or len(warnings) >= 2:
            size_mult = max(0.5, size_mult * strategy_weight)
            reasoning.append(f"Reduced position size to {size_mult:.1f}x due to caution signals")
        
        # Determine entry timing based on patterns
        entry_timing = "immediate"
        if any(p.name.endswith("_pullback") for p in matched_patterns):
            entry_timing = "wait_pullback"
        elif regime == "VOLATILE":
            entry_timing = "scale_in"
            reasoning.append("Scale-in entry recommended due to volatile conditions")
        
        # Suggested hold time from patterns
        if matched_patterns:
            avg_hold = sum(p.avg_hold_time for p in matched_patterns) / len(matched_patterns)
            suggested_hold = avg_hold
        else:
            suggested_hold = 60.0  # Default 1 hour
        
        # Risk parameter adjustments
        stop_mult = 1.0
        tp_mult = 1.0
        
        if regime == "VOLATILE":
            stop_mult = 1.5
            tp_mult = 1.3
        elif regime == "RANGING":
            stop_mult = 0.8
            tp_mult = 0.8
        elif historical_success > 0.6:
            tp_mult = 1.2  # Let winners run more
        
        # ML prediction can adjust take profit expectations
        if ml_prediction:
            expected_pnl = ml_prediction.get("expected_pnl_pct", 0)
            if expected_pnl > 2.0:
                tp_mult *= 1.1
            elif expected_pnl < -1.0:
                stop_mult *= 0.9  # Tighter stop on predicted losers
        
        # =====================================================================
        # 9. Final Recommendation
        # =====================================================================
        recommended_action = signal_direction
        
        # ML strongly predicts loss
        if ml_prediction and ml_prediction.get("predicted_outcome") == "LOSS":
            if ml_prediction.get("confidence", 0) > 0.6:
                recommended_action = "WAIT"
                ml_reasoning.append("Strong ML prediction of loss - recommending WAIT")
        
        if signal_direction in ("BUY", "SELL") and len(warnings) >= 3:
            recommended_action = "WAIT"
            reasoning.append("Multiple warnings suggest waiting for better setup")
        
        # Combine all reasoning
        all_reasoning = reasoning + ml_reasoning
        
        return AdaptiveRecommendation(
            recommended_action=recommended_action,
            confidence_adjustment=confidence_adj,
            size_multiplier=size_mult,
            preferred_strategy=regime_recommended_strategy or (best_strategy if best_weight > strategy_weight + 0.3 else strategy_name),
            strategy_weight=strategy_weight,
            matched_patterns=matched_patterns,
            pattern_confidence_boost=pattern_boost,
            ml_prediction=ml_prediction,
            ml_confidence_adjustment=ml_confidence_adj,
            ml_reasoning=ml_reasoning,
            entry_timing=entry_timing,
            suggested_hold_time=suggested_hold,
            stop_loss_multiplier=stop_mult,
            take_profit_multiplier=tp_mult,
            similar_past_trades=similar_trades["count"],
            historical_success_rate=historical_success,
            regime_analysis=regime_analysis_dict,
            regime_recommended_strategy=regime_recommended_strategy,
            reasoning=all_reasoning,
            warnings=warnings,
        )
    
    def _find_similar_trades(
        self,
        symbol: str,
        direction: str,
        regime: str,
        strategy: str,
    ) -> dict:
        """Find similar historical trades."""
        with session_scope() as s:
            # Query closed trades with similar characteristics
            trades = s.query(PaperTrade).filter(
                PaperTrade.status == "closed",
                PaperTrade.symbol == symbol,
                PaperTrade.side == direction,
            ).order_by(PaperTrade.closed_at.desc()).limit(50).all()
            
            if not trades:
                return {"count": 0, "success_rate": 0.5, "avg_pnl": 0}
            
            wins = sum(1 for t in trades if (t.realized_pnl or 0) > 0)
            total_pnl = sum(float(t.realized_pnl or 0) for t in trades)
            
            return {
                "count": len(trades),
                "success_rate": wins / len(trades) if trades else 0.5,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
            }
    
    def _matches_mistake(self, state: dict, direction: str, mistake: dict) -> bool:
        """Check if current conditions match a known mistake pattern."""
        content = mistake.get("content", "").lower()
        
        # Simple heuristic matching
        if "overbought" in content and direction == "BUY" and state.get("rsi", 50) > 70:
            return True
        if "oversold" in content and direction == "SELL" and state.get("rsi", 50) < 30:
            return True
        if "low volume" in content and state.get("volume_ratio", 1.0) < 0.7:
            return True
        if "against trend" in content:
            trend = state.get("trend", "").upper()
            if direction == "BUY" and trend == "DOWN":
                return True
            if direction == "SELL" and trend == "UP":
                return True
        
        return False
    
    def learn_from_trade(
        self,
        trade_id: int,
        patterns_at_entry: list[str],
        strategy_name: str,
        regime: str,
    ):
        """
        Called when a trade closes to update learning data.
        
        Args:
            trade_id: ID of the closed trade
            patterns_at_entry: Pattern names that matched at entry
            strategy_name: Strategy used for this trade
            regime: Market regime at entry
        """
        with session_scope() as s:
            trade = s.get(PaperTrade, trade_id)
            if not trade or trade.status != "closed":
                return
            
            pnl = float(trade.realized_pnl or 0)
            is_win = pnl > 0
            
            # Calculate hold time
            hold_time = 60.0  # Default
            if trade.opened_at and trade.closed_at:
                delta = trade.closed_at - trade.opened_at
                hold_time = delta.total_seconds() / 60.0
            
            # Update pattern statistics
            for pattern_name in patterns_at_entry:
                self.pattern_recognizer.update_pattern_from_trade(
                    pattern_name=pattern_name,
                    trade_outcome={
                        "success": is_win,
                        "pnl": pnl,
                        "hold_time_min": hold_time,
                    }
                )
            
            # Update strategy performance
            self.strategy_tracker.record_trade(
                strategy_name=strategy_name,
                regime=regime,
                pnl=pnl,
                is_win=is_win,
            )
            
            # Log the learning
            s.add(ActivityLog(
                category="learning",
                level="info",
                message=f"Learned from trade #{trade_id}: {strategy_name} in {regime}, "
                        f"PnL=${pnl:.2f}, patterns={patterns_at_entry}",
            ))
    
    def get_learning_stats(self) -> dict:
        """Get comprehensive learning statistics."""
        with session_scope() as s:
            pattern_count = s.query(AILearningMemory).filter(
                AILearningMemory.category == "pattern"
            ).count()
            
            strategy_count = s.query(AILearningMemory).filter(
                AILearningMemory.category == "strategy_performance"
            ).count()
            
            total_trades = s.query(PaperTrade).filter(
                PaperTrade.status == "closed"
            ).count()
        
        # Get top patterns
        top_patterns = []
        for name, pattern in sorted(
            self.pattern_recognizer.learned_patterns.items(),
            key=lambda x: x[1].success_rate * x[1].sample_count,
            reverse=True
        )[:5]:
            if pattern.sample_count >= 5:
                top_patterns.append({
                    "name": name,
                    "success_rate": round(pattern.success_rate * 100, 1),
                    "samples": pattern.sample_count,
                    "avg_return": round(pattern.avg_return, 2),
                })
        
        return {
            "patterns_learned": pattern_count,
            "strategy_regimes_tracked": strategy_count,
            "total_trades_analyzed": total_trades,
            "top_patterns": top_patterns,
            "strategy_performance": self.strategy_tracker.get_performance_summary(),
        }


# ============================================================================
# Singleton instance
# ============================================================================

_engine: Optional[AdaptiveLearningEngine] = None


def get_adaptive_engine() -> AdaptiveLearningEngine:
    """Get the singleton adaptive learning engine."""
    global _engine
    if _engine is None:
        _engine = AdaptiveLearningEngine()
    return _engine


def analyze_signal(
    signal_direction: str,
    signal_confidence: float,
    strategy_name: str,
    market_state: dict,
    symbol: str,
    wallet_id: Optional[int] = None,
) -> AdaptiveRecommendation:
    """Convenience function to analyze a signal."""
    return get_adaptive_engine().analyze(
        signal_direction=signal_direction,
        signal_confidence=signal_confidence,
        strategy_name=strategy_name,
        market_state=market_state,
        symbol=symbol,
        wallet_id=wallet_id,
    )


def learn_from_trade(
    trade_id: int,
    patterns_at_entry: list[str],
    strategy_name: str,
    regime: str,
):
    """Convenience function to learn from a trade."""
    get_adaptive_engine().learn_from_trade(
        trade_id=trade_id,
        patterns_at_entry=patterns_at_entry,
        strategy_name=strategy_name,
        regime=regime,
    )
