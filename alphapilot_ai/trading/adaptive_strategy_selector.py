"""
Adaptive Strategy Selector
==========================

Automatically selects the optimal trading strategy based on:
1. Current market regime
2. Historical performance of each strategy in similar conditions
3. Recent strategy performance (momentum)
4. Risk-adjusted returns (Sharpe-like metrics)

The selector uses a bandit-like approach where strategies that perform
well get more allocation, while maintaining exploration to avoid
getting stuck with suboptimal choices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

from database.db import session_scope
from database.models import PaperTrade, ClaudeDecision
from trading.market_regime import MarketRegime, RegimeAnalysis
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


class Strategy(str, Enum):
    """Available trading strategies."""
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    SCALPING = "Scalping"
    TREND_FOLLOWING = "Trend Following"
    BREAKOUT = "Breakout"
    VOLATILITY_BREAKOUT = "Volatility Breakout"
    PROBABILITY_EDGE = "Probability Edge"


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy: Strategy
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    recent_momentum: float = 0.0  # Performance trend over last N trades
    exploration_bonus: float = 0.1  # UCB exploration term
    
    # Performance by regime
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_score(self, regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate selection score for this strategy.
        
        Uses UCB1 (Upper Confidence Bound) formula:
        score = exploitation_value + exploration_bonus
        
        Where exploitation_value is based on risk-adjusted returns
        and exploration_bonus encourages trying less-used strategies.
        """
        # Base score from win rate and profit factor
        base_score = (self.win_rate * 0.4 + 
                      min(self.profit_factor / 3, 1) * 0.3 +
                      min(self.sharpe_ratio / 2, 1) * 0.3)
        
        # Regime-specific adjustment
        if regime and regime.value in self.regime_performance:
            regime_stats = self.regime_performance[regime.value]
            regime_wr = regime_stats.get("win_rate", 0.5)
            regime_weight = min(regime_stats.get("trades", 0) / 10, 1.0)  # More trades = more confidence
            base_score = base_score * (1 - regime_weight * 0.3) + regime_wr * regime_weight * 0.3
        
        # Recent momentum adjustment (hot hand effect)
        momentum_adj = self.recent_momentum * 0.1
        
        # Exploration bonus (UCB)
        exploration = self.exploration_bonus
        
        return base_score + momentum_adj + exploration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "recent_momentum": round(self.recent_momentum, 3),
            "regime_performance": self.regime_performance,
        }


@dataclass
class StrategySelection:
    """Result of strategy selection process."""
    selected_strategy: Strategy
    confidence: float
    reasoning: List[str]
    
    # Alternative strategies with their scores
    alternatives: List[Tuple[Strategy, float]]
    
    # Context used for selection
    regime: Optional[MarketRegime]
    regime_confidence: float
    
    # Performance data
    selected_performance: StrategyPerformance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected_strategy.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "alternatives": [(s.value, round(score, 3)) for s, score in self.alternatives],
            "regime": self.regime.value if self.regime else None,
            "regime_confidence": round(self.regime_confidence, 3),
            "performance": self.selected_performance.to_dict(),
        }


class AdaptiveStrategySelector:
    """
    Adaptive strategy selector using multi-armed bandit approach.
    
    Balances exploitation (using best-performing strategies) with
    exploration (trying less-used strategies to gather data).
    """
    
    def __init__(
        self,
        exploration_factor: float = 1.0,
        momentum_window: int = 20,
        min_trades_for_confidence: int = 10,
    ):
        """
        Initialize the selector.
        
        Args:
            exploration_factor: UCB exploration parameter (higher = more exploration)
            momentum_window: Number of recent trades to consider for momentum
            min_trades_for_confidence: Minimum trades before trusting a strategy's metrics
        """
        self.exploration_factor = exploration_factor
        self.momentum_window = momentum_window
        self.min_trades = min_trades_for_confidence
        
        # Performance tracking
        self._performance: Dict[Strategy, StrategyPerformance] = {}
        self._total_selections: int = 0
        self._selection_counts: Dict[Strategy, int] = defaultdict(int)
        
        # Regime-strategy mapping (prior knowledge)
        self._regime_priors: Dict[MarketRegime, List[Strategy]] = {
            MarketRegime.TRENDING_UP: [Strategy.MOMENTUM, Strategy.TREND_FOLLOWING],
            MarketRegime.TRENDING_DOWN: [Strategy.MOMENTUM, Strategy.TREND_FOLLOWING],
            MarketRegime.RANGING: [Strategy.MEAN_REVERSION, Strategy.SCALPING],
            MarketRegime.VOLATILE: [Strategy.SCALPING, Strategy.VOLATILITY_BREAKOUT],
            MarketRegime.ACCUMULATION: [Strategy.MEAN_REVERSION, Strategy.TREND_FOLLOWING],
            MarketRegime.DISTRIBUTION: [Strategy.MEAN_REVERSION, Strategy.TREND_FOLLOWING],
            MarketRegime.BREAKOUT: [Strategy.BREAKOUT, Strategy.MOMENTUM],
            MarketRegime.CONSOLIDATION: [Strategy.BREAKOUT, Strategy.SCALPING],
        }
        
        self._loaded = False
    
    def load_performance_data(self, lookback_days: int = 90) -> None:
        """Load historical performance data from database."""
        cutoff = utcnow() - timedelta(days=lookback_days)
        
        # Initialize all strategies
        for strategy in Strategy:
            self._performance[strategy] = StrategyPerformance(strategy=strategy)
        
        with session_scope() as s:
            # Get closed trades with strategy info from ClaudeDecision
            trades = (
                s.query(PaperTrade)
                .filter(
                    PaperTrade.status == "closed",
                    PaperTrade.closed_at >= cutoff,
                )
                .order_by(PaperTrade.closed_at.asc())
                .all()
            )
            
            # Track trades by strategy
            strategy_trades: Dict[Strategy, List[Dict]] = defaultdict(list)
            
            for trade in trades:
                # Find associated decision to get strategy
                decision = (
                    s.query(ClaudeDecision)
                    .filter(
                        ClaudeDecision.wallet_id == trade.wallet_id,
                        ClaudeDecision.symbol == trade.symbol,
                        ClaudeDecision.created_at <= trade.opened_at,
                    )
                    .order_by(ClaudeDecision.created_at.desc())
                    .first()
                )
                
                # Determine strategy from decision context
                strategy_name = "Momentum"  # Default
                regime_name = "UNKNOWN"
                
                if decision:
                    try:
                        import json
                        ctx = json.loads(decision.context_snapshot) if isinstance(
                            decision.context_snapshot, str
                        ) else (decision.context_snapshot or {})
                        
                        signal = ctx.get("technical_signal", {})
                        strategy_name = signal.get("strategy", "Momentum")
                        
                        regime_ctx = ctx.get("market_regime", {})
                        regime_name = regime_ctx.get("regime", "UNKNOWN")
                    except Exception:
                        pass
                
                # Map strategy name to enum
                try:
                    strategy = Strategy(strategy_name)
                except ValueError:
                    strategy = Strategy.MOMENTUM
                
                pnl = float(trade.realized_pnl or 0)
                entry = float(trade.entry_price or 1)
                qty = float(trade.qty or 1)
                pnl_pct = pnl / (entry * qty) * 100 if entry * qty > 0 else 0
                
                strategy_trades[strategy].append({
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "win": pnl > 0,
                    "regime": regime_name,
                    "timestamp": trade.closed_at,
                })
            
            # Calculate performance metrics for each strategy
            for strategy, trades_list in strategy_trades.items():
                perf = self._performance[strategy]
                
                if not trades_list:
                    continue
                
                perf.total_trades = len(trades_list)
                perf.wins = sum(1 for t in trades_list if t["win"])
                perf.losses = perf.total_trades - perf.wins
                perf.total_pnl = sum(t["pnl"] for t in trades_list)
                
                if perf.total_trades > 0:
                    perf.win_rate = perf.wins / perf.total_trades
                
                # Calculate avg win/loss
                wins_pnl = [t["pnl"] for t in trades_list if t["win"]]
                losses_pnl = [abs(t["pnl"]) for t in trades_list if not t["win"]]
                
                perf.avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
                perf.avg_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0
                
                # Profit factor
                gross_win = sum(wins_pnl)
                gross_loss = sum(losses_pnl)
                perf.profit_factor = gross_win / gross_loss if gross_loss > 0 else (
                    float('inf') if gross_win > 0 else 1.0
                )
                
                # Simplified Sharpe ratio (mean / std of returns)
                returns = [t["pnl_pct"] for t in trades_list]
                if len(returns) >= 5:
                    mean_ret = sum(returns) / len(returns)
                    var_ret = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                    std_ret = math.sqrt(var_ret) if var_ret > 0 else 1
                    perf.sharpe_ratio = mean_ret / std_ret if std_ret > 0 else 0
                
                # Recent momentum (last N trades vs overall)
                recent = trades_list[-self.momentum_window:]
                recent_wr = sum(1 for t in recent if t["win"]) / len(recent) if recent else 0.5
                perf.recent_momentum = recent_wr - perf.win_rate
                
                # Performance by regime
                regime_stats: Dict[str, Dict[str, float]] = defaultdict(
                    lambda: {"wins": 0, "losses": 0, "trades": 0, "pnl": 0}
                )
                for t in trades_list:
                    regime = t["regime"]
                    regime_stats[regime]["trades"] += 1
                    regime_stats[regime]["pnl"] += t["pnl"]
                    if t["win"]:
                        regime_stats[regime]["wins"] += 1
                    else:
                        regime_stats[regime]["losses"] += 1
                
                # Calculate win rate per regime
                for regime, stats in regime_stats.items():
                    if stats["trades"] > 0:
                        stats["win_rate"] = stats["wins"] / stats["trades"]
                
                perf.regime_performance = dict(regime_stats)
        
        # Calculate exploration bonus (UCB)
        self._update_exploration_bonuses()
        
        self._loaded = True
        logger.info(f"[ADAPTIVE_SELECTOR] Loaded performance data for {len(strategy_trades)} strategies")
    
    def _update_exploration_bonuses(self) -> None:
        """Update UCB exploration bonuses for all strategies."""
        total = sum(p.total_trades for p in self._performance.values()) + 1
        
        for strategy, perf in self._performance.items():
            n = perf.total_trades + 1
            # UCB1 formula: sqrt(2 * ln(total) / n)
            perf.exploration_bonus = self.exploration_factor * math.sqrt(
                2 * math.log(total) / n
            )
    
    def select_strategy(
        self,
        regime: Optional[RegimeAnalysis] = None,
        symbol: str = "",
        exclude_strategies: Optional[List[Strategy]] = None,
    ) -> StrategySelection:
        """
        Select the optimal strategy for current conditions.
        
        Args:
            regime: Current market regime analysis
            symbol: Trading symbol (for symbol-specific adjustments)
            exclude_strategies: Strategies to exclude from selection
        
        Returns:
            StrategySelection with selected strategy and reasoning
        """
        if not self._loaded:
            self.load_performance_data()
        
        exclude = set(exclude_strategies or [])
        available = [s for s in Strategy if s not in exclude]
        
        if not available:
            available = list(Strategy)
        
        # Get regime info
        current_regime = regime.regime if regime else None
        regime_confidence = regime.confidence if regime else 0.5
        
        # Calculate scores for all strategies
        scores: List[Tuple[Strategy, float, List[str]]] = []
        
        for strategy in available:
            perf = self._performance.get(strategy, StrategyPerformance(strategy=strategy))
            score = perf.get_score(current_regime)
            reasons = []
            
            # Apply regime prior boost
            if current_regime and current_regime in self._regime_priors:
                preferred = self._regime_priors[current_regime]
                if strategy in preferred:
                    boost = 0.15 * regime_confidence
                    score += boost
                    reasons.append(f"Preferred for {current_regime.value} regime (+{boost:.2f})")
            
            # Performance-based reasons
            if perf.total_trades >= self.min_trades:
                if perf.win_rate > 0.6:
                    reasons.append(f"Strong win rate: {perf.win_rate:.1%}")
                if perf.profit_factor > 1.5:
                    reasons.append(f"Good profit factor: {perf.profit_factor:.2f}")
                if perf.recent_momentum > 0.1:
                    reasons.append(f"Positive momentum: +{perf.recent_momentum:.1%}")
                elif perf.recent_momentum < -0.1:
                    score -= 0.1
                    reasons.append(f"Negative momentum: {perf.recent_momentum:.1%}")
            else:
                reasons.append(f"Limited data ({perf.total_trades} trades)")
            
            # Check regime-specific performance
            if current_regime and current_regime.value in perf.regime_performance:
                regime_stats = perf.regime_performance[current_regime.value]
                if regime_stats.get("trades", 0) >= 5:
                    regime_wr = regime_stats.get("win_rate", 0.5)
                    if regime_wr > 0.6:
                        reasons.append(f"Strong in {current_regime.value}: {regime_wr:.1%}")
                    elif regime_wr < 0.4:
                        score -= 0.1
                        reasons.append(f"Weak in {current_regime.value}: {regime_wr:.1%}")
            
            scores.append((strategy, score, reasons))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best strategy
        best_strategy, best_score, best_reasons = scores[0]
        
        # Calculate confidence based on score margin
        if len(scores) > 1:
            margin = best_score - scores[1][1]
            confidence = min(0.95, 0.5 + margin * 2)
        else:
            confidence = 0.7
        
        # Track selection
        self._total_selections += 1
        self._selection_counts[best_strategy] += 1
        
        return StrategySelection(
            selected_strategy=best_strategy,
            confidence=confidence,
            reasoning=best_reasons,
            alternatives=[(s, score) for s, score, _ in scores[1:4]],
            regime=current_regime,
            regime_confidence=regime_confidence,
            selected_performance=self._performance.get(
                best_strategy, StrategyPerformance(strategy=best_strategy)
            ),
        )
    
    def get_all_performance(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        if not self._loaded:
            self.load_performance_data()
        
        return {
            "strategies": {
                s.value: p.to_dict() for s, p in self._performance.items()
            },
            "total_selections": self._total_selections,
            "selection_distribution": {
                s.value: count for s, count in self._selection_counts.items()
            },
        }
    
    def get_best_strategy_for_regime(self, regime: MarketRegime) -> Tuple[Strategy, float]:
        """Get the best performing strategy for a specific regime."""
        if not self._loaded:
            self.load_performance_data()
        
        best_strategy = Strategy.MOMENTUM
        best_score = 0.0
        
        for strategy, perf in self._performance.items():
            if regime.value in perf.regime_performance:
                stats = perf.regime_performance[regime.value]
                if stats.get("trades", 0) >= 5:
                    score = stats.get("win_rate", 0) * 0.6 + min(stats.get("pnl", 0) / 100, 0.4)
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
        
        # Fall back to prior knowledge if no data
        if best_score == 0 and regime in self._regime_priors:
            best_strategy = self._regime_priors[regime][0]
            best_score = 0.5
        
        return best_strategy, best_score
    
    def record_trade_outcome(
        self,
        strategy: Strategy,
        regime: MarketRegime,
        pnl: float,
        pnl_pct: float,
    ) -> None:
        """
        Record a trade outcome for online learning.
        
        This allows the selector to update without reloading from DB.
        """
        perf = self._performance.get(strategy)
        if not perf:
            perf = StrategyPerformance(strategy=strategy)
            self._performance[strategy] = perf
        
        perf.total_trades += 1
        if pnl > 0:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.total_pnl += pnl
        perf.win_rate = perf.wins / perf.total_trades if perf.total_trades > 0 else 0.5
        
        # Update regime-specific stats
        if regime.value not in perf.regime_performance:
            perf.regime_performance[regime.value] = {
                "wins": 0, "losses": 0, "trades": 0, "pnl": 0, "win_rate": 0.5
            }
        
        stats = perf.regime_performance[regime.value]
        stats["trades"] += 1
        stats["pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.5
        
        # Update exploration bonuses
        self._update_exploration_bonuses()


# Singleton instance
_selector: Optional[AdaptiveStrategySelector] = None


def get_strategy_selector() -> AdaptiveStrategySelector:
    """Get the singleton strategy selector instance."""
    global _selector
    if _selector is None:
        _selector = AdaptiveStrategySelector()
    return _selector


def select_best_strategy(
    regime: Optional[RegimeAnalysis] = None,
    symbol: str = "",
) -> StrategySelection:
    """Convenience function for strategy selection."""
    selector = get_strategy_selector()
    return selector.select_strategy(regime=regime, symbol=symbol)
