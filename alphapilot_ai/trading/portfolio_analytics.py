"""
Portfolio Analytics Module
==========================
Comprehensive trading performance metrics and analysis.

Provides:
- Win rate, profit factor, expectancy
- Sharpe ratio, Sortino ratio
- Maximum drawdown analysis
- Performance by strategy, symbol, time period
- Risk metrics and position sizing analysis
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import defaultdict

from database.db import session_scope
from database.models import PaperTrade, Wallet
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Core trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # P&L metrics
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Rate metrics
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Average metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_win_loss_ratio: float = 0.0
    
    # Streak metrics
    current_streak: int = 0
    current_streak_type: str = "none"  # "win", "loss", "none"
    max_winning_streak: int = 0
    max_losing_streak: int = 0
    
    # Extremes
    largest_win: float = 0.0
    largest_loss: float = 0.0
    largest_win_symbol: str = ""
    largest_loss_symbol: str = ""


@dataclass
class RiskMetrics:
    """Risk and volatility metrics."""
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Volatility
    returns_std_dev: float = 0.0
    downside_deviation: float = 0.0
    
    # Value at Risk (simplified)
    var_95: float = 0.0
    var_99: float = 0.0
    
    # Position metrics
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    avg_holding_period_hours: float = 0.0


@dataclass
class PeriodPerformance:
    """Performance for a specific time period."""
    period: str  # "daily", "weekly", "monthly", "yearly"
    start_date: datetime
    end_date: datetime
    
    trades: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class SymbolPerformance:
    """Performance for a specific trading symbol."""
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    total_volume: float = 0.0


@dataclass 
class StrategyPerformance:
    """Performance for a specific trading strategy."""
    strategy_id: int
    strategy_name: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_confidence: float = 0.0


@dataclass
class PortfolioAnalytics:
    """Complete portfolio analytics report."""
    generated_at: datetime = field(default_factory=utcnow)
    
    # Core metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    
    # Breakdowns
    by_symbol: list[SymbolPerformance] = field(default_factory=list)
    by_strategy: list[StrategyPerformance] = field(default_factory=list)
    by_period: list[PeriodPerformance] = field(default_factory=list)
    
    # Equity curve data
    equity_curve: list[dict] = field(default_factory=list)
    
    # Open positions summary
    open_positions: int = 0
    open_exposure: float = 0.0


class PortfolioAnalyzer:
    """Analyzes trading portfolio and generates comprehensive reports."""
    
    def __init__(self, wallet_id: int | None = None):
        self.wallet_id = wallet_id
        self._risk_free_rate = 0.02  # 2% annual risk-free rate assumption
    
    def get_full_analytics(self) -> PortfolioAnalytics:
        """Generate complete portfolio analytics report."""
        trades = self._fetch_trades()
        open_trades = [t for t in trades if t.get("status") == "open"]
        closed_trades = [t for t in trades if t.get("status") == "closed"]
        
        analytics = PortfolioAnalytics()
        
        # Calculate core metrics
        analytics.performance = self._calculate_performance_metrics(closed_trades)
        analytics.risk = self._calculate_risk_metrics(closed_trades)
        
        # Calculate breakdowns
        analytics.by_symbol = self._calculate_by_symbol(closed_trades)
        analytics.by_strategy = self._calculate_by_strategy(closed_trades)
        analytics.by_period = self._calculate_by_period(closed_trades)
        
        # Equity curve
        analytics.equity_curve = self._build_equity_curve(closed_trades)
        
        # Open positions
        analytics.open_positions = len(open_trades)
        analytics.open_exposure = sum(
            float(t.get("entry_price", 0)) * float(t.get("qty", 0)) 
            for t in open_trades
        )
        
        return analytics
    
    def _fetch_trades(self) -> list[dict]:
        """Fetch all trades from database."""
        with session_scope() as s:
            query = s.query(PaperTrade)
            if self.wallet_id:
                query = query.filter(PaperTrade.wallet_id == self.wallet_id)
            
            trades = []
            for t in query.all():
                trades.append({
                    "id": t.id,
                    "wallet_id": t.wallet_id,
                    "strategy_id": t.strategy_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": float(t.qty or 0),
                    "entry_price": float(t.entry_price or 0),
                    "exit_price": float(t.exit_price or 0) if t.exit_price else None,
                    "realized_pnl": float(t.realized_pnl or 0),
                    "unrealized_pnl": float(t.unrealized_pnl or 0),
                    "fees": float(t.fees or 0),
                    "confidence": float(t.confidence or 0),
                    "status": t.status,
                    "opened_at": t.opened_at,
                    "closed_at": t.closed_at,
                    "exit_reason": t.exit_reason,
                })
            
            return trades
    
    def _calculate_performance_metrics(self, trades: list[dict]) -> PerformanceMetrics:
        """Calculate core performance metrics."""
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        metrics.total_trades = len(trades)
        
        # Categorize trades
        for t in trades:
            pnl = t.get("realized_pnl", 0)
            if pnl > 0:
                metrics.winning_trades += 1
                metrics.gross_profit += pnl
                if pnl > metrics.largest_win:
                    metrics.largest_win = pnl
                    metrics.largest_win_symbol = t.get("symbol", "")
            elif pnl < 0:
                metrics.losing_trades += 1
                metrics.gross_loss += abs(pnl)
                if pnl < metrics.largest_loss:
                    metrics.largest_loss = pnl
                    metrics.largest_loss_symbol = t.get("symbol", "")
            else:
                metrics.break_even_trades += 1
        
        metrics.total_realized_pnl = metrics.gross_profit - metrics.gross_loss
        
        # Rate calculations
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100
            metrics.loss_rate = metrics.losing_trades / metrics.total_trades * 100
        
        # Profit factor
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        elif metrics.gross_profit > 0:
            metrics.profit_factor = float('inf')
        
        # Averages
        if metrics.winning_trades > 0:
            metrics.avg_win = metrics.gross_profit / metrics.winning_trades
        if metrics.losing_trades > 0:
            metrics.avg_loss = metrics.gross_loss / metrics.losing_trades
        if metrics.total_trades > 0:
            metrics.avg_trade = metrics.total_realized_pnl / metrics.total_trades
        
        # Win/Loss ratio
        if metrics.avg_loss > 0:
            metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss
        
        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        metrics.expectancy = (
            (metrics.win_rate / 100 * metrics.avg_win) - 
            (metrics.loss_rate / 100 * metrics.avg_loss)
        )
        
        # Calculate streaks
        self._calculate_streaks(trades, metrics)
        
        return metrics
    
    def _calculate_streaks(self, trades: list[dict], metrics: PerformanceMetrics) -> None:
        """Calculate winning and losing streaks."""
        # Sort by closed_at
        sorted_trades = sorted(
            [t for t in trades if t.get("closed_at")],
            key=lambda x: x.get("closed_at") or datetime.min
        )
        
        current_streak = 0
        current_type = "none"
        max_win_streak = 0
        max_loss_streak = 0
        
        for t in sorted_trades:
            pnl = t.get("realized_pnl", 0)
            
            if pnl > 0:
                if current_type == "win":
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = "win"
                max_win_streak = max(max_win_streak, current_streak)
            elif pnl < 0:
                if current_type == "loss":
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = "loss"
                max_loss_streak = max(max_loss_streak, current_streak)
            # Break-even doesn't break streak
        
        metrics.current_streak = current_streak
        metrics.current_streak_type = current_type
        metrics.max_winning_streak = max_win_streak
        metrics.max_losing_streak = max_loss_streak
    
    def _calculate_risk_metrics(self, trades: list[dict]) -> RiskMetrics:
        """Calculate risk and volatility metrics."""
        metrics = RiskMetrics()
        
        if not trades:
            return metrics
        
        # Sort trades by close time
        sorted_trades = sorted(
            [t for t in trades if t.get("closed_at")],
            key=lambda x: x.get("closed_at") or datetime.min
        )
        
        if not sorted_trades:
            return metrics
        
        # Build equity curve for drawdown calculation
        returns = [t.get("realized_pnl", 0) for t in sorted_trades]
        cumulative = []
        total = 0
        for r in returns:
            total += r
            cumulative.append(total)
        
        # Max drawdown
        peak = 0
        max_dd = 0
        max_dd_start = 0
        max_dd_end = 0
        dd_start_idx = 0
        
        for i, equity in enumerate(cumulative):
            if equity > peak:
                peak = equity
                dd_start_idx = i
            
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_start = dd_start_idx
                max_dd_end = i
        
        metrics.max_drawdown = max_dd
        
        # Current drawdown
        if cumulative:
            current_peak = max(cumulative)
            metrics.current_drawdown = current_peak - cumulative[-1]
        
        # Drawdown percentages (relative to peak)
        if peak > 0:
            metrics.max_drawdown_pct = (max_dd / peak) * 100
            metrics.current_drawdown_pct = (metrics.current_drawdown / peak) * 100 if peak > 0 else 0
        
        # Returns statistics
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            metrics.returns_std_dev = math.sqrt(variance)
            
            # Downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                neg_variance = sum((r - mean_return) ** 2 for r in negative_returns) / len(negative_returns)
                metrics.downside_deviation = math.sqrt(neg_variance)
            
            # Sharpe ratio (simplified daily)
            # Assuming daily returns, annualize by sqrt(252)
            if metrics.returns_std_dev > 0:
                daily_rf = self._risk_free_rate / 252
                excess_return = mean_return - daily_rf
                metrics.sharpe_ratio = (excess_return / metrics.returns_std_dev) * math.sqrt(252)
            
            # Sortino ratio
            if metrics.downside_deviation > 0:
                metrics.sortino_ratio = (mean_return / metrics.downside_deviation) * math.sqrt(252)
            
            # Calmar ratio
            if metrics.max_drawdown > 0:
                annual_return = sum(returns)  # Simplified
                metrics.calmar_ratio = annual_return / metrics.max_drawdown
            
            # VaR (simple percentile)
            sorted_returns = sorted(returns)
            if len(sorted_returns) >= 20:
                idx_95 = int(len(sorted_returns) * 0.05)
                idx_99 = int(len(sorted_returns) * 0.01)
                metrics.var_95 = sorted_returns[idx_95] if idx_95 < len(sorted_returns) else sorted_returns[0]
                metrics.var_99 = sorted_returns[idx_99] if idx_99 < len(sorted_returns) else sorted_returns[0]
        
        # Position metrics
        position_sizes = [t.get("entry_price", 0) * t.get("qty", 0) for t in trades]
        if position_sizes:
            metrics.avg_position_size = sum(position_sizes) / len(position_sizes)
            metrics.max_position_size = max(position_sizes)
        
        # Average holding period
        holding_periods = []
        for t in sorted_trades:
            if t.get("opened_at") and t.get("closed_at"):
                delta = t["closed_at"] - t["opened_at"]
                holding_periods.append(delta.total_seconds() / 3600)  # Hours
        
        if holding_periods:
            metrics.avg_holding_period_hours = sum(holding_periods) / len(holding_periods)
        
        return metrics
    
    def _calculate_by_symbol(self, trades: list[dict]) -> list[SymbolPerformance]:
        """Calculate performance breakdown by trading symbol."""
        symbol_data: dict[str, SymbolPerformance] = {}
        
        for t in trades:
            symbol = t.get("symbol", "UNKNOWN")
            if symbol not in symbol_data:
                symbol_data[symbol] = SymbolPerformance(symbol=symbol)
            
            sp = symbol_data[symbol]
            sp.trades += 1
            sp.pnl += t.get("realized_pnl", 0)
            sp.total_volume += t.get("entry_price", 0) * t.get("qty", 0)
            
            if t.get("realized_pnl", 0) > 0:
                sp.wins += 1
            elif t.get("realized_pnl", 0) < 0:
                sp.losses += 1
        
        # Calculate rates
        for sp in symbol_data.values():
            if sp.trades > 0:
                sp.win_rate = sp.wins / sp.trades * 100
                sp.avg_pnl_per_trade = sp.pnl / sp.trades
        
        # Sort by PnL
        return sorted(symbol_data.values(), key=lambda x: x.pnl, reverse=True)
    
    def _calculate_by_strategy(self, trades: list[dict]) -> list[StrategyPerformance]:
        """Calculate performance breakdown by strategy."""
        strategy_data: dict[int, StrategyPerformance] = {}
        
        for t in trades:
            strategy_id = t.get("strategy_id") or 0
            if strategy_id not in strategy_data:
                strategy_data[strategy_id] = StrategyPerformance(
                    strategy_id=strategy_id,
                    strategy_name=f"Strategy {strategy_id}" if strategy_id else "Manual"
                )
            
            sp = strategy_data[strategy_id]
            sp.trades += 1
            sp.pnl += t.get("realized_pnl", 0)
            sp.avg_confidence += t.get("confidence", 0)
            
            if t.get("realized_pnl", 0) > 0:
                sp.wins += 1
            elif t.get("realized_pnl", 0) < 0:
                sp.losses += 1
        
        # Calculate rates and averages
        for sp in strategy_data.values():
            if sp.trades > 0:
                sp.win_rate = sp.wins / sp.trades * 100
                sp.avg_confidence = sp.avg_confidence / sp.trades
            
            gross_profit = sum(
                t.get("realized_pnl", 0) for t in trades 
                if t.get("strategy_id") == sp.strategy_id and t.get("realized_pnl", 0) > 0
            )
            gross_loss = abs(sum(
                t.get("realized_pnl", 0) for t in trades 
                if t.get("strategy_id") == sp.strategy_id and t.get("realized_pnl", 0) < 0
            ))
            
            if gross_loss > 0:
                sp.profit_factor = gross_profit / gross_loss
        
        return sorted(strategy_data.values(), key=lambda x: x.pnl, reverse=True)
    
    def _calculate_by_period(self, trades: list[dict]) -> list[PeriodPerformance]:
        """Calculate performance breakdown by time period."""
        now = utcnow()
        periods = []
        
        # Daily (last 7 days)
        for i in range(7):
            start = (now - timedelta(days=i+1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            period_trades = [
                t for t in trades 
                if t.get("closed_at") and start <= t["closed_at"] < end
            ]
            
            if period_trades:
                wins = sum(1 for t in period_trades if t.get("realized_pnl", 0) > 0)
                pnl = sum(t.get("realized_pnl", 0) for t in period_trades)
                
                gross_profit = sum(t.get("realized_pnl", 0) for t in period_trades if t.get("realized_pnl", 0) > 0)
                gross_loss = abs(sum(t.get("realized_pnl", 0) for t in period_trades if t.get("realized_pnl", 0) < 0))
                
                periods.append(PeriodPerformance(
                    period=start.strftime("%Y-%m-%d"),
                    start_date=start,
                    end_date=end,
                    trades=len(period_trades),
                    pnl=pnl,
                    win_rate=wins / len(period_trades) * 100 if period_trades else 0,
                    profit_factor=gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0,
                ))
        
        return periods
    
    def _build_equity_curve(self, trades: list[dict]) -> list[dict]:
        """Build equity curve data for charting."""
        sorted_trades = sorted(
            [t for t in trades if t.get("closed_at")],
            key=lambda x: x.get("closed_at") or datetime.min
        )
        
        curve = []
        cumulative = 0
        
        for t in sorted_trades:
            cumulative += t.get("realized_pnl", 0)
            curve.append({
                "timestamp": t["closed_at"].isoformat() if t.get("closed_at") else None,
                "pnl": t.get("realized_pnl", 0),
                "cumulative": cumulative,
                "symbol": t.get("symbol"),
            })
        
        return curve


def get_portfolio_summary(wallet_id: int | None = None) -> dict[str, Any]:
    """Get a quick portfolio summary for API responses."""
    analyzer = PortfolioAnalyzer(wallet_id)
    analytics = analyzer.get_full_analytics()
    
    perf = analytics.performance
    risk = analytics.risk
    
    return {
        "total_trades": perf.total_trades,
        "wins": perf.winning_trades,
        "losses": perf.losing_trades,
        "win_rate": round(perf.win_rate, 2),
        "profit_factor": round(perf.profit_factor, 2) if perf.profit_factor != float('inf') else "N/A",
        "total_pnl": round(perf.total_realized_pnl, 2),
        "avg_win": round(perf.avg_win, 2),
        "avg_loss": round(perf.avg_loss, 2),
        "expectancy": round(perf.expectancy, 2),
        "max_drawdown": round(risk.max_drawdown, 2),
        "max_drawdown_pct": round(risk.max_drawdown_pct, 2),
        "sharpe_ratio": round(risk.sharpe_ratio, 2),
        "sortino_ratio": round(risk.sortino_ratio, 2),
        "avg_holding_hours": round(risk.avg_holding_period_hours, 1),
        "current_streak": perf.current_streak,
        "current_streak_type": perf.current_streak_type,
        "max_win_streak": perf.max_winning_streak,
        "max_loss_streak": perf.max_losing_streak,
        "largest_win": round(perf.largest_win, 2),
        "largest_loss": round(perf.largest_loss, 2),
        "open_positions": analytics.open_positions,
        "open_exposure": round(analytics.open_exposure, 2),
    }
