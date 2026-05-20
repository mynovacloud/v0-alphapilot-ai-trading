"""
Advanced Position Sizing Engine
===============================

Determines HOW MUCH money to allocate to each trade using:
1. Kelly Criterion (optimal sizing based on win rate and payoff)
2. ATR-based volatility adjustment
3. Portfolio heat management (total risk exposure)
4. Conviction scaling (size up on higher confidence)
5. Drawdown-aware sizing (reduce size during losing streaks)

Key Principles:
- Never risk more than 2% of portfolio on a single trade
- Scale position size with conviction
- Reduce exposure during drawdowns
- Account for correlation between positions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from database.db import session_scope
from database.models import PaperTrade, Wallet
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_usd: float
    recommended_qty: float
    
    # Sizing factors
    base_size: float
    kelly_fraction: float
    volatility_adjustment: float
    conviction_multiplier: float
    drawdown_adjustment: float
    heat_adjustment: float
    
    # Risk metrics
    position_risk_pct: float  # Risk as % of portfolio
    portfolio_heat: float     # Total current risk exposure
    max_loss_usd: float       # Max loss if stop hit
    
    # Reasoning
    reasoning: str
    warnings: List[str]


class AdvancedPositionSizer:
    """
    Sophisticated position sizing that balances aggression with risk management.
    
    Uses a multi-factor approach:
    1. Start with base position size from config
    2. Apply Kelly Criterion for optimal sizing
    3. Adjust for current volatility (ATR-based)
    4. Scale by signal conviction
    5. Reduce during drawdowns
    6. Cap based on portfolio heat (total exposure)
    """
    
    def __init__(self):
        # Risk parameters
        self.max_risk_per_trade = 0.02    # 2% max risk per trade
        self.max_portfolio_heat = 0.15    # 15% max total exposure
        self.kelly_fraction = 0.25        # Use 25% of Kelly (conservative)
        
        # Drawdown parameters
        self.drawdown_levels = [
            (0.05, 0.8),   # 5% drawdown -> 80% size
            (0.10, 0.6),   # 10% drawdown -> 60% size
            (0.15, 0.4),   # 15% drawdown -> 40% size
            (0.20, 0.25),  # 20% drawdown -> 25% size
        ]
        
        # Conviction scaling
        self.conviction_levels = [
            (0.80, 1.2),   # 80%+ confidence -> 120% size
            (0.70, 1.0),   # 70-80% -> 100% size
            (0.60, 0.8),   # 60-70% -> 80% size
            (0.50, 0.6),   # 50-60% -> 60% size
        ]
    
    def calculate_size(
        self,
        wallet_id: int,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float,
        confidence: float,
        base_size_usd: float,
        signal_quality: str = "B",
    ) -> PositionSizeResult:
        """
        Calculate optimal position size for a trade.
        
        Args:
            wallet_id: Wallet to trade from
            symbol: Trading symbol
            entry_price: Expected entry price
            stop_loss_pct: Stop loss as decimal (0.05 = 5%)
            confidence: Signal confidence (0-1)
            base_size_usd: Default position size from config
            signal_quality: A+, A, B, C, F
        
        Returns:
            PositionSizeResult with recommended size and reasoning
        """
        warnings = []
        
        # Get wallet info and recent performance
        wallet_info = self._get_wallet_info(wallet_id)
        if not wallet_info:
            return self._minimum_size(base_size_usd, entry_price, "Wallet not found")
        
        portfolio_value = wallet_info["balance"]
        if portfolio_value <= 0:
            return self._minimum_size(base_size_usd, entry_price, "Zero balance")
        
        # 1. Calculate Kelly Criterion optimal size
        win_rate, avg_win, avg_loss = self._get_historical_stats(wallet_id)
        kelly = self._kelly_criterion(win_rate, avg_win, avg_loss)
        kelly_size = portfolio_value * kelly * self.kelly_fraction
        
        # 2. Calculate risk-based size (max risk per trade)
        if stop_loss_pct > 0:
            risk_based_size = (portfolio_value * self.max_risk_per_trade) / stop_loss_pct
        else:
            risk_based_size = base_size_usd
            warnings.append("No stop loss - using base size")
        
        # 3. Use the more conservative of Kelly and risk-based
        base_calculated = min(kelly_size, risk_based_size, base_size_usd * 2)
        base_calculated = max(base_calculated, base_size_usd * 0.25)  # Floor at 25% of base
        
        # 4. Apply conviction multiplier
        conviction_mult = self._conviction_multiplier(confidence, signal_quality)
        
        # 5. Apply drawdown adjustment
        current_dd = self._calculate_drawdown(wallet_id, portfolio_value)
        dd_mult = self._drawdown_multiplier(current_dd)
        if dd_mult < 1.0:
            warnings.append(f"Drawdown adjustment: {dd_mult:.0%} (DD={current_dd:.1%})")
        
        # 6. Apply portfolio heat check
        current_heat = self._calculate_portfolio_heat(wallet_id, portfolio_value)
        remaining_heat = max(0, self.max_portfolio_heat - current_heat)
        heat_mult = min(1.0, remaining_heat / self.max_risk_per_trade) if self.max_risk_per_trade > 0 else 1.0
        if heat_mult < 1.0:
            warnings.append(f"Portfolio heat: {current_heat:.1%} of {self.max_portfolio_heat:.1%} max")
        
        # Calculate final size
        final_size = base_calculated * conviction_mult * dd_mult * heat_mult
        
        # Apply hard caps
        max_size = min(
            portfolio_value * 0.20,  # Never more than 20% in one position
            wallet_info.get("max_position_usd", float("inf")),
            base_size_usd * 3,  # Never more than 3x base
        )
        final_size = min(final_size, max_size)
        final_size = max(final_size, 10.0)  # Minimum $10
        
        # Calculate quantity
        qty = final_size / entry_price if entry_price > 0 else 0
        
        # Calculate risk metrics
        max_loss = final_size * stop_loss_pct
        position_risk = max_loss / portfolio_value if portfolio_value > 0 else 0
        
        reasoning = (
            f"Base ${base_calculated:.0f} x {conviction_mult:.2f} conviction "
            f"x {dd_mult:.2f} DD adj x {heat_mult:.2f} heat = ${final_size:.0f}"
        )
        
        return PositionSizeResult(
            recommended_usd=round(final_size, 2),
            recommended_qty=round(qty, 6),
            base_size=base_calculated,
            kelly_fraction=kelly,
            volatility_adjustment=1.0,  # TODO: Add ATR-based adjustment
            conviction_multiplier=conviction_mult,
            drawdown_adjustment=dd_mult,
            heat_adjustment=heat_mult,
            position_risk_pct=position_risk,
            portfolio_heat=current_heat + position_risk,
            max_loss_usd=max_loss,
            reasoning=reasoning,
            warnings=warnings,
        )
    
    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion optimal bet size.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Win/Loss ratio (avg_win / avg_loss)
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.02  # Default 2%
        
        r = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        kelly = win_rate - ((1 - win_rate) / r)
        
        # Clamp to reasonable range
        return max(0.01, min(0.25, kelly))
    
    def _conviction_multiplier(self, confidence: float, quality: str) -> float:
        """Scale position size based on signal conviction."""
        # Base multiplier from confidence
        base_mult = 1.0
        for threshold, mult in self.conviction_levels:
            if confidence >= threshold:
                base_mult = mult
                break
        
        # Quality adjustment
        quality_adj = {
            "A+": 1.2,
            "A": 1.0,
            "B": 0.85,
            "C": 0.7,
            "F": 0.5,
        }.get(quality, 0.8)
        
        return base_mult * quality_adj
    
    def _drawdown_multiplier(self, drawdown: float) -> float:
        """Reduce position size during drawdowns."""
        for dd_threshold, mult in self.drawdown_levels:
            if drawdown >= dd_threshold:
                return mult
        return 1.0
    
    def _get_wallet_info(self, wallet_id: int) -> Optional[Dict]:
        """Get wallet balance and limits."""
        try:
            with session_scope() as s:
                wallet = s.query(Wallet).filter(Wallet.id == wallet_id).first()
                if wallet:
                    return {
                        "balance": float(wallet.paper_balance or 0),
                        "max_position_usd": float(wallet.max_position_usd or 1000),
                        "max_open_positions": int(wallet.max_open_positions or 10),
                    }
        except Exception as e:
            logger.error(f"Error getting wallet info: {e}")
        return None
    
    def _get_historical_stats(self, wallet_id: int) -> Tuple[float, float, float]:
        """Get win rate and average win/loss from recent trades."""
        try:
            from utils.helpers import utcnow_naive
            cutoff = utcnow_naive() - timedelta(days=30)
            with session_scope() as s:
                recent_trades = (
                    s.query(PaperTrade)
                    .filter(
                        PaperTrade.wallet_id == wallet_id,
                        PaperTrade.status == "closed",
                        PaperTrade.closed_at >= cutoff,
                    )
                    .all()
                )
                
                if not recent_trades:
                    return 0.5, 0.03, 0.02  # Defaults: 50% WR, 3% avg win, 2% avg loss
                
                wins = [t for t in recent_trades if (t.pnl or 0) > 0]
                losses = [t for t in recent_trades if (t.pnl or 0) <= 0]
                
                win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
                
                avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.03
                avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.02
                
                # Convert to percentage of typical position
                avg_position = 100  # Assume $100 average position
                avg_win_pct = avg_win / avg_position if avg_position > 0 else 0.03
                avg_loss_pct = avg_loss / avg_position if avg_position > 0 else 0.02
                
                return win_rate, avg_win_pct, avg_loss_pct
                
        except Exception as e:
            logger.error(f"Error getting historical stats: {e}")
            return 0.5, 0.03, 0.02
    
    def _calculate_drawdown(self, wallet_id: int, current_balance: float) -> float:
        """Calculate current drawdown from peak."""
        try:
            with session_scope() as s:
                # Get starting balance (peak) - simplified
                wallet = s.query(Wallet).filter(Wallet.id == wallet_id).first()
                if not wallet:
                    return 0.0
                
                starting = float(wallet.starting_paper_balance or current_balance)
                peak = max(starting, current_balance)  # Simplified peak
                
                if peak <= 0:
                    return 0.0
                
                return (peak - current_balance) / peak
                
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def _calculate_portfolio_heat(self, wallet_id: int, portfolio_value: float) -> float:
        """Calculate current total risk exposure."""
        try:
            with session_scope() as s:
                open_trades = (
                    s.query(PaperTrade)
                    .filter(
                        PaperTrade.wallet_id == wallet_id,
                        PaperTrade.status == "open",
                    )
                    .all()
                )
                
                if not open_trades or portfolio_value <= 0:
                    return 0.0
                
                total_risk = 0.0
                for trade in open_trades:
                    entry = float(trade.entry_price or 0)
                    stop = float(trade.stop_loss_price or entry * 0.95)
                    qty = float(trade.quantity or 0)
                    
                    if entry > 0:
                        risk_pct = abs(entry - stop) / entry
                        position_value = entry * qty
                        trade_risk = position_value * risk_pct
                        total_risk += trade_risk
                
                return total_risk / portfolio_value
                
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            return 0.0
    
    def _minimum_size(self, base_size: float, price: float, reason: str) -> PositionSizeResult:
        """Return minimum position size."""
        min_size = min(base_size * 0.25, 25.0)
        return PositionSizeResult(
            recommended_usd=min_size,
            recommended_qty=min_size / price if price > 0 else 0,
            base_size=base_size,
            kelly_fraction=0.02,
            volatility_adjustment=1.0,
            conviction_multiplier=0.5,
            drawdown_adjustment=1.0,
            heat_adjustment=1.0,
            position_risk_pct=0.02,
            portfolio_heat=0.0,
            max_loss_usd=min_size * 0.05,
            reasoning=reason,
            warnings=[reason],
        )


# Global instance
_position_sizer: Optional[AdvancedPositionSizer] = None


def get_position_sizer() -> AdvancedPositionSizer:
    """Get or create the global position sizer instance."""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = AdvancedPositionSizer()
    return _position_sizer


def calculate_position_size(
    wallet_id: int,
    symbol: str,
    entry_price: float,
    stop_loss_pct: float,
    confidence: float,
    base_size_usd: float,
    signal_quality: str = "B",
) -> PositionSizeResult:
    """Convenience function to calculate position size."""
    sizer = get_position_sizer()
    return sizer.calculate_size(
        wallet_id, symbol, entry_price, stop_loss_pct,
        confidence, base_size_usd, signal_quality
    )
