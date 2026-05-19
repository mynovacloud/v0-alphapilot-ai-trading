"""
Advanced Exit Management Engine
================================

Determines WHEN to exit trades using:
1. Dynamic stop-loss adjustment (ATR-based, time-based)
2. Trailing stops that activate after profit threshold
3. Time-based exits for stale positions
4. Profit locking at key levels
5. Pattern-based exits (reversal detection)
6. Partial profit taking

Exit Philosophy:
- Let winners run, cut losers quickly
- Never let a winner turn into a loser (break-even stops)
- Scale out of winning positions
- Use Claude for complex exit decisions (sparingly)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from database.db import session_scope
from database.models import PaperTrade
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


class ExitReason(str, Enum):
    """Reason for exit recommendation."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    TIME_LIMIT = "time_limit"
    REVERSAL_DETECTED = "reversal_detected"
    MOMENTUM_LOSS = "momentum_loss"
    PARTIAL_PROFIT = "partial_profit"
    MANUAL = "manual"
    HOLD = "hold"


@dataclass
class ExitDecision:
    """Decision about whether to exit a position."""
    should_exit: bool
    reason: ExitReason
    exit_price: float
    
    # For partial exits
    exit_percentage: float = 1.0  # 1.0 = full exit
    
    # Context
    current_pnl_pct: float = 0.0
    time_in_trade_hours: float = 0.0
    distance_to_stop: float = 0.0
    distance_to_target: float = 0.0
    
    # Recommendations
    new_stop_price: Optional[float] = None
    new_target_price: Optional[float] = None
    reasoning: str = ""


class AdvancedExitManager:
    """
    Sophisticated exit management that maximizes profits and minimizes losses.
    
    Key Features:
    1. Break-even stops: Move stop to entry once trade is +2% profitable
    2. Trailing stops: Trail the high-water mark by ATR-based distance
    3. Partial profits: Take 50% at first target, let rest run
    4. Time decay: Reduce target expectations for stale trades
    5. Reversal detection: Exit if momentum reverses against position
    """
    
    def __init__(self):
        # Break-even parameters
        self.breakeven_trigger_pct = 0.02   # Move to break-even at 2% profit
        self.breakeven_buffer_pct = 0.005   # Place stop 0.5% above entry for buffer
        
        # Trailing stop parameters
        self.trailing_trigger_pct = 0.04    # Start trailing at 4% profit
        self.trailing_distance_pct = 0.025  # Trail 2.5% behind high
        
        # Partial profit parameters
        self.partial_trigger_pct = 0.06     # Take partial at 6%
        self.partial_exit_pct = 0.50        # Exit 50% of position
        
        # Time parameters
        self.time_limit_hours = 72          # Max hold time
        self.time_decay_start_hours = 24    # Start reducing targets after 24h
        self.time_decay_rate = 0.1          # Reduce target by 10% per day
        
        # Momentum parameters
        self.momentum_window = 5            # Candles to check for momentum
        self.momentum_reversal_threshold = 0.6  # 60% of candles against = reversal
    
    def evaluate_exit(
        self,
        trade_id: int,
        current_price: float,
        recent_candles: Optional[List[Dict]] = None,
    ) -> ExitDecision:
        """
        Evaluate whether a position should be exited.
        
        Args:
            trade_id: ID of the trade to evaluate
            current_price: Current market price
            recent_candles: Recent price candles for momentum analysis
        
        Returns:
            ExitDecision with recommendation
        """
        trade = self._get_trade(trade_id)
        if not trade:
            return ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                exit_price=current_price,
                reasoning="Trade not found",
            )
        
        entry_price = float(trade.entry_price or current_price)
        quantity = float(trade.quantity or 0)
        side = trade.side or "BUY"
        
        # Calculate P&L
        if side == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Calculate time in trade
        opened_at = trade.opened_at
        if opened_at:
            from utils.helpers import ensure_utc
            opened_utc = ensure_utc(opened_at)
            time_in_trade = (utcnow() - opened_utc).total_seconds() / 3600
        else:
            time_in_trade = 0
        
        # Get stop/target prices
        stop_price = float(trade.stop_loss_price or 0)
        target_price = float(trade.take_profit_price or 0)
        trailing_stop = float(trade.trailing_stop_pct or 0)
        high_water = float(trade.high_water_mark or current_price)
        
        # Track best price seen
        if side == "BUY":
            new_high_water = max(high_water, current_price)
        else:
            new_high_water = min(high_water, current_price) if high_water > 0 else current_price
        
        # Calculate distances
        distance_to_stop = abs(current_price - stop_price) / current_price if stop_price > 0 else 1.0
        distance_to_target = abs(target_price - current_price) / current_price if target_price > 0 else 1.0
        
        # =========================================
        # EXIT CHECKS (in priority order)
        # =========================================
        
        # 1. STOP LOSS HIT
        if stop_price > 0:
            if (side == "BUY" and current_price <= stop_price) or \
               (side == "SELL" and current_price >= stop_price):
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    current_pnl_pct=pnl_pct,
                    time_in_trade_hours=time_in_trade,
                    distance_to_stop=0,
                    distance_to_target=distance_to_target,
                    reasoning=f"Stop loss hit at {current_price:.4f} (entry {entry_price:.4f})",
                )
        
        # 2. TAKE PROFIT HIT
        if target_price > 0:
            if (side == "BUY" and current_price >= target_price) or \
               (side == "SELL" and current_price <= target_price):
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TAKE_PROFIT,
                    exit_price=current_price,
                    current_pnl_pct=pnl_pct,
                    time_in_trade_hours=time_in_trade,
                    reasoning=f"Take profit hit at {current_price:.4f} (target {target_price:.4f})",
                )
        
        # 3. TRAILING STOP (if activated)
        if trailing_stop > 0 and pnl_pct >= self.trailing_trigger_pct:
            if side == "BUY":
                trailing_stop_price = new_high_water * (1 - trailing_stop)
                if current_price <= trailing_stop_price:
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        current_pnl_pct=pnl_pct,
                        time_in_trade_hours=time_in_trade,
                        reasoning=f"Trailing stop hit. High: {new_high_water:.4f}, Stop: {trailing_stop_price:.4f}",
                    )
            else:
                trailing_stop_price = new_high_water * (1 + trailing_stop)
                if current_price >= trailing_stop_price:
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        current_pnl_pct=pnl_pct,
                        time_in_trade_hours=time_in_trade,
                        reasoning=f"Trailing stop hit. Low: {new_high_water:.4f}, Stop: {trailing_stop_price:.4f}",
                    )
        
        # 4. TIME LIMIT
        if time_in_trade >= self.time_limit_hours:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TIME_LIMIT,
                exit_price=current_price,
                current_pnl_pct=pnl_pct,
                time_in_trade_hours=time_in_trade,
                reasoning=f"Time limit reached ({time_in_trade:.1f}h > {self.time_limit_hours}h)",
            )
        
        # 5. MOMENTUM REVERSAL (check candles)
        if recent_candles and len(recent_candles) >= self.momentum_window:
            reversal_detected, reversal_strength = self._detect_reversal(
                recent_candles, side
            )
            if reversal_detected and pnl_pct > 0.01:  # Only if in profit
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.MOMENTUM_LOSS,
                    exit_price=current_price,
                    current_pnl_pct=pnl_pct,
                    time_in_trade_hours=time_in_trade,
                    reasoning=f"Momentum reversal detected (strength: {reversal_strength:.1%})",
                )
        
        # 6. PARTIAL PROFIT (suggest but don't force)
        if pnl_pct >= self.partial_trigger_pct:
            # Check if we've already taken partial
            partial_taken = getattr(trade, 'partial_exit_done', False)
            if not partial_taken:
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.PARTIAL_PROFIT,
                    exit_price=current_price,
                    exit_percentage=self.partial_exit_pct,
                    current_pnl_pct=pnl_pct,
                    time_in_trade_hours=time_in_trade,
                    reasoning=f"Partial profit target reached ({pnl_pct:.1%})",
                )
        
        # =========================================
        # STOP ADJUSTMENT RECOMMENDATIONS
        # =========================================
        
        new_stop = None
        new_target = None
        reasoning_parts = []
        
        # Move to break-even if profitable enough
        if pnl_pct >= self.breakeven_trigger_pct:
            if side == "BUY":
                breakeven_stop = entry_price * (1 + self.breakeven_buffer_pct)
                if stop_price < breakeven_stop:
                    new_stop = breakeven_stop
                    reasoning_parts.append(f"Move stop to break-even: {breakeven_stop:.4f}")
            else:
                breakeven_stop = entry_price * (1 - self.breakeven_buffer_pct)
                if stop_price > breakeven_stop:
                    new_stop = breakeven_stop
                    reasoning_parts.append(f"Move stop to break-even: {breakeven_stop:.4f}")
        
        # Tighten stops as time passes without hitting target
        if time_in_trade >= self.time_decay_start_hours:
            days_stale = (time_in_trade - self.time_decay_start_hours) / 24
            decay_factor = 1 - (self.time_decay_rate * days_stale)
            decay_factor = max(0.5, decay_factor)  # Don't decay below 50%
            
            if target_price > 0 and decay_factor < 1.0:
                if side == "BUY":
                    new_target = entry_price + (target_price - entry_price) * decay_factor
                else:
                    new_target = entry_price - (entry_price - target_price) * decay_factor
                reasoning_parts.append(f"Time decay: reduce target to {new_target:.4f}")
        
        # HOLD - no exit needed
        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            exit_price=current_price,
            current_pnl_pct=pnl_pct,
            time_in_trade_hours=time_in_trade,
            distance_to_stop=distance_to_stop,
            distance_to_target=distance_to_target,
            new_stop_price=new_stop,
            new_target_price=new_target,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else f"Holding. PnL: {pnl_pct:.1%}, Time: {time_in_trade:.1f}h",
        )
    
    def _detect_reversal(self, candles: List[Dict], side: str) -> Tuple[bool, float]:
        """
        Detect if momentum is reversing against our position.
        
        Returns (is_reversing, reversal_strength)
        """
        if len(candles) < self.momentum_window:
            return False, 0.0
        
        recent = candles[-self.momentum_window:]
        against_count = 0
        
        for candle in recent:
            open_price = float(candle.get("open", 0))
            close_price = float(candle.get("close", 0))
            
            if side == "BUY":
                # Bearish candle is against us
                if close_price < open_price:
                    against_count += 1
            else:
                # Bullish candle is against us
                if close_price > open_price:
                    against_count += 1
        
        reversal_strength = against_count / self.momentum_window
        is_reversing = reversal_strength >= self.momentum_reversal_threshold
        
        return is_reversing, reversal_strength
    
    def _get_trade(self, trade_id: int) -> Optional[PaperTrade]:
        """Get trade from database."""
        try:
            with session_scope() as s:
                return s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        except Exception as e:
            logger.error(f"Error getting trade: {e}")
            return None
    
    def calculate_dynamic_stops(
        self,
        entry_price: float,
        side: str,
        atr: float,
        confidence: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate initial stop-loss, take-profit, and trailing stop based on ATR.
        
        Args:
            entry_price: Entry price
            side: BUY or SELL
            atr: Current ATR value
            confidence: Signal confidence (0-1)
        
        Returns:
            (stop_loss_price, take_profit_price, trailing_stop_pct)
        """
        # ATR-based stop distance (2-3 ATRs depending on confidence)
        atr_multiplier = 3.0 - confidence  # Higher confidence = tighter stops
        stop_distance = atr * atr_multiplier
        
        # Minimum stop distance
        min_stop_pct = 0.03  # 3% minimum
        max_stop_pct = 0.10  # 10% maximum
        stop_pct = stop_distance / entry_price if entry_price > 0 else 0.05
        stop_pct = max(min_stop_pct, min(max_stop_pct, stop_pct))
        
        # Take profit is 2x the stop (2:1 R:R minimum)
        # Higher confidence = larger target
        rr_ratio = 2.0 + confidence  # 2.0 to 3.0
        target_pct = stop_pct * rr_ratio
        
        # Calculate prices
        if side == "BUY":
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + target_pct)
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - target_pct)
        
        # Trailing stop percentage
        trailing_pct = stop_pct * 0.6  # Trail at 60% of initial stop distance
        
        return stop_loss, take_profit, trailing_pct


# Global instance
_exit_manager: Optional[AdvancedExitManager] = None


def get_exit_manager() -> AdvancedExitManager:
    """Get or create the global exit manager instance."""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = AdvancedExitManager()
    return _exit_manager


def evaluate_exit(
    trade_id: int,
    current_price: float,
    recent_candles: Optional[List[Dict]] = None,
) -> ExitDecision:
    """Convenience function to evaluate an exit."""
    manager = get_exit_manager()
    return manager.evaluate_exit(trade_id, current_price, recent_candles)


def calculate_stops(
    entry_price: float,
    side: str,
    atr: float,
    confidence: float,
) -> Tuple[float, float, float]:
    """Convenience function to calculate stops."""
    manager = get_exit_manager()
    return manager.calculate_dynamic_stops(entry_price, side, atr, confidence)
