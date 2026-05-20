"""
Profit Maximizer - Asymmetric exit logic to let winners run.

The key insight is that profitable trades should be treated differently than losing trades:
- Winners: Let them run with trailing stops, take partial profits
- Losers: Cut quickly, don't hope for recovery

This module implements:
1. Dynamic profit targets based on volatility and trend strength
2. Scaled exit (take 25% at 1R, 25% at 2R, let rest run)
3. Break-even stop management
4. Winner extension (widen TP if momentum continues)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from utils.logger import get_logger
from utils.helpers import utcnow

logger = get_logger(__name__)


@dataclass
class ProfitAction:
    """Recommended action for a profitable position."""
    action: str  # "hold", "take_partial", "tighten_stop", "extend_target", "close"
    percentage: float = 0.0  # If taking partial, what % to close
    new_stop: Optional[float] = None  # New stop price if tightening
    new_target: Optional[float] = None  # New target if extending
    reasoning: str = ""


@dataclass
class LossAction:
    """Recommended action for a losing position."""
    action: str  # "hold", "cut", "dca", "close"
    urgency: float = 0.5  # 0-1, how urgent is the action
    reasoning: str = ""


class ProfitMaximizer:
    """
    Manages open positions to maximize profits and minimize losses.
    Implements asymmetric treatment of winners vs losers.
    """
    
    def __init__(self):
        # Profit taking levels (R-multiples)
        self._partial_1 = 1.0   # Take 25% at 1R (1x risk)
        self._partial_2 = 2.0   # Take 25% at 2R
        self._partial_3 = 3.0   # Take 25% at 3R
        # Let final 25% run with trailing stop
        
        # Break-even settings
        self._break_even_trigger = 1.5  # Move stop to break-even at 1.5R
        
        # Winner extension settings
        self._extend_trigger = 2.5  # If we hit 2.5R with strong momentum, extend target
        self._momentum_threshold = 0.6  # RSI above this to extend
        
        # Loser settings
        self._cut_threshold = -0.5  # R-multiple where we consider cutting (before stop)
        self._dca_threshold = -1.0  # R-multiple where DCA might make sense
    
    def analyze_winner(
        self,
        entry_price: float,
        current_price: float,
        stop_price: float,
        target_price: float,
        side: str,
        partial_taken: int = 0,
        momentum: float = 0.5,
        trend_strength: float = 0.5,
    ) -> ProfitAction:
        """
        Analyze a winning position and recommend next action.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            stop_price: Current stop-loss price
            target_price: Current take-profit price
            side: "BUY" or "SELL"
            partial_taken: How many partial exits already taken (0-3)
            momentum: Current momentum indicator (0-1)
            trend_strength: Current trend strength (0-1)
            
        Returns:
            ProfitAction with recommended action
        """
        # Calculate R-multiple (profit in terms of initial risk)
        initial_risk = abs(entry_price - stop_price)
        if initial_risk == 0:
            initial_risk = entry_price * 0.02  # Default 2% risk
        
        if side == "BUY":
            current_profit = current_price - entry_price
        else:
            current_profit = entry_price - current_price
        
        r_multiple = current_profit / initial_risk
        
        # Check for partial profit opportunities
        if partial_taken == 0 and r_multiple >= self._partial_1:
            return ProfitAction(
                action="take_partial",
                percentage=0.25,
                reasoning=f"Hit {r_multiple:.1f}R - taking 25% profit (1st partial)"
            )
        
        if partial_taken == 1 and r_multiple >= self._partial_2:
            return ProfitAction(
                action="take_partial",
                percentage=0.33,  # 33% of remaining = 25% of original
                reasoning=f"Hit {r_multiple:.1f}R - taking 25% profit (2nd partial)"
            )
        
        if partial_taken == 2 and r_multiple >= self._partial_3:
            return ProfitAction(
                action="take_partial",
                percentage=0.50,  # 50% of remaining = 25% of original
                reasoning=f"Hit {r_multiple:.1f}R - taking 25% profit (3rd partial)"
            )
        
        # Check for break-even stop move
        if r_multiple >= self._break_even_trigger:
            # Move stop to break-even (entry price + small buffer)
            buffer = initial_risk * 0.1  # 10% of risk as buffer
            if side == "BUY":
                new_stop = entry_price + buffer
            else:
                new_stop = entry_price - buffer
            
            # Only suggest if current stop is worse than break-even
            if (side == "BUY" and stop_price < new_stop) or \
               (side == "SELL" and stop_price > new_stop):
                return ProfitAction(
                    action="tighten_stop",
                    new_stop=new_stop,
                    reasoning=f"Hit {r_multiple:.1f}R - moving stop to break-even"
                )
        
        # Check for target extension (if momentum is strong)
        if r_multiple >= self._extend_trigger and momentum > self._momentum_threshold:
            # Extend target by 50%
            if side == "BUY":
                target_distance = target_price - entry_price
                new_target = target_price + (target_distance * 0.5)
            else:
                target_distance = entry_price - target_price
                new_target = target_price - (target_distance * 0.5)
            
            return ProfitAction(
                action="extend_target",
                new_target=new_target,
                reasoning=f"Strong momentum ({momentum:.2f}) at {r_multiple:.1f}R - extending target"
            )
        
        # Default: hold and let trailing stop do its job
        return ProfitAction(
            action="hold",
            reasoning=f"At {r_multiple:.1f}R - holding with trailing stop"
        )
    
    def analyze_loser(
        self,
        entry_price: float,
        current_price: float,
        stop_price: float,
        side: str,
        dca_count: int = 0,
        signal_still_valid: bool = True,
        time_in_trade_hours: float = 0,
    ) -> LossAction:
        """
        Analyze a losing position and recommend next action.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            stop_price: Current stop-loss price
            side: "BUY" or "SELL"
            dca_count: How many DCAs already done
            signal_still_valid: Is the original signal still valid?
            time_in_trade_hours: How long we've been in this trade
            
        Returns:
            LossAction with recommended action
        """
        # Calculate R-multiple (negative for losers)
        initial_risk = abs(entry_price - stop_price)
        if initial_risk == 0:
            initial_risk = entry_price * 0.02
        
        if side == "BUY":
            current_loss = entry_price - current_price
        else:
            current_loss = current_price - entry_price
        
        r_multiple = -current_loss / initial_risk  # Negative for losses
        
        # Urgent cut if signal invalidated and loss is significant
        if not signal_still_valid and r_multiple <= self._cut_threshold:
            return LossAction(
                action="cut",
                urgency=0.8,
                reasoning=f"Signal invalidated at {r_multiple:.1f}R - cut loss early"
            )
        
        # Consider DCA if signal still valid and not too many DCAs
        if signal_still_valid and r_multiple <= self._dca_threshold and dca_count < 3:
            return LossAction(
                action="dca",
                urgency=0.4,
                reasoning=f"Signal valid, at {r_multiple:.1f}R - consider DCA #{dca_count + 1}"
            )
        
        # Time decay: if we've been in a loser too long, consider cutting
        if time_in_trade_hours > 24 and r_multiple < -0.3:
            return LossAction(
                action="cut",
                urgency=0.6,
                reasoning=f"In losing trade for {time_in_trade_hours:.0f}h at {r_multiple:.1f}R - cut stale position"
            )
        
        # Default: hold and wait for stop or recovery
        return LossAction(
            action="hold",
            urgency=0.3,
            reasoning=f"At {r_multiple:.1f}R - holding, stop at {stop_price:.4f}"
        )
    
    def calculate_optimal_stops(
        self,
        entry_price: float,
        side: str,
        atr: float,
        confidence: float,
        trend_aligned: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Calculate optimal stop-loss and take-profit based on market conditions.
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price, trailing_stop_pct)
        """
        # Base stop is 1.5-2.5 ATR depending on confidence
        stop_atr_mult = 2.5 - (confidence * 1.0)  # Higher confidence = tighter stop
        stop_distance = atr * stop_atr_mult
        
        # R:R ratio based on trend alignment
        if trend_aligned:
            rr_ratio = 2.5  # Better R:R when trading with trend
        else:
            rr_ratio = 2.0  # Need better entries counter-trend
        
        profit_distance = stop_distance * rr_ratio
        
        # Trailing stop percentage
        trailing_pct = (stop_distance / entry_price) * 0.7  # Tighter trail than initial stop
        
        if side == "BUY":
            stop_price = entry_price - stop_distance
            target_price = entry_price + profit_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - profit_distance
        
        return stop_price, target_price, trailing_pct


# Singleton instance
_maximizer_instance: Optional[ProfitMaximizer] = None


def get_profit_maximizer() -> ProfitMaximizer:
    """Get the singleton profit maximizer instance."""
    global _maximizer_instance
    if _maximizer_instance is None:
        _maximizer_instance = ProfitMaximizer()
    return _maximizer_instance
