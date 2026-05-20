"""
Strategic Claude Integration - Uses Claude AI only when it adds real value.

Philosophy:
- Claude is expensive ($0.01-0.03 per call)
- Don't waste Claude on obvious decisions
- Use Claude for:
  1. Ambiguous signals that could go either way
  2. High-value/large positions where getting it right matters
  3. Unusual market conditions
  4. Learning from mistakes (post-trade reflection)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClaudeConsultation:
    """Result of deciding whether to consult Claude."""
    should_consult: bool
    reason: str
    priority: str = "normal"  # low, normal, high, critical
    expected_value: float = 0.0  # Estimated value of Claude's input


class StrategicClaudeRouter:
    """
    Intelligently routes decisions to Claude only when valuable.
    
    Claude should be used for:
    1. AMBIGUOUS signals (confidence 40-60%) where Claude can add clarity
    2. LARGE positions (>$500) where mistakes are costly
    3. REVERSAL signals (contradicting current position)
    4. POST-TRADE reflection (learning from wins/losses)
    5. UNUSUAL market conditions (high volatility, news events)
    
    Claude should NOT be used for:
    1. STRONG signals (confidence >70%) - technical is enough
    2. WEAK signals (confidence <30%) - just skip the trade
    3. SMALL positions (<$50) - not worth the API cost
    4. REPEATED decisions - use caching
    """
    
    # Configuration
    AMBIGUOUS_CONF_LOW = 0.40
    AMBIGUOUS_CONF_HIGH = 0.65
    LARGE_POSITION_THRESHOLD = 500  # USD
    MIN_POSITION_FOR_CLAUDE = 50    # USD
    
    def __init__(self):
        self._recent_consultations: dict[str, float] = {}  # symbol -> timestamp
        self._consultation_cooldown = 300  # 5 minutes between consultations per symbol
        self._daily_budget_used = 0
        self._daily_budget_limit = 50  # Max consultations per day
    
    def should_consult_claude(
        self,
        symbol: str,
        technical_confidence: float,
        position_size_usd: float,
        signal_side: str,
        has_existing_position: bool = False,
        existing_position_side: Optional[str] = None,
        market_volatility: str = "normal",
    ) -> ClaudeConsultation:
        """
        Determine if this decision warrants Claude consultation.
        """
        reasons = []
        priority = "normal"
        should_consult = False
        expected_value = 0.0
        
        # Check budget
        if self._daily_budget_used >= self._daily_budget_limit:
            return ClaudeConsultation(
                should_consult=False,
                reason="Daily Claude budget exhausted",
                priority="none",
            )
        
        # Check cooldown
        last_consultation = self._recent_consultations.get(symbol, 0)
        if time.time() - last_consultation < self._consultation_cooldown:
            return ClaudeConsultation(
                should_consult=False,
                reason=f"Recently consulted for {symbol}, using cache",
                priority="none",
            )
        
        # 1. Position too small - don't waste Claude on tiny trades
        if position_size_usd < self.MIN_POSITION_FOR_CLAUDE:
            return ClaudeConsultation(
                should_consult=False,
                reason=f"Position too small (${position_size_usd:.0f})",
                priority="none",
            )
        
        # 2. Strong signal - technical analysis is sufficient
        if technical_confidence > 0.70:
            return ClaudeConsultation(
                should_consult=False,
                reason=f"Strong technical signal ({technical_confidence:.0%})",
                priority="none",
            )
        
        # 3. Weak signal - just skip the trade
        if technical_confidence < 0.30:
            return ClaudeConsultation(
                should_consult=False,
                reason=f"Weak signal ({technical_confidence:.0%}), skip trade",
                priority="none",
            )
        
        # 4. AMBIGUOUS signal - Claude can add value here
        if self.AMBIGUOUS_CONF_LOW <= technical_confidence <= self.AMBIGUOUS_CONF_HIGH:
            should_consult = True
            reasons.append(f"Ambiguous signal ({technical_confidence:.0%})")
            expected_value += 0.3
        
        # 5. LARGE position - worth getting Claude's opinion
        if position_size_usd >= self.LARGE_POSITION_THRESHOLD:
            should_consult = True
            reasons.append(f"Large position (${position_size_usd:.0f})")
            priority = "high"
            expected_value += 0.4
        
        # 6. REVERSAL signal - contradicting existing position
        if has_existing_position and existing_position_side:
            if signal_side != existing_position_side:
                should_consult = True
                reasons.append("Reversal signal against existing position")
                priority = "critical"
                expected_value += 0.5
        
        # 7. HIGH VOLATILITY - unusual market conditions
        if market_volatility in ["high", "extreme"]:
            should_consult = True
            reasons.append(f"High volatility ({market_volatility})")
            priority = "high" if priority != "critical" else priority
            expected_value += 0.3
        
        if not reasons:
            return ClaudeConsultation(
                should_consult=False,
                reason="Standard trade, using technical analysis",
                priority="none",
            )
        
        return ClaudeConsultation(
            should_consult=should_consult,
            reason="; ".join(reasons),
            priority=priority,
            expected_value=expected_value,
        )
    
    def record_consultation(self, symbol: str):
        """Record that we consulted Claude for a symbol."""
        self._recent_consultations[symbol] = time.time()
        self._daily_budget_used += 1
        
        # Clean old entries
        cutoff = time.time() - 3600  # 1 hour
        self._recent_consultations = {
            s: t for s, t in self._recent_consultations.items() if t > cutoff
        }
    
    def reset_daily_budget(self):
        """Reset daily budget (call at midnight)."""
        self._daily_budget_used = 0


# =============================================================================
# POST-TRADE REFLECTION - Learn from wins and losses
# =============================================================================

@dataclass
class TradeReflection:
    """Analysis of a completed trade for learning."""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    hold_duration_minutes: float
    was_profitable: bool
    lessons: list[str]
    what_went_right: list[str]
    what_went_wrong: list[str]
    should_have_done: str


class PostTradeReflector:
    """
    Analyzes completed trades to learn from them.
    Uses Claude sparingly for high-value reflections.
    """
    
    # Only reflect on significant trades
    MIN_PNL_FOR_REFLECTION = 0.02  # 2% move either direction
    MIN_POSITION_FOR_REFLECTION = 100  # $100 minimum
    
    def should_reflect(
        self,
        pnl_pct: float,
        position_size_usd: float,
        was_stopped_out: bool,
    ) -> bool:
        """Determine if this trade warrants post-trade reflection."""
        # Always reflect on stop-outs (learning opportunity)
        if was_stopped_out and abs(pnl_pct) > 0.03:
            return True
        
        # Reflect on significant wins/losses
        if abs(pnl_pct) >= self.MIN_PNL_FOR_REFLECTION:
            if position_size_usd >= self.MIN_POSITION_FOR_REFLECTION:
                return True
        
        # Reflect on big wins (what went right)
        if pnl_pct > 0.05:  # 5%+ gain
            return True
        
        return False
    
    def generate_reflection_prompt(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        hold_duration_minutes: float,
        entry_reasoning: str,
        exit_reason: str,
        market_conditions: str,
    ) -> str:
        """Generate a prompt for Claude to reflect on the trade."""
        outcome = "profitable" if pnl_pct > 0 else "losing"
        
        return f"""Analyze this {outcome} trade and extract learning:

TRADE DETAILS:
- Symbol: {symbol}
- Side: {side}
- Entry: ${entry_price:.4f}
- Exit: ${exit_price:.4f}
- P&L: {pnl_pct:.2%}
- Hold time: {hold_duration_minutes:.0f} minutes
- Entry reasoning: {entry_reasoning}
- Exit reason: {exit_reason}
- Market conditions: {market_conditions}

Please provide:
1. What went RIGHT in this trade (even if it lost)
2. What went WRONG (even if it won)
3. What should have been done differently
4. Key lesson to remember for future trades

Be specific and actionable. Focus on process, not outcome."""


# =============================================================================
# SINGLETON ACCESSORS
# =============================================================================

_claude_router: Optional[StrategicClaudeRouter] = None
_reflector: Optional[PostTradeReflector] = None


def get_claude_router() -> StrategicClaudeRouter:
    """Get or create the StrategicClaudeRouter singleton."""
    global _claude_router
    if _claude_router is None:
        _claude_router = StrategicClaudeRouter()
    return _claude_router


def get_reflector() -> PostTradeReflector:
    """Get or create the PostTradeReflector singleton."""
    global _reflector
    if _reflector is None:
        _reflector = PostTradeReflector()
    return _reflector
