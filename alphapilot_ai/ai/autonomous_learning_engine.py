"""
Autonomous Learning Engine
===========================

This is the BRAIN of AlphaPilot that operates WITHOUT Claude API.
It learns from every single trade - wins and losses - and builds
sophisticated pattern recognition to make better decisions over time.

KEY DESIGN PRINCIPLES:
1. Every loss is encoded as a "mistake pattern" to avoid
2. Every win is encoded as a "success pattern" to replicate
3. Patterns are matched using similarity scoring
4. Confidence is calibrated based on historical accuracy
5. The system gets smarter with every trade

MEMORY BANKS:
- MistakeMemory: Patterns that led to losses (AVOID these)
- SuccessMemory: Patterns that led to wins (REPLICATE these)
- SymbolMemory: Per-symbol performance and optimal conditions
- TimingMemory: Best times to trade each pattern
- RegimeMemory: What works in each market regime
- SequenceMemory: Trade sequences (what works after a loss, etc.)

This engine should make the bot profitable even with Claude API disabled.
"""
from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict, Tuple, Literal
from functools import lru_cache

from database.db import session_scope
from database.models import (
    ActivityLog,
    AILearningMemory,
    PaperTrade,
    ClaudeDecision,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeContext:
    """Complete context of a trade for pattern learning."""
    # Identity
    symbol: str
    side: str  # BUY or SELL
    
    # Technical state at entry
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 25.0
    atr_percent: float = 2.0
    bb_percent: float = 0.5  # Position within Bollinger Bands
    volume_ratio: float = 1.0  # vs 20-period average
    
    # Price action
    price_vs_ema20: float = 0.0  # % deviation from EMA20
    price_vs_ema50: float = 0.0
    price_vs_ema200: float = 0.0
    return_1h: float = 0.0
    return_4h: float = 0.0
    return_24h: float = 0.0
    
    # Market regime
    regime: str = "UNKNOWN"
    btc_trend: str = "NEUTRAL"
    volatility_regime: str = "NORMAL"
    
    # Timing
    hour_utc: int = 12
    day_of_week: int = 0  # 0=Monday
    
    # Signal
    signal_confidence: float = 0.5
    signal_quality: str = "B"
    strategy: str = "Momentum"
    
    # Portfolio state
    open_positions: int = 0
    recent_win_rate: float = 0.5
    consecutive_losses: int = 0
    daily_pnl_percent: float = 0.0
    
    def to_fingerprint(self) -> str:
        """Create a coarse-grained pattern fingerprint for this context.

        WHY THIS IS COARSE
        ------------------
        The original fingerprint composed SEVEN features (rsi 10-pt
        buckets × macd_sign × adx_bucket × vol_bucket × regime × side ×
        hour_bucket), producing roughly 3,600 possible cells in fingerprint
        space. In a 200-trade paper session almost none of those cells
        got populated more than once — every `[AUTONOMOUS]` log line in
        the operator console showed `pattern=1tr` — which silently
        neutralized Phase B's exact-pattern calibration tier (it needs
        N >= 5 trades on a fingerprint before it considers the win-rate
        usable).

        The current set keeps the FIVE features that carry the most
        signal-per-bit and widens the noisiest bucket:

          * side          — trades in different directions don't share fate
          * regime        — pattern viability shifts dramatically by regime
          * rsi_bucket    — 20-point buckets (5 cells) instead of 10-point
                            (10 cells). 20 points still distinguishes
                            "oversold / neutral / overbought" but stops
                            single-tick RSI wiggles from splitting cells.
          * macd_sign     — momentum direction is fundamental
          * adx_bucket    — trend strength matters; noise/trend is the
                            same coarse split as before

        DROPPED (with reasoning):
          * vol_bucket    — relative_volume regime is partially captured
                            by adx_bucket already (high vol ↔ strong trend
                            most of the time); the orthogonal information
                            wasn't worth the 3x cell explosion.
          * hour_bucket   — crypto trades 24/7. The asia/london/us split
                            was tripling cell count with no evidence that
                            time-of-day predicts outcomes for our universe.
                            If we later find evidence the buckets DO
                            predict, we add it back as a SEPARATE
                            adjustment layer rather than a fingerprint
                            dimension.

        FURTHER DROPPED on the empirical-tuning pass:
          * adx_bucket    — substantially redundant with `regime`.
                            TRENDING_UP/DOWN regimes are by definition
                            strong-ADX; RANGING/UNKNOWN are weak. Keeping
                            both doubled cells with little orthogonal info.
                            The test_fingerprint_space_is_meaningfully_
                            smaller_than_old test caught this: 5 features
                            still left 200 sample contexts in 90 unique
                            cells (avg 2.2 trades/cell, below the
                            calibration threshold). Dropping adx pushed
                            it under 60 cells (~3.5 trades/cell average).

        Theoretical cell count now: 2 × ~6 × 5 × 2 = ~120 vs ~3,600 before.

        Note for migration: existing learned patterns persisted under the
        old fingerprint format stay in the database but never match an
        incoming trade again. Effectively orphaned (not actively harmful);
        new patterns build up under the new format. The kNN tier (which
        uses TradeContext.to_vector() instead of this fingerprint) is
        unaffected and keeps working through the transition.
        """
        key_features = {
            "side": self.side,
            "regime": self.regime,
            "rsi_bucket": round(self.rsi / 20) * 20,  # 20-pt buckets: 0/20/40/60/80/100
            "macd_sign": "pos" if self.macd_histogram > 0 else "neg",
        }
        fingerprint_str = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def to_vector(self) -> List[float]:
        """Convert to numeric vector for similarity calculations."""
        regime_map = {"TRENDING_UP": 1, "TRENDING_DOWN": -1, "RANGING": 0, "VOLATILE": 0.5}
        side_map = {"BUY": 1, "SELL": -1}
        
        return [
            (self.rsi - 50) / 50,
            self.macd_histogram * 100,
            (self.adx - 25) / 25,
            (self.atr_percent - 2) / 2,
            (self.bb_percent - 0.5) * 2,
            (self.volume_ratio - 1) / 2,
            self.price_vs_ema20 / 3,
            self.price_vs_ema50 / 5,
            self.return_1h * 20,
            self.return_4h * 10,
            self.return_24h * 5,
            regime_map.get(self.regime, 0),
            (self.hour_utc - 12) / 12,
            self.signal_confidence,
            side_map.get(self.side, 0),
            min(self.consecutive_losses / 3, 1),
        ]


@dataclass
class LearnedPattern:
    """A pattern learned from historical trades."""
    fingerprint: str
    side: str
    
    # Outcome statistics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.5
    
    # Timing stats
    avg_hold_minutes: float = 60.0
    best_hold_minutes: float = 60.0  # Hold time of best trade
    
    # Context stats
    best_regime: str = "UNKNOWN"
    best_hour: int = 12
    worst_hour: int = 3
    
    # Quality metrics
    profit_factor: float = 1.0
    expectancy: float = 0.0  # Expected PnL per trade
    confidence_calibration: float = 1.0  # Adjustment to signal confidence
    
    # Timestamps
    first_seen: datetime = field(default_factory=utcnow)
    last_seen: datetime = field(default_factory=utcnow)
    last_updated: datetime = field(default_factory=utcnow)
    
    def update(self, pnl: float, pnl_pct: float, hold_minutes: float, context: TradeContext):
        """Update pattern statistics with a new trade outcome."""
        self.total_trades += 1
        self.total_pnl += pnl_pct
        self.last_seen = utcnow()
        self.last_updated = utcnow()
        
        if pnl_pct > 0:
            self.winning_trades += 1
            if self.avg_win == 0:
                self.avg_win = pnl_pct
            else:
                self.avg_win = 0.9 * self.avg_win + 0.1 * pnl_pct
            if pnl_pct > self.avg_win:
                self.best_hold_minutes = hold_minutes
                self.best_regime = context.regime
                self.best_hour = context.hour_utc
        else:
            if self.avg_loss == 0:
                self.avg_loss = abs(pnl_pct)
            else:
                self.avg_loss = 0.9 * self.avg_loss + 0.1 * abs(pnl_pct)
            self.worst_hour = context.hour_utc
        
        # Update derived metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
        self.avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        self.avg_hold_minutes = 0.9 * self.avg_hold_minutes + 0.1 * hold_minutes
        
        # Profit factor
        total_wins = self.avg_win * self.winning_trades
        total_losses = self.avg_loss * (self.total_trades - self.winning_trades)
        self.profit_factor = total_wins / total_losses if total_losses > 0 else 2.0
        
        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        # Confidence calibration based on historical accuracy
        # If pattern has 60% win rate but signal said 70% confidence, calibration = 0.86
        self.confidence_calibration = min(1.5, max(0.5, self.win_rate / 0.5))
    
    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "side": self.side,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_rate": self.win_rate,
            "avg_hold_minutes": self.avg_hold_minutes,
            "best_hold_minutes": self.best_hold_minutes,
            "best_regime": self.best_regime,
            "best_hour": self.best_hour,
            "worst_hour": self.worst_hour,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "confidence_calibration": self.confidence_calibration,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "LearnedPattern":
        pattern = cls(
            fingerprint=d.get("fingerprint", ""),
            side=d.get("side", "BUY"),
        )
        pattern.total_trades = d.get("total_trades", 0)
        pattern.winning_trades = d.get("winning_trades", 0)
        pattern.total_pnl = d.get("total_pnl", 0)
        pattern.avg_pnl = d.get("avg_pnl", 0)
        pattern.avg_win = d.get("avg_win", 0)
        pattern.avg_loss = d.get("avg_loss", 0)
        pattern.win_rate = d.get("win_rate", 0.5)
        pattern.avg_hold_minutes = d.get("avg_hold_minutes", 60)
        pattern.best_hold_minutes = d.get("best_hold_minutes", 60)
        pattern.best_regime = d.get("best_regime", "UNKNOWN")
        pattern.best_hour = d.get("best_hour", 12)
        pattern.worst_hour = d.get("worst_hour", 3)
        pattern.profit_factor = d.get("profit_factor", 1.0)
        pattern.expectancy = d.get("expectancy", 0)
        pattern.confidence_calibration = d.get("confidence_calibration", 1.0)
        if d.get("first_seen"):
            pattern.first_seen = datetime.fromisoformat(d["first_seen"])
        if d.get("last_seen"):
            pattern.last_seen = datetime.fromisoformat(d["last_seen"])
        if d.get("last_updated"):
            pattern.last_updated = datetime.fromisoformat(d["last_updated"])
        return pattern


@dataclass
class MistakePattern:
    """A pattern that consistently leads to losses - AVOID."""
    fingerprint: str
    description: str
    loss_count: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    last_seen: datetime = field(default_factory=utcnow)
    
    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "description": self.description,
            "loss_count": self.loss_count,
            "total_loss": self.total_loss,
            "avg_loss": self.avg_loss,
            "conditions": self.conditions,
            "severity": self.severity,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MistakePattern":
        return cls(
            fingerprint=d.get("fingerprint", ""),
            description=d.get("description", ""),
            loss_count=d.get("loss_count", 0),
            total_loss=d.get("total_loss", 0),
            avg_loss=d.get("avg_loss", 0),
            conditions=d.get("conditions", {}),
            severity=d.get("severity", "medium"),
            last_seen=datetime.fromisoformat(d["last_seen"]) if d.get("last_seen") else utcnow(),
        )


@dataclass
class SymbolProfile:
    """Learned profile for a specific trading symbol."""
    symbol: str
    
    # Performance
    total_trades: int = 0
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Optimal conditions
    best_regime: str = "TRENDING_UP"
    best_side: str = "BUY"
    best_hour: int = 14
    best_day: int = 2  # Wednesday
    optimal_hold_minutes: float = 120.0
    optimal_rsi_entry: Tuple[float, float] = (30, 70)
    
    # Avoid conditions
    worst_hour: int = 4
    worst_regime: str = "VOLATILE"
    avoid_low_volume: bool = True
    
    # Volatility profile
    typical_atr_percent: float = 3.0
    typical_daily_range: float = 5.0
    
    # Correlations
    btc_correlation: float = 0.7
    sector: str = "unknown"
    
    last_updated: datetime = field(default_factory=utcnow)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "total_pnl": self.total_pnl,
            "best_regime": self.best_regime,
            "best_side": self.best_side,
            "best_hour": self.best_hour,
            "best_day": self.best_day,
            "optimal_hold_minutes": self.optimal_hold_minutes,
            "optimal_rsi_entry": self.optimal_rsi_entry,
            "worst_hour": self.worst_hour,
            "worst_regime": self.worst_regime,
            "avoid_low_volume": self.avoid_low_volume,
            "typical_atr_percent": self.typical_atr_percent,
            "typical_daily_range": self.typical_daily_range,
            "btc_correlation": self.btc_correlation,
            "sector": self.sector,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SymbolProfile":
        profile = cls(symbol=d.get("symbol", ""))
        profile.total_trades = d.get("total_trades", 0)
        profile.win_rate = d.get("win_rate", 0.5)
        profile.avg_pnl = d.get("avg_pnl", 0)
        profile.total_pnl = d.get("total_pnl", 0)
        profile.best_regime = d.get("best_regime", "TRENDING_UP")
        profile.best_side = d.get("best_side", "BUY")
        profile.best_hour = d.get("best_hour", 14)
        profile.best_day = d.get("best_day", 2)
        profile.optimal_hold_minutes = d.get("optimal_hold_minutes", 120)
        profile.optimal_rsi_entry = tuple(d.get("optimal_rsi_entry", [30, 70]))
        profile.worst_hour = d.get("worst_hour", 4)
        profile.worst_regime = d.get("worst_regime", "VOLATILE")
        profile.avoid_low_volume = d.get("avoid_low_volume", True)
        profile.typical_atr_percent = d.get("typical_atr_percent", 3)
        profile.typical_daily_range = d.get("typical_daily_range", 5)
        profile.btc_correlation = d.get("btc_correlation", 0.7)
        profile.sector = d.get("sector", "unknown")
        if d.get("last_updated"):
            profile.last_updated = datetime.fromisoformat(d["last_updated"])
        return profile


@dataclass 
class AutonomousDecision:
    """Decision made by the autonomous learning engine."""
    action: str  # BUY, SELL, HOLD, AVOID
    confidence: float  # 0-1
    
    # Reasoning
    reasoning: str = ""
    matched_patterns: List[str] = field(default_factory=list)
    avoided_patterns: List[str] = field(default_factory=list)
    
    # Adjustments
    size_multiplier: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    recommended_hold_minutes: float = 120.0
    
    # Quality metrics
    pattern_match_score: float = 0.0
    historical_expectancy: float = 0.0
    calibrated_confidence: float = 0.0


# =============================================================================
# AUTONOMOUS LEARNING ENGINE
# =============================================================================

class AutonomousLearningEngine:
    """
    Self-learning trading engine that operates WITHOUT Claude API.
    
    Learns from every trade outcome and builds pattern recognition
    to make better decisions over time.
    """
    
    def __init__(self):
        # In-memory caches (loaded from DB on startup)
        self._patterns: Dict[str, LearnedPattern] = {}
        self._mistakes: Dict[str, MistakePattern] = {}
        self._symbols: Dict[str, SymbolProfile] = {}
        
        # Feature vectors for similarity matching
        self._trade_vectors: List[Tuple[List[float], float, str]] = []  # (vector, pnl, fingerprint)
        
        # Performance tracking
        self._recent_trades: List[Dict] = []  # Last 100 trades for recency weighting
        self._loaded = False
        
    def _ensure_loaded(self):
        """Load memory banks from database."""
        if self._loaded:
            return
        
        try:
            with session_scope() as s:
                # Load patterns
                patterns_row = s.query(AILearningMemory).filter(
                    AILearningMemory.category == "autonomous_patterns"
                ).first()
                if patterns_row and patterns_row.content:
                    data = json.loads(patterns_row.content)
                    for fp, p_data in data.items():
                        self._patterns[fp] = LearnedPattern.from_dict(p_data)
                
                # Load mistakes
                mistakes_row = s.query(AILearningMemory).filter(
                    AILearningMemory.category == "autonomous_mistakes"
                ).first()
                if mistakes_row and mistakes_row.content:
                    data = json.loads(mistakes_row.content)
                    for fp, m_data in data.items():
                        self._mistakes[fp] = MistakePattern.from_dict(m_data)
                
                # Load symbol profiles
                symbols_row = s.query(AILearningMemory).filter(
                    AILearningMemory.category == "autonomous_symbols"
                ).first()
                if symbols_row and symbols_row.content:
                    data = json.loads(symbols_row.content)
                    for sym, s_data in data.items():
                        self._symbols[sym] = SymbolProfile.from_dict(s_data)
                
                # Load trade vectors
                vectors_row = s.query(AILearningMemory).filter(
                    AILearningMemory.category == "autonomous_vectors"
                ).first()
                if vectors_row and vectors_row.content:
                    self._trade_vectors = json.loads(vectors_row.content)
            
            self._loaded = True
            logger.info(f"[AUTONOMOUS] Loaded {len(self._patterns)} patterns, "
                       f"{len(self._mistakes)} mistakes, {len(self._symbols)} symbols")
        except Exception:
            # Boot-path failure: persistence is corrupt / DB is unavailable.
            # We log with full traceback (the old code lost it) and continue
            # with empty in-memory state. Crashing here would take down the
            # whole singleton and break trading; the operator needs to see
            # the engine forgot everything, not have the bot die. Surfaced
            # at ERROR so it stands out in the console.
            logger.exception("[AUTONOMOUS] Failed to load memory — engine starts empty")
            self._loaded = True  # Prevent retry loops
    
    def _persist(self):
        """Save memory banks to database."""
        try:
            with session_scope() as s:
                # Save patterns
                patterns_data = {fp: p.to_dict() for fp, p in self._patterns.items()}
                self._upsert_memory(s, "autonomous_patterns", json.dumps(patterns_data))
                
                # Save mistakes
                mistakes_data = {fp: m.to_dict() for fp, m in self._mistakes.items()}
                self._upsert_memory(s, "autonomous_mistakes", json.dumps(mistakes_data))
                
                # Save symbols
                symbols_data = {sym: sp.to_dict() for sym, sp in self._symbols.items()}
                self._upsert_memory(s, "autonomous_symbols", json.dumps(symbols_data))
                
                # Save vectors (limit to last 1000)
                self._upsert_memory(s, "autonomous_vectors", 
                                   json.dumps(self._trade_vectors[-1000:]))
                
                s.commit()
        except Exception:
            # Persistence failure: learning that happened this tick is lost
            # on next restart. Bumped from logger.error (no traceback) to
            # logger.exception so the operator can SEE which table or which
            # serialization step blew up. We still don't raise — the trade
            # path must continue even if learning storage is broken.
            logger.exception("[AUTONOMOUS] Failed to persist memory — learning this tick is lost")
    
    def _upsert_memory(self, session, category: str, content: str):
        """Insert or update a memory row."""
        row = session.query(AILearningMemory).filter(
            AILearningMemory.category == category
        ).first()
        if row:
            row.content = content
            row.weight = 1.0
        else:
            row = AILearningMemory(category=category, content=content, weight=1.0)
            session.add(row)
    
    # =========================================================================
    # LEARNING FROM TRADES
    # =========================================================================
    
    def learn_from_trade(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_minutes: float,
        exit_reason: str,
        context: Optional[TradeContext] = None,
    ):
        """
        CRITICAL: This is called after EVERY trade closes.
        Learn from the outcome to improve future decisions.
        """
        self._ensure_loaded()
        
        # Build context if not provided
        if context is None:
            context = self._build_context_from_trade(trade_id, symbol, side)
        
        fingerprint = context.to_fingerprint()
        is_win = pnl_pct > 0
        
        logger.info(f"[LEARN] Trade {trade_id}: {symbol} {side} "
                   f"{'WIN' if is_win else 'LOSS'} {pnl_pct:+.2%} "
                   f"fingerprint={fingerprint}")
        
        # 1. Update or create pattern
        if fingerprint not in self._patterns:
            self._patterns[fingerprint] = LearnedPattern(
                fingerprint=fingerprint,
                side=side,
            )
        self._patterns[fingerprint].update(pnl, pnl_pct, hold_minutes, context)
        
        # 2. Track mistakes (losing patterns)
        if not is_win:
            self._record_mistake(fingerprint, pnl_pct, context, exit_reason)
        
        # 3. Update symbol profile
        self._update_symbol_profile(symbol, side, pnl_pct, hold_minutes, context, is_win)
        
        # 4. Store feature vector for similarity matching
        vector = context.to_vector()
        self._trade_vectors.append((vector, pnl_pct, fingerprint))
        if len(self._trade_vectors) > 1000:
            self._trade_vectors = self._trade_vectors[-1000:]
        
        # 5. Track recent trades for recency analysis
        self._recent_trades.append({
            "symbol": symbol,
            "side": side,
            "pnl_pct": pnl_pct,
            "fingerprint": fingerprint,
            "is_win": is_win,
            "timestamp": utcnow().isoformat(),
        })
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]
        
        # 6. Persist to database
        self._persist()

        # 7. Log learning insights
        pattern = self._patterns[fingerprint]
        logger.info(f"[LEARN] Pattern {fingerprint}: {pattern.total_trades} trades, "
                   f"{pattern.win_rate:.1%} win rate, {pattern.expectancy:+.2%} expectancy")

        # 8. Surface the learning event into the UI activity log. The
        # python logger only writes to stdout/log files, so without this the
        # operator running a training session has no live signal that
        # autonomous fingerprints are actually accumulating — which is exactly
        # the visibility they need to verify the loop is closing. Wrapped so
        # an audit-log failure can never break the learn path.
        from utils.errors import swallow_with_reason
        with swallow_with_reason(logger, "autonomous-learn audit log is best-effort; trade learning still occurred"):
            with session_scope() as s:
                s.add(ActivityLog(
                    category="ai",
                    level="info",
                    message=(
                        f"[AUTONOMOUS] Trade #{trade_id} {symbol} {side} "
                        f"{'WIN' if is_win else 'LOSS'} {pnl_pct:+.2%} "
                        f"fp={fingerprint} pattern={pattern.total_trades}tr "
                        f"wr={pattern.win_rate:.0%} ev={pattern.expectancy:+.2%}"
                    ),
                ))
    
    def _record_mistake(self, fingerprint: str, pnl_pct: float, 
                        context: TradeContext, exit_reason: str):
        """Record a losing trade as a mistake pattern."""
        if fingerprint not in self._mistakes:
            self._mistakes[fingerprint] = MistakePattern(
                fingerprint=fingerprint,
                description=self._generate_mistake_description(context, exit_reason),
                conditions={
                    "rsi_range": (context.rsi - 5, context.rsi + 5),
                    "regime": context.regime,
                    "hour_range": (context.hour_utc - 2, context.hour_utc + 2),
                    "side": context.side,
                },
            )
        
        mistake = self._mistakes[fingerprint]
        mistake.loss_count += 1
        mistake.total_loss += abs(pnl_pct)
        mistake.avg_loss = mistake.total_loss / mistake.loss_count
        mistake.last_seen = utcnow()
        
        # Determine severity
        if mistake.loss_count >= 5 and mistake.avg_loss > 0.03:
            mistake.severity = "critical"
        elif mistake.loss_count >= 3 and mistake.avg_loss > 0.02:
            mistake.severity = "high"
        elif mistake.loss_count >= 2:
            mistake.severity = "medium"
        else:
            mistake.severity = "low"
        
        logger.warning(f"[MISTAKE] {fingerprint}: {mistake.loss_count} losses, "
                      f"avg={mistake.avg_loss:.2%}, severity={mistake.severity}")
    
    def _generate_mistake_description(self, context: TradeContext, exit_reason: str) -> str:
        """Generate human-readable description of the mistake."""
        parts = []
        
        if context.rsi > 70:
            parts.append("Bought overbought (RSI>70)")
        elif context.rsi < 30:
            parts.append("Sold oversold (RSI<30)")
        
        if context.volume_ratio < 0.7:
            parts.append("Low volume")
        
        if context.regime == "VOLATILE":
            parts.append("Volatile market")
        
        if context.consecutive_losses > 2:
            parts.append(f"After {context.consecutive_losses} losses (tilt?)")
        
        if 3 <= context.hour_utc <= 7:
            parts.append("Low-liquidity hours")
        
        if exit_reason == "stop_loss":
            parts.append("Hit stop-loss")
        
        return "; ".join(parts) if parts else f"Loss in {context.regime} regime"
    
    def _update_symbol_profile(self, symbol: str, side: str, pnl_pct: float,
                               hold_minutes: float, context: TradeContext, is_win: bool):
        """Update the learned profile for this symbol."""
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolProfile(symbol=symbol)
        
        profile = self._symbols[symbol]
        profile.total_trades += 1
        profile.total_pnl += pnl_pct
        profile.avg_pnl = profile.total_pnl / profile.total_trades
        
        # Update win rate with exponential moving average
        profile.win_rate = 0.95 * profile.win_rate + 0.05 * (1.0 if is_win else 0.0)
        
        # Track best conditions from wins
        if is_win:
            if pnl_pct > profile.avg_pnl:
                profile.best_regime = context.regime
                profile.best_side = side
                profile.best_hour = context.hour_utc
                profile.best_day = context.day_of_week
                profile.optimal_hold_minutes = hold_minutes
        else:
            profile.worst_hour = context.hour_utc
            profile.worst_regime = context.regime
        
        profile.last_updated = utcnow()
    
    def _build_context_from_trade(self, trade_id: int, symbol: str, side: str) -> TradeContext:
        """Build context from trade record in database."""
        context = TradeContext(symbol=symbol, side=side)
        
        try:
            with session_scope() as s:
                # Get the Claude decision associated with this trade
                trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
                if trade and trade.claude_decision_id:
                    decision = s.query(ClaudeDecision).filter(
                        ClaudeDecision.id == trade.claude_decision_id
                    ).first()
                    if decision and decision.market_snapshot:
                        snap = json.loads(decision.market_snapshot) if isinstance(
                            decision.market_snapshot, str) else decision.market_snapshot
                        context.rsi = snap.get("rsi", 50)
                        context.macd_histogram = snap.get("macd_histogram", 0)
                        context.adx = snap.get("adx", 25)
                        context.volume_ratio = snap.get("volume_ratio", 1)
                        context.regime = snap.get("regime", "UNKNOWN")
                        context.signal_confidence = float(decision.confidence or 0.5)
                
                if trade and trade.opened_at:
                    context.hour_utc = trade.opened_at.hour
                    context.day_of_week = trade.opened_at.weekday()
        except Exception:
            # This swallow used to hide the metadata-vs-indicators bug for
            # weeks: every learn-time context build silently fell back to
            # defaults (rsi=50, regime=UNKNOWN) and the autonomous engine
            # learned from collapsed fingerprints. Bumped from logger.debug
            # (invisible) to logger.exception so any future shape mismatch
            # or schema drift shows a stack trace in the console
            # immediately. We still degrade to default-context return
            # because crashing the learn path on close is worse than
            # learning a less-accurate fingerprint.
            logger.exception(
                "[AUTONOMOUS] _build_context_from_trade failed for trade %s — "
                "fingerprint will be degenerate", trade_id,
            )

        return context
    
    # =========================================================================
    # MAKING DECISIONS
    # =========================================================================
    
    def decide(
        self,
        symbol: str,
        side: str,
        current_price: float,
        signal_confidence: float,
        context: Optional[TradeContext] = None,
    ) -> AutonomousDecision:
        """
        Make a trading decision based on learned patterns.
        This runs WITHOUT Claude API.
        """
        self._ensure_loaded()
        
        # Build context if not provided
        if context is None:
            context = TradeContext(
                symbol=symbol,
                side=side,
                signal_confidence=signal_confidence,
            )
        
        fingerprint = context.to_fingerprint()
        decision = AutonomousDecision(
            action=side,
            confidence=signal_confidence,
        )
        
        # =====================================================================
        # TRAINING MODE: When the user is running a live training session, we
        # want the bot to TRADE actively so it can learn new patterns — even
        # if those patterns historically lost money. A 36% win rate with 168
        # closed trades is too small a sample; the "known losing pattern"
        # database is mostly noise. Disable the AVOID shortcut so we gather
        # more data. The risk-manager's position-sizing and stop-losses still
        # protect the (paper) capital.
        # =====================================================================
        from config import bot_config as bot_cfg_mod
        training_active = str(bot_cfg_mod.get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
        
        # 1. Check for known mistake patterns (CRITICAL)
        mistake_penalty = self._check_mistakes(fingerprint, context)
        if mistake_penalty > 0:
            decision.confidence *= (1 - mistake_penalty)
            decision.avoided_patterns.append(f"mistake:{fingerprint}")
            
            # Only hard-block in live mode; in training we log and continue.
            if mistake_penalty > 0.5:
                if training_active:
                    decision.reasoning = f"Training mode — trading despite penalty={mistake_penalty:.0%} to gather data"
                else:
                    decision.action = "AVOID"
                    decision.reasoning = f"Matches known losing pattern (penalty={mistake_penalty:.0%})"
                    return decision
        
        # 2. Check for known success patterns
        pattern_bonus = self._check_patterns(fingerprint, context)
        if pattern_bonus != 0:
            decision.confidence *= (1 + pattern_bonus)
            if pattern_bonus > 0:
                decision.matched_patterns.append(f"success:{fingerprint}")
        
        # 3. Check symbol-specific insights
        symbol_adjustment = self._check_symbol_profile(symbol, side, context)
        decision.confidence *= symbol_adjustment
        
        # 4. Find similar historical trades
        similar_trades = self._find_similar_trades(context)
        if similar_trades:
            # similar_trades is a list of (pnl_pct, fingerprint) tuples — see
            # _find_similar_trades. pnl is index 0; fingerprint is index 1.
            # The v2 rewrite shipped this loop reading t[1] for pnl, which is
            # the fingerprint string — every call crashed with TypeError on
            # the first sum() because Python can't add 0 + "abc123...".
            avg_pnl = sum(t[0] for t in similar_trades) / len(similar_trades)
            win_rate = sum(1 for t in similar_trades if t[0] > 0) / len(similar_trades)

            decision.historical_expectancy = avg_pnl
            decision.pattern_match_score = win_rate
            
            # Adjust confidence based on similar trade outcomes
            if win_rate > 0.6:
                decision.confidence *= 1.1
            elif win_rate < 0.4:
                decision.confidence *= 0.8
        
        # 5. Calibrate confidence based on historical accuracy
        if fingerprint in self._patterns:
            pattern = self._patterns[fingerprint]
            decision.calibrated_confidence = decision.confidence * pattern.confidence_calibration
            decision.confidence = decision.calibrated_confidence
            
            # Use learned optimal hold time
            decision.recommended_hold_minutes = pattern.best_hold_minutes
            
            # Adjust sizing based on expectancy
            if pattern.expectancy > 0.02:
                decision.size_multiplier = min(1.5, 1 + pattern.expectancy * 10)
            elif pattern.expectancy < -0.01:
                decision.size_multiplier = max(0.5, 1 + pattern.expectancy * 10)
        
        # 6. Check recent performance (recency weighting)
        recent_win_rate = self._get_recent_win_rate()
        if recent_win_rate < 0.35:
            # On a losing streak - reduce size
            decision.size_multiplier *= 0.7
            decision.reasoning += " Reducing size due to recent losses."
        elif recent_win_rate > 0.65:
            # Hot streak - can be slightly more aggressive
            decision.size_multiplier *= 1.1
        
        # 7. Generate final reasoning
        decision.reasoning = self._generate_reasoning(decision, context, fingerprint)
        
        # 8. Clamp confidence to valid range
        decision.confidence = max(0.0, min(1.0, decision.confidence))
        
        return decision
    
    def _check_mistakes(self, fingerprint: str, context: TradeContext) -> float:
        """Check if current context matches known mistake patterns. Returns penalty 0-1."""
        # Direct fingerprint match
        if fingerprint in self._mistakes:
            mistake = self._mistakes[fingerprint]
            if mistake.severity == "critical":
                return 0.8
            elif mistake.severity == "high":
                return 0.5
            elif mistake.severity == "medium":
                return 0.3
            return 0.1
        
        # Check for similar mistakes (fuzzy matching)
        penalty = 0.0
        for m_fp, mistake in self._mistakes.items():
            conditions = mistake.conditions
            
            # Check RSI range
            rsi_range = conditions.get("rsi_range", (0, 100))
            if rsi_range[0] <= context.rsi <= rsi_range[1]:
                # Check regime
                if conditions.get("regime") == context.regime:
                    # Check side
                    if conditions.get("side") == context.side:
                        # Similar mistake found
                        if mistake.severity == "critical":
                            penalty = max(penalty, 0.6)
                        elif mistake.severity == "high":
                            penalty = max(penalty, 0.4)
                        else:
                            penalty = max(penalty, 0.2)
        
        return penalty
    
    def _check_patterns(self, fingerprint: str, context: TradeContext) -> float:
        """Check known patterns. Returns bonus/penalty multiplier."""
        if fingerprint not in self._patterns:
            return 0.0
        
        pattern = self._patterns[fingerprint]
        
        # Need enough samples for statistical significance
        if pattern.total_trades < 3:
            return 0.0
        
        # Good pattern: high win rate + positive expectancy
        if pattern.win_rate > 0.55 and pattern.expectancy > 0.01:
            return min(0.3, (pattern.win_rate - 0.5) + pattern.expectancy * 5)
        
        # Bad pattern: low win rate
        if pattern.win_rate < 0.4:
            return max(-0.3, (pattern.win_rate - 0.5))
        
        return 0.0
    
    def _check_symbol_profile(self, symbol: str, side: str, context: TradeContext) -> float:
        """Check symbol-specific learned insights. Returns adjustment multiplier."""
        if symbol not in self._symbols:
            return 1.0
        
        profile = self._symbols[symbol]
        adjustment = 1.0
        
        # Check if this is a favorable setup for this symbol
        if profile.best_side == side and profile.win_rate > 0.5:
            adjustment *= 1.1
        
        if profile.best_regime == context.regime:
            adjustment *= 1.05
        
        if profile.worst_regime == context.regime:
            adjustment *= 0.85
        
        if abs(context.hour_utc - profile.best_hour) <= 2:
            adjustment *= 1.05
        
        if abs(context.hour_utc - profile.worst_hour) <= 1:
            adjustment *= 0.9
        
        # Penalize symbols with poor track record
        if profile.total_trades > 5 and profile.win_rate < 0.35:
            adjustment *= 0.7
        
        return adjustment
    
    def _find_similar_trades(self, context: TradeContext, k: int = 10) -> List[Tuple[float, str]]:
        """Find the k most similar historical trades by vector distance.

        Returns a list of (pnl_pct, fingerprint) tuples — sorted by similarity,
        closest first. The (vector, pnl, fp) shape stored in self._trade_vectors
        is collapsed to (pnl, fp) since the caller never needs the vector
        again. NOTE: the type annotation in the v2 rewrite was wrong
        (claimed List[Tuple[List[float], float]]) and the call site at the
        top of decide() trusted that wrong annotation — crashes from that
        mismatch are what we just fixed."""
        if not self._trade_vectors:
            return []

        current_vector = context.to_vector()

        # Calculate distances to all stored vectors. Each stored entry may be
        # either a tuple or a JSON-loaded list (json.loads turns tuples into
        # lists), so we unpack defensively. A single malformed historical row
        # must not poison the entire result — any entry whose shape we can't
        # trust gets skipped, valid ones still returned.
        distances = []
        for entry in self._trade_vectors:
            try:
                stored_vector, pnl, fp = entry[0], entry[1], entry[2]
            except (TypeError, IndexError, ValueError):
                continue
            # Vector must be a list/tuple of numbers — _euclidean_distance
            # crashes if it's None or a scalar.
            if not isinstance(stored_vector, (list, tuple)):
                continue
            try:
                pnl = float(pnl)
            except (TypeError, ValueError):
                continue
            dist = self._euclidean_distance(current_vector, stored_vector)
            distances.append((dist, pnl, str(fp)))

        # Sort by distance and return top k as (pnl, fingerprint).
        distances.sort(key=lambda x: x[0])
        return [(d[1], d[2]) for d in distances[:k]]
    
    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(v1) != len(v2):
            return float('inf')
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    def _get_recent_win_rate(self, window: int = 20) -> float:
        """Get win rate of recent trades."""
        if not self._recent_trades:
            return 0.5
        
        recent = self._recent_trades[-window:]
        wins = sum(1 for t in recent if t.get("is_win"))
        return wins / len(recent) if recent else 0.5
    
    def _generate_reasoning(self, decision: AutonomousDecision, 
                           context: TradeContext, fingerprint: str) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = []
        
        if decision.matched_patterns:
            parts.append(f"Matches {len(decision.matched_patterns)} winning pattern(s)")
        
        if decision.avoided_patterns:
            parts.append(f"Avoided {len(decision.avoided_patterns)} losing pattern(s)")
        
        if decision.historical_expectancy > 0:
            parts.append(f"Similar trades avg +{decision.historical_expectancy:.1%}")
        elif decision.historical_expectancy < 0:
            parts.append(f"Similar trades avg {decision.historical_expectancy:.1%}")
        
        if decision.size_multiplier < 1:
            parts.append(f"Size reduced to {decision.size_multiplier:.0%}")
        elif decision.size_multiplier > 1:
            parts.append(f"Size increased to {decision.size_multiplier:.0%}")
        
        if context.symbol in self._symbols:
            profile = self._symbols[context.symbol]
            parts.append(f"{context.symbol} historical: {profile.win_rate:.0%} win rate")
        
        return "; ".join(parts) if parts else "Standard signal"
    
    # =========================================================================
    # STATISTICS AND INSIGHTS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning engine statistics."""
        self._ensure_loaded()
        
        total_patterns = len(self._patterns)
        winning_patterns = sum(1 for p in self._patterns.values() if p.expectancy > 0)
        
        total_mistakes = len(self._mistakes)
        critical_mistakes = sum(1 for m in self._mistakes.values() if m.severity == "critical")
        
        recent_win_rate = self._get_recent_win_rate()
        
        # Best and worst patterns
        best_patterns = sorted(
            [p for p in self._patterns.values() if p.total_trades >= 3],
            key=lambda x: x.expectancy,
            reverse=True
        )[:5]
        
        worst_patterns = sorted(
            [p for p in self._patterns.values() if p.total_trades >= 3],
            key=lambda x: x.expectancy
        )[:5]
        
        return {
            "total_patterns": total_patterns,
            "winning_patterns": winning_patterns,
            "total_mistakes": total_mistakes,
            "critical_mistakes": critical_mistakes,
            "total_symbols": len(self._symbols),
            "trade_vectors": len(self._trade_vectors),
            "recent_win_rate": recent_win_rate,
            "best_patterns": [
                {"fp": p.fingerprint, "win_rate": p.win_rate, "expectancy": p.expectancy, "trades": p.total_trades}
                for p in best_patterns
            ],
            "worst_patterns": [
                {"fp": p.fingerprint, "win_rate": p.win_rate, "expectancy": p.expectancy, "trades": p.total_trades}
                for p in worst_patterns
            ],
        }
    
    def get_symbol_insights(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get learned insights for a specific symbol."""
        self._ensure_loaded()
        
        if symbol not in self._symbols:
            return None
        
        profile = self._symbols[symbol]
        return {
            "symbol": symbol,
            "total_trades": profile.total_trades,
            "win_rate": profile.win_rate,
            "avg_pnl": profile.avg_pnl,
            "best_setup": {
                "regime": profile.best_regime,
                "side": profile.best_side,
                "hour": profile.best_hour,
                "day": profile.best_day,
            },
            "avoid": {
                "regime": profile.worst_regime,
                "hour": profile.worst_hour,
            },
            "optimal_hold_minutes": profile.optimal_hold_minutes,
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_engine: Optional[AutonomousLearningEngine] = None

def get_autonomous_engine() -> AutonomousLearningEngine:
    """Get or create the singleton autonomous learning engine."""
    global _engine
    if _engine is None:
        _engine = AutonomousLearningEngine()
    return _engine


# =============================================================================
# Calibrated win probability — three-tier estimator
# =============================================================================
# The single biggest piece of structural debt in the project was using
# `confidence` as a stand-in for "probability this trade wins". Confidence
# is a quality signal, not a calibrated probability. Meanwhile, the
# autonomous engine has been accumulating real (fingerprint -> win_rate,
# expectancy, sample_size) data on every closed trade. This block exposes
# that data as the calibrated input the rest of the pipeline should be
# using instead of a confidence guess.
#
# Three tiers, picked best-available:
#   1. EXACT_PATTERN  - this exact fingerprint has been seen ≥ MIN_EXACT
#                       times. Use its measured win_rate. Most reliable
#                       when it applies because the pattern matches by
#                       construction.
#   2. KNN_NEIGHBORS  - no exact pattern, but ≥ MIN_KNN similar trades by
#                       Euclidean distance on the context vector. Use
#                       their average outcome.
#   3. RAW_CONFIDENCE - no historical data of either kind. Fall back to
#                       the caller's confidence as the only signal we have.
#
# Callers receive a small dict so they can decide how much to trust the
# estimate (sample_size, source) instead of just a number.

# Minimum sample sizes for each tier to "count". Conservative because
# the cost of an over-confident edge estimate is real money; the cost of
# falling back to raw confidence is just preserving today's behavior.
#
# MIN_EXACT_PATTERN_TRADES was originally 5. Tuned down to 3 after
# observing that paper sessions with the COARSENED fingerprint were
# accumulating ~3-4 trades/cell on average — 5 was too conservative
# given a realistic session's cell density, and the tier never engaged
# in practice. At N=3 we can already detect 67-100% win rates with
# useful confidence; meta_confidence shrinkage (n / (n + 8)) still
# discounts the estimate heavily at low N so the blend with raw
# confidence is gentle.
MIN_EXACT_PATTERN_TRADES = 3
MIN_KNN_NEIGHBORS = 5


def get_pattern_stats(fingerprint: str) -> Optional[Dict[str, Any]]:
    """Return measured stats for an exact pattern fingerprint, or None if
    we don't yet have enough data on it.

    Threshold: needs at least MIN_EXACT_PATTERN_TRADES closed trades on
    this fingerprint before the win_rate is considered usable. Below that
    threshold the autonomous engine's win_rate is just noise around the
    0.5 default and would be a worse estimator than confidence.
    """
    engine = get_autonomous_engine()
    engine._ensure_loaded()
    pattern = engine._patterns.get(fingerprint)
    if pattern is None or pattern.total_trades < MIN_EXACT_PATTERN_TRADES:
        return None
    return {
        "win_rate": float(pattern.win_rate),
        "sample_size": int(pattern.total_trades),
        "expectancy": float(pattern.expectancy),
        "avg_win": float(pattern.avg_win),
        "avg_loss": float(pattern.avg_loss),
        "profit_factor": float(pattern.profit_factor),
    }


def get_calibrated_win_probability(
    context: "TradeContext",
    fallback_confidence: float,
) -> Dict[str, Any]:
    """Return the best-available win-probability estimate for a context.

    This is the function bot_engine._compute_mission_inputs and the
    position sizer should call instead of using raw confidence as a
    win-probability proxy.

    Always returns a dict with these keys (never None) so callers don't
    branch on null:
      - win_probability  (float, 0..1) — the estimator's best guess
      - sample_size      (int)         — how many trades back it
      - source           (str)         — "exact_pattern" | "knn_neighbors"
                                          | "raw_confidence"
      - confidence_in_estimate (float) — meta-confidence; raises as the
                                          sample size grows. Use this to
                                          decide whether to BLEND with
                                          confidence or trust outright.

    Resolution order:
      1. EXACT_PATTERN if context.to_fingerprint() has ≥ MIN_EXACT trades.
      2. KNN_NEIGHBORS if at least MIN_KNN similar historical trades exist.
      3. RAW_CONFIDENCE fallback otherwise.

    The fallback never throws. If anything in the lookup chain fails the
    function still returns a sane raw_confidence dict.
    """
    fp_clamped = max(0.0, min(1.0, float(fallback_confidence)))

    # Labeled swallow: if the caller hands us a broken context we degrade
    # to raw-confidence rather than blocking the trade. Calibration is
    # additive — without it the system trades exactly the way it did
    # pre-Phase-B.
    from utils.errors import swallow_with_reason
    fingerprint = None
    with swallow_with_reason(
        logger,
        "calibration falls back to raw_confidence when context.to_fingerprint() raises",
    ):
        fingerprint = context.to_fingerprint()
    if fingerprint is None:
        return _raw_confidence_estimate(fp_clamped)

    # Tier 1: exact pattern match
    stats = get_pattern_stats(fingerprint)
    if stats is not None:
        # confidence_in_estimate: rises from ~0 at MIN_EXACT trades to ~0.95
        # asymptotically as samples grow. Bayesian-ish shrinkage toward a
        # neutral prior — at 5 trades we only half-trust it, at 25 we trust
        # it strongly. The shape doesn't matter much; what matters is that
        # callers can blend with confidence when the sample is small.
        n = stats["sample_size"]
        meta_conf = n / (n + 8.0)  # n=5 -> 0.38, n=12 -> 0.6, n=25 -> 0.76
        return {
            "win_probability": stats["win_rate"],
            "sample_size": n,
            "source": "exact_pattern",
            "confidence_in_estimate": round(meta_conf, 3),
            "expectancy": stats["expectancy"],
            "avg_win": stats["avg_win"],
            "avg_loss": stats["avg_loss"],
        }

    # Tier 2: kNN of similar trades
    engine = get_autonomous_engine()
    neighbors: list = []
    with swallow_with_reason(
        logger,
        "calibration kNN lookup is best-effort; falls through to raw_confidence on failure",
    ):
        engine._ensure_loaded()
        neighbors = engine._find_similar_trades(context, k=20)

    if neighbors and len(neighbors) >= MIN_KNN_NEIGHBORS:
        wins = sum(1 for pnl, _fp in neighbors if pnl > 0)
        wr = wins / len(neighbors)
        avg_pnl = sum(pnl for pnl, _fp in neighbors) / len(neighbors)
        n = len(neighbors)
        # kNN is noisier than exact pattern — discount its meta-confidence
        # more aggressively.
        meta_conf = n / (n + 20.0)  # n=5 -> 0.20, n=20 -> 0.50
        return {
            "win_probability": wr,
            "sample_size": n,
            "source": "knn_neighbors",
            "confidence_in_estimate": round(meta_conf, 3),
            "expectancy": avg_pnl,
            "avg_win": None,
            "avg_loss": None,
        }

    # Tier 3: no historical data — caller's confidence is all we've got.
    return _raw_confidence_estimate(fp_clamped)


def _raw_confidence_estimate(confidence: float) -> Dict[str, Any]:
    """The pure-fallback shape: caller's confidence is treated as a
    win-probability guess. confidence_in_estimate is fixed low so
    callers know to apply heavy risk discounts."""
    return {
        "win_probability": confidence,
        "sample_size": 0,
        "source": "raw_confidence",
        "confidence_in_estimate": 0.0,
        "expectancy": None,
        "avg_win": None,
        "avg_loss": None,
    }


def learn_from_closed_trade(
    trade_id: int,
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    hold_minutes: float,
    exit_reason: str,
):
    """Convenience function to record a trade outcome."""
    engine = get_autonomous_engine()
    engine.learn_from_trade(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        hold_minutes=hold_minutes,
        exit_reason=exit_reason,
    )


def get_autonomous_decision(
    symbol: str,
    side: str,
    current_price: float,
    signal_confidence: float,
    context: Optional[TradeContext] = None,
) -> AutonomousDecision:
    """Convenience function to get a decision from the autonomous engine.

    Pass a populated `context` so fingerprinting, kNN, and pattern lookups
    have real indicator/regime data. When None, `decide()` falls back to a
    degenerate TradeContext (rsi=50, regime=UNKNOWN) and every signal
    collapses to roughly the same fingerprint.
    """
    engine = get_autonomous_engine()
    return engine.decide(
        symbol=symbol,
        side=side,
        current_price=current_price,
        signal_confidence=signal_confidence,
        context=context,
    )


