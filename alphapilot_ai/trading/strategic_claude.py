"""
Strategic Claude Integration  (v2 — enhanced)

The traffic cop that decides "should we burn a Claude API call on this tick, or
trust the free engines?" — and the orchestrator that combines the always-on
autonomous learning engine with the occasional Claude consultation.

Philosophy (unchanged):
  - Claude is expensive ($0.01-0.03 / call). Don't waste it on obvious decisions.
  - Use Claude for ambiguous signals, large/high-stakes positions, reversals,
    unusual conditions, and post-trade reflection.

WHAT CHANGED FROM v1 (full CHANGELOG at the bottom):
  * FIXED (big one): the autonomous engine now receives a POPULATED context
    built from the live signal's indicators, instead of being called with no
    context (which collapsed every decision to one of two default fingerprints
    and blinded its pattern/mistake/symbol machinery). engine.decide() already
    accepts a context — v1 just never built one. This is the decision-side half
    of the fix; the close-side learning loop is batch 2.
  * FIXED: should_consult_claude no longer lets the "strong signal" early-return
    short-circuit the large-position / reversal / high-volatility escalations —
    those are exactly the high-stakes cases the docstring says warrant Claude.
  * FIXED: market_volatility is now derived from the signal and passed through,
    so the high-volatility trigger can actually fire (it was dead in v1).
  * FIXED: reversal detection works — v1 read wallet["open_positions"] which
    bot_engine never populates; we now fall back to a light DB lookup.
  * FIXED: budget reconciliation. The router now defers to the decision engine's
    real budget and only records a consultation when a REAL Claude call happened
    (v1 burned a slot + cooldown even when claude_decide bypassed to a technical
    passthrough — phantom consults).
  * NEW: expected_value / priority are now actually used — under budget pressure
    the router raises the EV bar so scarce calls go to the highest-stakes trades.
  * NEW: post-trade reflection also fires on high-confidence losses (the most
    instructive case).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


_PRIORITY_RANK = {"none": 0, "low": 1, "normal": 2, "high": 3, "critical": 4}


def _max_priority(a: str, b: str) -> str:
    return a if _PRIORITY_RANK.get(a, 0) >= _PRIORITY_RANK.get(b, 0) else b


def _grade_from_confidence(conf: float) -> str:
    if conf >= 0.80:
        return "A+"
    if conf >= 0.70:
        return "A"
    if conf >= 0.55:
        return "B"
    if conf >= 0.40:
        return "C"
    return "F"


# =============================================================================
# CONSULTATION GATE
# =============================================================================

@dataclass
class ClaudeConsultation:
    """Result of deciding whether to consult Claude."""
    should_consult: bool
    reason: str
    priority: str = "normal"          # none/low/normal/high/critical
    expected_value: float = 0.0       # estimated value of Claude's input


class StrategicClaudeRouter:
    """
    Decides when a Claude consultation adds enough value to be worth the cost.

    CONSULT for: ambiguous signals (40-65%), large positions (>$500), reversals
    against an open position, or unusual volatility — including when the
    technical signal is otherwise strong (those are the high-stakes cases).
    SKIP for: tiny positions, repeated decisions (cooldown), or a plain
    strong/weak signal with nothing high-stakes about it.
    """

    AMBIGUOUS_CONF_LOW = 0.40
    AMBIGUOUS_CONF_HIGH = 0.65
    STRONG_CONF = 0.70
    WEAK_CONF = 0.30
    LARGE_POSITION_THRESHOLD = 500    # USD
    MIN_POSITION_FOR_CLAUDE = 50      # USD

    def __init__(self):
        self._recent_consultations: dict[str, float] = {}   # symbol -> ts of last REAL consult
        self._consultation_cooldown = 300                   # 5 min between real consults per symbol
        self._daily_budget_used = 0                         # real Claude calls we routed today
        self._daily_budget_limit = 50                       # secondary cap; primary is the decision engine's
        self._total_skipped = 0

    def should_consult_claude(
        self,
        symbol: str,
        technical_confidence: float,
        position_size_usd: float,
        signal_side: str,
        has_existing_position: bool = False,
        existing_position_side: Optional[str] = None,
        market_volatility: str = "normal",
        budget_remaining: Optional[int] = None,
        expected_value_bar: Optional[float] = None,
    ) -> ClaudeConsultation:
        """Determine if this decision warrants a Claude consultation.

        v2 fix: escalation triggers (large / reversal / high-vol) are evaluated
        BEFORE the strong/weak confidence gates, so a high-conviction large
        position or a reversal can still reach Claude. v1's early returns made
        those triggers unreachable outside the 0.30-0.70 band.
        """

        def _no(reason: str) -> ClaudeConsultation:
            self._total_skipped += 1
            return ClaudeConsultation(should_consult=False, reason=reason, priority="none")

        # ----- Unconditional hard blocks ------------------------------------
        # Defer to the decision engine's real budget when provided.
        if budget_remaining is not None and budget_remaining <= 0:
            return _no("Daily Claude budget exhausted (decision engine)")
        if self._daily_budget_used >= self._daily_budget_limit:
            return _no("Daily Claude budget exhausted (router cap)")

        last = self._recent_consultations.get(symbol, 0)
        if time.time() - last < self._consultation_cooldown:
            return _no(f"Recently consulted {symbol}; on cooldown")

        if position_size_usd < self.MIN_POSITION_FOR_CLAUDE:
            return _no(f"Position too small (${position_size_usd:.0f})")

        # ----- Compute escalation triggers (high-stakes reasons) ------------
        reasons: list[str] = []
        priority = "normal"
        ev = 0.0

        ambiguous = self.AMBIGUOUS_CONF_LOW <= technical_confidence <= self.AMBIGUOUS_CONF_HIGH
        large = position_size_usd >= self.LARGE_POSITION_THRESHOLD
        reversal = bool(
            has_existing_position and existing_position_side
            and signal_side != existing_position_side
        )
        high_vol = market_volatility in {"high", "extreme"}
        high_stakes = large or reversal or high_vol

        if ambiguous:
            reasons.append(f"ambiguous ({technical_confidence:.0%})")
            ev += 0.30
        if large:
            reasons.append(f"large position (${position_size_usd:.0f})")
            priority = _max_priority(priority, "high")
            ev += 0.40
        if reversal:
            reasons.append(f"reversal vs open {existing_position_side}")
            priority = _max_priority(priority, "critical")
            ev += 0.50
        if high_vol:
            reasons.append(f"volatility {market_volatility}")
            priority = _max_priority(priority, "high")
            ev += 0.30

        # ----- Confidence gates, but ESCALATIONS override them --------------
        # Weak signal with nothing high-stakes -> just skip the trade.
        if technical_confidence < self.WEAK_CONF and not high_stakes:
            return _no(f"Weak signal ({technical_confidence:.0%}); skip")
        # Strong signal with nothing high-stakes -> technical is enough.
        if technical_confidence > self.STRONG_CONF and not high_stakes:
            return _no(f"Strong technical signal ({technical_confidence:.0%})")

        if not reasons:
            return _no("Standard trade; using free engines")

        # ----- Budget-aware EV prioritization (NEW) -------------------------
        # When calls are scarce, demand a higher expected value so the few
        # remaining go to the highest-stakes consultations.
        if expected_value_bar is not None and ev < expected_value_bar:
            return _no(f"EV {ev:.2f} below scarce-budget bar {expected_value_bar:.2f}")

        return ClaudeConsultation(
            should_consult=True,
            reason="; ".join(reasons),
            priority=priority,
            expected_value=ev,
        )

    def record_consultation(self, symbol: str) -> None:
        """Record a REAL Claude consultation (sets cooldown + increments budget).

        v2: callers must invoke this ONLY when an actual API call happened
        (decision.source == "claude"), not merely when they decided to consult —
        otherwise bypassed/cached/passthrough decisions burn phantom slots.
        """
        self._recent_consultations[symbol] = time.time()
        self._daily_budget_used += 1
        cutoff = time.time() - 3600
        self._recent_consultations = {s: t for s, t in self._recent_consultations.items() if t > cutoff}

    def reset_daily_budget(self) -> None:
        """Reset the router's daily counter (call at midnight UTC)."""
        self._daily_budget_used = 0

    def get_router_stats(self) -> dict:
        return {
            "daily_budget_used": self._daily_budget_used,
            "daily_budget_limit": self._daily_budget_limit,
            "symbols_on_cooldown": len(self._recent_consultations),
            "total_skipped": self._total_skipped,
        }


# =============================================================================
# POST-TRADE REFLECTION
# =============================================================================

@dataclass
class TradeReflection:
    """Analysis of a completed trade for learning (in-memory dataclass; distinct
    from the database.models.TradeReflection ORM row)."""
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
    """Decides which closed trades are worth a (paid) Claude reflection."""

    MIN_PNL_FOR_REFLECTION = 0.02         # 2% move either direction
    MIN_POSITION_FOR_REFLECTION = 100     # $100 minimum

    def should_reflect(
        self,
        pnl_pct: float,
        position_size_usd: float,
        was_stopped_out: bool,
        signal_confidence: Optional[float] = None,
    ) -> bool:
        """Reflect on the most instructive trades.

        v2 adds the confidence/outcome-mismatch case: a HIGH-confidence trade
        that LOST is the single most valuable thing to reflect on, because it
        means our conviction model was wrong. (Backward-compatible: the new
        signal_confidence arg is optional.)
        """
        # High-confidence loss — the prediction was confidently wrong.
        if signal_confidence is not None and signal_confidence >= 0.70 and pnl_pct < 0:
            return True
        # Always reflect on meaningful stop-outs.
        if was_stopped_out and abs(pnl_pct) > 0.03:
            return True
        # Significant wins/losses on a real position.
        if abs(pnl_pct) >= self.MIN_PNL_FOR_REFLECTION and position_size_usd >= self.MIN_POSITION_FOR_REFLECTION:
            return True
        # Big wins (what went right).
        if pnl_pct > 0.05:
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
# CONTEXT BUILDER  (the fix that un-collapses the autonomous fingerprint)
# =============================================================================

def _derive_regime(ind: dict) -> str:
    """Single source of truth for the regime label.

    Delegates to claude_decision_engine._derive_regime_hint so the
    decide-time fingerprint (built here) and the persisted market_snapshot
    (also built via _derive_regime_hint through _extract_market_state)
    produce the SAME regime string. If these two diverge — even on
    fallback labels like RANGING vs DRIFT_UP — the autonomous engine's
    learn-time fingerprint won't match the decide-time one, and the
    pattern/kNN/mistake tables silently fragment.

    Cross-module reach into an underscore-prefixed helper is intentional:
    the helper is the canonical regime classifier for the pipeline.
    """
    try:
        from ai.claude_decision_engine import _derive_regime_hint
        return str(_derive_regime_hint(ind).get("regime") or "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _build_autonomous_context(symbol: str, side: str, technical_signal, strategy_type: str, tech_confidence: float):
    """Build a POPULATED TradeContext from the live signal indicators so the
    autonomous engine's fingerprint/vector carry real data.

    Returns None if the engine/context type can't be imported (the engine then
    falls back to its own default context — same as v1, no regression)."""
    try:
        from ai.autonomous_learning_engine import TradeContext
    except Exception:
        return None

    ind = getattr(technical_signal, "indicators", {}) or {}

    def f(*keys, default):
        for k in keys:
            v = ind.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    now = datetime.now(timezone.utc)
    ctx = TradeContext(symbol=symbol, side=side)

    # Fingerprint-critical fields (these are what was collapsing to defaults).
    ctx.rsi = f("rsi", "rsi_fast", default=50.0)
    ctx.macd_histogram = f("macd_histogram", default=0.0)
    ctx.macd = f("macd", default=0.0)
    ctx.adx = f("adx", default=25.0)
    ctx.volume_ratio = f("relative_volume", "volume_ratio", default=1.0)
    ctx.bb_percent = f("bb_percent_b", "bb_percent", default=0.5)
    atr_pct = ind.get("atr_pct")
    if atr_pct is not None:
        try:
            ctx.atr_percent = float(atr_pct) * 100.0   # engine expects percent units
        except (TypeError, ValueError):
            pass
    ctx.regime = _derive_regime(ind)
    ctx.hour_utc = now.hour
    ctx.day_of_week = now.weekday()

    # Signal metadata.
    ctx.signal_confidence = float(tech_confidence)
    ctx.signal_quality = _grade_from_confidence(float(tech_confidence))
    ctx.strategy = strategy_type or getattr(technical_signal, "strategy", "Momentum") or "Momentum"

    # Best-effort returns/velocity (used by the similarity vector, not the
    # fingerprint). Velocity over a few bars is a reasonable short-return proxy.
    v3 = ind.get("velocity_3bar")
    if v3 is not None:
        try:
            ctx.return_1h = float(v3)
        except (TypeError, ValueError):
            pass
    return ctx


def _volatility_label(ind: dict) -> str:
    """Derive a coarse volatility label from the signal so the high-volatility
    consult trigger can fire (v1 never passed this, so it was dead)."""
    exp = ind.get("atr_expansion")
    atr_pct = ind.get("atr_pct")
    try:
        if exp is not None and float(exp) >= 2.5:
            return "extreme"
        if exp is not None and float(exp) >= 1.8:
            return "high"
        if atr_pct is not None and float(atr_pct) >= 0.05:
            return "high"
    except (TypeError, ValueError):
        pass
    return "normal"


def _existing_position_side(wallet: dict, symbol: str) -> Optional[str]:
    """Side of any open position on this symbol (for reversal detection).

    Prefers wallet["open_positions"] if present (bot_engine may populate it in
    batch 2), otherwise a light indexed DB lookup. v1 only read the wallet dict,
    which bot_engine never fills, so reversal detection was always dead."""
    for pos in (wallet.get("open_positions") or []):
        if isinstance(pos, dict) and pos.get("symbol") == symbol:
            return pos.get("side")
    try:
        from database.db import session_scope
        from database.models import PaperTrade
        with session_scope() as s:
            row = (
                s.query(PaperTrade.side)
                .filter(
                    PaperTrade.wallet_id == int(wallet["id"]),
                    PaperTrade.symbol == symbol,
                    PaperTrade.status == "open",
                )
                .first()
            )
            return row[0] if row else None
    except Exception:
        return None


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class StrategicRouter:
    """
    Main decision router combining:
      1. Autonomous learning engine (FREE — always runs, now with REAL context)
      2. Claude (PAID — only when the gate says it adds value)

    Returns a TradeDecision (from ai.claude_decision_engine).
    """

    def __init__(self):
        self._claude_router = get_claude_router()
        self._reflector = get_reflector()

    def decide(
        self,
        wallet: dict,
        symbol: str,
        price: float,
        technical_signal,                 # Signal from strategy engine
        strategy_type: str,
        position_size_usd: float = 100,
    ):
        # Lazy imports to avoid circular dependencies.
        from ai.claude_decision_engine import TradeDecision, decide as claude_decide
        try:
            from ai.claude_decision_engine import get_api_usage_stats
        except Exception:
            get_api_usage_stats = None  # type: ignore

        side = technical_signal.side
        tech_confidence = float(technical_signal.confidence or 0.5)
        indicators = getattr(technical_signal, "indicators", {}) or {}

        # 1. ALWAYS get the autonomous decision (FREE) — now with a POPULATED
        #    context so it stops deciding on a collapsed default fingerprint.
        autonomous_decision = self._autonomous_decide(
            symbol, side, price, tech_confidence, technical_signal, strategy_type
        )

        # 2. Autonomous AVOID short-circuit.
        if autonomous_decision.action == "AVOID":
            logger.info("[AUTONOMOUS] Avoiding %s: %s", symbol, autonomous_decision.reasoning)
            d = TradeDecision(
                action="HOLD", confidence=0.0, size_multiplier=0.0,
                stop_loss_pct=0.05, take_profit_pct=0.10,
                rationale=f"[AUTONOMOUS_AVOID] {autonomous_decision.reasoning}",
                key_factors=list(getattr(autonomous_decision, "avoided_patterns", []) or []),
                risk_flags=["autonomous_avoid"], source="autonomous",
            )
            _set_quality(d, 0.0)
            return d

        adjusted_confidence = autonomous_decision.confidence
        size_multiplier = autonomous_decision.size_multiplier

        # 3. Gather the context the consult gate needs (now actually populated).
        existing_side = _existing_position_side(wallet, symbol)
        has_position = existing_side is not None
        market_vol = _volatility_label(indicators)

        budget_remaining = None
        if get_api_usage_stats is not None:
            try:
                budget_remaining = int(get_api_usage_stats().get("remaining", 0))
            except Exception:
                budget_remaining = None
        ev_bar = self._ev_bar_for_budget(budget_remaining)

        consultation = self._claude_router.should_consult_claude(
            symbol=symbol,
            technical_confidence=adjusted_confidence,
            position_size_usd=position_size_usd * max(0.0, size_multiplier),
            signal_side=side,
            has_existing_position=has_position,
            existing_position_side=existing_side,
            market_volatility=market_vol,
            budget_remaining=budget_remaining,
            expected_value_bar=ev_bar,
        )

        # 4. Consult Claude when the gate says it's worth it.
        if consultation.should_consult:
            logger.info("[CLAUDE] Consulting %s (%s, EV=%.2f): %s",
                        symbol, consultation.priority, consultation.expected_value, consultation.reason)
            try:
                claude_decision = claude_decide(
                    wallet=wallet, symbol=symbol, price=price,
                    technical_signal=technical_signal, strategy_type=strategy_type,
                )
                # v2: only burn a slot/cooldown if a REAL API call happened.
                if getattr(claude_decision, "source", "") == "claude":
                    self._claude_router.record_consultation(symbol)
                # Blend autonomous sizing conviction with Claude's call.
                claude_decision.size_multiplier = _clamp(
                    claude_decision.size_multiplier * max(0.0, size_multiplier), 0.0, 1.0
                )
                claude_decision.rationale = f"[CLAUDE+AUTO] {claude_decision.rationale}"
                return claude_decision
            except Exception as e:
                logger.warning("[CLAUDE] consult failed, using autonomous: %s", e)

        # 5. Autonomous-only path. Lenient pre-filter floor; bot_engine's
        #    configured floor remains authoritative downstream.
        floor = self._lenient_floor()
        action = side if adjusted_confidence >= floor else "HOLD"
        d = TradeDecision(
            action=action,
            confidence=adjusted_confidence,
            size_multiplier=size_multiplier,
            stop_loss_pct=autonomous_decision.stop_loss_pct,
            take_profit_pct=autonomous_decision.take_profit_pct,
            rationale=f"[AUTONOMOUS] {autonomous_decision.reasoning}",
            key_factors=list(getattr(autonomous_decision, "matched_patterns", []) or [])[:3],
            risk_flags=[],
            source="autonomous",
        )
        _set_quality(d, adjusted_confidence)
        # Persist a ClaudeDecision row even for the no-Claude path so the
        # autonomous engine has a market_snapshot to learn from at close
        # time. Without this, every signal that's "not worth a Claude call"
        # — typically the majority of ticks — opens with claude_decision_id=None
        # and the learn-side loop falls back to degenerate context, defeating
        # Phase A on the bulk of the trade flow. Side-effects d.claude_decision_id.
        try:
            from ai.claude_decision_engine import persist_decision
            persist_decision(
                wallet=wallet,
                symbol=symbol,
                price=price,
                technical_signal=technical_signal,
                decision=d,
                prompt_used="[AUTONOMOUS_ONLY]",
                source_override="autonomous",
            )
        except Exception:
            # Persistence is best-effort: a logging failure must never
            # cancel a trade that the engines already approved.
            logger.exception("Failed to persist autonomous-only decision for %s", symbol)
        logger.debug("[AUTONOMOUS] %s: %s conf=%.2f size=%.1fx",
                     symbol, action, adjusted_confidence, size_multiplier)
        return d

    # ----- helpers ----------------------------------------------------------

    def _autonomous_decide(self, symbol, side, price, tech_confidence, technical_signal, strategy_type):
        """Call the autonomous engine WITH a populated context. Degrades safely
        to the contextless convenience call if anything is unavailable."""
        try:
            from ai.autonomous_learning_engine import get_autonomous_engine
            engine = get_autonomous_engine()
            ctx = _build_autonomous_context(symbol, side, technical_signal, strategy_type, tech_confidence)
            try:
                return engine.decide(
                    symbol=symbol, side=side, current_price=price,
                    signal_confidence=tech_confidence, context=ctx,
                )
            except TypeError:
                # Engine signature without context kwarg — fall back.
                return engine.decide(
                    symbol=symbol, side=side, current_price=price, signal_confidence=tech_confidence,
                )
        except Exception:
            from ai.autonomous_learning_engine import get_autonomous_decision
            return get_autonomous_decision(
                symbol=symbol, side=side, current_price=price, signal_confidence=tech_confidence,
            )

    @staticmethod
    def _ev_bar_for_budget(remaining: Optional[int]) -> Optional[float]:
        """Raise the EV bar as the real budget runs low, so the last few calls
        go to the highest-stakes consults."""
        if remaining is None:
            return None
        if remaining > 10:
            return 0.0      # plenty left — no extra gate
        if remaining > 4:
            return 0.40     # require at least a 'large' or 'reversal' class reason
        return 0.70         # nearly out — reversals / stacked reasons only

    @staticmethod
    def _lenient_floor() -> float:
        """A LENIENT pre-filter floor (so the router doesn't override the user's
        configured floor). bot_engine's effective floor is authoritative."""
        try:
            from config import bot_config as cfg
            training = str(cfg.get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            training = False
        return 0.15 if training else 0.30


# =============================================================================
# SHARED TINY HELPERS
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def _set_quality(decision, confidence: float) -> None:
    """Set the conviction grade if the TradeDecision supports it (it does in
    decision-engine v2; older versions silently ignore the attribute)."""
    try:
        decision.quality = _grade_from_confidence(float(confidence))
    except Exception:
        pass


# =============================================================================
# SINGLETON ACCESSORS
# =============================================================================

_claude_router: Optional[StrategicClaudeRouter] = None
_reflector: Optional[PostTradeReflector] = None
_strategic_router: Optional["StrategicRouter"] = None


def get_claude_router() -> StrategicClaudeRouter:
    global _claude_router
    if _claude_router is None:
        _claude_router = StrategicClaudeRouter()
    return _claude_router


def get_reflector() -> PostTradeReflector:
    global _reflector
    if _reflector is None:
        _reflector = PostTradeReflector()
    return _reflector


def get_strategic_router() -> "StrategicRouter":
    global _strategic_router
    if _strategic_router is None:
        _strategic_router = StrategicRouter()
    return _strategic_router


# =============================================================================
# CHANGELOG (v1 -> v2)
# =============================================================================
# BUG FIXES
#   1. Autonomous engine now receives a POPULATED TradeContext built from the
#      live signal indicators (_build_autonomous_context). v1 called it with no
#      context, collapsing every decision to one of two default fingerprints and
#      blinding its pattern/mistake/symbol logic. engine.decide() already
#      accepts context — we just build and pass it now.
#      *** BEHAVIOR CHANGE — backtest. (Decision-side half; the close-side
#          learning loop is batch 2.) ***
#   2. should_consult_claude: escalation triggers (large / reversal / high-vol)
#      are evaluated BEFORE the strong/weak confidence gates and override the
#      strong-signal skip. v1's early returns made a high-conviction large
#      position or reversal unable to reach Claude.
#   3. market_volatility is derived from the signal (_volatility_label) and
#      passed in, so the high-volatility trigger can fire (dead in v1).
#   4. Reversal detection works: _existing_position_side falls back to a light
#      DB lookup instead of relying on wallet["open_positions"], which
#      bot_engine never populates.
#   5. Budget reconciliation: the gate defers to the decision engine's real
#      remaining budget, and record_consultation fires ONLY on a real Claude
#      call (decision.source == "claude"). v1 burned slots/cooldown on bypassed
#      consults (phantom consults).
#
# CAPABILITY UPGRADES
#   6. expected_value / priority are now used: under budget pressure the EV bar
#      rises (_ev_bar_for_budget), so scarce calls go to the highest-stakes
#      consults. (v1 computed EV/priority and ignored them.)
#   7. PostTradeReflector.should_reflect also triggers on high-confidence losses
#      (confidence >= 0.70 and pnl < 0) — the most instructive case.
#   8. Conviction grade set on autonomous-path decisions so bot_engine's sizer
#      gets a real grade instead of the default 'B'.
#   9. get_router_stats() for monitoring.
#
# CONTRACT PRESERVED
#   - StrategicRouter.decide(wallet, symbol, price, technical_signal,
#     strategy_type, position_size_usd=100) -> TradeDecision  (unchanged).
#   - should_consult_claude / record_consultation / reset_daily_budget /
#     should_reflect / generate_reflection_prompt keep their names; new args are
#     optional & keyword (backward-compatible).
#   - ClaudeConsultation / TradeReflection dataclasses unchanged.
#   - get_claude_router / get_reflector / get_strategic_router singletons kept.
#
# DEFERRED (batch 2)
#   - Close-side learning loop: wire autonomous_learning_engine.learn_from_trade
#     (and verify learn_from_closed_trade is actually called on close) and store
#     the entry context so learned fingerprints MATCH the rich decision-time
#     fingerprints this file now produces. Until then the decision side sends
#     real context but the memory banks it queries may still be sparse.
#   - bot_engine populating wallet["open_positions"] + market_volatility would
#     let us drop the per-symbol DB lookup here.