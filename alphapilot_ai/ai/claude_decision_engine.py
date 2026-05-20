"""
Claude-driven trade decision engine.  (v2 — enhanced)

This module is the "advisor brain" of AlphaPilot. For every candidate trade the
router decides is worth a Claude consultation, we hand Claude a structured packet
containing:

  - the technical signal from strategy_engine (side, confidence, *and now its
    full indicator dict* — see the metadata fix below)
  - a derived market-regime hint computed from those indicators
  - the wallet's current open positions and recent realized P&L
  - the wallet's recent history *on this specific symbol* (so Claude can
    self-correct: "we lost on PEPE-USD 4 of the last 5 entries")
  - the wallet's risk profile + caps
  - the learned playbook (only when backed by REAL closed-trade reflections)
  - optional external intelligence (fear/greed, derivatives, social, MTF) when
    the relevant connectors are available

Claude returns a strict JSON object describing what it wants to do:

    {
      "action": "BUY" | "SELL" | "HOLD" | "CLOSE",
      "confidence": 0.0 - 1.0,
      "size_multiplier": 0.0 - 1.0,
      "stop_loss_pct": 0.005 - 0.20,
      "take_profit_pct": 0.005 - 0.50,
      "rationale": "short paragraph",
      "key_factors": ["..."],
      "risk_flags": ["..."]
    }

The engine NEVER blindly trusts Claude:
  - sizing/stop/take requests are clamped to safe bounds
  - confidence is post-calibrated against this wallet's historical
    predicted-vs-realized accuracy (anti-overconfidence)
  - on any failure (not configured, timeout, invalid JSON) we fall back to the
    raw technical signal — trading never stops because the LLM hiccupped
  - every decision (Claude's *and* every fallback/passthrough) is persisted to
    ClaudeDecision for the audit trail

WHAT CHANGED FROM v1 (see CHANGELOG at bottom of module for detail):
  * FIXED: the prompt now actually receives the signal's indicators
    (v1 read a non-existent ``.metadata`` attribute and always sent ``{}``).
  * FIXED: decision cache is now price-aware AND wallet-scoped (v1 ignored
    price entirely and bled decisions across wallets).
  * FIXED: indicator legend now matches the keys strategy_engine truly emits.
  * NEW:   per-symbol recent-trade history injected into the prompt.
  * NEW:   confidence calibration loop (damps systematic overconfidence).
  * NEW:   derived regime token always present, even without the (dead)
           advanced_signal_engine.
  * NEW:   adaptive-learning market_state is now built from the live signal
           indicators instead of an empty extra_context.
  * NEW:   conviction grade (quality) emitted so position sizing stops
           treating every trade as a 'B'.
  * NEW:   optional, OFF-by-default second-opinion cross-check for high stakes.

COST OPTIMIZATION:
  - Decision caching: same (wallet, symbol) + similar price = cached decision
  - Training passthrough: directional technical signals above the floor bypass
    Claude entirely
  - Daily budget: hard limit on API calls per day (in-memory, per process)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai.claude_learning import build_playbook
from ai.adaptive_learning_engine import analyze_signal
from database.db import session_scope
from database.models import ClaudeDecision, PaperTrade
from services.claude_client import chat as claude_chat
from services.claude_client import is_configured as claude_is_configured
from trading.strategy_engine import Signal
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# TUNABLES
# =============================================================================

# --- Decision cache -------------------------------------------------------- #
# Key is (wallet_id, symbol). A decision is reused only if it is younger than
# DECISION_CACHE_TTL *and* the price has not moved more than
# PRICE_CHANGE_THRESHOLD since it was cached. (v1 ignored price completely.)
DECISION_CACHE_TTL = 1800          # 30 minutes
PRICE_CHANGE_THRESHOLD = 0.02      # 2% move invalidates a cached decision
_DECISION_CACHE_MAX = 500          # hard cap on cached entries

# --- Daily API budget ------------------------------------------------------ #
# NOTE: this counter is in-memory and resets when the process restarts. The
# strategic_claude router keeps its OWN, larger budget; reconciling the two is
# a batch-2 task (see CHANGELOG). At ~$0.01-0.03/call this caps daily spend.
DAILY_API_BUDGET = 25

# --- Model call ------------------------------------------------------------ #
DECISION_MAX_TOKENS = 700
DECISION_TEMPERATURE = 0.2         # decisions should be near-deterministic

# --- Soft response-time SLA ------------------------------------------------ #
# We cannot force a hard timeout from here (that lives in services.claude_client),
# but we measure latency and flag stale responses on fast-moving symbols.
DECISION_SLA_SECONDS = 4.0

# --- Confidence calibration ------------------------------------------------ #
# If this wallet's Claude trades historically realize a far lower win rate than
# the confidence Claude assigned, we damp incoming confidence by the ratio.
CALIBRATION_MIN_SAMPLE = 15        # need this many closed trades to calibrate
CALIBRATION_FLOOR = 0.60           # never damp below 60% of stated confidence
CALIBRATION_CEIL = 1.15            # allow a small boost if under-confident
CALIBRATION_TTL = 300              # recompute at most every 5 min per wallet

# --- Second opinion (OFF by default) --------------------------------------- #
# When enabled, high-stakes directional decisions are re-checked with a
# devil's-advocate prompt; disagreement downgrades the trade to HOLD. Costs an
# extra API call + budget slot, so it ships disabled.
ENABLE_SECOND_OPINION = False
SECOND_OPINION_SIZE_THRESHOLD = 1.0    # size_multiplier at/above this is "high stakes"
SECOND_OPINION_CONF_THRESHOLD = 0.85   # or confidence at/above this

# --- Passthrough --------------------------------------------------------- #
STRONG_PASSTHROUGH_FLOOR = 0.62    # technical confidence >= this bypasses Claude


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def _now_ts() -> float:
    return time.time()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class _TTLCache:
    """Tiny time-bounded cache for optional global/per-symbol intelligence."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._d: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        entry = self._d.get(key)
        if not entry:
            return None
        ts, val = entry
        if _now_ts() - ts > self.ttl:
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: Any, val: Any) -> None:
        self._d[key] = (_now_ts(), val)


# Caches for optional external data that does not change per-tick.
_global_md_cache = _TTLCache(ttl=300)     # fear/greed, derivatives, MTF
_calibration_cache = _TTLCache(ttl=CALIBRATION_TTL)


# =============================================================================
# DAILY API BUDGET
# =============================================================================

_api_calls_today: dict[str, int] = {}     # date_str -> count


def _check_api_budget() -> tuple[bool, int]:
    """Return (can_call, remaining_calls) for today."""
    calls = _api_calls_today.get(_today_str(), 0)
    remaining = DAILY_API_BUDGET - calls
    return remaining > 0, remaining


def _increment_api_calls() -> None:
    today = _today_str()
    _api_calls_today[today] = _api_calls_today.get(today, 0) + 1
    # Drop stale dates to avoid unbounded growth.
    for d in [d for d in _api_calls_today if d != today]:
        del _api_calls_today[d]


def get_api_usage_stats() -> dict:
    """Current API usage statistics for monitoring/UI."""
    today = _today_str()
    calls = _api_calls_today.get(today, 0)
    return {
        "date": today,
        "calls_today": calls,
        "daily_budget": DAILY_API_BUDGET,
        "remaining": max(0, DAILY_API_BUDGET - calls),
        "budget_pct_used": round(calls / DAILY_API_BUDGET * 100, 1) if DAILY_API_BUDGET else 0,
        "cache_size": len(_decision_cache),
    }


# =============================================================================
# DECISION CACHE  (wallet-scoped + price-aware)
# =============================================================================

# key -> (timestamp, price_at_cache, decision)
_decision_cache: dict[tuple[int, str], tuple[float, float, "TradeDecision"]] = {}


def _cache_key(wallet_id: int, symbol: str) -> tuple[int, str]:
    return (int(wallet_id), symbol)


def _get_cached_decision(wallet_id: int, symbol: str, current_price: float) -> "TradeDecision | None":
    """Return a cached decision iff it is fresh AND price hasn't moved much."""
    key = _cache_key(wallet_id, symbol)
    entry = _decision_cache.get(key)
    if not entry:
        return None

    cached_ts, cached_price, cached_decision = entry

    if _now_ts() - cached_ts > DECISION_CACHE_TTL:
        _decision_cache.pop(key, None)
        return None

    # Price-move invalidation (the bug v1 left unimplemented).
    if cached_price > 0:
        move = abs(current_price - cached_price) / cached_price
        if move > PRICE_CHANGE_THRESHOLD:
            _decision_cache.pop(key, None)
            logger.debug(
                "[CACHE_INVALIDATE] %s: price moved %.2f%% (> %.2f%%)",
                symbol, move * 100, PRICE_CHANGE_THRESHOLD * 100,
            )
            return None

    logger.debug("[CACHE_HIT] %s (wallet %s): %ds old", symbol, wallet_id, int(_now_ts() - cached_ts))
    return cached_decision


def _cache_decision(wallet_id: int, symbol: str, price: float, decision: "TradeDecision") -> None:
    _decision_cache[_cache_key(wallet_id, symbol)] = (_now_ts(), float(price), decision)
    if len(_decision_cache) > _DECISION_CACHE_MAX:
        oldest = sorted(_decision_cache.keys(), key=lambda k: _decision_cache[k][0])[:100]
        for k in oldest:
            del _decision_cache[k]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_BASE = """You are AlphaPilot, an autonomous trading copilot operating in PAPER mode.

Your job is to convert a technical signal + market context into a single tradeable \
decision. You are NOT a passive analyst — the operator wants you to trade actively \
when the technical engine surfaces a directional signal so they can observe fills, \
reflect on outcomes, and improve. Refusing to trade on every borderline signal \
produces zero learning and is the WORST possible outcome.

DECISION POLICY
  - When technical_side is BUY or SELL and technical_confidence meets or exceeds \
    operator_calibration.min_confidence_floor, you MUST return that direction. \
    Override to HOLD only if extra_context (or risk_flags) contains a concrete, \
    explicitly-stated risk (e.g. "kill_switch_engaged: true", "duplicate_position: \
    true"). Vague worries about "weak signal" or "single indicator" are NOT grounds \
    to override — the operator's floor IS the calibration.
  - You have NO memory of past trades except recent_history.last_10_closed_trades \
    and recent_history.symbol_recent_trades in the payload. If those are empty, you \
    have no history. Do NOT invent rules about "high-confidence trades that lost" or \
    "consecutive losses" — those are hallucinations.
  - Start from confidence_adjustments.adjusted_confidence, which already folds in any \
    external intelligence that was available.

USING THE CONTEXT (refine confidence/risk; do not veto trades that meet the floor)
  - advanced_indicators: RSI (<30 oversold, >70 overbought), MACD histogram (sign = \
    momentum direction), Bollinger %B (>1 above upper band, <0 below lower), ADX \
    (>25 trending), relative_volume (>1.5 confirms), velocity_*bar (is price moving \
    NOW in the signal's direction), cross_age_bars (fresh EMA crosses are stronger), \
    body_direction (recent candles closing strong/weak).
  - market_regime.regime: TRENDING_UP/DOWN favor momentum + trailing stops; RANGING \
    favors mean reversion at the extremes; VOLATILE → smaller size, wider stops.
  - recent_history.symbol_recent_trades: if this exact symbol+side has been losing, \
    LOWER confidence and tighten risk. This is your most direct feedback channel.
  - Optional blocks (derivatives_intelligence, fear_greed_index, social_sentiment, \
    multi_timeframe_analysis) may be null when their data source is unavailable; \
    ignore nulls, never fabricate values.

RISK SHAPING
  - Wider stop in high volatility, tighter in low; larger take-profit in trending \
    regimes, smaller in ranging. Reduce size_multiplier when volume is unconfirmed \
    or warnings are present.

HARD RULES (the only firm vetoes)
  1. NEVER size_multiplier > 1.0.
  2. stop_loss_pct REQUIRED on every BUY/SELL, in [0.005, 0.20].
  3. take_profit_pct REQUIRED on every BUY/SELL, in [0.005, 0.50].
  4. If a kill switch / spent loss budget is EXPLICITLY stated, return HOLD.
  5. Never invent indicators or history not present in the payload.

OUTPUT
  A single JSON object with EXACTLY these keys:
    action, confidence, size_multiplier, stop_loss_pct, take_profit_pct,
    rationale, key_factors, risk_flags
  No prose. No markdown. No code fences. JSON only.
"""

# Compact, NEUTRAL format-anchoring examples. They demonstrate the JSON shape and
# reasoning discipline only — they deliberately avoid real symbols and do not bias
# direction (one BUY, one risk-flag HOLD override).
_FEWSHOT_EXAMPLES = """FORMAT EXAMPLES (shape only — do not copy values):

Example A (clean directional follow-through):
{"action":"BUY","confidence":0.71,"size_multiplier":0.9,"stop_loss_pct":0.03,\
"take_profit_pct":0.07,"rationale":"Fresh EMA cross (cross_age 2b) with positive \
3-bar velocity and relative_volume 1.6x; regime TRENDING_UP supports holding.",\
"key_factors":["fresh_cross","vol_confirm","regime_trending_up"],"risk_flags":[]}

Example B (explicit risk overrides an otherwise-valid signal):
{"action":"HOLD","confidence":0.0,"size_multiplier":0.0,"stop_loss_pct":0.03,\
"take_profit_pct":0.06,"rationale":"Signal was BUY but extra_context states \
duplicate_position: true; do not stack.","key_factors":["duplicate_position"],\
"risk_flags":["duplicate_position"]}"""


# =============================================================================
# TradeDecision
# =============================================================================

@dataclass
class TradeDecision:
    """Final, clamped decision the bot acts on.

    Field contract is preserved from v1 (downstream consumers read action,
    confidence, size_multiplier, stop_loss_pct, take_profit_pct, rationale,
    key_factors, risk_flags, source, model). ``quality`` is NEW and additive:
    bot_engine already does ``getattr(decision, 'quality', 'B')`` for position
    sizing, so emitting a real grade makes sizing conviction-aware instead of
    treating every trade as a 'B'.
    """
    action: str = "HOLD"
    confidence: float = 0.0
    size_multiplier: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    rationale: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    source: str = "technical"          # claude / technical / cache / *_passthrough / ...
    model: str = ""
    raw_text: str = ""
    quality: str = "B"                 # A+/A/B/C/F conviction grade (additive)
    # Primary key of the persisted ClaudeDecision row this decision came from.
    # Threaded onto PaperTrade.claude_decision_id at open time so the close-side
    # learn hook can rebuild entry-time market context from market_snapshot.
    # None when the decision wasn't routed through _persist_decision (e.g.
    # synthesized in strategic_claude's autonomous-only path).
    claude_decision_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "size_multiplier": round(self.size_multiplier, 4),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "rationale": self.rationale,
            "key_factors": self.key_factors,
            "risk_flags": self.risk_flags,
            "source": self.source,
            "model": self.model,
            "quality": self.quality,
        }


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
# CONFIG KNOBS
# =============================================================================

def _read_calibration_knobs() -> tuple[float, bool]:
    """Read the operator's floor + training flag ONCE.

    We must NOT use ``or 0.55`` after float() — the string "0.0" is truthy but
    float 0.0 is falsy, which historically bumped an explicit 0.0 floor back up
    to 0.55 and broke the training-mode bypass.
    """
    floor = 0.55
    is_training = False
    try:
        from config.bot_config import get as cfg_get
        raw_floor = cfg_get("bot_min_confidence")
        if raw_floor is not None and str(raw_floor).strip() != "":
            floor = float(raw_floor)
        is_training = (cfg_get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass
    return floor, is_training


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def decide(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    extra_context: dict[str, Any] | None = None,
) -> TradeDecision:
    """Produce a trade decision for one (wallet, symbol) candidate.

    Always returns a TradeDecision (never raises). Falls back to the technical
    signal if Claude is not configured, errors out, or returns invalid output.
    Every call persists a ClaudeDecision row for auditability.
    """
    wallet_id = int(wallet["id"])
    fallback = _technical_fallback(technical_signal)

    floor, is_training = _read_calibration_knobs()
    side = (technical_signal.side or "HOLD").upper()
    tech_conf = float(technical_signal.confidence or 0.0)

    # Build the live market_state ONCE from the signal's own indicators (this is
    # the fix that lets the adaptive engine actually see RSI/MACD/etc. instead of
    # an empty extra_context).
    indicators = getattr(technical_signal, "indicators", {}) or {}
    market_state = _extract_market_state(indicators, extra_context or {})

    # ----- Adaptive learning enhancement ---------------------------------- #
    adaptive_rec = None
    adaptive_context: dict[str, Any] = {}
    try:
        adaptive_rec = analyze_signal(
            signal_direction=side,
            signal_confidence=tech_conf,
            strategy_name=strategy_type,
            market_state=market_state,
            symbol=symbol,
            wallet_id=wallet_id,
        )
        if adaptive_rec:
            adjusted_conf = max(0.0, min(1.0, tech_conf + adaptive_rec.confidence_adjustment))
            adaptive_context = {
                "adaptive_learning": {
                    "confidence_adjustment": adaptive_rec.confidence_adjustment,
                    "adjusted_confidence": adjusted_conf,
                    "pattern_matches": [p.name for p in adaptive_rec.matched_patterns],
                    "pattern_boost": adaptive_rec.pattern_confidence_boost,
                    "strategy_weight": adaptive_rec.strategy_weight,
                    "preferred_strategy": adaptive_rec.preferred_strategy,
                    "historical_success_rate": adaptive_rec.historical_success_rate,
                    "similar_trades_count": adaptive_rec.similar_past_trades,
                    "entry_timing": adaptive_rec.entry_timing,
                    "size_multiplier": adaptive_rec.size_multiplier,
                    "recommended_action": adaptive_rec.recommended_action,
                    "reasoning": adaptive_rec.reasoning[:3],
                    "warnings": adaptive_rec.warnings[:2],
                }
            }
            tech_conf = adjusted_conf  # bypass check uses the adjusted value
            logger.info(
                "[ADAPTIVE] %s: conf %.2f -> %.2f (adj=%+.2f) patterns=%s w=%.2f",
                symbol, float(technical_signal.confidence or 0.0), adjusted_conf,
                adaptive_rec.confidence_adjustment,
                [p.name for p in adaptive_rec.matched_patterns],
                adaptive_rec.strategy_weight,
            )
    except Exception as e:
        logger.warning("[ADAPTIVE] error: %s", e)

    # ----- Strong-signal / training passthrough --------------------------- #
    bypass_threshold = max(0.0, min(STRONG_PASSTHROUGH_FLOOR, floor)) if is_training else STRONG_PASSTHROUGH_FLOOR
    if side in {"BUY", "SELL"} and tech_conf >= bypass_threshold:
        size_mult, stop_pct, take_pct = 1.0, 0.05, 0.10
        key_factors = [f"strategy={technical_signal.strategy}", f"floor={bypass_threshold:.2f}"]
        risk_flags: list[str] = []
        if adaptive_rec:
            size_mult = adaptive_rec.size_multiplier
            stop_pct = 0.05 * adaptive_rec.stop_loss_multiplier
            take_pct = 0.10 * adaptive_rec.take_profit_multiplier
            risk_flags = list(adaptive_rec.warnings or [])
            if adaptive_rec.matched_patterns:
                key_factors.append(f"patterns={[p.name for p in adaptive_rec.matched_patterns[:3]]}")
            key_factors.append(f"strategy_weight={adaptive_rec.strategy_weight:.2f}")
            if adaptive_rec.historical_success_rate != 0.5:
                key_factors.append(f"hist_success={adaptive_rec.historical_success_rate*100:.0f}%")

        passthrough = TradeDecision(
            action=side,
            confidence=tech_conf,
            size_multiplier=_clamp(size_mult, 0.0, 1.0),
            stop_loss_pct=_clamp(stop_pct, 0.005, 0.20),
            take_profit_pct=_clamp(take_pct, 0.005, 0.50),
            rationale=(
                (f"Training-mode passthrough (conf {tech_conf:.2f} >= floor {bypass_threshold:.2f}). "
                 if is_training else
                 f"Technical passthrough (conf {tech_conf:.2f} >= {STRONG_PASSTHROUGH_FLOOR:.2f}). ")
                + (technical_signal.reasoning or "")
                + (f" | Adaptive: {', '.join(adaptive_rec.reasoning[:2])}"
                   if adaptive_rec and adaptive_rec.reasoning else "")
            ),
            key_factors=key_factors,
            risk_flags=risk_flags,
            source="training_passthrough" if is_training else "technical_strong",
        )
        passthrough.quality = _grade_from_confidence(passthrough.confidence)
        _persist_decision(wallet, symbol, price, technical_signal, passthrough, prompt_used="", extra_context=extra_context)
        return passthrough

    # ----- HOLD signals don't need Claude --------------------------------- #
    if side == "HOLD":
        hold = TradeDecision(
            action="HOLD",
            confidence=tech_conf,
            size_multiplier=0.0,
            rationale=f"Technical signal is HOLD - no trade opportunity. {technical_signal.reasoning}",
            key_factors=["tech_hold", f"conf={tech_conf:.2f}"],
            source="tech_hold",
        )
        hold.quality = _grade_from_confidence(tech_conf)
        _persist_decision(wallet, symbol, price, technical_signal, hold, prompt_used="[TECH_HOLD]", extra_context=extra_context)
        return hold

    if not claude_is_configured():
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="", extra_context=extra_context)
        return fallback

    # ----- Claude path ---------------------------------------------------- #
    try:
        # Cache (wallet-scoped, price-aware).
        cached = _get_cached_decision(wallet_id, symbol, price)
        if cached is not None:
            cached_copy = TradeDecision(
                action=cached.action,
                confidence=cached.confidence,
                size_multiplier=cached.size_multiplier,
                stop_loss_pct=cached.stop_loss_pct,
                take_profit_pct=cached.take_profit_pct,
                rationale=f"[CACHED] {cached.rationale}",
                key_factors=cached.key_factors,
                risk_flags=cached.risk_flags,
                source="cache",
                quality=cached.quality,
            )
            logger.info("[CACHE_HIT] %s: reuse (%s, conf=%.2f)", symbol, cached.action, cached.confidence)
            _persist_decision(wallet, symbol, price, technical_signal, cached_copy, prompt_used="[CACHED]", extra_context=extra_context)
            return cached_copy

        can_call, remaining = _check_api_budget()
        if not can_call:
            logger.warning("[BUDGET_EXCEEDED] daily Claude budget spent; technical fallback for %s", symbol)
            budget_fb = TradeDecision(
                action=fallback.action,
                confidence=fallback.confidence * 0.9,
                size_multiplier=fallback.size_multiplier,
                stop_loss_pct=fallback.stop_loss_pct,
                take_profit_pct=fallback.take_profit_pct,
                rationale=f"[BUDGET_LIMIT] Technical signal only - daily API budget exhausted. {fallback.rationale}",
                key_factors=fallback.key_factors,
                risk_flags=["api_budget_exhausted"],
                source="budget_fallback",
            )
            budget_fb.quality = _grade_from_confidence(budget_fb.confidence)
            _persist_decision(wallet, symbol, price, technical_signal, budget_fb, prompt_used="[BUDGET_EXCEEDED]", extra_context=extra_context)
            return budget_fb

        if remaining <= 5 or remaining % 10 == 0:
            logger.info("[API_BUDGET] %d Claude calls remaining today", remaining)

        enhanced_context = dict(extra_context or {})
        enhanced_context.update(adaptive_context)

        system_prompt = _build_system_prompt(wallet)
        prompt = _build_user_prompt(
            wallet=wallet,
            symbol=symbol,
            price=price,
            technical_signal=technical_signal,
            strategy_type=strategy_type,
            indicators=indicators,
            extra_context=enhanced_context,
            floor=floor,
            is_training=is_training,
        )

        started = _now_ts()
        result = claude_chat(
            prompt=prompt,
            system=system_prompt,
            max_tokens=DECISION_MAX_TOKENS,
            temperature=DECISION_TEMPERATURE,
        )
        elapsed = _now_ts() - started
        _increment_api_calls()

        if not result.get("ok"):
            logger.warning("Claude decision call failed: %s", result.get("error"))
            _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used=prompt, extra_context=enhanced_context)
            return fallback

        text = result.get("text", "")
        parsed = _parse_decision_json(text)
        if parsed is None:
            logger.warning("Claude returned non-JSON; falling back. text=%r", text[:200])
            _persist_decision(
                wallet, symbol, price, technical_signal, fallback,
                prompt_used=prompt, raw_text=text, source_override="fallback",
                extra_context=enhanced_context,
            )
            return fallback

        decision = _normalize_and_clamp(parsed, wallet)
        decision.source = "claude"
        decision.model = (result.get("raw", {}) or {}).get("model", "") or ""
        decision.raw_text = text

        # Soft SLA: flag a slow response (we still use it — we already paid).
        if elapsed > DECISION_SLA_SECONDS:
            decision.risk_flags = (decision.risk_flags or []) + [f"slow_response_{elapsed:.1f}s"]
            logger.warning("[SLA] Claude took %.1fs for %s (> %.1fs)", elapsed, symbol, DECISION_SLA_SECONDS)

        # Confidence calibration (anti-overconfidence) for directional trades.
        if decision.action in {"BUY", "SELL"}:
            factor = _confidence_calibration_factor(wallet_id)
            if factor != 1.0:
                before = decision.confidence
                decision.confidence = _clamp(decision.confidence * factor, 0.0, 1.0)
                decision.key_factors = (decision.key_factors or []) + [f"calib×{factor:.2f}"]
                logger.info("[CALIBRATION] %s: conf %.2f -> %.2f (×%.2f)", symbol, before, decision.confidence, factor)

        # Conviction grade for position sizing.
        decision.quality = _grade_from_confidence(decision.confidence)

        # Optional high-stakes second opinion (off by default).
        if ENABLE_SECOND_OPINION:
            decision = _maybe_second_opinion(
                decision=decision, wallet=wallet, symbol=symbol, price=price,
                technical_signal=technical_signal, strategy_type=strategy_type,
                base_prompt=prompt, system_prompt=system_prompt,
            )

        _cache_decision(wallet_id, symbol, price, decision)
        _persist_decision(wallet, symbol, price, technical_signal, decision, prompt_used=prompt, raw_text=text, extra_context=enhanced_context)
        return decision

    except Exception as e:
        logger.exception("Claude decision engine raised: %s", e)
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="", raw_text=str(e), extra_context=extra_context)
        return fallback


# =============================================================================
# SECOND OPINION (optional)
# =============================================================================

def _is_high_stakes(decision: TradeDecision) -> bool:
    if decision.action not in {"BUY", "SELL"}:
        return False
    return (
        decision.size_multiplier >= SECOND_OPINION_SIZE_THRESHOLD
        or decision.confidence >= SECOND_OPINION_CONF_THRESHOLD
        or bool(decision.risk_flags)
    )


def _maybe_second_opinion(
    *,
    decision: TradeDecision,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    base_prompt: str,
    system_prompt: str,
) -> TradeDecision:
    """Re-check a high-stakes directional decision; disagreement => HOLD."""
    if not _is_high_stakes(decision):
        return decision
    can_call, _ = _check_api_budget()
    if not can_call:
        return decision

    challenge = (
        base_prompt
        + "\n\nSECOND PASS — act as a skeptical risk reviewer. The first pass chose "
        + f"{decision.action} at confidence {decision.confidence:.2f}. Argue the strongest "
        + "case AGAINST this trade given the same payload, then return your honest JSON "
        + "decision. If the case against is weak, confirm the original direction."
    )
    try:
        result = claude_chat(prompt=challenge, system=system_prompt,
                             max_tokens=DECISION_MAX_TOKENS, temperature=DECISION_TEMPERATURE)
        _increment_api_calls()
        if not result.get("ok"):
            return decision
        parsed = _parse_decision_json(result.get("text", ""))
        if parsed is None:
            return decision
        second = _normalize_and_clamp(parsed, wallet)
    except Exception as e:
        logger.warning("[SECOND_OPINION] failed: %s", e)
        return decision

    if second.action != decision.action:
        logger.info("[SECOND_OPINION] disagreement (%s vs %s) -> HOLD %s",
                    decision.action, second.action, symbol)
        decision.action = "HOLD"
        decision.size_multiplier = 0.0
        decision.confidence = min(decision.confidence, second.confidence)
        decision.risk_flags = (decision.risk_flags or []) + ["second_opinion_disagreement"]
        decision.rationale = f"[2ND-OPINION HOLD] reviewer disagreed. {decision.rationale}"
    else:
        # Agreement: small, bounded confidence reinforcement.
        decision.confidence = _clamp(decision.confidence + 0.03, 0.0, 1.0)
        decision.key_factors = (decision.key_factors or []) + ["second_opinion_confirmed"]
    decision.quality = _grade_from_confidence(decision.confidence)
    return decision


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def _build_system_prompt(wallet: dict[str, Any]) -> str:
    """Base rules + wallet risk profile + (optional) learned playbook + few-shot.

    The playbook is suppressed unless backed by at least one REAL closed-trade
    reflection, so seed rules don't get parroted back as if learned on a fresh DB.
    """
    try:
        from database.models import TradeReflection
        with session_scope() as s:
            real_reflections = s.query(TradeReflection).count()
    except Exception:
        real_reflections = 0
    playbook = build_playbook(limit=25) if real_reflections > 0 else []

    risk_block = "WALLET RISK PROFILE:\n" + "\n".join([
        f"  - wallet_name: {wallet.get('name')}",
        f"  - platform: {wallet.get('platform')}",
        f"  - trading_mode: {wallet.get('trading_mode', 'paper')}",
        f"  - max_position_usd: {wallet.get('max_position_usd', 0)}",
        f"  - max_open_positions: {wallet.get('max_open_positions', 0)}",
        f"  - max_leverage: {wallet.get('max_leverage', 1.0)}",
        f"  - futures_enabled: {wallet.get('futures_enabled', False)}",
    ])

    playbook_block = ""
    if playbook:
        playbook_block = (
            "LEARNED PLAYBOOK (earned from REAL closed-trade reflections):\n"
            + "\n".join(f"  - {p}" for p in playbook)
            + "\n\nApply as priors, not overrides — the operator's floor is still the threshold."
        )

    parts = [SYSTEM_PROMPT_BASE, risk_block]
    if playbook_block:
        parts.append(playbook_block)
    parts.append(_FEWSHOT_EXAMPLES)
    return "\n\n".join(parts).strip()


# Master legend keyed by the indicator names strategy_engine ACTUALLY emits.
# (v1's legend documented ema_fast/ret_6/ret_24/atr_pct — keys that never exist.)
_INDICATOR_LEGEND = {
    "ema_fast": "fast EMA of close (short-term trend)",
    "ema_slow": "slow EMA of close (medium-term trend)",
    "gap_pct": "normalized EMA fast-slow gap (+ = bullish stack)",
    "return_lb": "lookback return over the strategy window",
    "rsi": "Relative Strength Index (<30 oversold, >70 overbought)",
    "rsi_fast": "short-period RSI used by scalping",
    "macd_histogram": "MACD histogram (+ bullish momentum, - bearish)",
    "relative_volume": "current volume / rolling average (>1.5 confirms)",
    "buying_pressure": "fraction of recent volume on up-closes (0..1)",
    "velocity_1bar": "1-bar % price change (is it moving NOW)",
    "velocity_2bar": "2-bar % price change",
    "velocity_3bar": "3-bar % price change (freshness gate)",
    "cross_age_bars": "bars since the EMA fast/slow cross (lower = fresher)",
    "body_direction": "recent candle body bias (+ closing strong, - weak)",
    "sma": "simple moving average (mean-reversion anchor)",
    "stdev": "rolling standard deviation",
    "z": "z-score of close vs SMA (mean-reversion stretch)",
    "vol_pct": "realized volatility as % of price",
    "bb_percent_b": "Bollinger %B (>1 above upper band, <0 below lower)",
    "bb_upper": "Bollinger upper band price",
    "bb_lower": "Bollinger lower band price",
    "adx": "ADX trend strength (>25 trending)",
    "plus_di": "+DI directional index",
    "minus_di": "-DI directional index",
    "recent_high": "recent range high (breakout reference)",
    "recent_low": "recent range low (breakdown reference)",
    "current_price": "most recent close used by breakout",
    "atr": "Average True Range (volatility, price units)",
}


def _build_user_prompt(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    indicators: dict[str, Any],
    extra_context: dict[str, Any],
    floor: float,
    is_training: bool,
) -> str:
    """Compact, machine-readable context. Keep lean — every token costs."""

    # Indicators (THE fix): read straight off the Signal the router passed.
    ind_rounded = {k: _round_or_str(v) for k, v in (indicators or {}).items()}
    # Legend limited to the keys actually present, so it never lies.
    legend = {k: _INDICATOR_LEGEND[k] for k in ind_rounded if k in _INDICATOR_LEGEND}

    open_positions, recent_trades = _wallet_recent_history(int(wallet["id"]))
    symbol_recent = _symbol_recent_history(int(wallet["id"]), symbol)

    # Derived regime hint — always present, no dependency on the dead engine.
    regime_hint = _derive_regime_hint(indicators)

    # advanced_indicators is now POPULATED (v1 sent None) from the live signal.
    advanced_indicators = {
        "rsi_14": indicators.get("rsi", indicators.get("rsi_fast")),
        "macd_histogram": indicators.get("macd_histogram"),
        "bollinger_percent_b": indicators.get("bb_percent_b"),
        "adx_trend_strength": indicators.get("adx"),
        "relative_volume": indicators.get("relative_volume"),
        "velocity_3bar": indicators.get("velocity_3bar"),
        "cross_age_bars": indicators.get("cross_age_bars"),
        "body_direction": indicators.get("body_direction"),
    }
    advanced_indicators = {k: _round_or_str(v) for k, v in advanced_indicators.items() if v is not None}

    # Optional external intelligence (best-effort, cached, never fatal).
    fear_greed_context = _fetch_fear_greed()
    derivatives_context = _fetch_derivatives(symbol)
    mtf_context = _fetch_mtf(symbol, float(technical_signal.confidence or 0))
    social_context = _fetch_social(symbol)

    # Aggregate external confidence adjustment.
    adjustments: list[tuple[str, float]] = []
    if mtf_context and mtf_context.get("confidence_boost"):
        adjustments.append(("MTF", mtf_context["confidence_boost"]))
    if derivatives_context and derivatives_context.get("confidence_adjustment"):
        adjustments.append(("Derivatives", derivatives_context["confidence_adjustment"]))
    if fear_greed_context and fear_greed_context.get("confidence_adjustment"):
        adjustments.append(("Fear&Greed", fear_greed_context["confidence_adjustment"]))
    total_adj = max(-0.25, min(0.25, sum(a for _, a in adjustments)))
    base_conf = float(technical_signal.confidence or 0)

    payload = {
        "operator_calibration": {
            "min_confidence_floor": round(floor, 4),
            "is_training_session": is_training,
            "instruction": (
                f"Trade in technical_side direction when technical_confidence >= {floor:.2f} "
                "unless extra_context contains a concrete contradiction."
            ),
        },
        "candidate": {
            "symbol": symbol,
            "price": round(price, 8),
            "strategy_type": strategy_type,
            "technical_side": technical_signal.side,
            "technical_confidence": round(base_conf, 4),
            "technical_reasoning": technical_signal.reasoning,
            "indicators": ind_rounded,
            "indicators_legend": legend,
        },
        "market_regime": {
            "regime": regime_hint["regime"],
            "basis": regime_hint["basis"],
            "note": "Heuristic regime derived from this signal's indicators (no external regime feed).",
        },
        "advanced_indicators": advanced_indicators or None,
        "wallet_state": {
            "paper_balance": round(float(wallet.get("paper_balance", 0)), 2),
            "open_position_count": len(open_positions),
            "open_positions": open_positions,
        },
        "recent_history": {
            "last_10_closed_trades": recent_trades,
            "symbol_recent_trades": symbol_recent,
            "note": (
                "Empty lists mean no closed history yet — normal at the start of a "
                "training session; do NOT treat emptiness as a reason to refuse trading."
            ) if not recent_trades and not symbol_recent else "",
        },
        "derivatives_intelligence": derivatives_context,
        "fear_greed_index": fear_greed_context,
        "social_sentiment": social_context,
        "multi_timeframe_analysis": mtf_context,
        "confidence_adjustments": {
            "sources": adjustments,
            "total_adjustment": round(total_adj, 3),
            "adjusted_confidence": round(min(1.0, max(0.0, base_conf + total_adj)), 4),
            "note": "Sum of external boosts/penalties. Apply to your final confidence.",
        },
        "extra_context": extra_context,
        "now_utc": utcnow().isoformat(),
    }

    return (
        "Decide the next action for this candidate. "
        "Return ONLY the JSON object specified in your instructions.\n\n"
        f"{json.dumps(payload, default=str)}"
    )


# =============================================================================
# HISTORY HELPERS
# =============================================================================

def _wallet_recent_history(wallet_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compact wallet-wide recent-history snapshot for prompt context."""
    with session_scope() as s:
        opens = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "open")
            .order_by(PaperTrade.opened_at.desc()).limit(10).all()
        )
        closed = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc()).limit(10).all()
        )
        open_positions = [
            {
                "symbol": t.symbol, "side": t.side, "qty": float(t.qty),
                "entry": float(t.entry_price), "unrealized_pnl": float(t.unrealized_pnl or 0),
                "age_min": _minutes_since(t.opened_at),
            } for t in opens
        ]
        recent_closed = [
            {
                "symbol": t.symbol, "side": t.side, "pnl": float(t.realized_pnl or 0),
                "confidence": float(t.confidence or 0), "held_min": _minutes_between(t.opened_at, t.closed_at),
            } for t in closed
        ]
    return open_positions, recent_closed


def _symbol_recent_history(wallet_id: int, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    """Recent closed trades for THIS exact symbol — the direct feedback channel
    that lets Claude self-correct ("we lost on this symbol 4 of the last 5")."""
    try:
        with session_scope() as s:
            rows = (
                s.query(PaperTrade)
                .filter(
                    PaperTrade.wallet_id == wallet_id,
                    PaperTrade.symbol == symbol,
                    PaperTrade.status == "closed",
                )
                .order_by(PaperTrade.closed_at.desc()).limit(limit).all()
            )
            out = []
            for t in rows:
                pnl = float(t.realized_pnl or 0)
                out.append({
                    "side": t.side,
                    "pnl": round(pnl, 4),
                    "outcome": "WIN" if pnl > 0 else "LOSS",
                    "confidence": round(float(t.confidence or 0), 3),
                    "held_min": _minutes_between(t.opened_at, t.closed_at),
                    "exit_reason": getattr(t, "exit_reason", None),
                })
            return out
    except Exception as e:
        logger.debug("[SYMBOL_HISTORY] %s: %s", symbol, e)
        return []


# =============================================================================
# MARKET STATE + REGIME
# =============================================================================

def _extract_market_state(indicators: dict, extra_context: dict) -> dict:
    """Build the adaptive engine's market_state from the LIVE signal indicators
    first, then fill any gaps from extra_context. (v1 read only extra_context,
    which is empty in the live flow, so the adaptive engine saw all defaults.)"""
    ind = indicators or {}
    ec_adv = (extra_context or {}).get("advanced_indicators", {}) or {}

    def pick(*keys, default=None):
        for src in (ind, ec_adv):
            for k in keys:
                if k in src and src[k] is not None:
                    return src[k]
        return default

    state = {
        "rsi": pick("rsi", "rsi_fast", "rsi_14", default=50),
        "macd_histogram": pick("macd_histogram", default=0),
        "bb_percent_b": pick("bb_percent_b", "bollinger_percent_b", default=0.5),
        "adx": pick("adx", "adx_trend_strength", default=0),
        "volume_ratio": pick("relative_volume", "volume_ratio", default=1.0),
        "trend": pick("trend_direction", "trend", default="NEUTRAL"),
        "momentum": pick("momentum_signal", "momentum", default="NEUTRAL"),
        "volatility_state": pick("volatility_state", default="NORMAL"),
        "trend_strength": pick("adx", "adx_trend_strength", default=0),
        "volatility_percentile": pick("volatility_percentile", default=50),
    }

    regime = (extra_context or {}).get("market_regime", {})
    state["regime"] = regime.get("regime") if isinstance(regime, dict) else None
    if not state["regime"]:
        state["regime"] = _derive_regime_hint(ind)["regime"]

    fg = (extra_context or {}).get("fear_greed_index", {}) or {}
    state["fear_greed"] = fg.get("value", 50)

    mtf = (extra_context or {}).get("multi_timeframe_analysis", {}) or {}
    state["mtf_alignment"] = mtf.get("alignment_score", 0.5)
    state["mtf_bias"] = mtf.get("overall_bias", "NEUTRAL")

    deriv = (extra_context or {}).get("derivatives_intelligence", {}) or {}
    state["funding_rate"] = deriv.get("funding_rate_pct", 0)
    state["long_ratio"] = deriv.get("long_ratio", 50)
    return state


def _derive_regime_hint(indicators: dict) -> dict:
    """Coarse regime label from whatever indicators the strategy provided.

    This replaces the always-None market_regime block from v1 with a real (if
    heuristic) signal Claude can reason over. Uses ADX/DI when available, else
    falls back to velocity + EMA gap for direction and vol_pct/atr for volatility.
    """
    ind = indicators or {}
    adx = _to_float(ind.get("adx"), None)
    plus_di = _to_float(ind.get("plus_di"), None)
    minus_di = _to_float(ind.get("minus_di"), None)
    vol_pct = _to_float(ind.get("vol_pct"), None)
    gap = _to_float(ind.get("gap_pct"), 0.0)
    vel3 = _to_float(ind.get("velocity_3bar"), 0.0)

    # Trending (needs trend-strength evidence).
    if adx is not None and adx >= 25 and plus_di is not None and minus_di is not None:
        if plus_di > minus_di:
            return {"regime": "TRENDING_UP", "basis": f"ADX {adx:.0f}, +DI>-DI"}
        return {"regime": "TRENDING_DOWN", "basis": f"ADX {adx:.0f}, -DI>+DI"}

    # Volatile.
    if vol_pct is not None and vol_pct >= 0.04:
        return {"regime": "VOLATILE", "basis": f"vol {vol_pct:.1%}"}

    # Weak/ranging when trend strength is explicitly low.
    if adx is not None and adx < 18:
        return {"regime": "RANGING", "basis": f"ADX {adx:.0f} (weak)"}

    # Direction-only fallback from velocity / EMA gap.
    direction = gap + vel3
    if direction > 0.002:
        return {"regime": "DRIFT_UP", "basis": f"gap+vel {direction:+.2%}"}
    if direction < -0.002:
        return {"regime": "DRIFT_DOWN", "basis": f"gap+vel {direction:+.2%}"}
    return {"regime": "UNKNOWN", "basis": "insufficient indicators"}


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

def _confidence_calibration_factor(wallet_id: int) -> float:
    """Multiplier that damps systematic overconfidence (or gently boosts under-
    confidence) based on this wallet's predicted-vs-realized track record.

    Compares the mean confidence stamped on recent closed trades against the
    realized win rate. ratio = win_rate / mean_confidence, clamped to a safe band.
    Cached per wallet for CALIBRATION_TTL seconds.
    """
    cached = _calibration_cache.get(wallet_id)
    if cached is not None:
        return cached

    factor = 1.0
    try:
        with session_scope() as s:
            rows = (
                s.query(PaperTrade.confidence, PaperTrade.realized_pnl)
                .filter(
                    PaperTrade.wallet_id == wallet_id,
                    PaperTrade.status == "closed",
                    PaperTrade.confidence.isnot(None),
                )
                .order_by(PaperTrade.closed_at.desc()).limit(60).all()
            )
        confs = [float(c) for c, _ in rows if c is not None and float(c) > 0]
        if len(confs) >= CALIBRATION_MIN_SAMPLE:
            mean_pred = sum(confs) / len(confs)
            wins = sum(1 for _, p in rows if (p or 0) > 0)
            win_rate = wins / len(rows) if rows else 0.5
            if mean_pred > 0:
                factor = _clamp(win_rate / mean_pred, CALIBRATION_FLOOR, CALIBRATION_CEIL)
    except Exception as e:
        logger.debug("[CALIBRATION] unavailable: %s", e)
        factor = 1.0

    _calibration_cache.set(wallet_id, factor)
    return factor


# =============================================================================
# OPTIONAL EXTERNAL INTELLIGENCE  (best-effort, cached, never fatal)
# =============================================================================

def _fetch_fear_greed() -> dict | None:
    cached = _global_md_cache.get("fear_greed")
    if cached is not None:
        return cached or None
    out = None
    try:
        from connectors.fear_greed import get_fear_greed_signal
        fg = get_fear_greed_signal()
        if fg and fg.get("available"):
            out = {
                "value": fg.get("value"),
                "classification": fg.get("classification"),
                "sentiment": fg.get("sentiment"),
                "signal": fg.get("signal"),
                "confidence_adjustment": fg.get("confidence_adjustment", 0),
                "summary": fg.get("summary", ""),
            }
    except Exception:
        out = None
    _global_md_cache.set("fear_greed", out or {})
    return out


def _fetch_derivatives(symbol: str) -> dict | None:
    cached = _global_md_cache.get(("deriv", symbol))
    if cached is not None:
        return cached or None
    out = None
    try:
        from connectors.coinglass import get_funding_signal
        d = get_funding_signal(symbol)
        if d:
            out = {
                "overall_signal": d.get("overall_signal", "NEUTRAL"),
                "confidence_adjustment": d.get("confidence_adjustment", 0),
                "funding_rate_pct": d.get("funding_rate"),
                "long_ratio": d.get("long_ratio"),
                "summary": d.get("summary", ""),
            }
    except Exception:
        out = None
    _global_md_cache.set(("deriv", symbol), out or {})
    return out


def _fetch_mtf(symbol: str, confidence: float) -> dict | None:
    if confidence < 0.50:   # MTF is the expensive one; only for plausible signals
        return None
    cached = _global_md_cache.get(("mtf", symbol))
    if cached is not None:
        return cached or None
    out = None
    try:
        from trading.multi_timeframe import get_mtf_signal_boost
        m = get_mtf_signal_boost(symbol)
        if m and m.get("bias") != "UNKNOWN":
            out = {
                "overall_bias": m["bias"],
                "alignment_score": round(m.get("alignment", 0), 2),
                "confidence_boost": m.get("boost", 0),
                "summary": m.get("summary", ""),
            }
    except Exception:
        out = None
    _global_md_cache.set(("mtf", symbol), out or {})
    return out


def _fetch_social(symbol: str) -> dict | None:
    try:
        from connectors.lunarcrush import get_social_metrics
        m = get_social_metrics(symbol)
        if not m:
            return None
        ctx = {
            "galaxy_score": m.galaxy_score,
            "alt_rank": m.alt_rank,
            "sentiment": ("BULLISH" if m.sentiment_score > 0.2
                          else "BEARISH" if m.sentiment_score < -0.2 else "NEUTRAL"),
            "sentiment_score": round(m.sentiment_score, 3),
            "social_volume_change_24h": f"{m.social_volume_change_24h:+.1f}%",
            "alerts": [],
        }
        if m.is_buzzing:
            ctx["alerts"].append("HIGH_BUZZ")
        if m.is_fading:
            ctx["alerts"].append("FADING_INTEREST")
        if m.has_influencer_pump:
            ctx["alerts"].append("INFLUENCER_PUMP")
        return ctx
    except Exception:
        return None


# =============================================================================
# PARSING + CLAMPING
# =============================================================================

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Balanced-brace extraction (handles nested objects + braces inside strings,
    which the v1 greedy ``\\{.*\\}`` regex could mangle)."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _parse_decision_json(text: str) -> dict[str, Any] | None:
    """Tolerantly extract a JSON object from Claude's response."""
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    block = _extract_first_json_object(cleaned)
    if not block:
        return None
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_and_clamp(parsed: dict[str, Any], wallet: dict[str, Any]) -> TradeDecision:
    """Coerce Claude's JSON into a safe TradeDecision (parsing/clamping only)."""
    action = str(parsed.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "SELL", "HOLD", "CLOSE"}:
        action = "HOLD"

    conf = _clamp(_to_float(parsed.get("confidence"), 0.0), 0.0, 1.0)
    size_mult = _clamp(_to_float(parsed.get("size_multiplier"), 1.0), 0.0, 1.0)
    sl = _clamp(_to_float(parsed.get("stop_loss_pct"), 0.05), 0.005, 0.20)
    tp = _clamp(_to_float(parsed.get("take_profit_pct"), 0.10), 0.005, 0.50)

    factors = parsed.get("key_factors") or []
    if not isinstance(factors, list):
        factors = [str(factors)]
    factors = [str(x)[:200] for x in factors][:8]

    flags = parsed.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(x)[:200] for x in flags][:8]

    rationale = str(parsed.get("rationale", ""))[:1500]

    d = TradeDecision(
        action=action, confidence=conf, size_multiplier=size_mult,
        stop_loss_pct=sl, take_profit_pct=tp, rationale=rationale,
        key_factors=factors, risk_flags=flags,
    )
    d.quality = _grade_from_confidence(conf)
    return d


def _technical_fallback(signal: Signal) -> TradeDecision:
    side = (signal.side or "HOLD").upper()
    if side not in {"BUY", "SELL", "HOLD"}:
        side = "HOLD"
    d = TradeDecision(
        action=side,
        confidence=float(signal.confidence or 0.0),
        size_multiplier=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        rationale=signal.reasoning or "Fallback: technical signal only.",
        key_factors=[],
        risk_flags=["claude_unavailable"],
        source="technical",
    )
    d.quality = _grade_from_confidence(d.confidence)
    return d


# =============================================================================
# PERSISTENCE
# =============================================================================

def _persist_decision(
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    decision: TradeDecision,
    *,
    prompt_used: str = "",
    raw_text: str = "",
    source_override: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> None:
    """Write a ClaudeDecision row. Captures the entry-time indicator/regime
    snapshot into market_snapshot so the autonomous learning engine can rebuild
    a populated TradeContext at close time. Never lets logging fail a trade."""
    try:
        snapshot = _extract_market_state(
            getattr(technical_signal, "metadata", {}) or {},
            extra_context or {},
        )
        new_id: int | None = None
        with session_scope() as s:
            row = ClaudeDecision(
                wallet_id=int(wallet["id"]),
                symbol=symbol,
                price=float(price),
                technical_side=technical_signal.side,
                technical_confidence=float(technical_signal.confidence or 0),
                action=decision.action,
                confidence=float(decision.confidence),
                size_multiplier=float(decision.size_multiplier),
                stop_loss_pct=float(decision.stop_loss_pct),
                take_profit_pct=float(decision.take_profit_pct),
                rationale=decision.rationale or "",
                key_factors=json.dumps(decision.key_factors)[:4000],
                risk_flags=json.dumps(decision.risk_flags)[:2000],
                source=source_override or decision.source,
                model=decision.model or "",
                prompt_used=(prompt_used or "")[:8000],
                raw_response=(raw_text or decision.raw_text or "")[:8000],
                market_snapshot=json.dumps(snapshot)[:4000],
            )
            s.add(row)
            # Flush to allocate the autoincrement id while the row is still
            # attached to the session. We capture the id locally and only
            # publish it onto the decision after the surrounding session_scope
            # commits — otherwise a rollback would leave decision pointing at
            # a row that never landed.
            s.flush()
            new_id = int(row.id)
        if new_id is not None:
            decision.claude_decision_id = new_id
    except Exception:
        logger.exception("Failed to persist ClaudeDecision row.")


# =============================================================================
# TINY HELPERS
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(v: Any, default):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _round_or_str(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str)) or v is None:
        return v
    return str(v)


def _minutes_since(dt: Any) -> int:
    if not dt:
        return 0
    try:
        return int((utcnow() - dt).total_seconds() // 60)
    except Exception:
        return 0


def _minutes_between(a: Any, b: Any) -> int:
    if not a or not b:
        return 0
    try:
        return int((b - a).total_seconds() // 60)
    except Exception:
        return 0


# =============================================================================
# CHANGELOG (v1 -> v2)
# =============================================================================
# BUG FIXES
#   1. metadata->indicators: _build_user_prompt now reads
#      getattr(technical_signal, "indicators", {}); v1 read a non-existent
#      ".metadata" and shipped an empty indicators block on every call.
#   2. Indicator legend rewritten to the keys strategy_engine actually emits
#      (gap_pct/return_lb/rsi/macd_histogram/velocity_*bar/cross_age_bars/...).
#   3. Decision cache is price-aware (stores price, invalidates on >2% move) and
#      wallet-scoped (key is (wallet_id, symbol)); v1 ignored price and bled
#      decisions across wallets. Docstrings now match real behavior (30-min TTL).
#   4. Removed dead/unused imports (get_adaptive_engine, Wallet).
#   5. Global market-data caching replaced the fragile mid-function `global`
#      hacks with a small _TTLCache.
#
# CAPABILITY UPGRADES
#   6. Per-symbol recent-trade history injected into the prompt
#      (recent_history.symbol_recent_trades) — direct self-correction channel.
#   7. Confidence calibration loop: damps systematic overconfidence using this
#      wallet's predicted-vs-realized win rate (clamped 0.60..1.15).
#   8. Derived market-regime token always present (no dependency on the dead
#      advanced_signal_engine).
#   9. Adaptive engine's market_state now built from the LIVE signal indicators
#      instead of an empty extra_context — its pattern layer can finally see data.
#  10. advanced_indicators block is populated (was always None in v1).
#  11. Conviction grade (TradeDecision.quality) emitted so bot_engine's position
#      sizer stops treating every trade as a 'B'. (Additive field; default 'B'.)
#  12. Few-shot format-anchoring examples in the system prompt (neutral, no
#      directional bias).
#  13. Robust JSON parsing (code-fence stripping + balanced-brace extraction).
#  14. Soft response-time SLA: slow Claude responses get a risk_flag.
#  15. Optional high-stakes second-opinion cross-check (ENABLE_SECOND_OPINION,
#      OFF by default so it never silently doubles spend).
#
# CONTRACT PRESERVED
#   - decide(*, wallet, symbol, price, technical_signal, strategy_type,
#     extra_context=None) -> TradeDecision  (unchanged)
#   - All TradeDecision fields consumers read are unchanged; `quality` is added.
#   - source labels unchanged ("claude" still means a real API call -> bot_engine's
#     claude_calls counter still works): technical / technical_strong /
#     training_passthrough / tech_hold / cache / budget_fallback / fallback / claude.
#   - _persist_decision writes the same ClaudeDecision columns as v1.
#
# DEFERRED (batch 2 — needs the other files)
#   - Reconciling THIS budget (25/day) with strategic_claude's budget (50/day).
#   - Populating ClaudeDecision.market_snapshot to feed autonomous_learning_engine's
#     _build_context_from_trade (only if that column exists — not touched here to
#     avoid a schema-mismatch crash).