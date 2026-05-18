"""
Claude-driven trade decision engine.

This module is the "brain" of AlphaPilot. For every candidate trade the bot
considers, we hand Claude a structured packet containing:

  - the technical signal from strategy_engine (momentum, mean reversion, etc.)
  - the live market snapshot (price, volatility proxy, recent returns)
  - the wallet's current open positions and recent realized P&L
  - the wallet's risk profile + caps (max position USD, leverage cap)
  - the current learned playbook of rules and mistakes (from AILearningMemory)
  - the global trading regime hints (kill switch, daily loss budget, etc.)

Claude returns a strict JSON object describing what it wants to do:

    {
      "action": "BUY" | "SELL" | "HOLD" | "CLOSE",
      "confidence": 0.0 - 1.0,
      "size_multiplier": 0.0 - 1.0,    # fraction of the bot's default size
      "stop_loss_pct": 0.0 - 0.20,
      "take_profit_pct": 0.0 - 0.50,
      "rationale": "short paragraph",
      "key_factors": ["..."],
      "risk_flags": ["..."]
    }

The engine NEVER blindly trusts Claude:
  - all sizing/leverage requests are clamped to wallet caps
  - if Claude fails, times out, or returns invalid JSON, we fall back to the
    raw technical signal (graceful degradation — trading never stops because
    the LLM hiccupped)
  - every decision (Claude's *and* the fallback) is persisted to ClaudeDecision
    so the Training Center can show audit history and so post-trade reflection
    can correlate fills with the reasoning that led to them.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ai.claude_learning import build_playbook
from ai.adaptive_learning_engine import get_adaptive_engine, analyze_signal
from database.db import session_scope
from database.models import ClaudeDecision, PaperTrade, Wallet
from services.claude_client import chat as claude_chat
from services.claude_client import is_configured as claude_is_configured
from trading.strategy_engine import Signal
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


# How long to wait for Claude before falling back to the technical signal.
# We keep this aggressive: the bot tick cadence is short, and a stale LLM
# response is worse than a fresh technical signal.
DECISION_MAX_TOKENS = 700
DECISION_TEMPERATURE = 0.2  # decisions should be near-deterministic


SYSTEM_PROMPT_BASE = """You are AlphaPilot, an autonomous trading copilot operating in PAPER mode.

Your job is to convert a technical signal + comprehensive market intelligence into a \
tradeable decision. You are NOT a passive analyst — the operator wants you to trade \
actively when the technical engine surfaces a directional signal, so they can observe \
the resulting fills, reflect on outcomes, and improve over time. Refusing to trade on \
every borderline signal produces zero learning and is the WORST possible outcome.

Decision policy:
  - When the technical signal direction is BUY or SELL and the technical \
    confidence meets or exceeds the operator's min_confidence_floor, you MUST \
    return that direction. Only override to HOLD if extra_context contains a \
    concrete, explicitly-stated risk_flag (e.g. "kill_switch_engaged: true" or \
    "duplicate_position: true"). Vague concerns about "weak signals" or \
    "single indicator" are NOT grounds to override — the operator's floor IS \
    the calibration.
  - You have NO prior memory of past trades except what is explicitly listed \
    under recent_history.last_10_closed_trades in the user payload. If that \
    list is empty, you have no history. Do not invent rules about "high-confidence \
    trades that lost money" or "consecutive losses" — these are hallucinations.
  - Use the confidence_adjustments.adjusted_confidence as your starting point, \
    which already factors in MTF alignment, derivatives sentiment, and Fear & Greed.

=== MARKET INTELLIGENCE INTEGRATION ===

You now receive comprehensive market intelligence. Use it to REFINE confidence and \
risk parameters, NOT to veto trades that meet the floor.

1. ADVANCED INDICATORS (advanced_indicators):
   - rsi_14: Relative Strength Index. <30 = oversold (buy), >70 = overbought (sell)
   - macd_histogram: Momentum. Positive = bullish, Negative = bearish
   - stoch_k/stoch_d: Stochastic. <20 = oversold, >80 = overbought
   - bollinger_percent_b: Price position in bands. >1 = above upper, <0 = below lower
   - adx_trend_strength: Trend strength. >25 = trending, <20 = ranging
   - relative_volume: Volume vs average. >1.5 = high volume confirms move
   - trend_direction/momentum_signal: Derived signals (BULLISH/BEARISH/NEUTRAL)
   - volatility_state: HIGH/NORMAL/LOW
   - volume_confirmation: True if volume supports the move

2. MULTI-TIMEFRAME ANALYSIS (multi_timeframe_analysis):
   - overall_bias: Aggregate trend across 5m, 15m, 1h, 4h, 1d
   - alignment_score: 0-1, how aligned timeframes are (>0.7 = strong)
   - entry_timing: NOW, WAIT_PULLBACK, WAIT_BREAKOUT, NO_TRADE
   - higher_tf_support: True if higher timeframes confirm direction
   - divergence_warning: True if lower TF diverging from higher (caution!)
   - Use confidence_boost to adjust confidence (already in adjusted_confidence)

3. MARKET REGIME (market_regime):
   - regime: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, ACCUMULATION, DISTRIBUTION
   - recommended_strategy: What strategy works best in this regime
   - position_size_multiplier: Suggested size adjustment for regime
   - trading_rules: Specific rules for this regime (bias, stop multipliers, etc.)
   - For VOLATILE regime: Reduce size, widen stops
   - For RANGING: Use mean reversion, trade at support/resistance
   - For TRENDING: Use momentum, trail stops

4. DERIVATIVES INTELLIGENCE (derivatives_intelligence):
   - funding_rate_pct: Perpetual funding rate. Very negative = shorts paying = bullish
   - long_ratio/short_ratio: Market positioning. Extreme = contrarian signal
   - overall_signal: BULLISH, BEARISH, NEUTRAL from derivatives data
   - warnings: Specific alerts like "Extreme shorts - potential squeeze"

5. FEAR & GREED INDEX (fear_greed_index):
   - value: 0-100 (0=extreme fear, 100=extreme greed)
   - sentiment: EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED
   - signal: STRONG_BUY (extreme fear), BUY, NEUTRAL, SELL, STRONG_SELL (extreme greed)
   - CONTRARIAN: Buy in fear, sell in greed
   - Extreme Fear (<25) = +0.15 confidence boost (buying opportunity)
   - Extreme Greed (>75) = -0.12 confidence penalty (caution)

6. SOCIAL SENTIMENT (social_sentiment):
   - galaxy_score: 0-100 social health
   - sentiment: BULLISH/NEUTRAL/BEARISH from social posts
   - alerts: HIGH_BUZZ, FADING_INTEREST, INFLUENCER_PUMP

=== HOW TO USE INTELLIGENCE ===

For BUY signals:
- Boost confidence if: MTF aligned bullish, derivatives bullish, fear/greed in fear
- Reduce confidence if: MTF divergence warning, extreme greed, low volume
- Adjust stop_loss: Wider in high volatility, tighter in low volatility
- Adjust take_profit: Larger in trending regime, smaller in ranging

For SELL signals:
- Boost confidence if: MTF aligned bearish, derivatives bearish, fear/greed in greed
- Similar adjustments apply

Position Sizing:
- Use market_regime.position_size_multiplier as a guide
- In VOLATILE regime: size_multiplier = 0.5-0.7
- In TRENDING regime with confirmation: size_multiplier = 1.0
- If volume_confirmation = False: reduce size_multiplier by 0.2

Hard rules (these are the only firm vetoes):
  1. NEVER recommend size_multiplier > 1.0 or leverage > the wallet's max_leverage.
  2. Stop-loss is REQUIRED on every BUY/SELL action (in [0.005, 0.20]).
  3. Take-profit is REQUIRED on every BUY/SELL action (in [0.005, 0.50]).
  4. If the kill switch is engaged or daily loss budget is spent (only when EXPLICITLY \
     stated in extra_context), return HOLD.
  5. Do not invent indicators or history that are not in the provided payload.

Your output MUST be a single JSON object with exactly these keys:
  action, confidence, size_multiplier, stop_loss_pct, take_profit_pct,
  rationale, key_factors, risk_flags

No prose. No markdown. No code fences. Just JSON.
"""


@dataclass
class TradeDecision:
    """Final, clamped decision the bot will act on."""
    action: str = "HOLD"           # BUY / SELL / HOLD / CLOSE
    confidence: float = 0.0        # 0..1
    size_multiplier: float = 1.0   # 0..1 (fraction of cfg.position_size_usd)
    stop_loss_pct: float = 0.02    # 2% stop loss (tighter for faster iteration)
    take_profit_pct: float = 0.03  # 3% take profit (lock in gains quickly)
    rationale: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    source: str = "technical"      # "claude" / "technical" / "fallback"
    model: str = ""
    raw_text: str = ""

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
        }


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #


def decide(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    extra_context: dict[str, Any] | None = None,
) -> TradeDecision:
    """
    Produce a trade decision for one (wallet, symbol) candidate.

    Always returns a TradeDecision (never raises). Falls back to the technical
    signal if Claude is not configured, errors out, or returns invalid output.
    Every call results in a persisted ClaudeDecision row for auditability.
    """
    fallback = _technical_fallback(technical_signal)

    # Read the operator's calibration knobs ONCE up front. The whole point of
    # the floor + training-mode flag is that the operator has explicitly told
    # the system "I want trades at this level of evidence". Claude has been
    # consistently overriding that floor (citing pre-existing seed rules about
    # "high-confidence trades losing money" and pulling every 0.51 signal to
    # 0.50 HOLD) — so when training mode is active, we treat the operator's
    # floor as authoritative and bypass Claude entirely on directional signals.
    floor = 0.55
    is_training = False
    try:
        from config.bot_config import get as cfg_get
        # cfg_get returns a string from the AppSetting table, or None if unset.
        # We must NOT use `or 0.55` here — the string "0.0" is truthy but the
        # FLOAT 0.0 is falsy, and earlier code paths that did `or 0.55` after
        # float() were silently bumping the user's floor of 0.0 back up to
        # 0.55, causing the training-mode bypass to never fire.
        raw_floor = cfg_get("bot_min_confidence")
        if raw_floor is not None and str(raw_floor).strip() != "":
            floor = float(raw_floor)
        is_training = (cfg_get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass
    side = (technical_signal.side or "HOLD").upper()
    tech_conf = float(technical_signal.confidence or 0.0)

    # ----- Adaptive Learning Enhancement -------------------------------------
    # Query the adaptive learning engine for pattern matches and historical
    # context. This provides confidence adjustments based on:
    # 1. Recognized market patterns with historical success rates
    # 2. Strategy performance in current market regime
    # 3. Similar historical trades for this symbol
    adaptive_rec = None
    adaptive_context = {}
    try:
        market_state = _extract_market_state(extra_context or {})
        adaptive_rec = analyze_signal(
            signal_direction=side,
            signal_confidence=tech_conf,
            strategy_name=strategy_type,
            market_state=market_state,
            symbol=symbol,
            wallet_id=wallet.get("id"),
        )
        
        # Apply confidence adjustment from adaptive learning
        if adaptive_rec:
            adjusted_conf = tech_conf + adaptive_rec.confidence_adjustment
            adjusted_conf = max(0.0, min(1.0, adjusted_conf))
            
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
                    "reasoning": adaptive_rec.reasoning[:3],  # Top 3 reasons
                    "warnings": adaptive_rec.warnings[:2],  # Top 2 warnings
                }
            }
            
            # Use adjusted confidence for bypass check
            tech_conf = adjusted_conf
            
            logger.info(
                f"[ADAPTIVE] {symbol}: conf {technical_signal.confidence:.2f} -> {adjusted_conf:.2f} "
                f"(adj={adaptive_rec.confidence_adjustment:+.2f}), "
                f"patterns={[p.name for p in adaptive_rec.matched_patterns]}, "
                f"strategy_weight={adaptive_rec.strategy_weight:.2f}"
            )
    except Exception as e:
        logger.warning(f"[ADAPTIVE] Error in adaptive learning: {e}")
    # -------------------------------------------------------------------------

    # ----- Strong-signal passthrough -------------------------------------
    # Path A: Always bypass Claude when the technical signal is very confident.
    # Path B: In training mode, bypass Claude whenever the technical signal is
    #         directional AND meets the operator's floor. This is the path
    #         that actually produces fills during a training session — the
    #         operator picked floor=0.00 deliberately to see lots of trades.
    bypass_threshold = max(0.0, min(0.62, floor)) if is_training else 0.62
    if side in {"BUY", "SELL"} and tech_conf >= bypass_threshold:
        # Use adaptive learning recommendations for sizing and risk parameters
        size_mult = 1.0
        stop_pct = 0.02
        take_pct = 0.03
        key_factors = [f"strategy={technical_signal.strategy}", f"floor={bypass_threshold:.2f}"]
        
        if adaptive_rec:
            size_mult = adaptive_rec.size_multiplier
            stop_pct = 0.02 * adaptive_rec.stop_loss_multiplier
            take_pct = 0.03 * adaptive_rec.take_profit_multiplier
            
            # Add pattern matches to key factors
            if adaptive_rec.matched_patterns:
                key_factors.append(f"patterns={[p.name for p in adaptive_rec.matched_patterns[:3]]}")
            key_factors.append(f"strategy_weight={adaptive_rec.strategy_weight:.2f}")
            if adaptive_rec.historical_success_rate != 0.5:
                key_factors.append(f"hist_success={adaptive_rec.historical_success_rate*100:.0f}%")
        
        passthrough = TradeDecision(
            action=side,
            confidence=tech_conf,
            size_multiplier=size_mult,
            stop_loss_pct=stop_pct,
            take_profit_pct=take_pct,
            rationale=(
                f"Training-mode passthrough (tech conf {tech_conf:.2f} >= floor {bypass_threshold:.2f}). "
                f"{technical_signal.reasoning}"
                if is_training
                else f"Technical passthrough (conf {tech_conf:.2f} >= 0.62): {technical_signal.reasoning}"
            ) + (f" | Adaptive: {', '.join(adaptive_rec.reasoning[:2])}" if adaptive_rec and adaptive_rec.reasoning else ""),
            key_factors=key_factors,
            risk_flags=adaptive_rec.warnings if adaptive_rec else [],
            source="technical_strong" if not is_training else "training_passthrough",
        )
        _persist_decision(wallet, symbol, price, technical_signal, passthrough, prompt_used="")
        return passthrough
    # ---------------------------------------------------------------------

    if not claude_is_configured():
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="")
        return fallback

    try:
        # Merge adaptive learning context into extra_context for Claude
        enhanced_context = dict(extra_context or {})
        if adaptive_context:
            enhanced_context.update(adaptive_context)
        
        prompt = _build_user_prompt(
            wallet=wallet,
            symbol=symbol,
            price=price,
            technical_signal=technical_signal,
            strategy_type=strategy_type,
            extra_context=enhanced_context,
        )
        system_prompt = _build_system_prompt(wallet)

        result = claude_chat(
            prompt=prompt,
            system=system_prompt,
            max_tokens=DECISION_MAX_TOKENS,
            temperature=DECISION_TEMPERATURE,
        )
        if not result.get("ok"):
            logger.warning("Claude decision call failed: %s", result.get("error"))
            _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used=prompt)
            return fallback

        text = result.get("text", "")
        parsed = _parse_decision_json(text)
        if parsed is None:
            logger.warning("Claude returned non-JSON; falling back. text=%r", text[:200])
            _persist_decision(
                wallet, symbol, price, technical_signal, fallback,
                prompt_used=prompt, raw_text=text, source_override="fallback",
            )
            return fallback

        decision = _normalize_and_clamp(parsed, wallet)
        decision.source = "claude"
        decision.model = result.get("raw", {}).get("model", "") or ""
        decision.raw_text = text

        _persist_decision(wallet, symbol, price, technical_signal, decision, prompt_used=prompt, raw_text=text)
        return decision

    except Exception as e:
        logger.exception("Claude decision engine raised: %s", e)
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="", raw_text=str(e))
        return fallback


# ---------------------------------------------------------------------------- #
# Prompt construction
# ---------------------------------------------------------------------------- #


def _build_system_prompt(wallet: dict[str, Any]) -> str:
    """Base rules + the wallet-specific risk profile + the learned playbook.

    The playbook is intentionally suppressed when there are no real trade
    reflections backing it. Otherwise the seed rules ("avoid low-liquidity
    markets after repeated slippage losses") get parroted back as if they
    were learned, which biases Claude toward HOLD on a brand-new database
    with zero closed trades.
    """
    # Only inject playbook rules that are backed by at least one real reflection.
    # On a fresh DB this collapses to an empty list and we omit the block entirely.
    try:
        from database.db import session_scope
        from database.models import TradeReflection
        with session_scope() as s:
            real_reflections = s.query(TradeReflection).count()
    except Exception:
        real_reflections = 0
    playbook = build_playbook(limit=25) if real_reflections > 0 else []

    risk_lines = [
        f"  - wallet_name: {wallet.get('name')}",
        f"  - platform: {wallet.get('platform')}",
        f"  - trading_mode: {wallet.get('trading_mode', 'paper')}",
        f"  - max_position_usd: {wallet.get('max_position_usd', 0)}",
        f"  - max_open_positions: {wallet.get('max_open_positions', 0)}",
        f"  - max_leverage: {wallet.get('max_leverage', 1.0)}",
        f"  - futures_enabled: {wallet.get('futures_enabled', False)}",
    ]
    risk_block = "WALLET RISK PROFILE:\n" + "\n".join(risk_lines)
    playbook_block = ""
    if playbook:
        playbook_block = (
            "LEARNED PLAYBOOK (rules earned from REAL closed-trade reflections):\n"
            + "\n".join(f"  - {p}" for p in playbook)
            + "\n\nThese are derived from actual past trades, not seed assumptions. "
            "Apply them as priors, not as overrides — the operator's min_confidence_floor "
            "is still the threshold."
        )
    return f"{SYSTEM_PROMPT_BASE}\n\n{risk_block}\n\n{playbook_block}".strip()


def _build_user_prompt(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    extra_context: dict[str, Any],
) -> str:
    """Compact, machine-readable context. Keep this lean — every token costs."""
    sig_meta = getattr(technical_signal, "metadata", {}) or {}

    # Pull recent realized PnL + open positions from this wallet.
    open_positions, recent_trades = _wallet_recent_history(int(wallet["id"]))

    # Read the operator's calibration knobs so Claude doesn't invent its own.
    # See note above: avoid `or 0.55` which would clobber an explicit 0.0.
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

    indicators = {k: _round_or_str(v) for k, v in sig_meta.items()}
    indicators_doc = {
        "ema_fast": "EMA-12 of close (short-term trend)",
        "ema_slow": "EMA-26 of close (medium-term trend)",
        "ret_6": "6-bar log return (positive=bullish)",
        "ret_24": "24-bar log return",
        "atr_pct": "ATR/price (volatility proxy)",
        "bars": "candles available for this lookback",
        "granularity_s": "candle granularity in seconds",
        "last_price": "most recent close",
        # Note: Advanced indicators are in a separate advanced_indicators block
        # with RSI, MACD, Bollinger, ADX, Stochastic, Volume analysis
    }

    # Fetch social sentiment from LunarCrush (if available)
    social_context = None
    try:
        from connectors.lunarcrush import get_social_metrics
        social_metrics = get_social_metrics(symbol)
        if social_metrics:
            social_context = {
                "galaxy_score": social_metrics.galaxy_score,
                "alt_rank": social_metrics.alt_rank,
                "sentiment": "BULLISH" if social_metrics.sentiment_score > 0.2 else (
                    "BEARISH" if social_metrics.sentiment_score < -0.2 else "NEUTRAL"
                ),
                "sentiment_score": round(social_metrics.sentiment_score, 3),
                "bullish_pct": round(social_metrics.bullish_pct, 1),
                "bearish_pct": round(social_metrics.bearish_pct, 1),
                "social_volume": social_metrics.social_volume,
                "social_volume_change_24h": f"{social_metrics.social_volume_change_24h:+.1f}%",
                "social_engagements": social_metrics.social_engagements,
                "influencer_mentions": social_metrics.influencer_mentions,
                "news_articles": social_metrics.news_articles,
                "volume_trend": social_metrics.social_volume_trend,
                "sentiment_trend": social_metrics.sentiment_trend,
                "alerts": [],
            }
            if social_metrics.is_buzzing:
                social_context["alerts"].append("HIGH_BUZZ: Social volume spiking - potential breakout")
            if social_metrics.is_fading:
                social_context["alerts"].append("FADING_INTEREST: Declining engagement - caution")
            if social_metrics.has_influencer_pump:
                social_context["alerts"].append("INFLUENCER_PUMP: Notable account activity - possible pump")
    except Exception as e:
        # Social data is optional - don't fail the decision
        pass

    # =========================================================================
    # NEW: Advanced Market Intelligence (with CACHING to avoid API spam)
    # =========================================================================
    
    # Use module-level cache for data that doesn't change per-symbol
    global _cached_fear_greed, _cached_fg_time
    global _cached_derivatives, _cached_deriv_time
    
    cache_ttl = 300  # 5 minutes cache for global market data
    now_ts = time.time()
    
    # 1. Fear & Greed Index (GLOBAL - same for all symbols, cache it)
    fear_greed_context = None
    try:
        if '_cached_fear_greed' not in globals() or '_cached_fg_time' not in globals():
            _cached_fear_greed = None
            _cached_fg_time = 0
        
        if now_ts - _cached_fg_time > cache_ttl or _cached_fear_greed is None:
            from connectors.fear_greed import get_fear_greed_signal
            _cached_fear_greed = get_fear_greed_signal()
            _cached_fg_time = now_ts
        
        fg_data = _cached_fear_greed
        if fg_data and fg_data.get("available"):
            fear_greed_context = {
                "value": fg_data.get("value"),
                "classification": fg_data.get("classification"),
                "sentiment": fg_data.get("sentiment"),
                "signal": fg_data.get("signal"),
                "confidence_adjustment": fg_data.get("confidence_adjustment", 0),
                "summary": fg_data.get("summary", ""),
            }
    except Exception:
        pass
    
    # 2. Derivatives Intelligence (per-symbol but with caching)
    derivatives_context = None
    try:
        if '_cached_derivatives' not in globals() or '_cached_deriv_time' not in globals():
            _cached_derivatives = {}
            _cached_deriv_time = {}
        
        cache_key = symbol
        if cache_key not in _cached_deriv_time or now_ts - _cached_deriv_time.get(cache_key, 0) > cache_ttl:
            from connectors.coinglass import get_funding_signal
            _cached_derivatives[cache_key] = get_funding_signal(symbol)
            _cached_deriv_time[cache_key] = now_ts
        
        deriv_data = _cached_derivatives.get(cache_key)
        if deriv_data:
            derivatives_context = {
                "overall_signal": deriv_data.get("overall_signal", "NEUTRAL"),
                "confidence_adjustment": deriv_data.get("confidence_adjustment", 0),
                "funding_rate_pct": deriv_data.get("funding_rate"),
                "summary": deriv_data.get("summary", ""),
            }
    except Exception:
        pass
    
    # 3. Multi-Timeframe Analysis - SKIP for now (too slow per-symbol)
    # Only fetch MTF for high-confidence signals to save API calls
    mtf_context = None
    if float(technical_signal.confidence or 0) >= 0.50:
        try:
            from trading.multi_timeframe import get_mtf_signal_boost
            mtf_data = get_mtf_signal_boost(symbol)
            if mtf_data and mtf_data.get("bias") != "UNKNOWN":
                mtf_context = {
                    "overall_bias": mtf_data["bias"],
                    "alignment_score": round(mtf_data.get("alignment", 0), 2),
                    "confidence_boost": mtf_data.get("boost", 0),
                    "summary": mtf_data.get("summary", ""),
                }
        except Exception:
            pass
    
    # 4. Market Regime - SKIP expensive per-symbol computation
    # Use a simplified approach: just pass the technical indicators
    market_regime_context = None
    
    # 5. Advanced Technical Indicators - SKIP (already in technical_signal)
    # The strategy engine already computes these, don't double-fetch
    advanced_indicators = None
    
    # Calculate aggregate confidence adjustment from all sources
    confidence_adjustments = []
    if mtf_context and mtf_context.get("confidence_boost"):
        confidence_adjustments.append(("MTF", mtf_context["confidence_boost"]))
    if derivatives_context and derivatives_context.get("confidence_adjustment"):
        confidence_adjustments.append(("Derivatives", derivatives_context["confidence_adjustment"]))
    if fear_greed_context and fear_greed_context.get("confidence_adjustment"):
        confidence_adjustments.append(("Fear&Greed", fear_greed_context["confidence_adjustment"]))
    
    total_adjustment = sum(adj for _, adj in confidence_adjustments)
    # Cap total adjustment to ±0.25
    total_adjustment = max(-0.25, min(0.25, total_adjustment))

    payload = {
        "operator_calibration": {
            # Tell Claude exactly what threshold the operator already set, so it
            # doesn't unilaterally invent a higher floor (the old prompt's "<0.55"
            # rule made every weak signal collapse to 0.50/HOLD).
            "min_confidence_floor": round(floor, 4),
            "is_training_session": is_training,
            "instruction": (
                "Trade in the technical_side direction whenever technical_confidence "
                f">= {floor:.2f} unless extra_context contains a concrete contradiction."
            ),
        },
        "candidate": {
            "symbol": symbol,
            "price": round(price, 8),
            "strategy_type": strategy_type,
            "technical_side": technical_signal.side,
            "technical_confidence": round(float(technical_signal.confidence or 0), 4),
            "technical_reasoning": technical_signal.reasoning,
            "indicators": indicators,
            "indicators_legend": indicators_doc,
        },
        "wallet_state": {
            "paper_balance": round(float(wallet.get("paper_balance", 0)), 2),
            "open_position_count": len(open_positions),
            "open_positions": open_positions,
        },
        "recent_history": {
            "last_10_closed_trades": recent_trades,
            "note": (
                "An empty list means this wallet has no closed-trade history yet. "
                "That is normal at the start of a training session — do NOT treat "
                "an empty history as a reason to refuse trading."
            ) if not recent_trades else "",
        },
        # =====================================================================
        # MARKET INTELLIGENCE SUITE
        # =====================================================================
        "advanced_indicators": advanced_indicators,
        "multi_timeframe_analysis": mtf_context,
        "market_regime": market_regime_context,
        "derivatives_intelligence": derivatives_context,
        "fear_greed_index": fear_greed_context,
        "social_sentiment": social_context,
        # Aggregate confidence adjustment from all intelligence sources
        "confidence_adjustments": {
            "sources": confidence_adjustments,
            "total_adjustment": round(total_adjustment, 3),
            "adjusted_confidence": round(
                min(1.0, max(0.0, float(technical_signal.confidence or 0) + total_adjustment)), 4
            ),
            "note": (
                "This is the sum of confidence boosts/penalties from MTF alignment, "
                "derivatives sentiment, and Fear & Greed index. Apply to your final "
                "confidence output."
            ),
        },
        "extra_context": extra_context,
        "now_utc": utcnow().isoformat(),
    }

    return (
        "Decide the next action for this candidate. "
        "Return ONLY the JSON object specified in your instructions.\n\n"
        f"{json.dumps(payload, default=str)}"
    )


def _wallet_recent_history(wallet_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compact recent-history snapshot used for prompt context."""
    with session_scope() as s:
        opens = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "open")
            .order_by(PaperTrade.opened_at.desc())
            .limit(10)
            .all()
        )
        closed = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc())
            .limit(10)
            .all()
        )
        open_positions = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "qty": float(t.qty),
                "entry": float(t.entry_price),
                "unrealized_pnl": float(t.unrealized_pnl or 0),
                "age_min": _minutes_since(t.opened_at),
            }
            for t in opens
        ]
        recent_closed = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "pnl": float(t.realized_pnl or 0),
                "confidence": float(t.confidence or 0),
                "held_min": _minutes_between(t.opened_at, t.closed_at),
            }
            for t in closed
        ]
    return open_positions, recent_closed


def _extract_market_state(extra_context: dict) -> dict:
    """Extract market state indicators for adaptive learning analysis."""
    state = {}
    
    # From advanced_indicators
    indicators = extra_context.get("advanced_indicators", {})
    state["rsi"] = indicators.get("rsi_14", 50)
    state["macd_histogram"] = indicators.get("macd_histogram", 0)
    state["bb_percent_b"] = indicators.get("bollinger_percent_b", 0.5)
    state["adx"] = indicators.get("adx_trend_strength", 0)
    state["volume_ratio"] = indicators.get("relative_volume", 1.0)
    state["trend"] = indicators.get("trend_direction", "NEUTRAL")
    state["momentum"] = indicators.get("momentum_signal", "NEUTRAL")
    state["volatility_state"] = indicators.get("volatility_state", "NORMAL")
    
    # From market_regime
    regime = extra_context.get("market_regime", {})
    state["regime"] = regime.get("regime", "UNKNOWN")
    
    # From fear_greed_index
    fg = extra_context.get("fear_greed_index", {})
    state["fear_greed"] = fg.get("value", 50)
    
    # From multi_timeframe_analysis
    mtf = extra_context.get("multi_timeframe_analysis", {})
    state["mtf_alignment"] = mtf.get("alignment_score", 0.5)
    state["mtf_bias"] = mtf.get("overall_bias", "NEUTRAL")
    
    # From derivatives_intelligence
    deriv = extra_context.get("derivatives_intelligence", {})
    state["funding_rate"] = deriv.get("funding_rate_pct", 0)
    state["long_ratio"] = deriv.get("long_ratio", 50)
    
    return state


# ---------------------------------------------------------------------------- #
# Parsing + clamping
# ---------------------------------------------------------------------------- #


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_decision_json(text: str) -> dict[str, Any] | None:
    """Tolerantly extract a JSON object from Claude's response."""
    if not text:
        return None
    # Fast path: the whole thing is JSON.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the first {...} block.
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_and_clamp(parsed: dict[str, Any], wallet: dict[str, Any]) -> TradeDecision:
    """Coerce Claude's JSON into a safe TradeDecision."""
    action = str(parsed.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "SELL", "HOLD", "CLOSE"}:
        action = "HOLD"

    conf = _to_float(parsed.get("confidence"), 0.0)
    conf = max(0.0, min(conf, 1.0))

    size_mult = _to_float(parsed.get("size_multiplier"), 1.0)
    size_mult = max(0.0, min(size_mult, 1.0))

    sl = _to_float(parsed.get("stop_loss_pct"), 0.05)
    sl = max(0.005, min(sl, 0.20))

    tp = _to_float(parsed.get("take_profit_pct"), 0.10)
    tp = max(0.005, min(tp, 0.50))

    factors = parsed.get("key_factors") or []
    if not isinstance(factors, list):
        factors = [str(factors)]
    factors = [str(x)[:200] for x in factors][:8]

    flags = parsed.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(x)[:200] for x in flags][:8]

    rationale = str(parsed.get("rationale", ""))[:1500]

    return TradeDecision(
        action=action,
        confidence=conf,
        size_multiplier=size_mult,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        rationale=rationale,
        key_factors=factors,
        risk_flags=flags,
    )


def _technical_fallback(signal: Signal) -> TradeDecision:
    side = (signal.side or "HOLD").upper()
    if side not in {"BUY", "SELL", "HOLD"}:
        side = "HOLD"
    return TradeDecision(
        action=side,
        confidence=float(signal.confidence or 0.0),
        size_multiplier=1.0,
        stop_loss_pct=0.02,   # 2% stop loss
        take_profit_pct=0.03, # 3% take profit
        rationale=signal.reasoning or "Fallback: technical signal only.",
        key_factors=[],
        risk_flags=["claude_unavailable"],
        source="technical",
    )


# ---------------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------------- #


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
) -> None:
    try:
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
            )
            s.add(row)
    except Exception:
        # Never let logging fail the trade.
        logger.exception("Failed to persist ClaudeDecision row.")


# ---------------------------------------------------------------------------- #
# Tiny helpers
# ---------------------------------------------------------------------------- #


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _round_or_str(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
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
