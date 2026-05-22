"""
Autonomous bot engine.

This is the loop that wakes up on a schedule, walks the configured universe,
asks the AIEngine for a decision per symbol, and (if confident enough) routes
the resulting trade through the existing PaperTradingEngine — which already
enforces wallet caps, risk manager checks, fees, and slippage.

Design notes:
  - This module owns NO scheduling. `services/scheduler.py` calls `tick()`.
  - This module owns NO HTTP. The web UI calls `BotEngine.tick()` directly
    or starts/stops the scheduler.
  - It is fully idempotent and safe to call concurrently — every DB write
    goes through `session_scope()` which is transactional.
  - When `bot_enabled=false` or the wallet's `bot_paused=true`, the loop
    is a no-op (besides logging that it was skipped).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai.ai_engine import AIEngine
from ai.claude_decision_engine import decide as claude_decide
from config.bot_config import BotConfig
from connectors.live_prices import get_price
from connectors.universe import coinbase_usd_universe
from database.db import session_scope
from database.models import (
    ActivityLog,
    PaperTrade,
    Strategy,
    Wallet,
)
from trading.paper_trading_engine import PaperTradingEngine
from trading.portfolio_intelligence import (
    PortfolioIntelligence,
    execute_portfolio_action,
)
from trading.position_monitor import PositionMonitor
from trading.risk_manager import RiskManager
from trading.strategy_engine import evaluate_symbol, evaluate_entry_quality, get_entry_candles
# Advanced trading modules (signal engine temporarily disabled - uses evaluate_symbol instead)
from trading.advanced_position_sizer import get_position_sizer
from risk.daily_mission_controller import get_mission_controller
from trading.market_intelligence import get_market_intelligence
from trading.strategic_claude import get_strategic_router
from trading.trade_filter import get_trade_filter, FilterResult
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TickResult:
    started_at: str
    ended_at: str
    universe_size: int
    wallets_evaluated: int
    decisions: int
    actions: int
    skipped: int
    errors: int
    notes: list[str] = field(default_factory=list)


class BotEngine:
    """Singleton-ish: one BotEngine per process. Safe to call .tick() concurrently."""

    def __init__(self) -> None:
        self.ai = AIEngine()
        self.paper = PaperTradingEngine()
        self.position_monitor = PositionMonitor()
        self.portfolio_intel = PortfolioIntelligence()
        self._lock = threading.Lock()
        # In-memory ring buffer of recent tick results so the UI can show "what the bot did".
        self._recent_ticks: list[TickResult] = []
        self._max_recent = 50
        
        # LOSING STREAK CIRCUIT BREAKER
        # Prevents the bot from opening new positions after consecutive losses
        self._consecutive_losses = 0
        self._max_consecutive_losses = 3  # Pause after 3 losses in a row
        self._cooldown_until: datetime | None = None
        self._cooldown_minutes = 5  # Wait 5 minutes after hitting streak limit

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker state. Call this when starting a new session."""
        self._consecutive_losses = 0
        self._cooldown_until = None
        self._log("bot", "Circuit breaker reset.", level="info")

    def record_trade_result(self, pnl: float) -> None:
        """Call this when a trade closes to update the losing streak tracker."""
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._max_consecutive_losses:
                self._cooldown_until = utcnow() + timedelta(minutes=self._cooldown_minutes)
                self._log(
                    "bot",
                    f"CIRCUIT BREAKER: {self._consecutive_losses} consecutive losses. "
                    f"Pausing new entries for {self._cooldown_minutes} minutes.",
                    level="warn",
                )
        else:
            # Reset on a win
            self._consecutive_losses = 0
            self._cooldown_until = None

    def is_in_cooldown(self) -> bool:
        """Check if we're in a cooldown period from losing streak."""
        if self._cooldown_until is None:
            return False
        if utcnow() >= self._cooldown_until:
            # Cooldown expired, reset
            self._cooldown_until = None
            self._consecutive_losses = 0
            self._log("bot", "Circuit breaker cooldown expired. Resuming trading.", level="info")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def tick(self, *, manual: bool = False) -> TickResult:
        """One pass of the autonomous loop. Returns a structured result."""
        # Serialize ticks within a single process. Two ticks running at once
        # would race on wallet.paper_balance and order counts.
        if not self._lock.acquire(blocking=False):
            return self._note_skip("tick already in progress")

        try:
            return self._run_tick(manual=manual)
        finally:
            self._lock.release()

    def recent_ticks(self, limit: int = 20) -> list[TickResult]:
        return list(self._recent_ticks[-limit:][::-1])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    def _run_tick(self, *, manual: bool) -> TickResult:
        cfg = BotConfig.load()
        started = utcnow()

        result = TickResult(
            started_at=started.isoformat(),
            ended_at=started.isoformat(),
            universe_size=0,
            wallets_evaluated=0,
            decisions=0,
            actions=0,
            skipped=0,
            errors=0,
        )

        # HEARTBEAT: Always log that a tick started so we know the scheduler is alive
        self._log(
            "bot",
            f"TICK STARTED (manual={manual}, enabled={cfg.bot_enabled}, dry_run={cfg.dry_run})",
            level="info",
        )

        if not cfg.bot_enabled and not manual:
            result.notes.append("bot_disabled")
            self._log("bot", "Tick skipped: bot is disabled.", level="info")
            return self._record(result)

        # Global kill switch — fast-path before we touch the network.
        if RiskManager.kill_switch_status():
            result.notes.append("kill_switch")
            self._log("bot", "Tick skipped: global kill switch is engaged.", level="warn")
            return self._record(result)

        # Build the universe once per tick.
        universe = self._load_universe(cfg)
        result.universe_size = len(universe)
        if not universe:
            result.notes.append("empty_universe")
            self._log("bot", "Tick skipped: universe is empty.", level="warn")
            return self._record(result)

        # -----------------------------------------------------------------
        # PREFETCH PRICES: Warm the cache with all universe + open position
        # prices in a single batch call. This is MUCH faster than fetching
        # each price individually during position monitoring.
        # -----------------------------------------------------------------
        try:
            from connectors.live_prices import get_prices_batch
            from database.db import session_scope
            from database.models import PaperTrade
            
            # Collect all symbols we need prices for
            # Universe items are dicts with "product_id" key, not objects with .symbol
            symbols_needed = [u["product_id"] for u in universe if isinstance(u, dict) and "product_id" in u]
            with session_scope() as s:
                open_positions = s.query(PaperTrade.symbol).filter(PaperTrade.status == "open").distinct().all()
                symbols_needed.extend([p[0] for p in open_positions])
            
            # Batch fetch all prices at once (populates cache)
            price_map = get_prices_batch(list(set(symbols_needed)))
            self._log("bot", f"Prefetched {len(price_map)} prices", level="debug")
        except Exception as e:
            import traceback
            self._log("bot", f"Price prefetch failed: {e}\n{traceback.format_exc()}", level="warn")
            price_map = {}

        # -----------------------------------------------------------------
        # POSITION MONITORING: Check all open positions for auto-exits
        # before evaluating new entries. This ensures SL/TP/trailing stops
        # are processed at the same frequency as new entry signals.
        # -----------------------------------------------------------------
        try:
            auto_exits = self._monitor_positions(cfg, universe, result, price_map)
            if auto_exits > 0:
                result.notes.append(f"auto_exits={auto_exits}")
        except Exception as e:
            self._log("bot", f"Position monitoring error: {e}", level="warn")
            result.errors += 1

        # -----------------------------------------------------------------
        # PORTFOLIO INTELLIGENCE: Proactive portfolio management.
        # This is where the "smart" behavior happens - instead of passively
        # waiting for losing positions to recover, we:
        #   1. DCA into losers at better prices
        #   2. Scale into winners that keep working
        #   3. Open offset trades to balance underwater positions
        # This runs EVERY tick, regardless of slot availability.
        # -----------------------------------------------------------------
        try:
            intel_actions = self._run_portfolio_intelligence(cfg, universe, result, price_map)
            if intel_actions > 0:
                result.notes.append(f"portfolio_intel_actions={intel_actions}")
        except Exception as e:
            self._log("bot", f"Portfolio intelligence error: {e}", level="warn")
            result.errors += 1

        # Walk every active wallet.
        with session_scope() as s:
            wallets = (
                s.query(Wallet)
                .filter(Wallet.bot_paused.is_(False))
                .all()
            )
            wallet_snapshots = [
                {
                    "id": w.id,
                    "name": w.name,
                    "platform": w.platform,
                    "trading_mode": w.trading_mode,
                    "paper_balance": float(w.paper_balance or 0),
                    "max_open_positions": int(w.max_open_positions or 0),
                    "max_position_usd": float(w.max_position_usd or 0),
                }
                for w in wallets
            ]

        for wallet in wallet_snapshots:
            result.wallets_evaluated += 1
            try:
                self._evaluate_wallet(cfg, wallet, universe, result)
            except Exception as e:  # never let one wallet kill the tick
                import traceback
                tb = traceback.format_exc()
                result.errors += 1
                result.notes.append(f"wallet_{wallet['id']}_error: {e}")
                self._log(
                    "bot",
                    f"Wallet {wallet['name']} (#{wallet['id']}) tick error: {e}\n{tb}",
                    wallet_id=wallet["id"],
                    level="warn",
                )

        result.ended_at = utcnow().isoformat()
        self._log(
            "bot",
            (
                f"Tick complete: universe={result.universe_size}, "
                f"wallets={result.wallets_evaluated}, "
                f"decisions={result.decisions}, "
                f"actions={result.actions}, "
                f"skipped={result.skipped}, errors={result.errors}"
            ),
            level="info",
        )
        return self._record(result)

    # ------------------------------------------------------------------ #
    # Per-wallet evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_wallet(
        self,
        cfg: BotConfig,
        wallet: dict[str, Any],
        universe: list[dict[str, Any]],
        result: TickResult,
    ) -> None:
        # Count current open paper positions for this wallet — respect caps.
        with session_scope() as s:
            open_count = (
                s.query(PaperTrade)
                .filter(
                    PaperTrade.wallet_id == wallet["id"],
                    PaperTrade.status == "open",
                )
                .count()
            )
            # Get symbols we already hold to ensure diversification
            held_symbols = set(
                row[0] for row in s.query(PaperTrade.symbol)
                .filter(PaperTrade.wallet_id == wallet["id"], PaperTrade.status == "open")
                .all()
            )
            # Pick the wallet's "default" strategy if any (the first one assigned to it).
            strat = (
                s.query(Strategy)
                .filter(Strategy.wallet_id == wallet["id"])
                .order_by(Strategy.id.asc())
                .first()
            )
            strategy_id = strat.id if strat else None
            strategy_type = strat.strategy_type if strat else cfg.default_strategy_type

        # USE cfg.max_open_per_wallet as the limit (not wallet's limit which may be lower).
        # This allows the user to set high diversification via bot config.
        cap = cfg.max_open_per_wallet
        if wallet["max_open_positions"] and wallet["max_open_positions"] > 0:
            cap = max(cap, wallet["max_open_positions"])  # Take the HIGHER value for diversity
        
        slots_left = max(0, cap - open_count)
        
        # Log detailed slot info for debugging
        self._log(
            "bot",
            f"[SLOT_CHECK] {wallet['name']}: open={open_count}, cap={cap}, slots_left={slots_left}, held={len(held_symbols)}, cfg.max_open={cfg.max_open_per_wallet}, wallet.max_open={wallet['max_open_positions']}",
            wallet_id=wallet["id"],
            level="info",
        )
        
        # Even if slots are full, log but DON'T return - Portfolio Intelligence
        # can still DCA into existing positions below.
        if slots_left <= 0:
            self._log(
                "bot",
                f"{wallet['name']}: No new entry slots ({open_count}/{cap}), but checking for DCA opportunities.",
                wallet_id=wallet["id"],
                level="info",
            )
            # Note: We continue so portfolio intelligence can still act on existing positions

        # CIRCUIT BREAKER CHECK: Track consecutive losses but DON'T block completely
        # Just add a note and log it - we still want to evaluate and potentially trade
        circuit_breaker_active = self.is_in_cooldown()
        if circuit_breaker_active:
            self._log(
                "bot",
                f"{wallet['name']}: Circuit breaker active ({self._consecutive_losses} losses, "
                f"cooldown until {self._cooldown_until}). Will be more conservative.",
                wallet_id=wallet["id"],
                level="warn",
            )
            result.notes.append("circuit_breaker_active")
            # Don't return - still evaluate but we'll be more selective

        # Track best per-tick candidate so we always log a useful summary
        # even when nothing crosses the confidence threshold.
        best: dict[str, Any] | None = None
        evaluated = 0
        below_conf = 0
        claude_calls = 0  # Track how many times we call Claude

        # Shuffle universe for better diversification - don't always evaluate same symbols first
        import random
        shuffled_universe = list(universe)
        random.shuffle(shuffled_universe)

        # Sweep the universe. DON'T break when slots are full - we still want to
        # call Claude for existing positions (re-evaluate, DCA opportunities, etc.)
        symbols_evaluated = 0
        symbols_skipped_held = 0
        symbols_skipped_noslots = 0
        
        for product in shuffled_universe:
            symbol = product["product_id"]
            
            # For NEW entries, skip if we already hold AND have no slots
            # But if we DO have slots, we should diversify into NEW symbols
            is_held = symbol in held_symbols
            
            # Skip this symbol for NEW entries if:
            # 1. We already hold it (diversification), OR
            # 2. We have no slots left AND we don't hold it (can't act anyway)
            if is_held:
                # We might want to DCA - let portfolio intelligence handle that
                symbols_skipped_held += 1
                continue
            if slots_left <= 0:
                # No slots for new positions - can't open new trades
                # Just track a few "best signals" for monitoring then stop
                symbols_skipped_noslots += 1
                if evaluated >= 10:
                    break
                continue
            
            price_payload = get_price(symbol)
            if not price_payload.get("ok"):
                result.skipped += 1
                continue
            price = float(price_payload["price"])
            if price <= 0:
                result.skipped += 1
                continue

            # =====================================================================
            # MARKET INTELLIGENCE CHECK
            # Analyzes BTC correlation, liquidity, entry timing, sentiment
            # This is what makes us smarter than average traders
            # =====================================================================
            market_intel = get_market_intelligence()
            market_intel.update_price(symbol, price)  # Track price history
            
            # Get market context (BTC trend, overall sentiment)
            market_ctx = market_intel.get_market_context()
            
            # =====================================================================
            # ADVANCED MULTI-FACTOR SIGNAL GENERATION
            # Uses the new advanced_signal_engine with:
            # - Trend analysis (EMA alignment, ADX)
            # - Momentum (RSI, MACD histogram, rate of change)
            # - Volatility (ATR, Bollinger Band position)
            # - Volume confirmation
            # - Pattern recognition
            # - Quality grades (A+, A, B, C, F)
            # =====================================================================
            
            # Use the ORIGINAL evaluate_symbol which works reliably
            # The advanced_signal_engine.analyze() requires candle data we don't have here
            signal = evaluate_symbol(symbol, strategy_type, tick_seconds=cfg.tick_seconds)
            
            # Store for later reference
            advanced_signal = None
            
            evaluated += 1
            result.decisions += 1
            
            # Get intelligence on this specific trade opportunity
            intel = market_intel.analyze_trade_opportunity(
                symbol=symbol,
                current_price=price,
                side=signal.side,
                volume_24h=price_payload.get("volume_24h", 0),
                bid=price_payload.get("bid", price * 0.999),
                ask=price_payload.get("ask", price * 1.001),
            )
            
            # Skip if market intelligence says don't trade
            if not intel.should_trade and intel.confidence_adjustment < 0.6:
                self._log(
                    "bot",
                    f"[INTEL] Skip {symbol}: {intel.skip_reason}",
                    wallet_id=wallet["id"],
                    level="debug",
                )
                continue
            
            # Skip longs if BTC is dumping (alts dump harder)
            if market_ctx.avoid_longs and signal.side == "BUY" and symbol != "BTC-USD":
                self._log(
                    "bot",
                    f"[INTEL] Skip {symbol} long: {market_ctx.reasoning}",
                    wallet_id=wallet["id"],
                    level="info",
                )
                continue

            # ALWAYS call Claude for BUY/SELL signals, even if confidence is low.
            # Let Claude be the arbiter. Only skip truly empty HOLD signals.
            if (
                signal.side == "HOLD"
                and float(signal.confidence or 0) <= 0.05
                and not signal.indicators.get("ema_fast")
            ):
                # Still record the diagnostic so the user sees we evaluated it.
                if best is None:
                    best = {
                        "symbol": symbol,
                        "side": "HOLD",
                        "confidence": 0.0,
                        "reason": signal.reasoning,
                    }
                continue

            # =====================================================================
            # STRATEGIC CLAUDE ROUTING
            # Only calls expensive Claude API when it adds value:
            # - Ambiguous signals (40-65% confidence)
            # - Large positions (>$500)
            # - Reversal situations
            # - High volatility moments
            # Otherwise uses fast technical decision
            # =====================================================================
            strategic_router = get_strategic_router()
            decision = strategic_router.decide(
                wallet=wallet,
                symbol=symbol,
                price=price,
                technical_signal=signal,
                strategy_type=strategy_type,
                position_size_usd=cfg.position_size_usd,
            )
            
            # Track if we used Claude for logging
            if decision.source == "claude":
                claude_calls += 1

            side = decision.action
            confidence = float(decision.confidence or 0.0)
            
            # Apply market intelligence adjustments to confidence
            # Better entry timing = higher confidence, poor liquidity = lower confidence
            adjusted_confidence = confidence * intel.confidence_adjustment
            
            # Log intelligence impact if significant
            if abs(intel.confidence_adjustment - 1.0) > 0.1:
                self._log(
                    "bot",
                    f"[INTEL] {symbol}: conf {confidence:.2f} -> {adjusted_confidence:.2f} "
                    f"(entry={intel.entry_timing.entry_quality}, sector={intel.sector})",
                    wallet_id=wallet["id"],
                    level="debug",
                )
            
            confidence = adjusted_confidence
            
            # =====================================================================
            # TRADE QUALITY FILTER
            # Final checks before entry:
            # - Time-of-day (avoid low-volume hours)
            # - Position correlation (avoid overexposure)
            # - Market regime alignment
            # - Session awareness
            # =====================================================================
            trade_filter = get_trade_filter()
            
            # Get current positions for correlation check
            current_positions = [
                {"symbol": p["symbol"], "side": p["side"]}
                for p in wallet.get("open_positions", [])
            ]
            
            filter_result = trade_filter.apply_filters(
                symbol=symbol,
                side=side,
                confidence=confidence,
                current_positions=current_positions,
                # MarketContext exposes regime info via two separate fields
                # (`volatility_regime` and `correlation_regime`); the trade
                # filter only needs a single string, so we forward the
                # volatility regime which is what its rules key off of.
                market_regime=(
                    getattr(market_ctx, "volatility_regime", None)
                    if market_ctx else None
                ),
                signal_strategy=signal.strategy if signal else None,
            )
            
            if not filter_result.should_trade:
                self._log(
                    "bot",
                    f"[FILTER] {symbol} {side} rejected: {', '.join(filter_result.reasons)}",
                    wallet_id=wallet["id"],
                    level="info",
                )
                continue
            
            # Apply filter adjustments
            confidence *= filter_result.confidence_adjustment

            # =====================================================================
            # SUPPORT / RESISTANCE HEADROOM GATE
            # Reject BUYs sitting right under a wall of resistance (and SELLs
            # right above support). Without this, even a "valid" signal fires
            # into a structure ceiling that price can't punch through, which
            # is exactly the "buy and watch it do nothing then drift down"
            # pattern the user is seeing.
            # =====================================================================
            entry_candles = get_entry_candles(symbol, tick_seconds=cfg.tick_seconds, lookback_bars=80)
            sr_check = evaluate_entry_quality(entry_candles, side)
            if not sr_check.get("accept", True):
                self._log(
                    "bot",
                    f"[S/R] {symbol} {side} rejected: {sr_check.get('reason')}",
                    wallet_id=wallet["id"],
                    level="info",
                )
                continue

            # =====================================================================
            # ENTRY QUALITY + FUNDAMENTAL ALIGNMENT (advisory only)
            #
            # Earlier iterations of this block stacked too many multipliers
            # (quality 0.65..1.20, fundamentals 0.85..1.10, R:R hard floor of
            # 1.2) and ended up rejecting every realistic scalper BUY in chop.
            # The S/R `accept` flag above is already the structural gate;
            # everything below is just a small nudge to break ties between
            # otherwise-similar candidates. Total swing here is capped at
            # roughly ±10% of confidence so good signals always survive.
            # =====================================================================
            sr_quality_raw = float(sr_check.get("quality_score", 1.0))
            potential_rr = float(sr_check.get("potential_rr", 0.0))

            # Compress quality_score from [0.65, 1.20] -> [0.95, 1.05].
            # We trust the S/R headroom gate to do the heavy lifting; this
            # multiplier just gives a tiny edge to setups with great room.
            sr_quality = 0.95 + (sr_quality_raw - 0.65) * (0.10 / 0.55)
            sr_quality = max(0.95, min(1.05, sr_quality))
            confidence *= sr_quality

            try:
                fg_score = float(getattr(market_ctx, "market_fear_greed", 50.0))
            except (TypeError, ValueError):
                fg_score = 50.0
            btc_trend = (getattr(market_ctx, "btc_trend", "") or "").lower()

            fundamental_mult = 1.0
            fundamental_notes: list[str] = []

            if side == "BUY":
                if fg_score >= 85:  # raised from 80 - only the most extreme greed
                    fundamental_mult *= 0.95
                    fundamental_notes.append(f"extreme greed ({fg_score:.0f})")
                elif fg_score <= 25 and signal.strategy in ("Mean Reversion", "Scalping"):
                    fundamental_mult *= 1.03
                    fundamental_notes.append(f"fear ({fg_score:.0f}) + reversion")
                if btc_trend in ("strong_uptrend", "uptrend", "bullish"):
                    fundamental_mult *= 1.03
                    fundamental_notes.append(f"BTC {btc_trend}")
                elif btc_trend in ("strong_downtrend", "bearish") and symbol != "BTC-USD":
                    fundamental_mult *= 0.95
                    fundamental_notes.append(f"BTC {btc_trend} - alts follow")
            else:  # SELL
                if fg_score <= 15:  # raised from 20 - only the most extreme fear
                    fundamental_mult *= 0.95
                    fundamental_notes.append(f"extreme fear ({fg_score:.0f})")
                elif fg_score >= 75 and signal.strategy in ("Mean Reversion", "Scalping"):
                    fundamental_mult *= 1.03
                    fundamental_notes.append(f"greed ({fg_score:.0f}) + reversion")
                if btc_trend in ("strong_downtrend", "bearish"):
                    fundamental_mult *= 1.03
                    fundamental_notes.append(f"BTC {btc_trend}")
                elif btc_trend in ("strong_uptrend", "bullish") and symbol != "BTC-USD":
                    fundamental_mult *= 0.95
                    fundamental_notes.append(f"BTC {btc_trend}")

            confidence *= fundamental_mult

            if fundamental_notes or abs(sr_quality - 1.0) > 0.02:
                self._log(
                    "bot",
                    (
                        f"[QUALITY] {symbol} {side}: sr_q={sr_quality:.2f} "
                        f"(R:R={potential_rr:.1f}), fundamentals={fundamental_mult:.2f} "
                        f"({'; '.join(fundamental_notes) or 'neutral'}) -> conf={confidence:.2f}"
                    ),
                    wallet_id=wallet["id"],
                    level="debug",
                )

            # Always remember the strongest candidate we saw so the tick log
            # reads like "best was BTC-USD BUY 0.42 (below 0.55 floor)".
            if best is None or confidence > best.get("confidence", 0.0):
                best = {
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "reason": (decision.rationale or signal.reasoning or "")[:120],
                }

            if side not in {"BUY", "SELL"}:
                continue
            
            # ---- Risk gating -------------------------------------------------
            # Two layers gate every entry:
            #
            #   1. Adaptive confidence floor. The user's `bot_min_confidence`
            #      is the FLOOR — adaptive_risk can raise it in real time
            #      when rolling WR drops, but never lowers it below the
            #      user's setting. This is what was missing: the static
            #      0.45/0.60 floor never reacted to the bot bleeding all
            #      night, even though the learning engine "knew".
            #
            #   2. P&L-based circuit breaker. The legacy consecutive-loss
            #      counter is still here for fast reaction (3 instant
            #      losses → tighten), but the new rolling-P&L breaker
            #      catches the slow bleed that scalp wins were hiding.
            #
            # Both also respect a per-symbol blacklist that auto-engages
            # for symbols with sustained <25% WR in this session.
            from trading.adaptive_risk import AdaptiveRiskManager
            effective_min_conf = AdaptiveRiskManager.effective_floor(cfg.min_confidence)
            if circuit_breaker_active:
                # Stack the legacy fast-reaction bump on top.
                effective_min_conf = min(0.85, effective_min_conf + 0.05)
            pnl_breaker_active, _ = AdaptiveRiskManager.is_breaker_active()
            if pnl_breaker_active:
                # Rolling P&L is bleeding — demand strong conviction only.
                effective_min_conf = max(effective_min_conf, 0.75)

            if confidence < effective_min_conf:
                below_conf += 1
                # Log blocked trades so user knows WHY nothing is executing
                if side in {"BUY", "SELL"} and confidence >= 0.40:
                    self._log(
                        "bot",
                        f"[BLOCKED] {symbol} {side}: conf {confidence:.2f} < min {effective_min_conf:.2f}",
                        wallet_id=wallet["id"],
                        level="debug",
                    )
                continue

            # Per-symbol auto-blacklist (session-scoped, auto-expires).
            if AdaptiveRiskManager.is_symbol_blocked(symbol):
                self._log(
                    "bot",
                    f"[BLOCKED] {symbol} {side}: symbol auto-blacklisted (sustained low WR)",
                    wallet_id=wallet["id"],
                    level="debug",
                )
                continue

            # =====================================================================
            # ADVANCED POSITION SIZING
            # Uses Kelly Criterion, drawdown adjustment, portfolio heat management
            # =====================================================================
            position_sizer = get_position_sizer()
            size_result = position_sizer.calculate_size(
                wallet_id=wallet["id"],
                symbol=symbol,
                entry_price=price,
                stop_loss_pct=decision.stop_loss_pct,
                confidence=confidence,
                base_size_usd=cfg.position_size_usd,
                signal_quality=getattr(decision, 'quality', 'B'),
            )
            
            position_usd = size_result.recommended_usd
            qty = size_result.recommended_qty
            
            if qty <= 0 or position_usd < 10:
                self._log(
                    "bot",
                    f"[SKIP] {symbol}: Position too small (${position_usd:.2f}, qty={qty}). {size_result.reasoning}",
                    wallet_id=wallet["id"],
                    level="info",
                )
                continue
            
            # Log sizing decision
            self._log(
                "bot",
                f"[SIZE OK] {symbol}: ${position_usd:.0f}, qty={qty:.6f} ({size_result.conviction_multiplier:.1f}x conv)",
                wallet_id=wallet["id"],
                level="info",
            )
            if size_result.warnings:
                self._log(
                    "bot",
                    f"[SIZE WARN] {symbol}: {', '.join(size_result.warnings[:2])}",
                    wallet_id=wallet["id"],
                    level="info",
                )

            # =====================================================================
            # DAILY MISSION CONTROLLER — the boss layer.
            # When the operator has enabled mission_controller_enabled in bot_config,
            # every trade decision passes through the controller. It can:
            #   - reject the trade (mode is LOCK/KILL, daily limits hit, edge too
            #     small for the current mode, confidence below mode-specific floor,
            #     symbol or strategy or combo currently quarantined for a loss
            #     streak)
            #   - override position_usd with a mode-scaled approved_notional
            #     (SCOUT = 25% of proposed, BUILD = 100%, ATTACK = 120%, etc.)
            # When the flag is off, get_mission_controller() returns a no-op
            # that approves everything and leaves sizing as-is, so this block is
            # safe to land before the UI toggle exists.
            # =====================================================================
            mission = get_mission_controller()
            mission_inputs = _compute_mission_inputs(
                signal=signal,
                decision=decision,
                confidence=confidence,
                proposed_notional=position_usd,
                # Phase B: pass symbol/side/strategy so the helper can build
                # an autonomous TradeContext and look up the calibrated
                # historical win rate for this fingerprint. Without these,
                # the calibration falls back to raw-confidence and we get
                # the pre-Phase-B behavior.
                symbol=symbol,
                side=side,
                strategy_type=strategy_type,
            )
            mission_decision = mission.approve_trade(
                symbol=symbol,
                strategy=strategy_type,
                confidence=confidence,
                proposed_notional=position_usd,
                expected_net_edge=mission_inputs["expected_net_edge"],
                spread_bps=mission_inputs["spread_bps"],
                volatility_score=mission_inputs["volatility_score"],
                market_quality_score=mission_inputs["market_quality_score"],
                router_wants_claude=(decision.source == "claude"),
                metadata={
                    "strategy_type": strategy_type,
                    "decision_source": decision.source,
                    # Phase B: record which tier of the win-probability
                    # estimator backed this approval so post-hoc audits
                    # can separate "approved on real data" from "approved
                    # on a confidence guess".
                    "calibration_source": mission_inputs.get("win_probability_source"),
                    "calibration_sample_size": mission_inputs.get("win_probability_sample_size"),
                    "calibration_win_prob": mission_inputs.get("win_probability_used"),
                },
            )
            if not mission_decision.approved:
                # Include the calibration source + sample so the operator
                # can tell from the console whether this rejection was
                # backed by measured fingerprint data or just a confidence
                # guess. After Phase B + the fingerprint coarsening, we
                # want to see "exact_pattern(n=N)" appearing on more
                # rejections over time — that's evidence the learning loop
                # is closing.
                src = mission_inputs.get("win_probability_source", "?")
                ssz = mission_inputs.get("win_probability_sample_size", 0)
                self._log(
                    "bot",
                    f"[MISSION REJECT] {symbol} {side}: {mission_decision.rejection_code.value} "
                    f"— {mission_decision.reason} [calib={src} n={ssz}]",
                    wallet_id=wallet["id"],
                    level="info",
                )
                continue
            # If the controller is the real one (not the no-op), let it
            # override the position size. The mode multipliers in MissionConfig
            # are the authoritative sizing once the flag is on.
            if getattr(mission_decision, "approved_notional", None) is not None and mission_decision.approved_notional > 0:
                if mission_decision.approved_notional != position_usd:
                    position_usd = float(mission_decision.approved_notional)
                    qty = position_usd / price if price > 0 else 0.0
                    self._log(
                        "bot",
                        f"[MISSION SIZE] {symbol}: mode={mission_decision.mode.value} "
                        f"size ${size_result.recommended_usd:.0f} -> ${position_usd:.0f} (×{mission_decision.size_multiplier})",
                        wallet_id=wallet["id"],
                        level="info",
                    )

            if cfg.dry_run:
                self._log(
                    "bot",
                    (
                        f"[DRY RUN] {wallet['name']}: {decision.source} would {side} {qty} "
                        f"{symbol} @ {price:.4f} (conf={confidence:.2f}, sl={decision.stop_loss_pct:.2%}, "
                        f"tp={decision.take_profit_pct:.2%}) — {decision.rationale[:200]}"
                    ),
                    wallet_id=wallet["id"],
                    level="info",
                )
                result.actions += 1
                slots_left -= 1
                continue

            # Real path: open a paper trade through the existing engine.
            self._log(
                "bot",
                f"[EXECUTING] {wallet['name']}: {side} {qty:.6f} {symbol} @ ${price:.4f} (conf={confidence:.2f})",
                wallet_id=wallet["id"],
                level="info",
            )
            
            # Resolve the holding profile for this trade. The mode comes
            # from Settings; `mixed` tiers it by confidence; `ai_decide`
            # reads the holding intent out of the take-profit Claude
            # already chose. The resolved name governs every exit rule.
            from trading.holding_profiles import resolve_profile_name, profile_from_claude_targets
            ai_choice = None
            if cfg.holding_mode == "ai_decide":
                ai_choice = profile_from_claude_targets(getattr(decision, "take_profit_pct", None))
            holding_profile_name = resolve_profile_name(
                cfg.holding_mode, confidence, ai_choice
            )

            try:
                outcome = self.paper.open_trade(
                    wallet_id=wallet["id"],
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    entry_price=price,
                    confidence=confidence,
                    market_type="Crypto",
                    strategy_id=strategy_id,
                    holding_profile_name=holding_profile_name,
                    notes=(
                        f"bot/{decision.source}/{strategy_type}: {decision.rationale[:400]}"
                    ),
                    claude_decision_id=getattr(decision, "claude_decision_id", None),
                    # Persist the Phase B calibration tier so the scorecard
                    # on /training can show how many of today's trades were
                    # backed by measured pattern data vs raw confidence.
                    calibration_source=mission_inputs.get("win_probability_source"),
                    calibration_sample_size=mission_inputs.get("win_probability_sample_size"),
                )
            except Exception as e:
                # A single bad trade (attribute error, DB hiccup, etc.) must
                # NOT abort the whole wallet's tick. Log it and continue so
                # the rest of the universe still gets a chance.
                import traceback
                self._log(
                    "bot",
                    f"[OPEN_TRADE ERROR] {symbol} {side}: {e}\n{traceback.format_exc()[:500]}",
                    wallet_id=wallet["id"],
                    level="warn",
                )
                result.errors += 1
                continue

            if outcome.get("ok"):
                # Surface which calibration tier backed the decision so
                # the operator can audit live whether trades are being
                # approved on measured data (exact_pattern / knn) or on
                # raw confidence. Over a session the mix should shift
                # toward exact_pattern as fingerprints accumulate.
                src = mission_inputs.get("win_probability_source", "?")
                ssz = mission_inputs.get("win_probability_sample_size", 0)
                self._log(
                    "bot",
                    f"[TRADE OPENED] {symbol} {side} - trade_id={outcome.get('trade_id')} "
                    f"[calib={src} n={ssz}]",
                    wallet_id=wallet["id"],
                    level="info",
                )
                result.actions += 1
                slots_left -= 1
                # Exit thresholds (stop-loss, take-profit, trailing stop,
                # and the hard time cap) were all set by open_trade from
                # the resolved holding profile — the single source of
                # truth for how this trade exits. Surface it for the
                # operator so the training console shows which profile
                # each trade is running under.
                self._log(
                    "bot",
                    f"[PROFILE] {symbol}: holding profile '{holding_profile_name}' "
                    f"(mode={cfg.holding_mode})",
                    wallet_id=wallet["id"],
                    level="info",
                )
                self._log(
                    "trade",
                    (
                        f"OPEN {side} {qty} {symbol} @ {price:.4f} "
                        f"on {wallet['name']} (conf={confidence:.2f}, src={decision.source})"
                    ),
                    wallet_id=wallet["id"],
                    level="success",
                )
                try:
                    from services.notifier import notify
                    notify(
                        f"Opened {side} {qty} {symbol} @ {price:.4f} "
                        f"on {wallet['name']} (conf={confidence:.2f}, src={decision.source})",
                        level="info",
                        category="trade",
                    )
                except Exception:
                    pass
            else:
                result.skipped += 1
                self._log(
                    "trade",
                    (
                        f"REJECTED {side} {qty} {symbol} @ {price:.4f} on {wallet['name']}: "
                        f"{outcome.get('error') or outcome.get('reason') or 'unknown'}"
                    ),
                    wallet_id=wallet["id"],
                    level="warn",
                )

        # Per-wallet tick summary so the live console always has a heartbeat.
        if best:
            best_line = (
                f"best={best['symbol']} {best['side']} conf={best['confidence']:.2f}"
            )
        else:
            best_line = "no signals"
        self._log(
            "bot",
            (
                f"{wallet['name']}: evaluated {evaluated}/{len(universe)} symbols, "
                f"claude_calls={claude_calls}, floor={cfg.min_confidence:.2f}, below={below_conf}, "
                f"slots={slots_left}/{cap}, held={len(held_symbols)}, skipped_held={symbols_skipped_held}, "
                f"skipped_noslots={symbols_skipped_noslots}, {best_line}."
            ),
            wallet_id=wallet["id"],
            level="info",
        )

    # ------------------------------------------------------------------ #
    # Portfolio Intelligence
    # ------------------------------------------------------------------ #

    def _run_portfolio_intelligence(
        self,
        cfg: BotConfig,
        universe: list[dict[str, Any]],
        result: TickResult,
        price_map: dict[str, float] | None = None,
    ) -> int:
        """
        Proactive portfolio management - DCA, scale-in, offset trades.
        
        This is what makes the bot ACTIVE instead of passive.
        Instead of opening 3 positions and hoping they recover,
        we continuously look for ways to improve portfolio P&L.
        
        Returns the number of actions executed.
        """
        # Use the prefetched price_map if provided, otherwise build one minimally
        if not price_map:
            price_map = {}
            # Only fetch prices for symbols we actually need (open positions)
            with session_scope() as s:
                open_symbols = [p[0] for p in s.query(PaperTrade.symbol).filter(PaperTrade.status == "open").distinct().all()]
            for symbol in open_symbols:
                price_payload = get_price(symbol)
                if price_payload.get("ok"):
                    price_map[symbol] = float(price_payload["price"])
        
        if not price_map:
            return 0
        
        # Get all wallets with positions
        with session_scope() as s:
            wallet_ids = (
                s.query(PaperTrade.wallet_id)
                .filter(PaperTrade.status == "open")
                .distinct()
                .all()
            )
            wallet_ids = [w[0] for w in wallet_ids]
        
        actions_executed = 0
        
        for wallet_id in wallet_ids:
            # Generate portfolio improvement actions
            actions = self.portfolio_intel.generate_actions(
                wallet_id=wallet_id,
                price_map=price_map,
                universe=universe,
                cfg=cfg,
                max_actions=3,  # Max 3 actions per wallet per tick
            )
            
            for action in actions:
                self._log(
                    "portfolio",
                    f"Portfolio Intel: {action.action_type.upper()} {action.symbol} - {action.reason}",
                    wallet_id=wallet_id,
                    level="info",
                )
                
                # Execute the action
                outcome = execute_portfolio_action(
                    action=action,
                    wallet_id=wallet_id,
                    paper_engine=self.paper,
                    cfg=cfg,
                )
                
                if outcome.get("ok"):
                    actions_executed += 1
                    self._log(
                        "portfolio",
                        f"Portfolio Intel SUCCESS: {action.action_type} {action.symbol} executed",
                        wallet_id=wallet_id,
                        level="success",
                    )
                    
                    # Notify
                    try:
                        from services.notifier import notify
                        notify(
                            f"Portfolio Intel: {action.action_type.upper()} {action.symbol} - {action.reason[:100]}",
                            level="info",
                            category="portfolio_intel",
                        )
                    except Exception:
                        pass
                else:
                    self._log(
                        "portfolio",
                        f"Portfolio Intel FAILED: {action.action_type} {action.symbol} - {outcome.get('error')}",
                        wallet_id=wallet_id,
                        level="warn",
                    )
        
        return actions_executed

    # ------------------------------------------------------------------ #
    # Position Monitoring
    # ------------------------------------------------------------------ #

    def _monitor_positions(
        self,
        cfg: BotConfig,
        universe: list[dict[str, Any]],
        result: TickResult,
        price_map: dict[str, float] | None = None,
    ) -> int:
        """
        Check all open positions for auto-exit conditions (SL/TP/trailing/time).

        Returns the number of positions that were auto-closed.
        """
        # Use provided price_map or build one (for backwards compatibility)
        if price_map is None:
            price_map = {}
            for product in universe:
                symbol = product["product_id"]
                price_payload = get_price(symbol)
                if price_payload.get("ok"):
                    price_map[symbol] = float(price_payload["price"])

        # Ensure open position symbols have prices (fetch if not in map)
        with session_scope() as s:
            open_symbols = (
                s.query(PaperTrade.symbol)
                .filter(PaperTrade.status == "open")
                .distinct()
                .all()
            )
            open_symbols = [row[0] for row in open_symbols]
        
        for symbol in open_symbols:
            if symbol not in price_map:
                price_payload = get_price(symbol)
                if price_payload.get("ok"):
                    price_map[symbol] = float(price_payload["price"])

        # Fetch all wallets with open positions
        with session_scope() as s:
            wallet_ids = (
                s.query(PaperTrade.wallet_id)
                .filter(PaperTrade.status == "open")
                .distinct()
                .all()
            )
            wallet_ids = [w[0] for w in wallet_ids]

        closed_count = 0
        self._log("bot", f"[MONITOR] Checking {len(wallet_ids)} wallets, {len(price_map)} prices: {list(price_map.keys())[:10]}...", level="info")
        for wallet_id in wallet_ids:
            try:
                exits = self.position_monitor.check_all_positions(wallet_id, price_map)
                self._log("bot", f"[MONITOR] Wallet {wallet_id}: {len(exits)} exit signals", level="info")
                for exit_signal in exits:
                    self._log("bot", f"[MONITOR] Processing exit: {exit_signal.symbol} reason={exit_signal.reason}", level="info")
                    # Close the position through the paper engine
                    outcome = self.paper.close_trade(
                        trade_id=exit_signal.trade_id,
                        exit_price=exit_signal.current_price,
                        notes=f"auto-exit/{exit_signal.reason}: triggered at ${exit_signal.current_price:.4f}",
                    )
                    self._log("bot", f"[MONITOR] close_trade result: {outcome}", level="info")
                    if outcome.get("ok"):
                        closed_count += 1
                        result.actions += 1
                        # Update circuit breaker with trade result
                        pnl = outcome.get("realized_pnl", 0) or 0
                        self.record_trade_result(pnl)
                        # Feed the adaptive risk manager. This is the loop
                        # that closes the gap between "the learning engine
                        # noticed losses" and "the bot stopped taking the
                        # same kind of trades": rolling-WR drives the
                        # confidence floor, rolling-P&L drives the breaker,
                        # symbol stats drive the blacklist.
                        try:
                            from trading.adaptive_risk import AdaptiveRiskManager
                            AdaptiveRiskManager.record_trade(
                                symbol=exit_signal.symbol,
                                pnl=float(pnl),
                            )
                        except Exception as exc:
                            # A logging failure here must NEVER prevent the
                            # close from being committed — adaptive state
                            # is best-effort.
                            self._log("bot", f"[ADAPTIVE] record_trade failed: {exc}", level="warn")
                        # Update the exit_reason on the trade record
                        with session_scope() as s:
                            trade = s.query(PaperTrade).filter(PaperTrade.id == exit_signal.trade_id).first()
                            if trade:
                                trade.exit_reason = exit_signal.reason
                        # Log it
                        self._log(
                            "bot",
                            f"AUTO-EXIT ({exit_signal.reason}): {exit_signal.symbol} "
                            f"closed at ${exit_signal.current_price:.4f} "
                            f"(P&L: {exit_signal.pnl_pct:+.2%}, streak: {self._consecutive_losses}L)",
                            level="info",
                        )
                        # Notify
                        try:
                            from services.notifier import notify
                            notify(
                                f"Auto-exit ({exit_signal.reason}): {exit_signal.symbol} "
                                f"closed at ${exit_signal.current_price:.4f} "
                                f"(P&L: {exit_signal.pnl_pct:+.2%})",
                                level="info",
                                category="auto_exit",
                            )
                        except Exception:
                            pass
            except Exception as e:
                self._log("bot", f"[MONITOR] Error processing wallet {wallet_id}: {e}", level="error")

        return closed_count

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_snapshot(symbol: str, price: float) -> dict[str, Any]:
        # Cheap deterministic-ish placeholder values until a real feature pipeline
        # is wired in. The DecisionEngine only reads liquidity / volatility / probs.
        # Using stable mid-range numbers prevents the bot from acting purely on noise.
        return {
            "symbol": symbol,
            "platform": "Coinbase",
            "market_type": "Crypto",
            "current_price": price,
            "fair_value": price,
            "ai_probability": 0.55,
            "market_probability": 0.50,
            "liquidity": 0.7,
            "volatility": 0.4,
        }

    @staticmethod
    def _load_universe(cfg: BotConfig) -> list[dict[str, Any]]:
        if cfg.universe == "coinbase_usd":
            return coinbase_usd_universe(limit=cfg.universe_limit)
        # Future: other universes (e.g. equity, prediction markets).
        return coinbase_usd_universe(limit=cfg.universe_limit)

    def _note_skip(self, reason: str) -> TickResult:
        now = utcnow().isoformat()
        result = TickResult(
            started_at=now,
            ended_at=now,
            universe_size=0,
            wallets_evaluated=0,
            decisions=0,
            actions=0,
            skipped=1,
            errors=0,
            notes=[reason],
        )
        return self._record(result)

    def _record(self, result: TickResult) -> TickResult:
        self._recent_ticks.append(result)
        if len(self._recent_ticks) > self._max_recent:
            self._recent_ticks = self._recent_ticks[-self._max_recent:]
        return result

    @staticmethod
    def _log(category: str, message: str, wallet_id: int | None = None, level: str = "info") -> None:
        with session_scope() as s:
            s.add(
                ActivityLog(
                    category=category,
                    level=level,
                    wallet_id=wallet_id,
                    message=message,
                )
            )


# =============================================================================
# MISSION-CONTROLLER INPUT DERIVATION
# =============================================================================
# The Daily Mission Controller's approve_trade() expects four numeric inputs
# the rest of the pipeline doesn't currently track as first-class values:
#
#   expected_net_edge  - dollar edge expected after fees/slippage/loss-prob
#   spread_bps         - bid-ask spread in basis points
#   volatility_score   - normalized 0..1
#   market_quality_score - composite 0..1
#
# We synthesize them from the technical signal's indicators + the decision's
# SL/TP percentages. This is a lossy approximation — once we have live order
# book data the spread/quality scores should become real measurements. Until
# then the heuristics below give the controller plausible inputs so its
# mode/edge gates can do useful work in paper trading.

def _compute_mission_inputs(
    *,
    signal,
    decision,
    confidence: float,
    proposed_notional: float,
    symbol: str = "",
    side: str = "",
    strategy_type: str = "",
) -> dict:
    indicators = getattr(signal, "indicators", {}) or {}
    sl_pct = max(0.001, float(decision.stop_loss_pct or 0.025))
    tp_pct = max(0.001, float(decision.take_profit_pct or 0.05))

    # --- Calibrated win probability (Phase B) ----------------------------
    # The old version used `confidence` as a stand-in for "probability this
    # trade wins". That's a quality signal, not a calibrated probability.
    # Meanwhile the autonomous engine has been accumulating real
    # (fingerprint -> win_rate, expectancy, sample_size) data on every
    # closed trade. We now use that as the source of truth when we have
    # enough samples, and blend toward confidence when we don't.
    #
    # Resolution order (see autonomous_learning_engine.get_calibrated_win_probability):
    #   1. exact_pattern   - same fingerprint, n ≥ 5 closed trades
    #   2. knn_neighbors   - k similar trades by Euclidean distance, n ≥ 5
    #   3. raw_confidence  - no historical data; fall back to confidence
    #
    # We blend the measured estimate with raw confidence using the
    # estimate's own meta-confidence (which grows with sample size). At
    # n=0 the formula collapses to today's behavior; at n>=25 we trust
    # the data ~75%.
    win_prob_meta = {
        "source": "raw_confidence",
        "sample_size": 0,
        "win_probability": max(0.0, min(1.0, float(confidence))),
        "confidence_in_estimate": 0.0,
        "expectancy": None,
        "avg_win": None,
        "avg_loss": None,
    }
    try:
        from trading.strategic_claude import _build_autonomous_context
        from ai.autonomous_learning_engine import get_calibrated_win_probability

        ctx = _build_autonomous_context(
            symbol=symbol or getattr(signal, "symbol", "") or "?",
            side=side or getattr(signal, "side", "") or "BUY",
            technical_signal=signal,
            strategy_type=strategy_type or getattr(signal, "strategy", "Momentum"),
            tech_confidence=float(confidence),
        )
        if ctx is not None:
            win_prob_meta = get_calibrated_win_probability(ctx, float(confidence))
    except Exception:
        # Calibration is best-effort. If anything in the lookup chain
        # fails we keep the raw-confidence default already in win_prob_meta
        # and trade exactly the way we did before Phase B.
        pass

    measured = max(0.0, min(1.0, float(win_prob_meta["win_probability"])))
    meta_conf = max(0.0, min(1.0, float(win_prob_meta["confidence_in_estimate"])))
    raw_conf = max(0.0, min(1.0, float(confidence)))
    prob_win = meta_conf * measured + (1.0 - meta_conf) * raw_conf

    # --- Expected net edge (dollars) -------------------------------------
    # When we have a calibrated exact_pattern estimate with measured
    # avg_win / avg_loss magnitudes, use those instead of the decision's
    # tp_pct / sl_pct (which are forward targets, not realized averages).
    # The autonomous engine's averages already net out exits that hit
    # before TP, partial fills, etc. — they're closer to reality.
    avg_win_pct = win_prob_meta.get("avg_win")
    avg_loss_pct = win_prob_meta.get("avg_loss")
    if (
        win_prob_meta.get("source") == "exact_pattern"
        and avg_win_pct is not None
        and avg_loss_pct is not None
        and avg_win_pct > 0
        and avg_loss_pct > 0
    ):
        win_magnitude_pct = float(avg_win_pct)
        loss_magnitude_pct = float(avg_loss_pct)
    else:
        win_magnitude_pct = tp_pct
        loss_magnitude_pct = sl_pct

    expected_win_pnl = prob_win * win_magnitude_pct * proposed_notional
    expected_loss_pnl = (1.0 - prob_win) * loss_magnitude_pct * proposed_notional
    fee_drag = 0.005 * proposed_notional  # 50 bps round-trip
    expected_net_edge = expected_win_pnl - expected_loss_pnl - fee_drag

    # --- Volatility score (0..1) -----------------------------------------
    # atr_pct measured fractionally (e.g. 0.02 = 2%). Map 0% -> 0.0,
    # 5% -> 1.0, clamp. Anything above 5% atr is effectively "extreme".
    atr_pct = float(indicators.get("atr_pct") or 0.0)
    volatility_score = max(0.0, min(1.0, atr_pct / 0.05))

    # --- Spread (bps) ----------------------------------------------------
    # We do not yet track bid-ask spread on the paper side — every fill is
    # at the polled mid price. Until we wire WebSocket order books, we
    # approximate "effective spread cost" as half the atr in bps (volatile
    # markets have wider spreads on Coinbase). Conservative bias, but the
    # mode-specific max_spread_bps values are forgiving enough on Scout
    # (12bps) that this won't deadlock training sessions.
    spread_bps = min(40.0, atr_pct * 100.0 * 5.0)  # atr_pct=2% -> 10 bps

    # --- Market quality (0..1) -------------------------------------------
    # Composite of (a) volume health, (b) trend strength, (c) signal
    # confidence. Built so reasonable markets score 0.6+ and only truly
    # poor conditions drop below the protect_min_market_quality of 0.76.
    vol_ratio = float(indicators.get("relative_volume") or indicators.get("volume_ratio") or 1.0)
    adx = float(indicators.get("adx") or indicators.get("adx_trend_strength") or 0.0)
    q = 0.30
    q += min(0.25, max(0.0, (vol_ratio - 0.5) / 1.5) * 0.25)  # 0.5x->0, 2.0x->+0.25
    q += min(0.20, max(0.0, (adx - 15.0) / 20.0) * 0.20)       # ADX 15->0, 35+->+0.20
    q += min(0.25, max(0.0, (confidence - 0.4)) * 0.50)        # conf 0.4->0, 0.9->+0.25
    market_quality_score = max(0.0, min(1.0, q))

    return {
        "expected_net_edge": round(expected_net_edge, 4),
        "spread_bps": round(spread_bps, 2),
        "volatility_score": round(volatility_score, 4),
        "market_quality_score": round(market_quality_score, 4),
        # Phase B observability: which estimator backed the edge math?
        # Surfaced so the mission decision's metadata records (and the
        # operator can audit) whether a given approval was based on
        # measured data or a confidence guess.
        "win_probability_source": win_prob_meta.get("source", "raw_confidence"),
        "win_probability_sample_size": int(win_prob_meta.get("sample_size") or 0),
        "win_probability_used": round(prob_win, 4),
    }


# Module-level singleton so the scheduler and HTTP routes share state.
bot_engine = BotEngine()
