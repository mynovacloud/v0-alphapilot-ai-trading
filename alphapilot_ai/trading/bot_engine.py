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
from trading.position_monitor import PositionMonitor, initialize_trade_sl_tp
from trading.risk_manager import RiskManager
from trading.strategy_engine import evaluate_symbol
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
            symbols_needed = [u.symbol for u in universe]
            with session_scope() as s:
                open_positions = s.query(PaperTrade.symbol).filter(PaperTrade.status == "open").distinct().all()
                symbols_needed.extend([p[0] for p in open_positions])
            
            # Batch fetch all prices at once (populates cache)
            price_map = get_prices_batch(list(set(symbols_needed)))
            self._log("bot", f"Prefetched {len(price_map)} prices", level="debug")
        except Exception as e:
            self._log("bot", f"Price prefetch failed: {e}", level="warn")
            price_map = {}

        # -----------------------------------------------------------------
        # POSITION MONITORING: Check all open positions for auto-exits
        # before evaluating new entries. This ensures SL/TP/trailing stops
        # are processed at the same frequency as new entry signals.
        # -----------------------------------------------------------------
        auto_exits = self._monitor_positions(cfg, universe, result, price_map)
        if auto_exits > 0:
            result.notes.append(f"auto_exits={auto_exits}")

        # -----------------------------------------------------------------
        # PORTFOLIO INTELLIGENCE: Proactive portfolio management.
        # This is where the "smart" behavior happens - instead of passively
        # waiting for losing positions to recover, we:
        #   1. DCA into losers at better prices
        #   2. Scale into winners that keep working
        #   3. Open offset trades to balance underwater positions
        # This runs EVERY tick, regardless of slot availability.
        # -----------------------------------------------------------------
        intel_actions = self._run_portfolio_intelligence(cfg, universe, result)
        if intel_actions > 0:
            result.notes.append(f"portfolio_intel_actions={intel_actions}")

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
                result.errors += 1
                result.notes.append(f"wallet_{wallet['id']}_error: {e}")
                self._log(
                    "bot",
                    f"Wallet {wallet['name']} (#{wallet['id']}) tick error: {e}",
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

        # Track best per-tick candidate so we always log a useful summary
        # even when nothing crosses the confidence threshold.
        best: dict[str, Any] | None = None
        evaluated = 0
        below_conf = 0
        claude_calls = 0  # Track how many times we call Claude

        # Sweep the universe. DON'T break when slots are full - we still want to
        # call Claude for existing positions (re-evaluate, DCA opportunities, etc.)
        for product in universe:
            symbol = product["product_id"]
            
            # For NEW entries, skip if we already hold AND have no slots
            # But if we DO have slots, we should diversify into NEW symbols
            is_held = symbol in held_symbols
            
            # Skip this symbol for NEW entries if:
            # 1. We already hold it (diversification), OR
            # 2. We have no slots left AND we don't hold it (can't act anyway)
            if is_held:
                # We might want to DCA - let portfolio intelligence handle that
                continue
            if slots_left <= 0:
                # No slots for new positions, but keep evaluating for logging/monitoring
                # Continue evaluating a few more for "best signal" tracking
                if evaluated >= 10:  # Evaluate at least 10 symbols even if slots full
                    break
            
            price_payload = get_price(symbol)
            if not price_payload.get("ok"):
                result.skipped += 1
                continue
            price = float(price_payload["price"])
            if price <= 0:
                result.skipped += 1
                continue

            # Compute the REAL signal from Coinbase candle data instead of a
            # synthetic snapshot. evaluate_symbol picks the strategy implementation
            # (Momentum / Mean Reversion / Volatility Breakout / Probability Edge)
            # and returns a Signal with confidence, reasoning, and indicators.
            # Granularity follows the tick rate so a 2s real-time session uses
            # 60s bars instead of stale 5-minute data.
            signal = evaluate_symbol(symbol, strategy_type, tick_seconds=cfg.tick_seconds)
            evaluated += 1
            result.decisions += 1

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

            # Hand the technical signal to Claude for the FINAL decision.
            claude_calls += 1
            decision = claude_decide(
                wallet=wallet,
                symbol=symbol,
                price=price,
                technical_signal=signal,
                strategy_type=strategy_type,
            )

            side = decision.action
            confidence = float(decision.confidence or 0.0)

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
            if confidence < cfg.min_confidence:
                below_conf += 1
                continue

            # Sizing: Claude's size_multiplier scales the bot's default size,
            # then we clamp to the wallet's hard cap.
            base_position_usd = min(
                cfg.position_size_usd,
                wallet["max_position_usd"] or cfg.position_size_usd,
            )
            position_usd = max(0.0, base_position_usd * float(decision.size_multiplier or 1.0))
            qty = round(position_usd / price, 6)
            if qty <= 0:
                continue

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
            outcome = self.paper.open_trade(
                wallet_id=wallet["id"],
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=price,
                confidence=confidence,
                market_type="Crypto",
                strategy_id=strategy_id,
                notes=(
                    f"bot/{decision.source}/{strategy_type}: {decision.rationale[:400]}"
                ),
            )

            if outcome.get("ok"):
                result.actions += 1
                slots_left -= 1
                # Initialize SL/TP based on Claude's recommendation
                trade_id = outcome.get("trade_id")
                if trade_id:
                    with session_scope() as s:
                        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
                        if trade:
                            initialize_trade_sl_tp(
                                trade,
                                stop_loss_pct=decision.stop_loss_pct,
                                take_profit_pct=decision.take_profit_pct,
                                trailing_stop_pct=None,  # Can be enabled later
                                max_loss_pct=0.10,
                                time_limit_hours=None,
                            )
                            s.commit()
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
                f"slots={cap - open_count}/{cap}, held={len(held_symbols)}, {best_line}."
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
    ) -> int:
        """
        Proactive portfolio management - DCA, scale-in, offset trades.
        
        This is what makes the bot ACTIVE instead of passive.
        Instead of opening 3 positions and hoping they recover,
        we continuously look for ways to improve portfolio P&L.
        
        Returns the number of actions executed.
        """
        # Build price map
        price_map: dict[str, float] = {}
        for product in universe:
            symbol = product["product_id"]
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
                            f"(P&L: {exit_signal.pnl_pct:+.2%})",
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


# Module-level singleton so the scheduler and HTTP routes share state.
bot_engine = BotEngine()
