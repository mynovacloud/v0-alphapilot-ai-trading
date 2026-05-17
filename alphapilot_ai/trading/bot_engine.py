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
        # POSITION MONITORING: Check all open positions for auto-exits
        # before evaluating new entries. This ensures SL/TP/trailing stops
        # are processed at the same frequency as new entry signals.
        # -----------------------------------------------------------------
        auto_exits = self._monitor_positions(cfg, universe, result)
        if auto_exits > 0:
            result.notes.append(f"auto_exits={auto_exits}")

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
            # Pick the wallet's "default" strategy if any (the first one assigned to it).
            strat = (
                s.query(Strategy)
                .filter(Strategy.wallet_id == wallet["id"])
                .order_by(Strategy.id.asc())
                .first()
            )
            strategy_id = strat.id if strat else None
            strategy_type = strat.strategy_type if strat else cfg.default_strategy_type

        cap = min(wallet["max_open_positions"] or 0, cfg.max_open_per_wallet)
        slots_left = max(0, cap - open_count)
        if slots_left <= 0:
            result.skipped += 1
            self._log(
                "bot",
                f"{wallet['name']}: skipped (no open slots — {open_count}/{cap} used).",
                wallet_id=wallet["id"],
                level="info",
            )
            return

        # Track best per-tick candidate so we always log a useful summary
        # even when nothing crosses the confidence threshold.
        best: dict[str, Any] | None = None
        evaluated = 0
        below_conf = 0

        # Sweep the universe. Stop once we've used all available slots.
        for product in universe:
            if slots_left <= 0:
                break

            symbol = product["product_id"]
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

            # Skip the LLM round-trip when we have nothing useful to ask about.
            # A truly empty technical signal (HOLD with zero confidence and no
            # computed indicators) gives Claude nothing to reason from — it
            # will respond "no indicators provided, HOLD" 100% of the time.
            # That wastes paid API calls AND fills the decision log with
            # noise that masks any real BUY/SELL signals.
            #
            # We always skip these — the previous version gated on
            # `cfg.min_confidence > 0.05`, but a training session at floor 0.0
            # would NOT skip and would dump dozens of useless HOLDs into the
            # decision log every tick.
            if (
                signal.side == "HOLD"
                and float(signal.confidence or 0) <= 0.0
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
                f"floor={cfg.min_confidence:.2f}, below={below_conf}, "
                f"slots={cap - open_count}/{cap}, {best_line}."
            ),
            wallet_id=wallet["id"],
            level="info",
        )

    # ------------------------------------------------------------------ #
    # Position Monitoring
    # ------------------------------------------------------------------ #

    def _monitor_positions(
        self,
        cfg: BotConfig,
        universe: list[dict[str, Any]],
        result: TickResult,
    ) -> int:
        """
        Check all open positions for auto-exit conditions (SL/TP/trailing/time).

        Returns the number of positions that were auto-closed.
        """
        # Build price map from universe for quick lookup
        price_map: dict[str, float] = {}
        for product in universe:
            symbol = product["product_id"]
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
        for wallet_id in wallet_ids:
            exits = self.position_monitor.check_all_positions(wallet_id, price_map)
            for exit_signal in exits:
                # Close the position through the paper engine
                outcome = self.paper.close_trade(
                    trade_id=exit_signal.trade_id,
                    exit_price=exit_signal.current_price,
                    notes=f"auto-exit/{exit_signal.reason}: triggered at ${exit_signal.current_price:.4f}",
                )
                if outcome.get("ok"):
                    closed_count += 1
                    # Update the exit_reason on the trade record
                    with session_scope() as s:
                        trade = s.query(PaperTrade).filter(PaperTrade.id == exit_signal.trade_id).first()
                        if trade:
                            trade.exit_reason = exit_signal.reason
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
