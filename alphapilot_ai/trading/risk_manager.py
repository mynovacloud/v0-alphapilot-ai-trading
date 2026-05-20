"""
Risk manager — central middleware that gates EVERY trade attempt.

Used by:
  - PaperTradingEngine.open_trade
  - LiveTradingEngine._dispatch
  - BotEngine (indirectly, via the engines)

The same `evaluate_trade()` is called from all paths, so paper and live obey
the same rules. When something rejects a trade, the reason string is stored
on the activity log so you can see WHY the bot refused to act.

Layered checks (in order):
  1. Global kill switch (AppSetting `kill_switch=true`)            -> hard stop, all wallets
  2. Wallet bot_paused                                              -> hard stop, this wallet
  3. Daily-loss circuit breaker for the wallet (auto-pauses wallet) -> hard stop + pause
  4. Post-loss cooldown (after N consecutive losses, freeze N min)  -> soft stop, time-based
  5. Wallet caps (max_position_usd, max_open_positions,
                  max_daily_trades, max_daily_loss_usd)             -> hard stop
  6. Paper-balance affordability (paper-only)                       -> hard stop (paper)
  7. Strategy caps (max_position_size, min_confidence, max_open_trades, max_daily_loss)
                                                                    -> hard stop

All checks are conservative-by-default: any failure returns
`RiskDecision(allowed=False, reason=...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from database.db import session_scope
from database.models import (
    ActivityLog,
    AppSetting,
    LiveOrder,
    PaperTrade,
    Strategy,
    Wallet,
)
from utils.helpers import utcnow, ensure_utc
from utils.logger import get_logger

logger = get_logger(__name__)


# Tunable cooldown rules (could be moved into AppSetting later)
COOLDOWN_AFTER_N_LOSSES = 3        # 3 losing trades in a row...
COOLDOWN_MINUTES = 15              # ...locks the wallet out for 15 minutes


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    # Optional: which check rejected the trade ("kill_switch", "wallet_cap", ...).
    code: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """Central trade-gate. Both paper and live engines call `evaluate_trade()`."""

    # Class-level flag to bypass cooldown during training sessions
    _training_mode_bypass = False

    @classmethod
    def set_training_mode(cls, enabled: bool) -> None:
        """Enable/disable training mode which bypasses cooldown checks."""
        cls._training_mode_bypass = enabled
        logger.info(f"[RISK] Training mode bypass: {enabled}")

    @classmethod
    def is_training_mode(cls) -> bool:
        """Check if training mode bypass is active."""
        return cls._training_mode_bypass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate_trade(
        self,
        wallet_id: int,
        qty: float,
        entry_price: float,
        confidence: float,
        strategy_id: int | None = None,
        *,
        is_paper: bool = True,
        leverage: float = 1.0,
        is_perp: bool = False,
    ) -> RiskDecision:
        """Run every gate in order. First failure short-circuits."""
        # 1. Global kill switch
        if self._kill_switch_on():
            return RiskDecision(False, "Global kill switch is engaged.", "kill_switch")

        notional = max(0.0, qty * entry_price)

        with session_scope() as s:
            wallet: Wallet | None = s.get(Wallet, wallet_id)
            if not wallet:
                return RiskDecision(False, "Wallet not found", "wallet_missing")

            # 2. Per-wallet pause flag (bypass in training mode)
            if wallet.bot_paused and not self._training_mode_bypass:
                return RiskDecision(False, "Wallet bot is paused.", "wallet_paused")

            # 3. Daily-loss circuit breaker (bypass in training mode)
            #    If breached outside training, auto-pause the wallet.
            if not self._training_mode_bypass:
                tripped, day_loss = self._daily_loss_tripped(s, wallet, is_paper=is_paper)
                if tripped:
                    self._auto_pause(s, wallet, reason=f"daily loss ${day_loss:,.2f} hit cap")
                    return RiskDecision(
                        False,
                        f"Daily loss cap hit (${day_loss:,.2f}). Wallet auto-paused.",
                        "daily_loss_breaker",
                    )

            # 4. Cooldown after consecutive losses (bypass in training mode)
            if not self._training_mode_bypass:
                cooldown_until = self._cooldown_until(s, wallet, is_paper=is_paper)
                if cooldown_until is not None:
                    return RiskDecision(
                        False,
                        f"Wallet in post-loss cooldown until {cooldown_until.isoformat()}",
                        "cooldown",
                    )

            # 5. Wallet caps
            if wallet.max_position_usd and notional > wallet.max_position_usd:
                return RiskDecision(
                    False,
                    f"Notional ${notional:,.2f} > wallet cap ${wallet.max_position_usd:,.2f}",
                    "wallet_position_cap",
                )

            # 5b. Futures / leverage gates
            if is_perp:
                if not getattr(wallet, "futures_enabled", False):
                    return RiskDecision(
                        False,
                        "Wallet has futures_enabled=False; cannot place perp order.",
                        "futures_disabled",
                    )
                wallet_max_lev = float(getattr(wallet, "max_leverage", 1.0) or 1.0)
                if leverage > wallet_max_lev:
                    return RiskDecision(
                        False,
                        f"Leverage {leverage}x > wallet max {wallet_max_lev}x.",
                        "leverage_cap",
                    )
            elif leverage and leverage > 1.0:
                # Spot trade asking for leverage — refuse rather than silently ignore.
                return RiskDecision(
                    False,
                    "Spot order cannot use leverage > 1.",
                    "spot_leverage_rejected",
                )

            open_count_paper, todays_count_paper = self._wallet_paper_today(s, wallet_id)
            open_count_live, todays_count_live = self._wallet_live_today(s, wallet_id)
            open_count = open_count_paper + open_count_live
            todays_count = todays_count_paper + todays_count_live

            # Use the HIGHER of wallet limit and config limit to allow session overrides.
            # During training sessions, the config setting should take precedence.
            config_max_open = self._get_config_max_open()
            effective_max_open = max(
                wallet.max_open_positions or 3,
                config_max_open or 3
            )
            
            logger.debug(
                f"[RISK_CHECK] wallet_id={wallet_id}: open={open_count}, "
                f"wallet_limit={wallet.max_open_positions}, config_limit={config_max_open}, "
                f"effective_limit={effective_max_open}"
            )
            
            if open_count >= effective_max_open:
                return RiskDecision(
                    False,
                    f"Wallet already has {open_count} open positions (cap {effective_max_open}).",
                    "wallet_open_cap",
                )

            if wallet.max_daily_trades and todays_count >= wallet.max_daily_trades:
                return RiskDecision(
                    False,
                    f"Wallet hit max_daily_trades={wallet.max_daily_trades}.",
                    "wallet_daily_count",
                )

            # 6. Paper-balance affordability (paper layer only)
            if is_paper and notional > (wallet.paper_balance or 0.0):
                return RiskDecision(
                    False,
                    f"Notional ${notional:,.2f} > paper balance ${wallet.paper_balance:,.2f}",
                    "insufficient_paper_balance",
                )

            # 7. Strategy caps
            strat: Strategy | None = s.get(Strategy, strategy_id) if strategy_id else None
            if strat:
                if strat.max_position_size and notional > strat.max_position_size:
                    return RiskDecision(
                        False,
                        f"Notional > strategy max_position_size ({strat.max_position_size})",
                        "strategy_position_cap",
                    )
                if confidence < (strat.min_confidence or 0):
                    # Training-session escape hatch: when the operator is on
                    # the AI Training page they explicitly set a global
                    # confidence floor (often 0.0) to see lots of trades.
                    # Without this override, the strategy table's seeded
                    # min_confidence (0.55-0.65) silently vetoes every
                    # training trade — that's the "Settings page wins over
                    # Training page" behaviour the operator reported.
                    training_floor = self._training_session_floor()
                    if training_floor is None or confidence < training_floor:
                        return RiskDecision(
                            False,
                            f"Confidence {confidence:.2f} < strategy min {strat.min_confidence:.2f}",
                            "strategy_confidence",
                        )

                strategy_today = (
                    s.query(PaperTrade)
                    .filter(
                        PaperTrade.wallet_id == wallet_id,
                        PaperTrade.strategy_id == strategy_id,
                    )
                    .all()
                )
                strat_day_pnl = sum(
                    t.realized_pnl
                    for t in strategy_today
                    if t.closed_at and t.closed_at.date() == utcnow().date()
                )
                if strat.max_daily_loss and strat_day_pnl <= -abs(strat.max_daily_loss):
                    return RiskDecision(
                        False,
                        f"Strategy daily loss reached ({strat_day_pnl:.2f})",
                        "strategy_daily_loss",
                    )

                strat_open = sum(1 for t in strategy_today if t.status == "open")
                if strat.max_open_trades and strat_open >= strat.max_open_trades:
                    return RiskDecision(
                        False,
                        f"Strategy at max_open_trades={strat.max_open_trades}",
                        "strategy_open_cap",
                    )

        return RiskDecision(True, "ok", "ok")

    # Backwards-compat shim: existing callers use `.evaluate(...)`.
    def evaluate(
        self,
        wallet_id: int,
        qty: float,
        entry_price: float,
        confidence: float,
        strategy_id: int | None = None,
    ) -> RiskDecision:
        result = self.evaluate_trade(
            wallet_id=wallet_id,
            qty=qty,
            entry_price=entry_price,
            confidence=confidence,
            strategy_id=strategy_id,
            is_paper=True,
        )
        
        # Log ALL rejections to activity log for debug console
        if not result.allowed:
            try:
                from database.db import session_scope
                from database.models import ActivityLog
                with session_scope() as s:
                    s.add(ActivityLog(
                        category="risk",
                        level="warn",
                        message=f"RISK REJECTED: {result.reason} (code={result.code}) - qty={qty}, price={entry_price}, conf={confidence}",
                        wallet_id=wallet_id,
                    ))
            except Exception:
                pass  # Don't let logging failures break the flow
        
        return result

    # ------------------------------------------------------------------ #
    # Kill switch (global, AppSetting-backed)
    # ------------------------------------------------------------------ #

    KILL_SWITCH_KEY = "kill_switch"

    @classmethod
    def _kill_switch_on(cls) -> bool:
        with session_scope() as s:
            row = s.query(AppSetting).filter(AppSetting.key == cls.KILL_SWITCH_KEY).first()
            return bool(row and (row.value or "").strip().lower() in {"1", "true", "on", "yes"})

    @classmethod
    def _training_session_floor(cls) -> float | None:
        """If a training session is active, return the operator's confidence
        floor. Otherwise return None and let the strategy's own min_confidence
        win. This is what makes the AI Training page's slider authoritative
        during a training session — without it, the strategy table's seeded
        floors silently veto the operator's intent.
        """
        with session_scope() as s:
            active_row = s.query(AppSetting).filter(AppSetting.key == "training_session_active").first()
            active = bool(active_row and (active_row.value or "").strip().lower() in {"1", "true", "on", "yes"})
            if not active:
                return None
            floor_row = s.query(AppSetting).filter(AppSetting.key == "bot_min_confidence").first()
            raw = (floor_row.value if floor_row else "") or ""
            if raw.strip() == "":
                return None
            try:
                return float(raw)
            except ValueError:
                return None

    @classmethod
    def _get_config_max_open(cls) -> int:
        """Get the config setting for max_open_per_wallet.
        This allows training sessions to override wallet-level caps.
        Uses BotConfig to ensure we get the latest setting including session overrides.
        """
        from config.bot_config import BotConfig
        cfg = BotConfig.load()
        return cfg.max_open_per_wallet

    @classmethod
    def set_kill_switch(cls, on: bool, *, reason: str = "") -> None:
        """Engage / release the global kill switch and log it."""
        from utils.helpers import utcnow as _now

        with session_scope() as s:
            row = s.query(AppSetting).filter(AppSetting.key == cls.KILL_SWITCH_KEY).first()
            value = "true" if on else "false"
            if row:
                row.value = value
                row.updated_at = _now()
            else:
                s.add(AppSetting(key=cls.KILL_SWITCH_KEY, value=value))

            s.add(
                ActivityLog(
                    category="risk",
                    level="warn" if on else "info",
                    message=(
                        f"KILL SWITCH ENGAGED ({reason})" if on
                        else f"Kill switch released ({reason or 'manual'})."
                    ),
                )
            )

        # Notify externally — never let a notifier failure break the kill switch.
        try:
            from services.notifier import notify
            if on:
                notify(f"KILL SWITCH ENGAGED — {reason or 'manual'}", level="error", category="risk")
            else:
                notify(f"Kill switch released ({reason or 'manual'}).", level="warn", category="risk")
        except Exception:
            pass

    @classmethod
    def kill_switch_status(cls) -> bool:
        return cls._kill_switch_on()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wallet_paper_today(s, wallet_id: int) -> tuple[int, int]:
        today = utcnow().date()
        rows: Iterable[PaperTrade] = (
            s.query(PaperTrade).filter(PaperTrade.wallet_id == wallet_id).all()
        )
        rows = list(rows)
        open_count = sum(1 for t in rows if t.status == "open")
        todays = sum(1 for t in rows if t.opened_at and t.opened_at.date() == today)
        return open_count, todays

    @staticmethod
    def _wallet_live_today(s, wallet_id: int) -> tuple[int, int]:
        today = utcnow().date()
        rows: Iterable[LiveOrder] = (
            s.query(LiveOrder).filter(LiveOrder.wallet_id == wallet_id).all()
        )
        rows = list(rows)
        open_count = sum(1 for o in rows if o.status in {"open", "pending_submit", "partially_filled"})
        todays = sum(1 for o in rows if o.submitted_at and o.submitted_at.date() == today)
        return open_count, todays

    @staticmethod
    def _daily_loss_tripped(s, wallet: Wallet, *, is_paper: bool) -> tuple[bool, float]:
        """Return (tripped?, day_pnl). Aggregates paper + live realized PnL today."""
        today = utcnow().date()
        cap = abs(wallet.max_daily_loss_usd or 0.0)
        if cap <= 0:
            return False, 0.0

        paper_pnl = sum(
            (t.realized_pnl or 0.0)
            for t in s.query(PaperTrade).filter(PaperTrade.wallet_id == wallet.id).all()
            if t.closed_at and t.closed_at.date() == today
        )
        live_pnl = sum(
            (o.realized_pnl or 0.0)
            for o in s.query(LiveOrder).filter(LiveOrder.wallet_id == wallet.id).all()
            if o.closed_at and o.closed_at.date() == today
        )
        day_pnl = paper_pnl + live_pnl
        return (day_pnl <= -cap), day_pnl

    @staticmethod
    def _cooldown_until(s, wallet: Wallet, *, is_paper: bool):
        """
        If the wallet just took N consecutive losses, lock it out for
        COOLDOWN_MINUTES from the most-recent loss.
        Returns the unlock time, or None if no cooldown is active.
        """
        recent: list = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet.id, PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc())
            .limit(COOLDOWN_AFTER_N_LOSSES)
            .all()
        )
        if len(recent) < COOLDOWN_AFTER_N_LOSSES:
            return None
        if not all((t.realized_pnl or 0.0) < 0 for t in recent):
            return None
        last_close = recent[0].closed_at
        if not last_close:
            return None
        # Ensure timezone-aware comparison (DB may store naive datetimes)
        last_close_utc = ensure_utc(last_close)
        unlock = last_close_utc + timedelta(minutes=COOLDOWN_MINUTES)
        if utcnow() >= unlock:
            return None
        return unlock

    @staticmethod
    def _auto_pause(s, wallet: Wallet, *, reason: str) -> None:
        if wallet.bot_paused:
            return
        wallet.bot_paused = True
        s.add(
            ActivityLog(
                category="risk",
                level="warn",
                wallet_id=wallet.id,
                message=f"Auto-pausing wallet '{wallet.name}': {reason}",
            )
        )
