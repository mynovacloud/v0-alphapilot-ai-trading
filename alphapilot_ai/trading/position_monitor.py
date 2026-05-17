"""
Position Monitor - Auto-exit engine for open positions.

Runs every tick to check if any open positions have hit their:
  - Stop-loss price
  - Take-profit price
  - Trailing stop price
  - Max loss percentage
  - Time limit

Also updates trailing stops as price moves favorably.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from database.db import session_scope
from database.models import PaperTrade, ActivityLog
from utils.helpers import utcnow

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class ExitSignal:
    """Represents a position that should be closed."""
    trade_id: int
    symbol: str
    reason: str  # "sl" / "tp" / "trailing" / "max_loss" / "time"
    current_price: float
    trigger_price: float | None
    pnl_pct: float


class PositionMonitor:
    """
    Monitors open positions and emits exit signals when thresholds are hit.

    Usage in bot_engine tick loop:
        monitor = PositionMonitor()
        exits = monitor.check_all_positions(wallet_id, price_map)
        for exit in exits:
            paper_engine.close_trade(exit.trade_id, exit.current_price, exit.reason)
    """

    def __init__(self, default_max_loss_pct: float = 0.10):
        self.default_max_loss_pct = default_max_loss_pct

    def check_all_positions(
        self,
        wallet_id: int,
        price_map: dict[str, float],
    ) -> list[ExitSignal]:
        """
        Check all open positions for a wallet against current prices.

        Args:
            wallet_id: The wallet to check positions for
            price_map: Dict of symbol -> current_price

        Returns:
            List of ExitSignal objects for positions that should be closed
        """
        exits: list[ExitSignal] = []

        with session_scope() as s:
            open_trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.wallet_id == wallet_id)
                .filter(PaperTrade.status == "open")
                .all()
            )

            for trade in open_trades:
                current_price = price_map.get(trade.symbol)
                if current_price is None:
                    continue

                exit_signal = self._check_single_position(s, trade, current_price)
                if exit_signal:
                    exits.append(exit_signal)
                else:
                    # No exit needed - update trailing stop if applicable
                    self._update_trailing_stop(s, trade, current_price)

            s.commit()

        return exits

    def _check_single_position(
        self,
        session: "Session",
        trade: PaperTrade,
        current_price: float,
    ) -> ExitSignal | None:
        """
        Check if a single position should be closed.

        Returns ExitSignal if position should close, None otherwise.
        """
        entry = float(trade.entry_price)
        qty = float(trade.qty)
        side = trade.side.upper()

        # Calculate P&L percentage
        if side == "BUY":
            pnl_pct = (current_price - entry) / entry if entry > 0 else 0
        else:  # SELL / SHORT
            pnl_pct = (entry - current_price) / entry if entry > 0 else 0

        # 1. Check max loss (hard cap)
        max_loss = float(trade.max_loss_pct or self.default_max_loss_pct)
        if pnl_pct <= -max_loss:
            self._log_exit(session, trade, "max_loss", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="max_loss",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # 2. Check stop-loss price
        if trade.stop_loss_price:
            sl_price = float(trade.stop_loss_price)
            if side == "BUY" and current_price <= sl_price:
                self._log_exit(session, trade, "sl", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="sl",
                    current_price=current_price,
                    trigger_price=sl_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price >= sl_price:
                self._log_exit(session, trade, "sl", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="sl",
                    current_price=current_price,
                    trigger_price=sl_price,
                    pnl_pct=pnl_pct,
                )

        # 3. Check trailing stop price
        if trade.trailing_stop_price:
            trailing_price = float(trade.trailing_stop_price)
            if side == "BUY" and current_price <= trailing_price:
                self._log_exit(session, trade, "trailing", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="trailing",
                    current_price=current_price,
                    trigger_price=trailing_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price >= trailing_price:
                self._log_exit(session, trade, "trailing", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="trailing",
                    current_price=current_price,
                    trigger_price=trailing_price,
                    pnl_pct=pnl_pct,
                )

        # 4. Check take-profit price
        if trade.take_profit_price:
            tp_price = float(trade.take_profit_price)
            if side == "BUY" and current_price >= tp_price:
                self._log_exit(session, trade, "tp", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="tp",
                    current_price=current_price,
                    trigger_price=tp_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price <= tp_price:
                self._log_exit(session, trade, "tp", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="tp",
                    current_price=current_price,
                    trigger_price=tp_price,
                    pnl_pct=pnl_pct,
                )

        # 5. Check time limit (only if position is flat or losing)
        # DEFAULT: If no time limit set, use 24 hours for paper trades
        time_limit = float(trade.time_limit_hours) if trade.time_limit_hours else 24.0
        if trade.opened_at:
            deadline = trade.opened_at + timedelta(hours=time_limit)
            if utcnow() >= deadline and pnl_pct <= 0.01:  # Allow small gains to run
                self._log_exit(session, trade, "time", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="time",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
        
        # 6. Close winning positions that have been profitable for a while
        # If up 1%+ for more than 2 hours, take profit (don't let winners reverse)
        if trade.opened_at and pnl_pct >= 0.01:
            age_hours = (utcnow() - trade.opened_at).total_seconds() / 3600
            if age_hours >= 2.0 and pnl_pct >= 0.015:  # 1.5%+ after 2 hours = take profit
                self._log_exit(session, trade, "profit_time", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="profit_time",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )

        return None

    def _update_trailing_stop(
        self,
        session: "Session",
        trade: PaperTrade,
        current_price: float,
    ) -> None:
        """
        Update the trailing stop price if price has moved favorably.

        For BUY: trailing stop rises when price rises (lock in gains)
        For SELL: trailing stop falls when price falls (lock in gains)
        """
        if not trade.trailing_stop_pct:
            return

        side = trade.side.upper()
        trail_pct = float(trade.trailing_stop_pct)
        high_water = float(trade.high_water_price or trade.entry_price)

        if side == "BUY":
            # Update high water mark if price is higher
            if current_price > high_water:
                trade.high_water_price = current_price
                high_water = current_price

            # Trailing stop is X% below high water mark
            new_trailing = high_water * (1 - trail_pct)

            # Only ratchet UP the trailing stop (never lower it)
            current_trailing = float(trade.trailing_stop_price or 0)
            if new_trailing > current_trailing:
                trade.trailing_stop_price = new_trailing

        else:  # SELL / SHORT
            # Update low water mark if price is lower
            low_water = float(trade.high_water_price or trade.entry_price)
            if current_price < low_water:
                trade.high_water_price = current_price
                low_water = current_price

            # Trailing stop is X% above low water mark
            new_trailing = low_water * (1 + trail_pct)

            # Only ratchet DOWN the trailing stop (never raise it)
            current_trailing = float(trade.trailing_stop_price or float("inf"))
            if new_trailing < current_trailing:
                trade.trailing_stop_price = new_trailing

    def _log_exit(
        self,
        session: "Session",
        trade: PaperTrade,
        reason: str,
        price: float,
        pnl_pct: float,
    ) -> None:
        """Log the auto-exit to activity log."""
        reason_labels = {
            "sl": "Stop-Loss",
            "tp": "Take-Profit",
            "trailing": "Trailing Stop",
            "max_loss": "Max Loss Cap",
            "time": "Time Limit",
        }
        session.add(
            ActivityLog(
                category="auto_exit",
                level="info",
                message=(
                    f"Auto-exit triggered: {trade.symbol} {trade.side} "
                    f"closed by {reason_labels.get(reason, reason)} "
                    f"at ${price:.4f} (P&L: {pnl_pct:+.2%})"
                ),
                wallet_id=trade.wallet_id,
            )
        )


def initialize_trade_sl_tp(
    trade: PaperTrade,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    max_loss_pct: float | None = None,
    time_limit_hours: float | None = None,
) -> None:
    """
    Set up stop-loss, take-profit, and trailing stop for a newly opened trade.

    Converts percentage-based inputs to absolute prices based on entry price.
    """
    entry = float(trade.entry_price)
    side = trade.side.upper()

    # Store original entry for DCA tracking
    if trade.original_entry is None:
        trade.original_entry = entry

    # Calculate and set stop-loss price
    if stop_loss_pct is not None and stop_loss_pct > 0:
        if side == "BUY":
            trade.stop_loss_price = entry * (1 - stop_loss_pct)
        else:
            trade.stop_loss_price = entry * (1 + stop_loss_pct)

    # Calculate and set take-profit price
    if take_profit_pct is not None and take_profit_pct > 0:
        if side == "BUY":
            trade.take_profit_price = entry * (1 + take_profit_pct)
        else:
            trade.take_profit_price = entry * (1 - take_profit_pct)

    # Set trailing stop percentage (actual price calculated on each tick)
    if trailing_stop_pct is not None and trailing_stop_pct > 0:
        trade.trailing_stop_pct = trailing_stop_pct
        trade.high_water_price = entry
        # Initial trailing stop is at the same level as a normal SL
        if side == "BUY":
            trade.trailing_stop_price = entry * (1 - trailing_stop_pct)
        else:
            trade.trailing_stop_price = entry * (1 + trailing_stop_pct)

    # Set max loss and time limit
    if max_loss_pct is not None:
        trade.max_loss_pct = max_loss_pct
    if time_limit_hours is not None:
        trade.time_limit_hours = time_limit_hours
