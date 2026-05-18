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
        import logging

        with session_scope() as s:
            open_trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.wallet_id == wallet_id)
                .filter(PaperTrade.status == "open")
                .all()
            )
            
            logging.info(f"[POSITION_MONITOR] Wallet {wallet_id}: {len(open_trades)} open trades, {len(price_map)} prices in map")

            for trade in open_trades:
                current_price = price_map.get(trade.symbol)
                if current_price is None:
                    # Try to fetch price directly if not in map
                    import logging
                    logging.warning(f"[POSITION_MONITOR] No price for {trade.symbol} in map, fetching...")
                    try:
                        from connectors.live_prices import get_price
                        price_result = get_price(trade.symbol)
                        if price_result.get("ok"):
                            current_price = float(price_result["price"])
                            logging.info(f"[POSITION_MONITOR] Fetched {trade.symbol} price: ${current_price}")
                        else:
                            logging.warning(f"[POSITION_MONITOR] Failed to fetch {trade.symbol}: {price_result}")
                            continue
                    except Exception as e:
                        logging.error(f"[POSITION_MONITOR] Error fetching {trade.symbol}: {e}")
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
        
        # Calculate P&L in USD
        pnl_usd = pnl_pct * entry * qty

        # =====================================================================
        # SCALPER MODE: Check micro-profit target (highest priority for profits)
        # =====================================================================
        # Get wallet settings - use explicit query to ensure fresh data
        from database.models import Wallet
        wallet = session.query(Wallet).filter(Wallet.id == trade.wallet_id).first()
        
        # Default values if columns don't exist yet
        trading_style = 'hybrid'
        micro_target_usd = 0.25
        min_profit_pct = 0.003
        
        if wallet:
            trading_style = getattr(wallet, 'trading_style', 'hybrid') or 'hybrid'
            micro_target_usd = getattr(wallet, 'micro_profit_target_usd', 0.25) or 0.25
            min_profit_pct = getattr(wallet, 'min_profit_pct', 0.003) or 0.003
        
        # Log for debugging
        import logging
        logging.info(f"[POSITION_MONITOR] {trade.symbol}: pnl_usd=${pnl_usd:.4f}, target=${micro_target_usd}, style={trading_style}")
        
        if pnl_usd > 0:  # Only check if we're in profit
            # Scalper mode: take ANY profit that hits the USD target
            if trading_style == "scalper" and pnl_usd >= micro_target_usd:
                logging.info(f"[SCALPER] TRIGGERING EXIT for {trade.symbol}: pnl_usd=${pnl_usd:.4f} >= target=${micro_target_usd}")
                self._log_exit(session, trade, "micro_profit", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="micro_profit",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # Hybrid mode: use both USD and percentage targets
            if trading_style == "hybrid":
                if pnl_usd >= micro_target_usd or pnl_pct >= min_profit_pct:
                    logging.info(f"[HYBRID] TRIGGERING EXIT for {trade.symbol}: pnl_usd=${pnl_usd:.4f}, pnl_pct={pnl_pct:.4%}")
                    self._log_exit(session, trade, "target_profit", current_price, pnl_pct)
                    return ExitSignal(
                        trade_id=trade.id,
                        symbol=trade.symbol,
                        reason="target_profit",
                        current_price=current_price,
                        trigger_price=None,
                        pnl_pct=pnl_pct,
                    )
            
            # Swing mode: only use percentage-based targets (traditional SL/TP)
            # Falls through to the standard SL/TP checks below

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
        # DEFAULT: If no time limit set, use 4 hours for paper trades (faster iteration)
        time_limit = float(trade.time_limit_hours) if trade.time_limit_hours else 4.0
        if trade.opened_at:
            from utils.helpers import ensure_utc
            opened_utc = ensure_utc(trade.opened_at)
            deadline = opened_utc + timedelta(hours=time_limit)
            if utcnow() >= deadline and pnl_pct <= 0.005:  # Close if flat or losing after time limit
                self._log_exit(session, trade, "time", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="time",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
        
        # 6. Take small profits after some time has passed
        # This ensures we lock in gains instead of letting them evaporate
        if trade.opened_at and pnl_pct > 0:
            from utils.helpers import time_since_minutes
            age_minutes = time_since_minutes(trade.opened_at)
            
            # After 30 mins: take 0.5%+ profit
            if age_minutes >= 30 and pnl_pct >= 0.005:
                self._log_exit(session, trade, "profit_30m", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="profit_30m",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # After 1 hour: take 0.3%+ profit  
            if age_minutes >= 60 and pnl_pct >= 0.003:
                self._log_exit(session, trade, "profit_1h", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="profit_1h",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # After 2 hours: take ANY profit (even $0.01)
            if age_minutes >= 120 and pnl_pct > 0.001:
                self._log_exit(session, trade, "profit_2h", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="profit_2h",
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
        Also handles breakeven stop activation.

        For BUY: trailing stop rises when price rises (lock in gains)
        For SELL: trailing stop falls when price falls (lock in gains)
        
        The trailing stop uses an ADAPTIVE algorithm:
        - At small profits: use the configured trailing_stop_pct
        - As profits grow: tighten the trailing stop to lock in more gains
        """
        import logging
        
        side = trade.side.upper()
        entry = float(trade.entry_price)
        
        # Calculate current profit percentage
        if side == "BUY":
            profit_pct = (current_price - entry) / entry if entry > 0 else 0
        else:
            profit_pct = (entry - current_price) / entry if entry > 0 else 0
        
        # =====================================================================
        # BREAKEVEN STOP: Once profit hits trigger, move stop to lock in small gain
        # =====================================================================
        breakeven_trigger = float(trade.breakeven_trigger_pct or 0)
        breakeven_stop = float(trade.breakeven_stop_pct or 0)
        
        if breakeven_trigger > 0 and not trade.breakeven_activated and profit_pct >= breakeven_trigger:
            # Activate breakeven stop!
            if side == "BUY":
                new_stop = entry * (1 + breakeven_stop)  # Stop above entry
            else:
                new_stop = entry * (1 - breakeven_stop)  # Stop below entry for shorts
            
            # Update stop loss to breakeven level
            current_sl = float(trade.stop_loss_price or 0)
            if side == "BUY" and new_stop > current_sl:
                trade.stop_loss_price = new_stop
                trade.breakeven_activated = True
                logging.info(f"[BREAKEVEN] {trade.symbol}: Activated! Stop moved from ${current_sl:.4f} to ${new_stop:.4f} (entry=${entry:.4f})")
            elif side == "SELL" and (current_sl == 0 or new_stop < current_sl):
                trade.stop_loss_price = new_stop
                trade.breakeven_activated = True
                logging.info(f"[BREAKEVEN] {trade.symbol}: Activated! Stop moved to ${new_stop:.4f}")
        
        # =====================================================================
        # TRAILING STOP: Ratchet stop up as price rises
        # =====================================================================
        if not trade.trailing_stop_pct:
            return

        trail_pct = float(trade.trailing_stop_pct)
        high_water = float(trade.high_water_price or entry)
        
        # ADAPTIVE TRAILING: Tighten the trailing % as profits grow
        # - At 1% profit: use base trailing %
        # - At 2% profit: tighten by 20%
        # - At 3% profit: tighten by 35%
        # - At 5%+ profit: tighten by 50%
        adaptive_multiplier = 1.0
        if profit_pct >= 0.05:
            adaptive_multiplier = 0.50  # 50% tighter trailing stop
        elif profit_pct >= 0.03:
            adaptive_multiplier = 0.65
        elif profit_pct >= 0.02:
            adaptive_multiplier = 0.80
        elif profit_pct >= 0.01:
            adaptive_multiplier = 0.90
        
        effective_trail_pct = trail_pct * adaptive_multiplier

        if side == "BUY":
            # Update high water mark if price is higher
            if current_price > high_water:
                trade.high_water_price = current_price
                high_water = current_price
                logging.debug(f"[TRAILING] {trade.symbol}: New high water ${high_water:.4f}")

            # Trailing stop is X% below high water mark
            new_trailing = high_water * (1 - effective_trail_pct)

            # Only ratchet UP the trailing stop (never lower it)
            current_trailing = float(trade.trailing_stop_price or 0)
            if new_trailing > current_trailing:
                trade.trailing_stop_price = new_trailing
                logging.info(
                    f"[TRAILING] {trade.symbol}: Stop raised ${current_trailing:.4f} -> ${new_trailing:.4f} "
                    f"(high=${high_water:.4f}, trail={effective_trail_pct:.2%}, profit={profit_pct:.2%})"
                )

        else:  # SELL / SHORT
            # Update low water mark if price is lower
            low_water = float(trade.high_water_price or entry)
            if current_price < low_water:
                trade.high_water_price = current_price
                low_water = current_price

            # Trailing stop is X% above low water mark
            new_trailing = low_water * (1 + effective_trail_pct)

            # Only ratchet DOWN the trailing stop (never raise it)
            current_trailing = float(trade.trailing_stop_price or float("inf"))
            if new_trailing < current_trailing:
                trade.trailing_stop_price = new_trailing
                logging.info(
                    f"[TRAILING] {trade.symbol} (SHORT): Stop lowered to ${new_trailing:.4f} "
                    f"(low=${low_water:.4f}, profit={profit_pct:.2%})"
                )

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
