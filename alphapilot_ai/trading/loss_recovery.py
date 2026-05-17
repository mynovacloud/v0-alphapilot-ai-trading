"""
Loss Recovery System - Strategies for managing losing positions.

Provides DCA (dollar-cost averaging), scale-out, and pivot strategies
to help recover from underwater positions or cut losses intelligently.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from database.db import session_scope
from database.models import PaperTrade, ActivityLog, Wallet
from utils.helpers import utcnow

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class RecoveryAction:
    """Represents a suggested recovery action for an underwater position."""
    trade_id: int
    symbol: str
    action: Literal["dca", "scale_out", "pivot", "hold", "close"]
    reason: str
    suggested_size_usd: float | None = None
    suggested_side: str | None = None  # For pivot: the new direction
    confidence: float = 0.5


class LossRecoveryEngine:
    """
    Analyzes underwater positions and suggests recovery strategies.

    Recovery strategies:
    - DCA (Dollar-Cost Average): Add to position at better price to lower avg entry
    - Scale-Out: Close partial position to reduce exposure
    - Pivot: Close losing position and open opposite direction
    - Hold: Wait for price to recover
    - Close: Cut losses entirely

    Risk guards prevent excessive averaging and ensure capital preservation.
    """

    def __init__(
        self,
        max_dca_count: int = 3,
        dca_threshold_pct: float = 0.05,  # Position must be down 5%+ to DCA
        scale_out_threshold_pct: float = 0.08,  # Position must be down 8%+ to consider scale-out
        pivot_threshold_pct: float = 0.07,  # Position must be down 7%+ to consider pivot
        max_loss_before_close_pct: float = 0.15,  # Force close at 15% loss
        cooldown_minutes: int = 30,  # Min time between recovery actions
    ):
        self.max_dca_count = max_dca_count
        self.dca_threshold_pct = dca_threshold_pct
        self.scale_out_threshold_pct = scale_out_threshold_pct
        self.pivot_threshold_pct = pivot_threshold_pct
        self.max_loss_before_close_pct = max_loss_before_close_pct
        self.cooldown_minutes = cooldown_minutes

    def analyze_position(
        self,
        trade: PaperTrade,
        current_price: float,
        signal_still_valid: bool = True,
        signal_reversed: bool = False,
    ) -> RecoveryAction:
        """
        Analyze an underwater position and suggest a recovery strategy.

        Args:
            trade: The PaperTrade to analyze
            current_price: Current market price
            signal_still_valid: Whether the original entry signal is still valid
            signal_reversed: Whether the signal has flipped to the opposite direction

        Returns:
            RecoveryAction with suggested strategy
        """
        entry = float(trade.entry_price)
        side = trade.side.upper()
        dca_count = trade.dca_count or 0

        # Calculate current loss percentage
        if side == "BUY":
            loss_pct = (entry - current_price) / entry if entry > 0 else 0
        else:  # SELL / SHORT
            loss_pct = (current_price - entry) / entry if entry > 0 else 0

        # If position is profitable, no recovery needed
        if loss_pct <= 0:
            return RecoveryAction(
                trade_id=trade.id,
                symbol=trade.symbol,
                action="hold",
                reason="Position is profitable, no recovery needed",
                confidence=0.9,
            )

        # 1. Check if we should force close (max loss exceeded)
        if loss_pct >= self.max_loss_before_close_pct:
            return RecoveryAction(
                trade_id=trade.id,
                symbol=trade.symbol,
                action="close",
                reason=f"Loss ({loss_pct:.1%}) exceeds maximum allowed ({self.max_loss_before_close_pct:.1%})",
                confidence=0.95,
            )

        # 2. Check if signal has reversed - consider pivot
        if signal_reversed and loss_pct >= self.pivot_threshold_pct:
            new_side = "SELL" if side == "BUY" else "BUY"
            return RecoveryAction(
                trade_id=trade.id,
                symbol=trade.symbol,
                action="pivot",
                reason=f"Signal reversed with {loss_pct:.1%} loss - consider flipping to {new_side}",
                suggested_side=new_side,
                suggested_size_usd=float(trade.qty * current_price),
                confidence=0.7,
            )

        # 3. Check if we can DCA (average down)
        if (
            signal_still_valid
            and dca_count < self.max_dca_count
            and loss_pct >= self.dca_threshold_pct
        ):
            # Suggest adding half the original position size
            original_size = float(trade.qty * (trade.original_entry or entry))
            suggested_add = original_size * 0.5
            return RecoveryAction(
                trade_id=trade.id,
                symbol=trade.symbol,
                action="dca",
                reason=f"Signal still valid, down {loss_pct:.1%} - DCA #{dca_count + 1} suggested",
                suggested_size_usd=suggested_add,
                confidence=0.65,
            )

        # 4. Check if we should scale out (reduce exposure)
        if loss_pct >= self.scale_out_threshold_pct and not signal_still_valid:
            return RecoveryAction(
                trade_id=trade.id,
                symbol=trade.symbol,
                action="scale_out",
                reason=f"Signal weakened with {loss_pct:.1%} loss - consider reducing exposure",
                suggested_size_usd=float(trade.qty * current_price * 0.5),  # Close half
                confidence=0.6,
            )

        # 5. Default: hold and wait
        return RecoveryAction(
            trade_id=trade.id,
            symbol=trade.symbol,
            action="hold",
            reason=f"Position down {loss_pct:.1%} - within tolerance, holding",
            confidence=0.5,
        )

    def execute_dca(
        self,
        trade: PaperTrade,
        add_qty: float,
        add_price: float,
    ) -> bool:
        """
        Execute a DCA (dollar-cost average) by adding to an existing position.

        Updates the trade's entry price to the weighted average and increments dca_count.
        """
        if (trade.dca_count or 0) >= self.max_dca_count:
            return False

        old_qty = float(trade.qty)
        old_entry = float(trade.entry_price)

        # Calculate new weighted average entry
        total_cost = (old_qty * old_entry) + (add_qty * add_price)
        new_qty = old_qty + add_qty
        new_entry = total_cost / new_qty if new_qty > 0 else old_entry

        with session_scope() as s:
            t = s.query(PaperTrade).filter(PaperTrade.id == trade.id).first()
            if not t or t.status != "open":
                return False

            # Preserve original entry if this is first DCA
            if t.original_entry is None:
                t.original_entry = old_entry

            t.qty = new_qty
            t.entry_price = new_entry
            t.dca_count = (t.dca_count or 0) + 1

            # Update SL/TP prices based on new entry
            self._recalculate_sl_tp(t)

            # Log the DCA action
            s.add(
                ActivityLog(
                    category="dca",
                    level="info",
                    message=(
                        f"DCA #{t.dca_count}: Added {add_qty:.6f} {t.symbol} at ${add_price:.4f}. "
                        f"New avg entry: ${new_entry:.4f} (was ${old_entry:.4f})"
                    ),
                    wallet_id=t.wallet_id,
                )
            )
            s.commit()

        return True

    def execute_scale_out(
        self,
        trade_id: int,
        close_fraction: float = 0.5,
        current_price: float = 0.0,
    ) -> float | None:
        """
        Close a portion of a position to reduce exposure.

        Returns the realized P&L from the closed portion, or None if failed.
        """
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
            if not trade or trade.status != "open":
                return None

            close_qty = float(trade.qty) * close_fraction
            remaining_qty = float(trade.qty) - close_qty

            if remaining_qty <= 0:
                # Would close entire position - just return None, let normal close handle it
                return None

            entry = float(trade.entry_price)
            side = trade.side.upper()

            # Calculate P&L on closed portion
            if side == "BUY":
                pnl = (current_price - entry) * close_qty
            else:
                pnl = (entry - current_price) * close_qty

            # Update trade with remaining quantity
            trade.qty = remaining_qty
            # Add to realized P&L (partial close)
            trade.realized_pnl = float(trade.realized_pnl or 0) + pnl

            # Log the scale-out
            s.add(
                ActivityLog(
                    category="scale_out",
                    level="info",
                    message=(
                        f"Scale-out: Closed {close_fraction:.0%} of {trade.symbol} position "
                        f"at ${current_price:.4f}. Realized: ${pnl:+.2f}. "
                        f"Remaining: {remaining_qty:.6f} units"
                    ),
                    wallet_id=trade.wallet_id,
                )
            )
            s.commit()

            return pnl

    def _recalculate_sl_tp(self, trade: PaperTrade) -> None:
        """
        Recalculate SL/TP prices after a DCA changes the entry price.

        Preserves the original percentage distances.
        """
        new_entry = float(trade.entry_price)
        side = trade.side.upper()

        # If there's a stop loss, recalculate based on original percentage
        if trade.stop_loss_price and trade.original_entry:
            original_entry = float(trade.original_entry)
            if side == "BUY":
                sl_pct = 1 - (float(trade.stop_loss_price) / original_entry)
                trade.stop_loss_price = new_entry * (1 - sl_pct)
            else:
                sl_pct = (float(trade.stop_loss_price) / original_entry) - 1
                trade.stop_loss_price = new_entry * (1 + sl_pct)

        # If there's a take profit, recalculate based on original percentage
        if trade.take_profit_price and trade.original_entry:
            original_entry = float(trade.original_entry)
            if side == "BUY":
                tp_pct = (float(trade.take_profit_price) / original_entry) - 1
                trade.take_profit_price = new_entry * (1 + tp_pct)
            else:
                tp_pct = 1 - (float(trade.take_profit_price) / original_entry)
                trade.take_profit_price = new_entry * (1 - tp_pct)

        # Update trailing stop if present
        if trade.trailing_stop_pct:
            trail_pct = float(trade.trailing_stop_pct)
            trade.high_water_price = new_entry
            if side == "BUY":
                trade.trailing_stop_price = new_entry * (1 - trail_pct)
            else:
                trade.trailing_stop_price = new_entry * (1 + trail_pct)


def get_underwater_positions(wallet_id: int, price_map: dict[str, float]) -> list[dict]:
    """
    Get all underwater (losing) positions for a wallet with their current loss %.

    Returns list of dicts with trade info and current loss metrics.
    """
    underwater = []

    with session_scope() as s:
        trades = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id)
            .filter(PaperTrade.status == "open")
            .all()
        )

        for trade in trades:
            current_price = price_map.get(trade.symbol)
            if current_price is None:
                continue

            entry = float(trade.entry_price)
            side = trade.side.upper()

            if side == "BUY":
                pnl_pct = (current_price - entry) / entry if entry > 0 else 0
            else:
                pnl_pct = (entry - current_price) / entry if entry > 0 else 0

            if pnl_pct < 0:
                underwater.append({
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "side": side,
                    "entry_price": entry,
                    "current_price": current_price,
                    "qty": float(trade.qty),
                    "loss_pct": abs(pnl_pct),
                    "loss_usd": abs(pnl_pct * entry * float(trade.qty)),
                    "dca_count": trade.dca_count or 0,
                    "can_dca": (trade.dca_count or 0) < 3,
                })

    return sorted(underwater, key=lambda x: x["loss_pct"], reverse=True)
