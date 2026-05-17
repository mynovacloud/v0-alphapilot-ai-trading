"""Risk manager: gate every trade attempt before it hits the engine."""
from __future__ import annotations

from dataclasses import dataclass

from database.db import session_scope
from database.models import PaperTrade, Strategy, Wallet
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """
    Validates trades against per-wallet and per-strategy risk caps.

    All checks are conservative-by-default: any failure returns
    `RiskDecision(allowed=False, reason=...)`.
    """

    def evaluate(
        self,
        wallet_id: int,
        qty: float,
        entry_price: float,
        confidence: float,
        strategy_id: int | None = None,
    ) -> RiskDecision:
        notional = qty * entry_price

        with session_scope() as s:
            wallet = s.get(Wallet, wallet_id)
            if not wallet:
                return RiskDecision(False, "Wallet not found")

            # Hard rule: paper-balance must cover notional
            if notional > wallet.paper_balance * 1.0:
                return RiskDecision(
                    False,
                    f"Trade notional ${notional:,.2f} exceeds paper balance ${wallet.paper_balance:,.2f}",
                )

            strat: Strategy | None = s.get(Strategy, strategy_id) if strategy_id else None

            if strat:
                if notional > strat.max_position_size:
                    return RiskDecision(
                        False, f"Position size > strategy max ({strat.max_position_size})"
                    )
                if confidence < strat.min_confidence:
                    return RiskDecision(
                        False, f"Confidence {confidence:.2f} < required {strat.min_confidence:.2f}"
                    )

                # Daily loss check
                today = utcnow().date()
                todays_trades = (
                    s.query(PaperTrade)
                    .filter(
                        PaperTrade.wallet_id == wallet_id,
                        PaperTrade.strategy_id == strategy_id,
                    )
                    .all()
                )
                day_pnl = sum(
                    t.realized_pnl for t in todays_trades if t.closed_at and t.closed_at.date() == today
                )
                if day_pnl <= -abs(strat.max_daily_loss):
                    return RiskDecision(False, f"Max daily loss reached ({day_pnl:.2f})")

                # Max open trades
                open_count = sum(1 for t in todays_trades if t.status == "open")
                if open_count >= strat.max_open_trades:
                    return RiskDecision(False, f"Max open trades ({strat.max_open_trades}) reached")

        return RiskDecision(True, "ok")
