"""
Paper trading engine.

Responsibilities:
- Validate trade through RiskManager
- Estimate fees + slippage
- Persist PaperTrade rows
- Open / close trades and update wallet paper balance
- Log every action via ActivityLog
"""
from __future__ import annotations

import random
from typing import Any

from database.db import session_scope
from database.models import ActivityLog, PaperTrade, Wallet
from trading.risk_manager import RiskManager
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


class PaperTradingEngine:
    def __init__(self) -> None:
        self.risk = RiskManager()

    # --- Public API -----------------------------------------------------

    def open_trade(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        confidence: float = 0.6,
        market_type: str = "Crypto",
        strategy_id: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        logger.info(
            f"[OPEN_TRADE] Attempting: wallet={wallet_id}, symbol={symbol}, "
            f"side={side}, qty={qty}, price={entry_price}, conf={confidence}"
        )

        decision = self.risk.evaluate(wallet_id, qty, entry_price, confidence, strategy_id)
        if not decision:
            logger.warning(f"[OPEN_TRADE] REJECTED by risk manager: {decision.reason} (code={decision.code})")
            self._log("risk", f"Trade rejected: {decision.reason}", wallet_id=wallet_id, level="warn")
            return {"ok": False, "reason": decision.reason, "code": decision.code}

        notional = qty * entry_price
        fees = self._estimate_fees(notional)
        slippage = self._estimate_slippage(notional)

        with session_scope() as s:
            wallet = s.get(Wallet, wallet_id)
            if not wallet:
                logger.error(f"[OPEN_TRADE] Wallet {wallet_id} not found")
                return {"ok": False, "reason": "Wallet not found"}

            # Reserve cash from paper balance
            wallet.paper_balance -= (notional + fees + slippage)
            wallet.last_synced = utcnow()

            trade = PaperTrade(
                wallet_id=wallet_id,
                strategy_id=strategy_id,
                symbol=symbol,
                market_type=market_type,
                side=side,
                qty=qty,
                entry_price=entry_price,
                fees=fees,
                slippage=slippage,
                confidence=confidence,
                status="open",
                notes=notes,
            )
            s.add(trade)
            s.flush()
            trade_id = trade.id

            s.add(
                ActivityLog(
                    category="paper_trade",
                    level="info",
                    wallet_id=wallet_id,
                    message=f"Opened {side} {qty} {symbol} @ {entry_price:.2f} (conf={confidence:.2f})",
                )
            )
            
            logger.info(f"[OPEN_TRADE] SUCCESS: trade_id={trade_id}, symbol={symbol}, notional=${notional:.2f}")

        return {"ok": True, "trade_id": trade_id, "fees": fees, "slippage": slippage}

    def close_trade(self, trade_id: int, exit_price: float, notes: str = "") -> dict[str, Any]:
        """Close a paper trade and record the P&L."""
        logger.info(f"[CLOSE_TRADE] Attempting to close trade {trade_id} at {exit_price}")
        try:
            with session_scope() as s:
                trade = s.get(PaperTrade, trade_id)
                if not trade:
                    return {"ok": False, "reason": "Trade not found"}
                if trade.status != "open":
                    return {"ok": False, "reason": f"Trade already {trade.status}"}

                wallet = s.get(Wallet, trade.wallet_id)
                
                # Safe type conversion
                entry_price = float(trade.entry_price or 0)
                qty = float(trade.qty or 0)
                fees = float(trade.fees or 0)
                slippage = float(trade.slippage or 0)
                side = (trade.side or "BUY").upper()
                
                if entry_price <= 0 or qty <= 0:
                    return {"ok": False, "reason": f"Invalid trade data: entry={entry_price}, qty={qty}"}
                
                direction = 1 if side == "BUY" else -1
                pnl = (exit_price - entry_price) * qty * direction
                pnl -= fees + slippage

                trade.exit_price = exit_price
                trade.realized_pnl = round(pnl, 2)
                trade.unrealized_pnl = 0.0
                trade.status = "closed"
                trade.closed_at = utcnow()

                # Return notional + pnl to paper balance
                notional = qty * entry_price
                if wallet:
                    wallet.paper_balance = float(wallet.paper_balance or 0) + notional + pnl
                    wallet.last_synced = utcnow()

                s.add(
                    ActivityLog(
                        category="paper_trade",
                        level="info",
                        wallet_id=trade.wallet_id,
                        message=f"Closed {side} {qty} {trade.symbol} @ {exit_price:.2f}, PnL={pnl:.2f}",
                    )
                )
        except Exception as e:
            logger.exception(f"[CLOSE_TRADE] Error closing trade {trade_id}")
            return {"ok": False, "reason": str(e)}

        # Trade is now closed. Kick the Claude reflection loop so it can
        # learn from this fill. Best-effort: reflection failures must never
        # break the close. The function is idempotent if called twice.
        try:
            from ai.claude_learning import record_trade_outcome
            record_trade_outcome(trade_id)
        except Exception:
            logger.exception("Reflection failed for trade %s", trade_id)

        return {"ok": True, "pnl": round(pnl, 2)}

    # --- Helpers --------------------------------------------------------

    @staticmethod
    def _estimate_fees(notional: float) -> float:
        # 10 bps fee assumption
        return round(notional * 0.001, 2)

    @staticmethod
    def _estimate_slippage(notional: float) -> float:
        return round(notional * random.uniform(0.0001, 0.001), 2)

    @staticmethod
    def _log(category: str, message: str, wallet_id: int | None = None, level: str = "info") -> None:
        with session_scope() as s:
            s.add(ActivityLog(category=category, level=level, message=message, wallet_id=wallet_id))
