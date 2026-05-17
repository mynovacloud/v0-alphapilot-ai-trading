"""
Live trading engine.

The single entry point the autonomous bot uses to place real orders.

Responsibilities:
  1. Resolve the wallet's connector + decrypted API keys.
  2. Verify wallet trading_mode and bot_paused state.
  3. Run the central RiskManager (rejects oversized / over-cap orders).
  4. Persist a `LiveOrder` row in `pending_submit` BEFORE hitting the exchange.
  5. Submit to the connector. Update the row with the exchange order_id.
  6. If the wallet is in `live_shadow` mode, ALSO record a paper trade
     so we can compare live vs. paper P&L over time.
  7. Log everything via ActivityLog so you can audit what the bot did.
"""
from __future__ import annotations

import json
from typing import Any

from config.settings import settings
from connectors.crypto_connectors import CoinbaseConnector
from connectors.registry import get_connector
from database.db import session_scope
from database.models import (
    ActivityLog,
    ApiCredentialPlaceholder,
    LiveOrder,
    Wallet,
)
from trading.paper_trading_engine import PaperTradingEngine
from trading.risk_manager import RiskManager
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_LIVE_PLATFORMS = {"Coinbase", "Coinbase Perp"}  # extend as more connectors implement live trading
PERP_LIVE_PLATFORMS = {"Coinbase Perp"}


class LiveTradingEngine:
    def __init__(self) -> None:
        self.risk = RiskManager()
        self.paper = PaperTradingEngine()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def place_market(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        quote_size: float | None = None,
        base_qty: float | None = None,
        confidence: float = 0.6,
        strategy_id: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return self._dispatch(
            wallet_id=wallet_id,
            symbol=symbol,
            side=side,
            order_type="market",
            base_qty=base_qty,
            quote_size=quote_size,
            confidence=confidence,
            strategy_id=strategy_id,
            notes=notes,
        )

    def place_perp_market(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        base_qty: float,
        leverage: float = 1.0,
        reduce_only: bool = False,
        confidence: float = 0.6,
        strategy_id: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Place a perpetual-futures market order. Routes through the same risk
        gate as spot orders, but additionally enforces wallet.futures_enabled
        and wallet.max_leverage inside RiskManager.evaluate_trade(is_perp=True).
        """
        return self._dispatch(
            wallet_id=wallet_id,
            symbol=symbol,
            side=side,
            order_type="perp_market",
            base_qty=base_qty,
            confidence=confidence,
            strategy_id=strategy_id,
            notes=notes,
            leverage=leverage,
            reduce_only=reduce_only,
        )

    def place_limit(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        base_qty: float,
        limit_price: float,
        confidence: float = 0.6,
        strategy_id: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return self._dispatch(
            wallet_id=wallet_id,
            symbol=symbol,
            side=side,
            order_type="limit",
            base_qty=base_qty,
            limit_price=limit_price,
            confidence=confidence,
            strategy_id=strategy_id,
            notes=notes,
        )

    def place_stop_limit(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        base_qty: float,
        stop_price: float,
        limit_price: float,
        confidence: float = 0.6,
        strategy_id: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        return self._dispatch(
            wallet_id=wallet_id,
            symbol=symbol,
            side=side,
            order_type="stop_limit",
            base_qty=base_qty,
            stop_price=stop_price,
            limit_price=limit_price,
            confidence=confidence,
            strategy_id=strategy_id,
            notes=notes,
        )

    def cancel_order(self, wallet_id: int, live_order_id: int) -> dict[str, Any]:
        with session_scope() as s:
            order = s.get(LiveOrder, live_order_id)
            if not order or order.wallet_id != wallet_id:
                return {"ok": False, "error": "Order not found"}
            if not order.exchange_order_id:
                return {"ok": False, "error": "Order never reached the exchange"}
            connector = self._build_connector(s, wallet_id)

        if not isinstance(connector, CoinbaseConnector):
            return {"ok": False, "error": "Cancel only implemented for Coinbase right now."}
        result = connector.cancel_orders([order.exchange_order_id])
        with session_scope() as s:
            order = s.get(LiveOrder, live_order_id)
            if order and result.get("ok"):
                order.status = "cancelled"
                order.closed_at = utcnow()
        return result

    def reconcile_order(self, live_order_id: int) -> dict[str, Any]:
        """Pull latest state from the exchange and update the DB row."""
        with session_scope() as s:
            order = s.get(LiveOrder, live_order_id)
            if not order or not order.exchange_order_id:
                return {"ok": False, "error": "No exchange_order_id to reconcile"}
            connector = self._build_connector(s, order.wallet_id)

        if not isinstance(connector, CoinbaseConnector):
            return {"ok": False, "error": "Reconcile only implemented for Coinbase right now."}

        state = connector.get_order(order.exchange_order_id)
        if not state.get("ok"):
            return state

        with session_scope() as s:
            order = s.get(LiveOrder, live_order_id)
            if not order:
                return {"ok": False, "error": "Order vanished"}
            cb_status = (state.get("status") or "").upper()
            mapping = {
                "OPEN": "open",
                "PENDING": "pending_submit",
                "FILLED": "filled",
                "CANCELLED": "cancelled",
                "EXPIRED": "cancelled",
                "FAILED": "failed",
            }
            new_status = mapping.get(cb_status, order.status)
            order.status = new_status
            order.filled_qty = state.get("filled_size", order.filled_qty)
            order.avg_fill_price = state.get("average_filled_price", order.avg_fill_price)
            order.fees = state.get("total_fees", order.fees)
            if new_status == "filled" and not order.filled_at:
                order.filled_at = utcnow()
            order.raw_payload = json.dumps(state.get("raw", {}))[:4000]
            return {"ok": True, "status": new_status}

    # ------------------------------------------------------------------ #
    # Internal: dispatcher + persistence + risk gating
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        order_type: str,
        base_qty: float | None = None,
        quote_size: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        confidence: float = 0.6,
        strategy_id: int | None = None,
        notes: str = "",
        leverage: float = 1.0,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        if not settings.live_trading_enabled:
            return {"ok": False, "error": "Live trading globally disabled (set LIVE_TRADING_ENABLED=true)."}

        # ---- Pull wallet state, build connector ----
        with session_scope() as s:
            wallet = s.get(Wallet, wallet_id)
            if not wallet:
                return {"ok": False, "error": "Wallet not found"}
            if wallet.bot_paused:
                return {"ok": False, "error": "Wallet bot is paused"}
            if wallet.trading_mode not in {"live", "live_shadow"}:
                return {
                    "ok": False,
                    "error": f"Wallet trading_mode={wallet.trading_mode}; must be 'live' or 'live_shadow'.",
                }
            if wallet.platform not in SUPPORTED_LIVE_PLATFORMS:
                return {
                    "ok": False,
                    "error": f"Live trading not yet supported on {wallet.platform}.",
                }
            connector = self._build_connector(s, wallet_id)
            shadow = wallet.trading_mode == "live_shadow"
            platform = wallet.platform

        # ---- Risk check ----
        # Use limit_price if provided, else estimate notional from quote_size or last price.
        if base_qty is not None and (limit_price or 0) > 0:
            notional = base_qty * (limit_price or 0)
            risk_qty = base_qty
            risk_price = limit_price or 0
        elif base_qty is not None:
            # market sell: estimate from a stored price would be ideal; use quote_size if we have it
            risk_qty = base_qty
            risk_price = quote_size / base_qty if (quote_size and base_qty) else 0
            notional = base_qty * risk_price if risk_price else (quote_size or 0)
        else:
            # market buy: quote_size IS the notional
            notional = quote_size or 0
            risk_qty = 0
            risk_price = 0

        if notional > 0 and not self._wallet_caps_ok(wallet_id, notional):
            return {"ok": False, "error": "Wallet cap exceeded (max_position_usd or max_daily_loss)."}

        is_perp = order_type == "perp_market"
        decision = self.risk.evaluate_trade(
            wallet_id=wallet_id,
            qty=risk_qty,
            entry_price=risk_price,
            confidence=confidence,
            strategy_id=strategy_id,
            is_paper=False,
            is_perp=is_perp,
            leverage=leverage,
        )
        if not decision and notional > 0:
            return {"ok": False, "error": f"Risk: {decision.reason}"}

        # ---- Persist a pending order BEFORE we hit the exchange ----
        with session_scope() as s:
            order = LiveOrder(
                wallet_id=wallet_id,
                strategy_id=strategy_id,
                client_order_id=CoinbaseConnector._new_client_order_id(),
                platform=platform,
                symbol=symbol,
                side=side.upper(),
                order_type=order_type,
                base_qty=base_qty,
                quote_size=quote_size,
                limit_price=limit_price,
                stop_price=stop_price,
                confidence=confidence,
                is_paper_shadow=shadow,
                status="pending_submit",
            )
            s.add(order)
            s.flush()
            order_id = order.id
            client_order_id = order.client_order_id

        # ---- Submit ----
        if order_type == "market":
            result = connector.place_market_order(
                symbol=symbol,
                side=side,
                base_qty=base_qty,
                quote_size=quote_size,
                client_order_id=client_order_id,
            )
        elif order_type == "limit":
            result = connector.place_limit_order(
                symbol=symbol,
                side=side,
                base_qty=base_qty,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == "stop_limit":
            result = connector.place_stop_limit_order(
                symbol=symbol,
                side=side,
                base_qty=base_qty,
                stop_price=stop_price,
                limit_price=limit_price,
                client_order_id=client_order_id,
                stop_direction="STOP_DIRECTION_STOP_DOWN" if side.upper() == "SELL" else "STOP_DIRECTION_STOP_UP",
            )
        else:
            result = {"ok": False, "error": f"Unsupported order_type: {order_type}"}

        # ---- Update DB with result ----
        with session_scope() as s:
            order = s.get(LiveOrder, order_id)
            if not order:
                return {"ok": False, "error": "Order vanished from DB"}
            order.raw_payload = json.dumps(result.get("raw", {}))[:4000]
            if result.get("ok"):
                order.exchange_order_id = result.get("order_id")
                order.status = "open"
                order.accepted_at = utcnow()
                msg = f"LIVE {order_type.upper()} {side} {symbol} accepted (id={order.exchange_order_id})"
                level = "info"
            else:
                order.status = "rejected"
                order.last_error = str(result.get("error", ""))[:500]
                order.closed_at = utcnow()
                msg = f"LIVE {order_type.upper()} {side} {symbol} REJECTED: {order.last_error}"
                level = "warn"
            s.add(ActivityLog(category="live_trade", level=level, wallet_id=wallet_id, message=msg))

        # ---- Optional: record a paper-shadow copy ----
        if shadow and result.get("ok") and base_qty and (limit_price or risk_price):
            self.paper.open_trade(
                wallet_id=wallet_id,
                symbol=symbol,
                side=side,
                qty=base_qty,
                entry_price=limit_price or risk_price,
                confidence=confidence,
                strategy_id=strategy_id,
                notes=f"shadow of live order {order_id}",
            )

        return {
            "ok": result.get("ok", False),
            "live_order_id": order_id,
            "exchange_order_id": result.get("order_id"),
            "error": result.get("error"),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_connector(session, wallet_id: int):
        wallet = session.get(Wallet, wallet_id)
        creds = (
            session.query(ApiCredentialPlaceholder)
            .filter(ApiCredentialPlaceholder.wallet_id == wallet_id)
            .first()
        )
        return get_connector(
            wallet.platform,
            api_key=creds.api_key if creds else "",
            api_secret=creds.api_secret if creds else "",
            api_passphrase=creds.api_passphrase if creds else "",
            account_id=creds.account_id if creds else "",
            sandbox=wallet.sandbox_mode,
        )

    @staticmethod
    def _wallet_caps_ok(wallet_id: int, notional: float) -> bool:
        with session_scope() as s:
            w = s.get(Wallet, wallet_id)
            if not w:
                return False
            if notional > (w.max_position_usd or 0):
                return False

            # Daily loss check across LIVE orders (closed today)
            today = utcnow().date()
            todays = (
                s.query(LiveOrder)
                .filter(LiveOrder.wallet_id == wallet_id)
                .all()
            )
            day_pnl = sum(
                o.realized_pnl
                for o in todays
                if o.closed_at and o.closed_at.date() == today
            )
            if day_pnl <= -abs(w.max_daily_loss_usd or 0):
                return False
            day_count = sum(1 for o in todays if o.submitted_at and o.submitted_at.date() == today)
            if day_count >= (w.max_daily_trades or 0):
                return False
        return True
