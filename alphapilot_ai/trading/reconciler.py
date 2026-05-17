"""
Reconciler — keeps our LiveOrder rows in sync with what the exchange thinks.

Why this matters:
  The bot's process can crash, the network can drop mid-submit, an order can
  be cancelled manually in the Coinbase app, fills can come in between ticks.
  Without reconciliation we'd lose track of state and either double-trade or
  miss exits. The reconciler runs on its own schedule and:

    1. For every LiveOrder in a non-terminal state (pending_submit, open,
       partially_filled), pulls the latest state from the exchange and updates
       our row. Records fills, fees, average price, closure time.

    2. For every "active" wallet, pulls the current OPEN orders list from the
       exchange and flags any orders we don't know about (manual orders the
       user placed in the Coinbase app). These get logged so the user can see
       drift but are NOT touched by the bot.

    3. Marks any LiveOrder rows that have been pending_submit for too long
       (default 90s) as 'failed' — it means the submit call never completed
       round-trip. The risk manager will refuse to retry until manually cleared.

Terminal states (we never re-poll these): filled, cancelled, rejected, failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from connectors.registry import get_connector
from database.db import session_scope
from database.models import ActivityLog, LiveOrder, Wallet
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

# Status sets
NON_TERMINAL = {"pending_submit", "open", "partially_filled"}
TERMINAL = {"filled", "cancelled", "rejected", "failed"}

# Coinbase status -> our internal status
CB_STATUS_MAP = {
    "OPEN": "open",
    "PENDING": "open",
    "FILLED": "filled",
    "CANCELLED": "cancelled",
    "EXPIRED": "cancelled",
    "FAILED": "failed",
    "UNKNOWN_ORDER_STATUS": "open",
}

# How long can a row sit in pending_submit before we declare it dead?
PENDING_SUBMIT_TIMEOUT_SECONDS = 90


@dataclass
class ReconcileResult:
    started_at: str = ""
    finished_at: str = ""
    orders_polled: int = 0
    orders_updated: int = 0
    orders_filled: int = 0
    orders_failed_timeout: int = 0
    drift_orders: int = 0
    errors: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "orders_polled": self.orders_polled,
            "orders_updated": self.orders_updated,
            "orders_filled": self.orders_filled,
            "orders_failed_timeout": self.orders_failed_timeout,
            "drift_orders": self.drift_orders,
            "errors": self.errors,
            "notes": self.notes[-10:],
        }


class Reconciler:
    """
    Reconciler is stateless across runs — every call to `reconcile()` starts
    fresh from the DB and the exchange. Designed to be safe to run on an
    interval (e.g. every 30s) without conflicting with the bot tick.
    """

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def reconcile(self) -> ReconcileResult:
        result = ReconcileResult(started_at=utcnow().isoformat())

        # Step 1: time out stale pending_submit rows ----------------------
        result.orders_failed_timeout = self._timeout_stale_submits()

        # Step 2: per-wallet poll of non-terminal orders ------------------
        with session_scope() as s:
            wallets = (
                s.query(Wallet)
                .filter(Wallet.platform.in_(["Coinbase"]))  # only platforms we have live trading for
                .filter(Wallet.api_key.isnot(None))
                .all()
            )
            wallet_specs = [
                {
                    "id": w.id,
                    "name": w.name,
                    "platform": w.platform,
                    "api_key": w.api_key,
                    "api_secret": w.api_secret,
                    "passphrase": w.passphrase,
                    "trading_mode": w.trading_mode or "paper",
                }
                for w in wallets
            ]

        for wallet in wallet_specs:
            try:
                self._reconcile_wallet(wallet, result)
            except Exception as e:
                result.errors += 1
                logger.exception("Reconciler error on wallet %s: %s", wallet["name"], e)
                self._log(
                    "reconciler",
                    f"Wallet {wallet['name']}: reconcile failed: {e}",
                    wallet_id=wallet["id"],
                    level="error",
                )

        result.finished_at = utcnow().isoformat()
        self._history.append(result.as_dict())
        if len(self._history) > 50:
            self._history = self._history[-50:]
        return result

    # ------------------------------------------------------------------
    def _timeout_stale_submits(self) -> int:
        cutoff = utcnow() - timedelta(seconds=PENDING_SUBMIT_TIMEOUT_SECONDS)
        count = 0
        with session_scope() as s:
            stale = (
                s.query(LiveOrder)
                .filter(LiveOrder.status == "pending_submit")
                .filter(LiveOrder.submitted_at < cutoff)
                .all()
            )
            for o in stale:
                o.status = "failed"
                o.last_error = (
                    o.last_error
                    or f"Timed out after {PENDING_SUBMIT_TIMEOUT_SECONDS}s in pending_submit. "
                       "The submit network call never completed; treating as failed."
                )
                o.closed_at = utcnow()
                count += 1
                s.add(
                    ActivityLog(
                        category="reconciler",
                        level="warn",
                        wallet_id=o.wallet_id,
                        message=(
                            f"LiveOrder {o.client_order_id} (wallet {o.wallet_id}) "
                            f"timed out in pending_submit and was marked failed."
                        ),
                    )
                )
        return count

    # ------------------------------------------------------------------
    def _reconcile_wallet(self, wallet: dict[str, Any], result: ReconcileResult) -> None:
        # Build the connector once per wallet
        connector = get_connector(
            wallet["platform"],
            api_key=wallet["api_key"],
            api_secret=wallet["api_secret"],
            passphrase=wallet["passphrase"],
        )

        # Pull our non-terminal orders for this wallet
        with session_scope() as s:
            ours = (
                s.query(LiveOrder)
                .filter(LiveOrder.wallet_id == wallet["id"])
                .filter(LiveOrder.status.in_(list(NON_TERMINAL)))
                .all()
            )
            our_specs = [
                {
                    "id": o.id,
                    "client_order_id": o.client_order_id,
                    "exchange_order_id": o.exchange_order_id,
                    "status": o.status,
                }
                for o in ours
            ]

        # Update each from the exchange
        for spec in our_specs:
            if not spec["exchange_order_id"]:
                # No exchange_order_id means submit didn't return success.
                # The timeout pass above handles these eventually.
                continue
            result.orders_polled += 1
            try:
                fresh = connector.get_order(spec["exchange_order_id"])
            except Exception as e:
                result.errors += 1
                result.notes.append(f"poll fail {spec['client_order_id']}: {e}")
                continue
            if not fresh.get("ok"):
                result.errors += 1
                continue

            new_status = CB_STATUS_MAP.get(str(fresh.get("status") or "").upper(), spec["status"])
            self._apply_update(
                live_order_id=spec["id"],
                new_status=new_status,
                filled_qty=float(fresh.get("filled_size") or 0),
                avg_price=float(fresh.get("average_filled_price") or 0),
                fees=float(fresh.get("total_fees") or 0),
                result=result,
            )

        # Drift detection: pull exchange OPEN orders, compare to our DB
        try:
            listing = connector.list_orders(status="OPEN", limit=250) if hasattr(connector, "list_orders") else {"ok": False, "orders": []}
        except Exception as e:
            listing = {"ok": False, "error": str(e), "orders": []}

        if listing.get("ok"):
            exchange_open_ids = {o["exchange_order_id"] for o in listing["orders"] if o.get("exchange_order_id")}
            with session_scope() as s:
                known = {
                    o.exchange_order_id
                    for o in s.query(LiveOrder)
                    .filter(LiveOrder.wallet_id == wallet["id"])
                    .filter(LiveOrder.exchange_order_id.in_(list(exchange_open_ids) or [""]))
                    .all()
                    if o.exchange_order_id
                }
            unknown = exchange_open_ids - known
            if unknown:
                result.drift_orders += len(unknown)
                self._log(
                    "reconciler",
                    (
                        f"Wallet {wallet['name']}: detected {len(unknown)} open order(s) "
                        f"on the exchange that the bot did not place. Bot will NOT touch them."
                    ),
                    wallet_id=wallet["id"],
                    level="warn",
                )

    # ------------------------------------------------------------------
    def _apply_update(
        self,
        *,
        live_order_id: int,
        new_status: str,
        filled_qty: float,
        avg_price: float,
        fees: float,
        result: ReconcileResult,
    ) -> None:
        with session_scope() as s:
            o = s.query(LiveOrder).filter(LiveOrder.id == live_order_id).first()
            if not o:
                return

            changed = False

            # If we now have fills, normalize partial vs full
            if new_status == "open" and filled_qty > 0 and o.base_qty:
                if filled_qty + 1e-9 < float(o.base_qty):
                    new_status = "partially_filled"

            if new_status != o.status:
                # transition
                o.status = new_status
                changed = True
                if new_status == "filled":
                    o.filled_at = utcnow()
                    o.closed_at = utcnow()
                    result.orders_filled += 1
                elif new_status in {"cancelled", "rejected", "failed"}:
                    o.closed_at = utcnow()

            if abs(filled_qty - float(o.filled_qty or 0)) > 1e-9:
                o.filled_qty = filled_qty
                changed = True
            if avg_price and abs(avg_price - float(o.avg_fill_price or 0)) > 1e-9:
                o.avg_fill_price = avg_price
                changed = True
            if fees and abs(fees - float(o.fees or 0)) > 1e-9:
                o.fees = fees
                changed = True

            if changed:
                result.orders_updated += 1
                s.add(
                    ActivityLog(
                        category="reconciler",
                        level="info",
                        wallet_id=o.wallet_id,
                        message=(
                            f"LiveOrder {o.client_order_id}: status={o.status}, "
                            f"filled={o.filled_qty}, avg={o.avg_fill_price}, fees={o.fees}"
                        ),
                    )
                )

    # ------------------------------------------------------------------
    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return list(reversed(self._history))[:limit]

    @staticmethod
    def _log(category: str, message: str, *, wallet_id: int | None = None, level: str = "info") -> None:
        with session_scope() as s:
            s.add(ActivityLog(category=category, level=level, message=message, wallet_id=wallet_id))


# Singleton
reconciler = Reconciler()
