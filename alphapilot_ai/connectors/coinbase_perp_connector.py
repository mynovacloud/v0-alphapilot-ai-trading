"""
Coinbase Perpetuals connector — uses the Coinbase Advanced Trade API perpetual
futures product family (CFM / INTX-style products like 'BTC-PERP-INTX').

This is a thin subclass of CoinbaseConnector that:
  - Adds `place_perp_order(symbol, side, base_qty, leverage, reduce_only)`.
  - Adds `set_leverage(product_id, leverage)`.
  - Adds `list_perp_positions()` to read open positions (with mark price,
    unrealized PnL, liquidation price, margin used).
  - Adds `close_perp_position(product_id)` to flatten a position with a single
    reduce-only market order in the opposite direction.

Notes:
  - Coinbase has both a US futures venue (CFM, retail) and Coinbase International
    Exchange (INTX, non-US perps). Both are accessed through the same Advanced
    Trade-style REST surface but with different product_ids and a different
    base URL for INTX. We expose `venue="cfm"` (default, US) or `venue="intx"`.
  - Live trading remains gated on `LIVE_TRADING_ENABLED`. With the gate off, every
    method returns {ok: False, error: "..."} just like the spot connector.
  - All methods return the same {ok, order_id, raw, error} shape so the
    autonomous bot doesn't care whether it's spot or perp.
"""
from __future__ import annotations

from typing import Any

import httpx

from config.settings import settings
from connectors.crypto_connectors import CoinbaseConnector


class CoinbasePerpConnector(CoinbaseConnector):
    """Perpetual-futures connector for Coinbase (CFM US futures or INTX perps)."""

    platform = "Coinbase Perp"

    # CFM (US futures, retail) shares the spot host.
    CFM_BASE_URL = "https://api.coinbase.com"
    # INTX (international, non-US) lives on a separate host.
    INTX_BASE_URL = "https://api.international.coinbase.com"

    def __init__(self, *args: Any, venue: str = "cfm", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.venue = venue if venue in {"cfm", "intx"} else "cfm"
        self.BASE_URL = self.INTX_BASE_URL if self.venue == "intx" else self.CFM_BASE_URL

    # ------------------------------------------------------------------ #
    # Position reads
    # ------------------------------------------------------------------ #

    def list_perp_positions(self) -> dict[str, Any]:
        """
        Return a normalized list of open perp positions:
          [{symbol, side, base_qty, mark_price, entry_price, unrealized_pnl,
            margin_used, leverage, liquidation_price}]
        """
        if not self.api_key or not self.api_secret:
            return {"ok": False, "error": "no credentials", "positions": []}
        # CFM positions endpoint:
        path = "/api/v3/brokerage/cfm/positions" if self.venue == "cfm" else "/api/v1/positions"
        try:
            with httpx.Client(timeout=15.0) as c:
                r = c.get(self.BASE_URL + path, headers=self._auth_headers("GET", path))
            data = r.json() if r.content else {}
            if r.status_code != 200:
                return {"ok": False, "error": data, "status": r.status_code, "positions": []}
            positions = data.get("positions") or data.get("results") or []
            normalized: list[dict[str, Any]] = []
            for p in positions:
                # Field names differ slightly between CFM and INTX. Pull both.
                normalized.append(
                    {
                        "symbol": p.get("product_id") or p.get("symbol"),
                        "side": (p.get("side") or p.get("position_side") or "LONG").upper(),
                        "base_qty": float(p.get("number_of_contracts") or p.get("net_size") or p.get("size") or 0),
                        "mark_price": float(p.get("mark_price") or p.get("vwap") or 0),
                        "entry_price": float(p.get("entry_vwap") or p.get("avg_entry_price") or 0),
                        "unrealized_pnl": float(p.get("unrealized_pnl") or 0),
                        "margin_used": float(p.get("margin") or p.get("im_contribution") or 0),
                        "leverage": float(p.get("leverage") or 0),
                        "liquidation_price": float(p.get("liquidation_price") or 0) or None,
                    }
                )
            return {"ok": True, "positions": normalized, "raw": data}
        except Exception as e:
            return {"ok": False, "error": str(e), "positions": []}

    # ------------------------------------------------------------------ #
    # Leverage / margin mode
    # ------------------------------------------------------------------ #

    def set_leverage(self, product_id: str, leverage: float) -> dict[str, Any]:
        """Apply a leverage setting to a product before placing a perp order."""
        gate = self._ensure_live_enabled()
        if gate:
            return gate
        if leverage <= 0:
            return {"ok": False, "error": "leverage must be > 0"}
        path = (
            "/api/v3/brokerage/cfm/intraday/margin_setting"
            if self.venue == "cfm"
            else "/api/v1/portfolios/margin"
        )
        payload = {"product_id": product_id, "leverage": float(leverage)}
        try:
            res = self._post_signed(path, payload)
            return {"ok": res["status_code"] == 200, "raw": res.get("json", {})}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

    def place_perp_order(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        *,
        leverage: float = 1.0,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a market order against a perpetual product. The bot uses this
        for both opens (reduce_only=False) and closes (reduce_only=True).

        symbol examples:
          - CFM:  "BTC-27DEC24-CDE"   (product id from Coinbase's perp listing)
          - INTX: "BTC-PERP-INTX"

        We always use a market order for now — the slippage from a 1-lot retail
        size is negligible compared to the model's edge, and limit-on-quote is
        a separate (later) feature.
        """
        gate = self._ensure_live_enabled()
        if gate:
            return gate
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            return {"ok": False, "error": f"invalid side {side}"}
        if base_qty <= 0:
            return {"ok": False, "error": "base_qty must be > 0"}

        # Best-effort: set leverage first. On INTX leverage is per-portfolio,
        # on CFM it's per-product. Either way we don't fail the order if this
        # call fails — the exchange will reject with a clearer error if needed.
        if leverage and leverage != 1.0:
            self.set_leverage(symbol, leverage)

        coid = client_order_id or self._new_client_order_id()
        path = "/api/v3/brokerage/orders"
        payload = {
            "client_order_id": coid,
            "product_id": symbol,
            "side": side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(base_qty),
                }
            },
        }
        if reduce_only:
            # Coinbase encodes reduce-only at the order level for perps.
            payload["leverage"] = str(leverage)
            payload["margin_type"] = "ISOLATED"
            payload["reduce_only"] = True
        try:
            res = self._post_signed(path, payload)
            j = res.get("json", {})
            ok = res["status_code"] == 200 and j.get("success", True)
            return {
                "ok": bool(ok),
                "client_order_id": coid,
                "order_id": (j.get("order_id") or j.get("success_response", {}).get("order_id")),
                "raw": j,
                "error": j.get("error_response") if not ok else None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "client_order_id": coid}

    # ------------------------------------------------------------------ #
    # Close-out helper
    # ------------------------------------------------------------------ #

    def close_perp_position(self, symbol: str) -> dict[str, Any]:
        """
        Flatten the open position in `symbol` with a single reduce-only market
        order. No-op (ok) if no position is open. Returns the order result so
        the caller can persist the fill.
        """
        positions = self.list_perp_positions()
        if not positions.get("ok"):
            return positions
        pos = next((p for p in positions["positions"] if p["symbol"] == symbol and p["base_qty"] != 0), None)
        if not pos:
            return {"ok": True, "skipped": "no_open_position"}
        # Opposite side, full size, reduce-only.
        opposite = "SELL" if pos["side"].startswith("LONG") or pos["side"] == "BUY" else "BUY"
        return self.place_perp_order(
            symbol=symbol,
            side=opposite,
            base_qty=abs(pos["base_qty"]),
            leverage=pos.get("leverage") or 1.0,
            reduce_only=True,
        )


# Sanity-check at import time that the parent has the methods we extend.
assert hasattr(CoinbasePerpConnector, "_post_signed"), "CoinbasePerpConnector requires CoinbaseConnector._post_signed"
