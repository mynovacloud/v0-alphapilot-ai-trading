"""
Crypto exchange connectors.

Three platforms have REAL authenticated implementations:
  - Coinbase Advanced Trade (read + LIVE TRADING with safety gate)
  - Binance (binance.com) - read only
  - Kraken - read only

The rest (Crypto.com, generic ones) fall back to a public-price-only mock so
the rest of the app keeps working. Live prices for every connector are
provided by CoinGecko via `connectors.live_prices`.

Live trading is gated by:
  1. settings.live_trading_enabled (env var)
  2. The wallet's `trading_mode` field ("live" or "live_shadow")
  3. RiskManager evaluation
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import uuid
from typing import Any

import httpx

from config.settings import settings
from connectors.base_connector import BaseConnector
from connectors.live_prices import get_price as live_price


# --------------------------------------------------------------------- #
# Mock fallback (public price only)
# --------------------------------------------------------------------- #


class _MockPriceConnector(BaseConnector):
    """For platforms we haven't wired up real auth for yet.

    Public price comes from CoinGecko (real). Auth/balance is mocked.
    """

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        result = live_price(symbol)
        if result.get("ok"):
            price = float(result["price"])
            return {
                "platform": self.platform,
                "symbol": symbol,
                "current_price": price,
                "bid": round(price * 0.999, 2),
                "ask": round(price * 1.001, 2),
                "source": result["source"],
                "live": True,
            }
        return {"platform": self.platform, "symbol": symbol, "error": result.get("error"), "live": False}

    def validate_credentials(self) -> dict[str, Any]:
        if not self.api_key and not self.api_secret:
            return {"valid": True, "mock": True, "note": "No keys provided. Public price feed only."}
        return {
            "valid": True,
            "mock": True,
            "note": f"{self.platform} authenticated calls aren't wired up yet — keys saved for future use.",
        }


class CryptocomConnector(_MockPriceConnector):
    platform = "Crypto.com"


# --------------------------------------------------------------------- #
# Coinbase Advanced Trade (REAL)
# --------------------------------------------------------------------- #


class CoinbaseConnector(BaseConnector):
    """Real Coinbase Advanced Trade connector (read-only)."""

    platform = "Coinbase"
    BASE_URL = "https://api.coinbase.com"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        return _market_data_via_coingecko(self.platform, symbol)

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = f"{timestamp}{method}{path}{body}"
        return hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        timestamp = str(int(time.time()))
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": self._sign(timestamp, method, path, body),
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def validate_credentials(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"valid": False, "error": "Coinbase requires both API key and API secret."}
        try:
            path = "/api/v3/brokerage/accounts"
            with httpx.Client(timeout=15.0) as c:
                r = c.get(self.BASE_URL + path, headers=self._auth_headers("GET", path))
            if r.status_code == 200:
                accounts = r.json().get("accounts", [])
                return {
                    "valid": True,
                    "platform": self.platform,
                    "accounts_visible": len(accounts),
                    "live": True,
                }
            return {"valid": False, "status": r.status_code, "error": r.text[:300]}
        except Exception as e:
            return {"valid": False, "error": f"Coinbase auth failed: {e}"}

    def fetch_balance(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"cash": 0.0, "currency": "USD", "live": False, "note": "no keys"}
        try:
            path = "/api/v3/brokerage/accounts"
            with httpx.Client(timeout=15.0) as c:
                r = c.get(self.BASE_URL + path, headers=self._auth_headers("GET", path))
            r.raise_for_status()
            accounts = r.json().get("accounts", [])
            total_usd = 0.0
            balances: list[dict[str, Any]] = []
            for a in accounts:
                bal = a.get("available_balance") or {}
                amt = float(bal.get("value", 0))
                cur = bal.get("currency", "")
                if amt > 0:
                    balances.append({"currency": cur, "amount": amt})
                if cur in {"USD", "USDC", "USDT"}:
                    total_usd += amt
            return {"cash": round(total_usd, 2), "currency": "USD", "balances": balances, "live": True}
        except Exception as e:
            return {"error": f"fetch_balance failed: {e}", "live": False}

    # =====================================================================
    # LIVE TRADING (Coinbase Advanced Trade)
    # =====================================================================
    #
    # Every order method:
    #   - Returns a dict with shape: {"ok": bool, "order_id": str, "raw": ..., "error": ...}
    #   - Generates a client_order_id (UUID) for idempotency
    #   - Refuses to run if settings.live_trading_enabled is False
    #
    # Coinbase Advanced Trade order payload shape:
    #   {
    #     "client_order_id": "...",
    #     "product_id": "BTC-USD",
    #     "side": "BUY" | "SELL",
    #     "order_configuration": { ... }  # one of: market_market_ioc, limit_limit_gtc,
    #                                     #         stop_limit_stop_limit_gtc, etc.
    #   }
    # ---------------------------------------------------------------------

    def _post_signed(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._auth_headers("POST", path, body)
        with httpx.Client(timeout=15.0) as c:
            r = c.post(self.BASE_URL + path, headers=headers, content=body)
        try:
            data = r.json()
        except Exception:
            data = {"raw_text": r.text}
        return {"status_code": r.status_code, "json": data}

    def _ensure_live_enabled(self) -> dict[str, Any] | None:
        if not settings.live_trading_enabled:
            return {
                "ok": False,
                "error": "Live trading disabled. Set LIVE_TRADING_ENABLED=true in .env.",
            }
        if not self.api_key or not self.api_secret:
            return {"ok": False, "error": "Coinbase requires both API key and API secret."}
        return None

    @staticmethod
    def _new_client_order_id() -> str:
        return str(uuid.uuid4())

    def place_market_order(
        self,
        symbol: str,
        side: str,
        base_qty: float | None = None,
        quote_size: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Market order. Specify EITHER base_qty (e.g. 0.001 BTC) OR quote_size
        (e.g. $25 of BTC). Coinbase requires:
          - BUY market: quote_size only (USD amount)
          - SELL market: base_size only (asset amount)
        """
        gate = self._ensure_live_enabled()
        if gate:
            return gate

        side = side.upper()
        coid = client_order_id or self._new_client_order_id()

        if side == "BUY":
            if quote_size is None:
                return {"ok": False, "error": "Coinbase market BUY requires quote_size (USD)."}
            cfg = {"market_market_ioc": {"quote_size": str(quote_size)}}
        elif side == "SELL":
            if base_qty is None:
                return {"ok": False, "error": "Coinbase market SELL requires base_qty (asset amount)."}
            cfg = {"market_market_ioc": {"base_size": str(base_qty)}}
        else:
            return {"ok": False, "error": f"Invalid side: {side}"}

        payload = {
            "client_order_id": coid,
            "product_id": symbol,
            "side": side,
            "order_configuration": cfg,
        }
        return self._submit_order(payload, coid)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        limit_price: float,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Limit order, GTC by default."""
        gate = self._ensure_live_enabled()
        if gate:
            return gate

        coid = client_order_id or self._new_client_order_id()
        cfg_key = "limit_limit_gtc" if time_in_force == "GTC" else "limit_limit_gtd"
        payload = {
            "client_order_id": coid,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                cfg_key: {
                    "base_size": str(base_qty),
                    "limit_price": str(limit_price),
                    "post_only": False,
                }
            },
        }
        return self._submit_order(payload, coid)

    def place_stop_limit_order(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        stop_price: float,
        limit_price: float,
        stop_direction: str = "STOP_DIRECTION_STOP_DOWN",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Stop-limit order. `stop_direction` is one of:
          STOP_DIRECTION_STOP_UP   -> trigger when price >= stop_price (used for SELL stop)
          STOP_DIRECTION_STOP_DOWN -> trigger when price <= stop_price (used for BUY stop)

        Common usage: SELL stop-loss below current price -> STOP_DIRECTION_STOP_DOWN.
        """
        gate = self._ensure_live_enabled()
        if gate:
            return gate

        coid = client_order_id or self._new_client_order_id()
        payload = {
            "client_order_id": coid,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(base_qty),
                    "limit_price": str(limit_price),
                    "stop_price": str(stop_price),
                    "stop_direction": stop_direction,
                }
            },
        }
        return self._submit_order(payload, coid)

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        base_qty: float,
        limit_price: float,
        stop_trigger_price: float,
        stop_limit_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Bracket order: a TP limit + a SL stop-limit attached to the entry.
        Coinbase exposes this as `trigger_bracket_gtc`.
        """
        gate = self._ensure_live_enabled()
        if gate:
            return gate

        coid = client_order_id or self._new_client_order_id()
        payload = {
            "client_order_id": coid,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                "trigger_bracket_gtc": {
                    "base_size": str(base_qty),
                    "limit_price": str(limit_price),
                    "stop_trigger_price": str(stop_trigger_price),
                }
            },
        }
        return self._submit_order(payload, coid)

    def _submit_order(self, payload: dict[str, Any], client_order_id: str) -> dict[str, Any]:
        path = "/api/v3/brokerage/orders"
        try:
            res = self._post_signed(path, payload)
        except Exception as e:
            return {"ok": False, "error": f"network: {e}", "client_order_id": client_order_id}

        body = res.get("json", {})
        if res["status_code"] in (200, 201) and body.get("success"):
            order_id = (body.get("success_response") or {}).get("order_id", "")
            return {
                "ok": True,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "raw": body,
            }
        # Coinbase failure
        err = (body.get("error_response") or {}).get("message") or body.get("error") or body.get("message") or "unknown error"
        return {
            "ok": False,
            "error": err,
            "client_order_id": client_order_id,
            "status_code": res["status_code"],
            "raw": body,
        }

    def cancel_orders(self, exchange_order_ids: list[str]) -> dict[str, Any]:
        """Cancel one or more open orders. Coinbase accepts a batch."""
        gate = self._ensure_live_enabled()
        if gate:
            return gate
        path = "/api/v3/brokerage/orders/batch_cancel"
        payload = {"order_ids": exchange_order_ids}
        try:
            res = self._post_signed(path, payload)
            return {"ok": res["status_code"] == 200, "raw": res.get("json", {})}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_orders(self, status: str = "OPEN", limit: int = 250) -> dict[str, Any]:
        """
        List orders for the authenticated account.

        status:
          - "OPEN"   -> only currently working orders
          - "FILLED" -> recently filled
          - "CANCELLED"
          - ""       -> all (Coinbase returns most recent first)
        """
        if not self.api_key or not self.api_secret:
            return {"ok": False, "error": "no credentials"}
        path = "/api/v3/brokerage/orders/historical/batch"
        query = f"?limit={limit}"
        if status:
            query += f"&order_status={status}"
        try:
            with httpx.Client(timeout=15.0) as c:
                r = c.get(
                    self.BASE_URL + path + query,
                    headers=self._auth_headers("GET", path),
                )
            data = r.json()
            if r.status_code != 200:
                return {"ok": False, "error": data, "status": r.status_code}
            return {
                "ok": True,
                "orders": [
                    {
                        "exchange_order_id": o.get("order_id"),
                        "client_order_id": o.get("client_order_id"),
                        "product_id": o.get("product_id"),
                        "side": o.get("side"),
                        "status": o.get("status"),
                        "filled_size": float(o.get("filled_size", 0) or 0),
                        "average_filled_price": float(o.get("average_filled_price", 0) or 0),
                        "total_fees": float(o.get("total_fees", 0) or 0),
                        "created_time": o.get("created_time"),
                    }
                    for o in data.get("orders", [])
                ],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_order(self, exchange_order_id: str) -> dict[str, Any]:
        """Fetch the latest state of one order."""
        if not self.api_key or not self.api_secret:
            return {"ok": False, "error": "no credentials"}
        path = f"/api/v3/brokerage/orders/historical/{exchange_order_id}"
        try:
            with httpx.Client(timeout=15.0) as c:
                r = c.get(self.BASE_URL + path, headers=self._auth_headers("GET", path))
            data = r.json()
            if r.status_code != 200:
                return {"ok": False, "error": data.get("error_response", data), "status": r.status_code}
            order = data.get("order", {})
            return {
                "ok": True,
                "status": order.get("status"),  # OPEN / FILLED / CANCELLED / FAILED / EXPIRED
                "filled_size": float(order.get("filled_size", 0) or 0),
                "average_filled_price": float(order.get("average_filled_price", 0) or 0),
                "total_fees": float(order.get("total_fees", 0) or 0),
                "raw": order,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_products(self, product_type: str = "SPOT") -> dict[str, Any]:
        """List available products (markets) from Coinbase."""
        path = f"/api/v3/brokerage/products?product_type={product_type}"
        try:
            with httpx.Client(timeout=15.0) as c:
                # public endpoint also works unauthenticated, but we sign anyway
                # so the same path is used in tests.
                if self.api_key and self.api_secret:
                    r = c.get(
                        self.BASE_URL + path,
                        headers=self._auth_headers("GET", "/api/v3/brokerage/products"),
                    )
                else:
                    r = c.get(self.BASE_URL + path)
            data = r.json()
            if r.status_code != 200:
                return {"ok": False, "error": data}
            return {
                "ok": True,
                "products": [
                    {
                        "product_id": p.get("product_id"),
                        "base_currency": p.get("base_currency_id"),
                        "quote_currency": p.get("quote_currency_id"),
                        "min_market_funds": p.get("quote_min_size"),
                        "base_increment": p.get("base_increment"),
                        "quote_increment": p.get("quote_increment"),
                    }
                    for p in data.get("products", [])
                ],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------- #
# Binance (REAL)
# --------------------------------------------------------------------- #


class BinanceConnector(BaseConnector):
    platform = "Binance"
    BASE_URL = "https://api.binance.com"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        return _market_data_via_coingecko(self.platform, symbol)

    def _sign(self, query: str) -> str:
        return hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def validate_credentials(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"valid": False, "error": "Binance requires both API key and API secret."}
        try:
            ts = int(time.time() * 1000)
            query = f"timestamp={ts}"
            sig = self._sign(query)
            url = f"{self.BASE_URL}/api/v3/account?{query}&signature={sig}"
            headers = {"X-MBX-APIKEY": self.api_key}
            with httpx.Client(timeout=15.0) as c:
                r = c.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                return {
                    "valid": True,
                    "platform": self.platform,
                    "can_trade": data.get("canTrade"),
                    "live": True,
                }
            return {"valid": False, "status": r.status_code, "error": r.text[:300]}
        except Exception as e:
            return {"valid": False, "error": f"Binance auth failed: {e}"}

    def fetch_balance(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"cash": 0.0, "currency": "USD", "live": False, "note": "no keys"}
        try:
            ts = int(time.time() * 1000)
            query = f"timestamp={ts}"
            sig = self._sign(query)
            url = f"{self.BASE_URL}/api/v3/account?{query}&signature={sig}"
            headers = {"X-MBX-APIKEY": self.api_key}
            with httpx.Client(timeout=15.0) as c:
                r = c.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            balances: list[dict[str, Any]] = []
            usd = 0.0
            for b in data.get("balances", []):
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                total = free + locked
                if total > 0:
                    balances.append({"currency": b["asset"], "amount": total})
                if b["asset"] in {"USDT", "USDC", "BUSD", "USD"}:
                    usd += total
            return {"cash": round(usd, 2), "currency": "USD", "balances": balances, "live": True}
        except Exception as e:
            return {"error": f"fetch_balance failed: {e}", "live": False}


# --------------------------------------------------------------------- #
# Kraken (REAL)
# --------------------------------------------------------------------- #


class KrakenConnector(BaseConnector):
    platform = "Kraken"
    BASE_URL = "https://api.kraken.com"

    def fetch_market_data(self, symbol: str) -> dict[str, Any]:
        return _market_data_via_coingecko(self.platform, symbol)

    def _sign(self, urlpath: str, data: dict[str, Any], nonce: str) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        try:
            secret = base64.b64decode(self.api_secret)
        except Exception:
            return ""
        sig = hmac.new(secret, message, hashlib.sha512)
        return base64.b64encode(sig.digest()).decode()

    def validate_credentials(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"valid": False, "error": "Kraken requires both API key and base64 API secret."}
        try:
            urlpath = "/0/private/Balance"
            nonce = str(int(time.time() * 1000))
            data = {"nonce": nonce}
            sig = self._sign(urlpath, data, nonce)
            if not sig:
                return {"valid": False, "error": "Kraken secret must be base64-encoded."}
            headers = {"API-Key": self.api_key, "API-Sign": sig}
            with httpx.Client(timeout=15.0) as c:
                r = c.post(self.BASE_URL + urlpath, headers=headers, data=data)
            payload = r.json()
            if payload.get("error"):
                return {"valid": False, "error": "; ".join(payload["error"])}
            return {"valid": True, "platform": self.platform, "live": True}
        except Exception as e:
            return {"valid": False, "error": f"Kraken auth failed: {e}"}

    def fetch_balance(self) -> dict[str, Any]:
        if not self.api_key or not self.api_secret:
            return {"cash": 0.0, "currency": "USD", "live": False, "note": "no keys"}
        try:
            urlpath = "/0/private/Balance"
            nonce = str(int(time.time() * 1000))
            data = {"nonce": nonce}
            sig = self._sign(urlpath, data, nonce)
            headers = {"API-Key": self.api_key, "API-Sign": sig}
            with httpx.Client(timeout=15.0) as c:
                r = c.post(self.BASE_URL + urlpath, headers=headers, data=data)
            payload = r.json()
            if payload.get("error"):
                return {"error": "; ".join(payload["error"]), "live": False}
            result = payload.get("result", {})
            balances: list[dict[str, Any]] = []
            usd = 0.0
            for k, v in result.items():
                amt = float(v)
                if amt <= 0:
                    continue
                balances.append({"currency": k, "amount": amt})
                if k in {"ZUSD", "USD", "USDT", "USDC"}:
                    usd += amt
            return {"cash": round(usd, 2), "currency": "USD", "balances": balances, "live": True}
        except Exception as e:
            return {"error": f"fetch_balance failed: {e}", "live": False}


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _market_data_via_coingecko(platform: str, symbol: str) -> dict[str, Any]:
    result = live_price(symbol)
    if result.get("ok"):
        price = float(result["price"])
        return {
            "platform": platform,
            "symbol": symbol,
            "current_price": price,
            "bid": round(price * 0.999, 2),
            "ask": round(price * 1.001, 2),
            "source": result["source"],
            "live": True,
        }
    return {"platform": platform, "symbol": symbol, "error": result.get("error"), "live": False}
