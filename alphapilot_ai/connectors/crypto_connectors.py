"""
Crypto exchange connectors.

Three platforms have REAL authenticated implementations:
  - Coinbase Advanced Trade
  - Binance (binance.com)
  - Kraken

The rest (Crypto.com, generic ones) fall back to a public-price-only mock so
the rest of the app keeps working. Live prices for every connector are
provided by CoinGecko via `connectors.live_prices`.

Live trading is still locked at the framework level (see BaseConnector).
These connectors only READ from real APIs (auth + balances + prices).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any

import httpx

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
