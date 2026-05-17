"""
Notifier — sends bot events to an external chat channel (Telegram or Discord).

Configured via AppSetting keys (saved from the Settings page):
  - notifier_provider: "telegram" | "discord" | "none"
  - notifier_telegram_bot_token
  - notifier_telegram_chat_id
  - notifier_discord_webhook_url
  - notifier_min_level: "info" | "warn" | "error"  (filter low-importance events)

All sends are best-effort and never raise — if the network is down the bot
keeps trading and the failure is logged to ActivityLog.
"""
from __future__ import annotations

from typing import Any

import httpx

from config import bot_config
from database.db import session_scope
from database.models import ActivityLog
from utils.logger import get_logger

logger = get_logger(__name__)

LEVEL_RANK = {"info": 0, "warn": 1, "error": 2}


def _cfg() -> dict[str, str]:
    return {
        "provider": bot_config.get("notifier_provider") or "none",
        "tg_token": bot_config.get("notifier_telegram_bot_token") or "",
        "tg_chat": bot_config.get("notifier_telegram_chat_id") or "",
        "discord_url": bot_config.get("notifier_discord_webhook_url") or "",
        "min_level": bot_config.get("notifier_min_level") or "info",
    }


def is_configured() -> bool:
    c = _cfg()
    if c["provider"] == "telegram":
        return bool(c["tg_token"] and c["tg_chat"])
    if c["provider"] == "discord":
        return bool(c["discord_url"])
    return False


def notify(message: str, *, level: str = "info", category: str = "bot") -> dict[str, Any]:
    """
    Send a notification. Returns {ok, provider, error?}. Never raises.
    """
    c = _cfg()
    provider = c["provider"]
    min_rank = LEVEL_RANK.get(c["min_level"], 0)
    msg_rank = LEVEL_RANK.get(level, 0)
    if msg_rank < min_rank:
        return {"ok": True, "skipped": "below_min_level"}
    if provider == "none":
        return {"ok": True, "skipped": "no_provider"}

    prefix = {"info": "", "warn": "[WARN] ", "error": "[ERROR] "}.get(level, "")
    text = f"{prefix}{category}: {message}"

    try:
        if provider == "telegram":
            return _send_telegram(c["tg_token"], c["tg_chat"], text)
        if provider == "discord":
            return _send_discord(c["discord_url"], text)
        return {"ok": False, "error": f"Unknown provider: {provider}"}
    except Exception as e:
        logger.warning("Notifier send failed: %s", e)
        with session_scope() as s:
            s.add(
                ActivityLog(
                    category="notifier",
                    level="warn",
                    message=f"Send failed via {provider}: {e}",
                )
            )
        return {"ok": False, "error": str(e)}


def _send_telegram(token: str, chat_id: str, text: str) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    with httpx.Client(timeout=8.0) as c:
        r = c.post(url, json={"chat_id": chat_id, "text": text[:4000]})
    return {"ok": r.status_code == 200, "provider": "telegram", "status": r.status_code}


def _send_discord(webhook_url: str, text: str) -> dict[str, Any]:
    with httpx.Client(timeout=8.0) as c:
        r = c.post(webhook_url, json={"content": text[:1900]})
    return {"ok": r.status_code in (200, 204), "provider": "discord", "status": r.status_code}


def send_test() -> dict[str, Any]:
    """User clicks 'Send test' in Settings."""
    return notify("AlphaPilot test notification — channel is wired up.", level="info", category="setup")
