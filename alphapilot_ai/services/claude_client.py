"""
Claude (Anthropic) client for AlphaPilot.

Reads its API key + model from bot_config so they can be edited from the
Settings page without redeploying. All network calls are best-effort and
never raise — failures return {"ok": False, "error": "..."} so trading
loops never die because the LLM is down.

Public API:
    is_configured() -> bool
    get_model() -> str
    chat(prompt, system=None, max_tokens=1024, temperature=0.4) -> dict
    send_test() -> dict
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from config import bot_config
from utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TIMEOUT = 30.0


def _api_key() -> str:
    return (bot_config.get("anthropic_api_key") or "").strip()


def get_model() -> str:
    return (bot_config.get("anthropic_model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def is_configured() -> bool:
    """Cheap check that does not hit the network."""
    return bool(_api_key())


def chat(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.4,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Send a single user message to Claude. Returns:
        {"ok": True,  "text": "...", "raw": <full response>}
        {"ok": False, "error": "..."}
    """
    key = _api_key()
    if not key:
        return {"ok": False, "error": "Anthropic API key not configured."}

    body: dict[str, Any] = {
        "model": (model or get_model()),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
            r = c.post(API_URL, headers=headers, json=body)
        if r.status_code != 200:
            # Anthropic returns structured errors — surface the message if present.
            try:
                err = r.json().get("error", {}).get("message") or r.text
            except Exception:
                err = r.text
            return {"ok": False, "error": f"HTTP {r.status_code}: {err}", "status": r.status_code}

        data = r.json()
        # `content` is a list of blocks; concatenate any text blocks.
        text_parts = [
            blk.get("text", "")
            for blk in data.get("content", [])
            if blk.get("type") == "text"
        ]
        return {"ok": True, "text": "".join(text_parts).strip(), "raw": data}
    except Exception as e:
        logger.exception("Claude chat failed.")
        return {"ok": False, "error": str(e)}


def send_test() -> dict[str, Any]:
    """Used by the Settings page Test button."""
    if not is_configured():
        return {"ok": False, "error": "No API key set."}
    res = chat(
        prompt="Reply with exactly the word: OK",
        max_tokens=10,
        temperature=0.0,
    )
    if not res.get("ok"):
        return res
    return {"ok": True, "model": get_model(), "text": res.get("text", "")}
