"""
Claude-driven trade decision engine.

This module is the "brain" of AlphaPilot. For every candidate trade the bot
considers, we hand Claude a structured packet containing:

  - the technical signal from strategy_engine (momentum, mean reversion, etc.)
  - the live market snapshot (price, volatility proxy, recent returns)
  - the wallet's current open positions and recent realized P&L
  - the wallet's risk profile + caps (max position USD, leverage cap)
  - the current learned playbook of rules and mistakes (from AILearningMemory)
  - the global trading regime hints (kill switch, daily loss budget, etc.)

Claude returns a strict JSON object describing what it wants to do:

    {
      "action": "BUY" | "SELL" | "HOLD" | "CLOSE",
      "confidence": 0.0 - 1.0,
      "size_multiplier": 0.0 - 1.0,    # fraction of the bot's default size
      "stop_loss_pct": 0.0 - 0.20,
      "take_profit_pct": 0.0 - 0.50,
      "rationale": "short paragraph",
      "key_factors": ["..."],
      "risk_flags": ["..."]
    }

The engine NEVER blindly trusts Claude:
  - all sizing/leverage requests are clamped to wallet caps
  - if Claude fails, times out, or returns invalid JSON, we fall back to the
    raw technical signal (graceful degradation — trading never stops because
    the LLM hiccupped)
  - every decision (Claude's *and* the fallback) is persisted to ClaudeDecision
    so the Training Center can show audit history and so post-trade reflection
    can correlate fills with the reasoning that led to them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ai.claude_learning import build_playbook
from database.db import session_scope
from database.models import ClaudeDecision, PaperTrade, Wallet
from services.claude_client import chat as claude_chat
from services.claude_client import is_configured as claude_is_configured
from trading.strategy_engine import Signal
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


# How long to wait for Claude before falling back to the technical signal.
# We keep this aggressive: the bot tick cadence is short, and a stale LLM
# response is worse than a fresh technical signal.
DECISION_MAX_TOKENS = 700
DECISION_TEMPERATURE = 0.2  # decisions should be near-deterministic


SYSTEM_PROMPT_BASE = """You are AlphaPilot, an autonomous trading copilot operating in PAPER mode.

You make conservative, data-driven decisions and you NEVER hallucinate market data. \
You only use the numbers and context provided in the user message. If the context \
is incomplete or contradictory, prefer HOLD.

Trading rules you must always obey:
  1. NEVER recommend a position larger than 1.0x the bot's default size.
  2. NEVER recommend leverage greater than the wallet's max_leverage.
  3. Stop-loss is REQUIRED on every BUY/SELL action (in [0.005, 0.20]).
  4. Take-profit is REQUIRED on every BUY/SELL action (in [0.005, 0.50]).
  5. If recent trades on this symbol have lost money, demand higher confidence.
  6. If the kill switch is engaged or the daily loss budget is spent, return HOLD.
  7. Confidence must reflect EVIDENCE strength, not optimism. If you only see
     one weak signal, confidence should be < 0.55.

Your output MUST be a single JSON object with exactly these keys:
  action, confidence, size_multiplier, stop_loss_pct, take_profit_pct,
  rationale, key_factors, risk_flags

No prose. No markdown. No code fences. Just JSON.
"""


@dataclass
class TradeDecision:
    """Final, clamped decision the bot will act on."""
    action: str = "HOLD"           # BUY / SELL / HOLD / CLOSE
    confidence: float = 0.0        # 0..1
    size_multiplier: float = 1.0   # 0..1 (fraction of cfg.position_size_usd)
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    rationale: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    source: str = "technical"      # "claude" / "technical" / "fallback"
    model: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "size_multiplier": round(self.size_multiplier, 4),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "rationale": self.rationale,
            "key_factors": self.key_factors,
            "risk_flags": self.risk_flags,
            "source": self.source,
            "model": self.model,
        }


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #


def decide(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    extra_context: dict[str, Any] | None = None,
) -> TradeDecision:
    """
    Produce a trade decision for one (wallet, symbol) candidate.

    Always returns a TradeDecision (never raises). Falls back to the technical
    signal if Claude is not configured, errors out, or returns invalid output.
    Every call results in a persisted ClaudeDecision row for auditability.
    """
    fallback = _technical_fallback(technical_signal)

    if not claude_is_configured():
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="")
        return fallback

    try:
        prompt = _build_user_prompt(
            wallet=wallet,
            symbol=symbol,
            price=price,
            technical_signal=technical_signal,
            strategy_type=strategy_type,
            extra_context=extra_context or {},
        )
        system_prompt = _build_system_prompt(wallet)

        result = claude_chat(
            prompt=prompt,
            system=system_prompt,
            max_tokens=DECISION_MAX_TOKENS,
            temperature=DECISION_TEMPERATURE,
        )
        if not result.get("ok"):
            logger.warning("Claude decision call failed: %s", result.get("error"))
            _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used=prompt)
            return fallback

        text = result.get("text", "")
        parsed = _parse_decision_json(text)
        if parsed is None:
            logger.warning("Claude returned non-JSON; falling back. text=%r", text[:200])
            _persist_decision(
                wallet, symbol, price, technical_signal, fallback,
                prompt_used=prompt, raw_text=text, source_override="fallback",
            )
            return fallback

        decision = _normalize_and_clamp(parsed, wallet)
        decision.source = "claude"
        decision.model = result.get("raw", {}).get("model", "") or ""
        decision.raw_text = text

        _persist_decision(wallet, symbol, price, technical_signal, decision, prompt_used=prompt, raw_text=text)
        return decision

    except Exception as e:
        logger.exception("Claude decision engine raised: %s", e)
        _persist_decision(wallet, symbol, price, technical_signal, fallback, prompt_used="", raw_text=str(e))
        return fallback


# ---------------------------------------------------------------------------- #
# Prompt construction
# ---------------------------------------------------------------------------- #


def _build_system_prompt(wallet: dict[str, Any]) -> str:
    """Base rules + the wallet-specific risk profile + the learned playbook."""
    playbook = build_playbook(limit=25)
    risk_lines = [
        f"  - wallet_name: {wallet.get('name')}",
        f"  - platform: {wallet.get('platform')}",
        f"  - trading_mode: {wallet.get('trading_mode', 'paper')}",
        f"  - max_position_usd: {wallet.get('max_position_usd', 0)}",
        f"  - max_open_positions: {wallet.get('max_open_positions', 0)}",
        f"  - max_leverage: {wallet.get('max_leverage', 1.0)}",
        f"  - futures_enabled: {wallet.get('futures_enabled', False)}",
    ]
    risk_block = "WALLET RISK PROFILE:\n" + "\n".join(risk_lines)
    playbook_block = ""
    if playbook:
        playbook_block = (
            "LEARNED PLAYBOOK (rules and lessons from past paper trading):\n"
            + "\n".join(f"  - {p}" for p in playbook)
            + "\n\nApply these rules. If a new decision contradicts a learned rule, prefer the rule unless evidence is overwhelming."
        )
    return f"{SYSTEM_PROMPT_BASE}\n\n{risk_block}\n\n{playbook_block}".strip()


def _build_user_prompt(
    *,
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    strategy_type: str,
    extra_context: dict[str, Any],
) -> str:
    """Compact, machine-readable context. Keep this lean — every token costs."""
    sig_meta = getattr(technical_signal, "metadata", {}) or {}

    # Pull recent realized PnL + open positions from this wallet.
    open_positions, recent_trades = _wallet_recent_history(int(wallet["id"]))

    payload = {
        "candidate": {
            "symbol": symbol,
            "price": round(price, 8),
            "strategy_type": strategy_type,
            "technical_side": technical_signal.side,
            "technical_confidence": round(float(technical_signal.confidence or 0), 4),
            "technical_reasoning": technical_signal.reasoning,
            "indicators": {k: _round_or_str(v) for k, v in sig_meta.items()},
        },
        "wallet_state": {
            "paper_balance": round(float(wallet.get("paper_balance", 0)), 2),
            "open_position_count": len(open_positions),
            "open_positions": open_positions,
        },
        "recent_history": {
            "last_10_closed_trades": recent_trades,
        },
        "extra_context": extra_context,
        "now_utc": utcnow().isoformat(),
    }

    return (
        "Decide the next action for this candidate. "
        "Return ONLY the JSON object specified in your instructions.\n\n"
        f"{json.dumps(payload, default=str)}"
    )


def _wallet_recent_history(wallet_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compact recent-history snapshot used for prompt context."""
    with session_scope() as s:
        opens = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "open")
            .order_by(PaperTrade.opened_at.desc())
            .limit(10)
            .all()
        )
        closed = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id, PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc())
            .limit(10)
            .all()
        )
        open_positions = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "qty": float(t.qty),
                "entry": float(t.entry_price),
                "unrealized_pnl": float(t.unrealized_pnl or 0),
                "age_min": _minutes_since(t.opened_at),
            }
            for t in opens
        ]
        recent_closed = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "pnl": float(t.realized_pnl or 0),
                "confidence": float(t.confidence or 0),
                "held_min": _minutes_between(t.opened_at, t.closed_at),
            }
            for t in closed
        ]
    return open_positions, recent_closed


# ---------------------------------------------------------------------------- #
# Parsing + clamping
# ---------------------------------------------------------------------------- #


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_decision_json(text: str) -> dict[str, Any] | None:
    """Tolerantly extract a JSON object from Claude's response."""
    if not text:
        return None
    # Fast path: the whole thing is JSON.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the first {...} block.
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_and_clamp(parsed: dict[str, Any], wallet: dict[str, Any]) -> TradeDecision:
    """Coerce Claude's JSON into a safe TradeDecision."""
    action = str(parsed.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "SELL", "HOLD", "CLOSE"}:
        action = "HOLD"

    conf = _to_float(parsed.get("confidence"), 0.0)
    conf = max(0.0, min(conf, 1.0))

    size_mult = _to_float(parsed.get("size_multiplier"), 1.0)
    size_mult = max(0.0, min(size_mult, 1.0))

    sl = _to_float(parsed.get("stop_loss_pct"), 0.05)
    sl = max(0.005, min(sl, 0.20))

    tp = _to_float(parsed.get("take_profit_pct"), 0.10)
    tp = max(0.005, min(tp, 0.50))

    factors = parsed.get("key_factors") or []
    if not isinstance(factors, list):
        factors = [str(factors)]
    factors = [str(x)[:200] for x in factors][:8]

    flags = parsed.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(x)[:200] for x in flags][:8]

    rationale = str(parsed.get("rationale", ""))[:1500]

    return TradeDecision(
        action=action,
        confidence=conf,
        size_multiplier=size_mult,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        rationale=rationale,
        key_factors=factors,
        risk_flags=flags,
    )


def _technical_fallback(signal: Signal) -> TradeDecision:
    side = (signal.side or "HOLD").upper()
    if side not in {"BUY", "SELL", "HOLD"}:
        side = "HOLD"
    return TradeDecision(
        action=side,
        confidence=float(signal.confidence or 0.0),
        size_multiplier=1.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        rationale=signal.reasoning or "Fallback: technical signal only.",
        key_factors=[],
        risk_flags=["claude_unavailable"],
        source="technical",
    )


# ---------------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------------- #


def _persist_decision(
    wallet: dict[str, Any],
    symbol: str,
    price: float,
    technical_signal: Signal,
    decision: TradeDecision,
    *,
    prompt_used: str = "",
    raw_text: str = "",
    source_override: str | None = None,
) -> None:
    try:
        with session_scope() as s:
            row = ClaudeDecision(
                wallet_id=int(wallet["id"]),
                symbol=symbol,
                price=float(price),
                technical_side=technical_signal.side,
                technical_confidence=float(technical_signal.confidence or 0),
                action=decision.action,
                confidence=float(decision.confidence),
                size_multiplier=float(decision.size_multiplier),
                stop_loss_pct=float(decision.stop_loss_pct),
                take_profit_pct=float(decision.take_profit_pct),
                rationale=decision.rationale or "",
                key_factors=json.dumps(decision.key_factors)[:4000],
                risk_flags=json.dumps(decision.risk_flags)[:2000],
                source=source_override or decision.source,
                model=decision.model or "",
                prompt_used=(prompt_used or "")[:8000],
                raw_response=(raw_text or decision.raw_text or "")[:8000],
            )
            s.add(row)
    except Exception:
        # Never let logging fail the trade.
        logger.exception("Failed to persist ClaudeDecision row.")


# ---------------------------------------------------------------------------- #
# Tiny helpers
# ---------------------------------------------------------------------------- #


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _round_or_str(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


def _minutes_since(dt: Any) -> int:
    if not dt:
        return 0
    try:
        return int((utcnow() - dt).total_seconds() // 60)
    except Exception:
        return 0


def _minutes_between(a: Any, b: Any) -> int:
    if not a or not b:
        return 0
    try:
        return int((b - a).total_seconds() // 60)
    except Exception:
        return 0
