"""
Claude learning system.

This is the long-term memory + reflection loop that turns every paper trade
into a lesson Claude can use for the NEXT trade. The flow:

  1. A trade is opened by the bot (with rationale captured in ClaudeDecision).
  2. The trade closes (PaperTradingEngine.close_trade) — we get fill price + PnL.
  3. record_trade_outcome() asks Claude:
        "You decided X for these reasons. The outcome was Y. What rule should
         AlphaPilot remember for next time?"
     and stores the answer as a TradeReflection + 1..N AILearningMemory rows.
  4. build_playbook() returns the top-N currently active rules, ordered by
     weight + recency, for injection into the system prompt of every future
     decision call.
  5. consolidate_lessons() periodically asks Claude to merge near-duplicate
     rules and re-rank weights, so the playbook stays compact and high-signal.

Everything here is best-effort. If Claude is misconfigured or down, learning
silently degrades — the bot keeps trading on its technical signals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from database.db import session_scope
from database.models import (
    ActivityLog,
    AILearningMemory,
    ClaudeDecision,
    PaperTrade,
    TradeReflection,
)
from services.claude_client import chat as claude_chat
from services.claude_client import is_configured as claude_is_configured
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


REFLECTION_SYSTEM_PROMPT = """You are AlphaPilot's reflection coach. Your job is to extract
durable, generalizable trading lessons from a single paper-trade outcome.

You will receive: the original decision (action, rationale, key factors, risk flags),
the live market context, and the realized P&L. You must return a JSON object:

{
  "verdict": "good_call" | "bad_call" | "lucky" | "unlucky" | "neutral",
  "score": -1.0 .. 1.0,             # negative = mistake, positive = skill
  "lessons": [
    {"category": "lesson"|"mistake"|"rule", "content": "...", "weight": 0.1 .. 2.0}
  ],
  "summary": "two-sentence postmortem"
}

Rules:
  - Up to 3 lessons. Quality over quantity.
  - "lucky" = positive PnL but the decision was unjustified by evidence.
  - "unlucky" = negative PnL but the decision was sound.
  - "score" must reflect process quality, not outcome alone.
  - Lessons must be ACTIONABLE generalizations ("when X, do Y"), not narration.
  - No prose outside the JSON.
"""


CONSOLIDATION_SYSTEM_PROMPT = """You are AlphaPilot's playbook editor. You will receive the
current list of learned rules with weights. Your job is to:

  1. Merge near-duplicates into a single, sharper rule.
  2. Strengthen weights of rules that are confirmed by multiple recent trades.
  3. Demote (or remove) rules that contradict newer, higher-weight rules.

Return JSON:
{
  "keep": [{"id": <int>, "new_weight": float, "new_content": "rewritten"|null}],
  "delete": [<id>, <id>, ...],
  "create": [{"category": "rule", "content": "...", "weight": float}]
}

No prose outside the JSON.
"""


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #


def build_playbook(limit: int = 25) -> list[str]:
    """Return the top-N learned rules as plain strings, sorted by weight desc."""
    with session_scope() as s:
        rows = (
            s.query(AILearningMemory)
            .order_by(AILearningMemory.weight.desc(), AILearningMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.content for r in rows if r.content]


def get_playbook_with_metadata(limit: int = 100) -> list[dict[str, Any]]:
    """Same as build_playbook but with row metadata for the Training UI."""
    with session_scope() as s:
        rows = (
            s.query(AILearningMemory)
            .order_by(AILearningMemory.weight.desc(), AILearningMemory.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "category": r.category,
                "content": r.content,
                "weight": float(r.weight or 0),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def record_trade_outcome(trade_id: int) -> dict[str, Any]:
    """
    Called when a PaperTrade transitions to closed. Computes a reflection via
    Claude (if configured) and writes both a TradeReflection row and any new
    AILearningMemory rows. Idempotent: if a reflection already exists for this
    trade, returns it unchanged.
    """
    with session_scope() as s:
        trade = s.get(PaperTrade, trade_id)
        if not trade:
            return {"ok": False, "error": "trade not found"}
        if trade.status != "closed":
            return {"ok": False, "error": "trade not yet closed"}

        existing = (
            s.query(TradeReflection)
            .filter(TradeReflection.trade_id == trade_id)
            .first()
        )
        if existing:
            return {"ok": True, "reflection_id": existing.id, "cached": True}

        # Pull the originating ClaudeDecision (most recent for the same wallet+symbol
        # before the trade opened) for context. This is a heuristic match — we
        # don't currently link decisions to fills directly.
        decision = (
            s.query(ClaudeDecision)
            .filter(
                ClaudeDecision.wallet_id == trade.wallet_id,
                ClaudeDecision.symbol == trade.symbol,
                ClaudeDecision.created_at <= trade.opened_at,
            )
            .order_by(ClaudeDecision.created_at.desc())
            .first()
        )
        decision_payload = _decision_to_dict(decision) if decision else None

        trade_payload = {
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "qty": float(trade.qty),
            "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price or 0),
            "fees": float(trade.fees or 0),
            "slippage": float(trade.slippage or 0),
            "realized_pnl": float(trade.realized_pnl or 0),
            "confidence": float(trade.confidence or 0),
            "held_minutes": _minutes_between(trade.opened_at, trade.closed_at),
            "notes": trade.notes or "",
        }

    # If Claude isn't available, still record a no-LLM reflection so the
    # Training UI shows that the trade was processed.
    if not claude_is_configured():
        return _save_reflection(
            trade_id=trade_id,
            verdict="neutral",
            score=0.0,
            summary="Reflection skipped: Claude not configured.",
            lessons=[],
            decision_payload=decision_payload,
            trade_payload=trade_payload,
            raw="",
        )

    prompt_payload = {
        "decision": decision_payload,
        "trade": trade_payload,
        "now_utc": utcnow().isoformat(),
    }
    result = claude_chat(
        prompt=(
            "Reflect on this paper trade and produce JSON per your instructions.\n\n"
            f"{json.dumps(prompt_payload, default=str)}"
        ),
        system=REFLECTION_SYSTEM_PROMPT,
        max_tokens=900,
        temperature=0.2,
    )
    if not result.get("ok"):
        return _save_reflection(
            trade_id=trade_id,
            verdict="neutral",
            score=0.0,
            summary=f"Claude unavailable: {result.get('error', '')}",
            lessons=[],
            decision_payload=decision_payload,
            trade_payload=trade_payload,
            raw=result.get("error", ""),
        )

    parsed = _parse_json_loose(result.get("text", ""))
    if not parsed:
        return _save_reflection(
            trade_id=trade_id,
            verdict="neutral",
            score=0.0,
            summary="Could not parse Claude reflection.",
            lessons=[],
            decision_payload=decision_payload,
            trade_payload=trade_payload,
            raw=result.get("text", ""),
        )

    return _save_reflection(
        trade_id=trade_id,
        verdict=str(parsed.get("verdict", "neutral"))[:40],
        score=_float(parsed.get("score"), 0.0, lo=-1.0, hi=1.0),
        summary=str(parsed.get("summary", ""))[:1500],
        lessons=_clean_lessons(parsed.get("lessons", [])),
        decision_payload=decision_payload,
        trade_payload=trade_payload,
        raw=result.get("text", ""),
    )


def consolidate_lessons() -> dict[str, Any]:
    """
    Ask Claude to compress the playbook. Returns a dict with the operations
    applied. Safe to call manually from the Training UI.
    """
    if not claude_is_configured():
        return {"ok": False, "error": "claude_not_configured"}

    rows = get_playbook_with_metadata(limit=200)
    if len(rows) < 5:
        return {"ok": True, "skipped": True, "reason": "playbook too small"}

    payload = [
        {"id": r["id"], "content": r["content"], "weight": r["weight"], "category": r["category"]}
        for r in rows
    ]
    res = claude_chat(
        prompt=(
            "Consolidate this playbook per your instructions.\n\n"
            f"{json.dumps(payload, default=str)}"
        ),
        system=CONSOLIDATION_SYSTEM_PROMPT,
        max_tokens=1500,
        temperature=0.2,
    )
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}

    parsed = _parse_json_loose(res.get("text", ""))
    if not parsed:
        return {"ok": False, "error": "unparseable", "raw": res.get("text", "")[:500]}

    keep = parsed.get("keep") or []
    delete = parsed.get("delete") or []
    create = parsed.get("create") or []

    applied = {"updated": 0, "deleted": 0, "created": 0}
    with session_scope() as s:
        for item in keep:
            try:
                row = s.get(AILearningMemory, int(item.get("id")))
                if not row:
                    continue
                nw = item.get("new_weight")
                if nw is not None:
                    row.weight = max(0.0, min(float(nw), 5.0))
                nc = item.get("new_content")
                if nc:
                    row.content = str(nc)[:2000]
                applied["updated"] += 1
            except Exception:
                continue
        for did in delete:
            try:
                row = s.get(AILearningMemory, int(did))
                if row:
                    s.delete(row)
                    applied["deleted"] += 1
            except Exception:
                continue
        for c in create:
            try:
                s.add(AILearningMemory(
                    category=str(c.get("category", "rule"))[:60],
                    content=str(c.get("content", ""))[:2000],
                    weight=max(0.0, min(_float(c.get("weight"), 1.0, lo=0.0, hi=5.0), 5.0)),
                ))
                applied["created"] += 1
            except Exception:
                continue

        s.add(ActivityLog(
            category="ai",
            level="info",
            message=f"Playbook consolidated: {applied}",
        ))

    return {"ok": True, "applied": applied}


def reset_playbook() -> dict[str, Any]:
    """Wipe the entire learned playbook. Used from the Training UI."""
    with session_scope() as s:
        n = s.query(AILearningMemory).delete()
        s.add(ActivityLog(
            category="ai",
            level="warn",
            message=f"Playbook reset: {n} rules deleted.",
        ))
    return {"ok": True, "deleted": int(n or 0)}


def readiness_score() -> dict[str, Any]:
    """
    A 0..100 score indicating how ready the Claude bot is to handle real money.
    Considers: total closed trades, win rate, average reflection score, playbook
    size, claude configuration. Used by the Training Center.
    """
    with session_scope() as s:
        closed = s.query(PaperTrade).filter(PaperTrade.status == "closed").count()
        wins = (
            s.query(PaperTrade)
            .filter(PaperTrade.status == "closed", PaperTrade.realized_pnl > 0)
            .count()
        )
        losses = (
            s.query(PaperTrade)
            .filter(PaperTrade.status == "closed", PaperTrade.realized_pnl < 0)
            .count()
        )
        rules = s.query(AILearningMemory).count()
        reflections = s.query(TradeReflection).count()
        avg_score_row = s.query(TradeReflection).order_by(TradeReflection.id.desc()).limit(50).all()
        avg_score = (
            sum(float(r.score or 0) for r in avg_score_row) / len(avg_score_row)
            if avg_score_row else 0.0
        )

    win_rate = (wins / closed) if closed else 0.0

    # Sub-scores (each 0..1)
    sample_score = min(closed / 250.0, 1.0)             # 250 trades = "well-sampled"
    win_score = max(0.0, min(win_rate / 0.55, 1.0))     # 55% = good crypto win rate
    process_score = max(0.0, min((avg_score + 1.0) / 2.0, 1.0))
    rules_score = min(rules / 30.0, 1.0)
    config_score = 1.0 if claude_is_configured() else 0.0

    composite = (
        0.30 * sample_score
        + 0.25 * win_score
        + 0.20 * process_score
        + 0.15 * rules_score
        + 0.10 * config_score
    )

    return {
        "score": round(composite * 100.0, 1),
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "rules": rules,
        "reflections": reflections,
        "avg_reflection_score": round(avg_score, 3),
        "claude_configured": claude_is_configured(),
        "components": {
            "sample": round(sample_score, 3),
            "win": round(win_score, 3),
            "process": round(process_score, 3),
            "rules": round(rules_score, 3),
            "config": round(config_score, 3),
        },
    }


def recent_reflections(limit: int = 25) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = (
            s.query(TradeReflection)
            .order_by(TradeReflection.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "trade_id": r.trade_id,
                "verdict": r.verdict,
                "score": float(r.score or 0),
                "summary": r.summary,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def recent_decisions(limit: int = 25) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = (
            s.query(ClaudeDecision)
            .order_by(ClaudeDecision.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "wallet_id": r.wallet_id,
                "symbol": r.symbol,
                "price": float(r.price or 0),
                "action": r.action,
                "confidence": float(r.confidence or 0),
                "size_multiplier": float(r.size_multiplier or 0),
                "stop_loss_pct": float(r.stop_loss_pct or 0),
                "take_profit_pct": float(r.take_profit_pct or 0),
                "rationale": r.rationale or "",
                "key_factors": _safe_json_list(r.key_factors),
                "risk_flags": _safe_json_list(r.risk_flags),
                "source": r.source,
                "model": r.model or "",
                "technical_side": r.technical_side or "",
                "technical_confidence": float(r.technical_confidence or 0),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------------- #


def _save_reflection(
    *,
    trade_id: int,
    verdict: str,
    score: float,
    summary: str,
    lessons: list[dict[str, Any]],
    decision_payload: dict[str, Any] | None,
    trade_payload: dict[str, Any],
    raw: str,
) -> dict[str, Any]:
    with session_scope() as s:
        ref = TradeReflection(
            trade_id=trade_id,
            verdict=verdict[:40],
            score=score,
            summary=summary[:1500],
            lessons_json=json.dumps(lessons)[:6000],
            decision_json=json.dumps(decision_payload, default=str)[:6000] if decision_payload else "",
            trade_json=json.dumps(trade_payload, default=str)[:4000],
            raw_response=(raw or "")[:6000],
        )
        s.add(ref)
        s.flush()
        ref_id = ref.id

        # Promote each lesson to AILearningMemory.
        for lesson in lessons:
            content = str(lesson.get("content", "")).strip()
            if not content:
                continue
            s.add(AILearningMemory(
                category=str(lesson.get("category", "lesson"))[:60],
                content=content[:2000],
                weight=max(0.05, min(float(lesson.get("weight", 1.0) or 1.0), 5.0)),
            ))

        s.add(ActivityLog(
            category="ai",
            level="info",
            message=(
                f"Reflection saved for trade #{trade_id}: verdict={verdict}, "
                f"score={score:.2f}, lessons={len(lessons)}"
            ),
        ))

    return {"ok": True, "reflection_id": ref_id, "lessons_added": len(lessons)}


def _decision_to_dict(d: ClaudeDecision) -> dict[str, Any]:
    return {
        "action": d.action,
        "confidence": float(d.confidence or 0),
        "rationale": d.rationale or "",
        "key_factors": _safe_json_list(d.key_factors),
        "risk_flags": _safe_json_list(d.risk_flags),
        "source": d.source,
        "model": d.model,
        "technical_side": d.technical_side,
        "technical_confidence": float(d.technical_confidence or 0),
        "created_at": d.created_at.isoformat() if d.created_at else "",
    }


def _safe_json_list(s: str | None) -> list[Any]:
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return []


def _clean_lessons(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        out.append({
            "category": str(item.get("category", "lesson"))[:60],
            "content": content[:2000],
            "weight": _float(item.get("weight"), 1.0, lo=0.05, hi=5.0),
        })
    return out


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to extract first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _float(v: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        x = float(v)
    except Exception:
        x = default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _minutes_between(a: Any, b: Any) -> int:
    if not a or not b:
        return 0
    try:
        return int((b - a).total_seconds() // 60)
    except Exception:
        return 0
