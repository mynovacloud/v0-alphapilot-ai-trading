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
    Wallet,
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
    Comprehensive 0..100 readiness score for graduating from paper to live.

    Weighs FIVE signal families derived from the entire simulation history:
      1. Sample size           - have we observed enough trades?
      2. Profitability         - win rate AND profit factor (gross win / gross loss)
      3. Edge magnitude        - expectancy (avg P&L per trade) vs starting bankroll
      4. Risk discipline       - max drawdown and average loss vs average win
      5. Process & coverage    - reflection scores, playbook depth, symbol diversity,
                                 Claude config, trading recency

    Returns a dict with the score, sub-scores, and the raw simulation metrics
    so the Training UI can render the complete picture.
    """
    import math
    from datetime import datetime, timedelta

    with session_scope() as s:
        # Materialize ONLY the columns we read, while the session is still open.
        # Holding ORM objects past `with session_scope()` triggers
        # DetachedInstanceError the moment lazy-loading kicks in (e.g. when
        # SQLAlchemy expires attributes after commit). The /training page
        # crashed with exactly that error overnight after the session expired
        # the cached attributes.
        all_trades = [
            {
                "status": t.status,
                "realized_pnl": float(t.realized_pnl or 0.0),
                "symbol": t.symbol,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
                "side": t.side,
                "entry_price": float(t.entry_price or 0.0),
                "exit_price": float(t.exit_price or 0.0) if t.exit_price is not None else None,
                # PaperTrade has no size_usd column — derive from qty * entry_price.
                "size_usd": float((t.qty or 0.0) * (t.entry_price or 0.0)),
                "unrealized_pnl": float(t.unrealized_pnl or 0.0),
            }
            for t in s.query(PaperTrade).all()
        ]
        rules = s.query(AILearningMemory).count()
        reflections_total = s.query(TradeReflection).count()
        avg_score_row = (
            s.query(TradeReflection)
            .order_by(TradeReflection.id.desc())
            .limit(50)
            .all()
        )
        avg_score = (
            sum(float(r.score or 0) for r in avg_score_row) / len(avg_score_row)
            if avg_score_row
            else 0.0
        )
        wallets = s.query(Wallet).all()
        starting_bankroll = sum(float(w.paper_balance or 0.0) for w in wallets) or 0.0

    # Partition trades. (`all_trades` is a list of plain dicts — see materialize
    # block above. Using ORM objects here would crash with DetachedInstanceError.)
    closed = [t for t in all_trades if t["status"] == "closed"]
    open_trades = [t for t in all_trades if t["status"] == "open"]
    wins = [t for t in closed if (t["realized_pnl"] or 0) > 0]
    losses = [t for t in closed if (t["realized_pnl"] or 0) < 0]
    flats = [t for t in closed if (t["realized_pnl"] or 0) == 0]

    closed_count = len(closed)
    win_count = len(wins)
    loss_count = len(losses)

    realized_pnl = sum(float(t["realized_pnl"] or 0) for t in closed)
    unrealized_pnl = sum(float(t["unrealized_pnl"] or 0) for t in open_trades)
    total_pnl = realized_pnl + unrealized_pnl

    gross_win = sum(float(t["realized_pnl"] or 0) for t in wins)
    gross_loss = abs(sum(float(t["realized_pnl"] or 0) for t in losses))
    avg_win = (gross_win / win_count) if win_count else 0.0
    avg_loss = (gross_loss / loss_count) if loss_count else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_win > 0 else 0.0
    )
    win_rate = (win_count / closed_count) if closed_count else 0.0
    expectancy = (realized_pnl / closed_count) if closed_count else 0.0  # $ per trade
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (
        float("inf") if avg_win > 0 else 0.0
    )

    # Equity curve & max drawdown across closed trades, ordered by close time.
    closed_sorted = sorted(
        closed,
        key=lambda t: t["closed_at"] or t["opened_at"] or datetime.min,
    )
    equity = starting_bankroll
    peak = starting_bankroll if starting_bankroll else 0.0
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    pnl_series: list[float] = []
    for t in closed_sorted:
        equity += float(t["realized_pnl"] or 0)
        pnl_series.append(float(t["realized_pnl"] or 0))
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd_abs:
            max_dd_abs = drawdown
            max_dd_pct = (drawdown / peak) if peak > 0 else 0.0

    # Sharpe-like consistency: mean / std of per-trade P&L.
    consistency = 0.0
    if len(pnl_series) >= 5:
        mean = sum(pnl_series) / len(pnl_series)
        var = sum((x - mean) ** 2 for x in pnl_series) / len(pnl_series)
        std = math.sqrt(var)
        consistency = (mean / std) if std > 0 else 0.0  # unitless, ~Sharpe ratio per trade

    avg_hold_min = 0.0
    holds = [
        ((t["closed_at"] - t["opened_at"]).total_seconds() / 60.0)
        for t in closed
        if t["closed_at"] and t["opened_at"]
    ]
    if holds:
        avg_hold_min = sum(holds) / len(holds)

    distinct_symbols = len({t["symbol"] for t in all_trades if t["symbol"]})

    # "Recency": did the bot trade within the last 7 days?
    last_close = max((t["closed_at"] for t in closed if t["closed_at"]), default=None)
    last_open = max((t["opened_at"] for t in all_trades if t["opened_at"]), default=None)
    last_activity = max([d for d in (last_close, last_open) if d], default=None)
    recency_days = (
        (datetime.utcnow() - last_activity).days if last_activity else 9999
    )

    return_pct = (realized_pnl / starting_bankroll * 100.0) if starting_bankroll else 0.0

    # ------- Sub-scores (each clamped to 0..1) -------
    sample_score = min(closed_count / 250.0, 1.0)  # 250 closed trades = "well-sampled"

    # Profitability: blend of win rate (target 55%) and profit factor (target 1.5).
    win_rate_score = max(0.0, min(win_rate / 0.55, 1.0))
    pf_score = (
        1.0 if profit_factor == float("inf") and gross_win > 0
        else max(0.0, min((profit_factor - 1.0) / 0.5, 1.0)) if profit_factor > 0
        else 0.0
    )
    profitability_score = 0.5 * win_rate_score + 0.5 * pf_score

    # Edge: expectancy as fraction of starting bankroll per trade.
    # 0.1% per trade = decent, 0.5% = excellent.
    expectancy_pct = (expectancy / starting_bankroll) if starting_bankroll else 0.0
    edge_score = max(0.0, min(expectancy_pct / 0.005, 1.0))

    # Discipline: penalize big drawdowns and skewed payoff.
    # 25%+ drawdown -> 0; flat -> 1.
    dd_score = max(0.0, 1.0 - (max_dd_pct / 0.25))
    payoff_score = (
        1.0 if payoff_ratio == float("inf") and avg_win > 0
        else max(0.0, min(payoff_ratio / 1.5, 1.0)) if payoff_ratio > 0
        else 0.0
    )
    consistency_score = max(0.0, min((consistency + 0.2) / 0.5, 1.0))
    discipline_score = 0.5 * dd_score + 0.3 * payoff_score + 0.2 * consistency_score

    # Process & coverage.
    process_score = max(0.0, min((avg_score + 1.0) / 2.0, 1.0))
    rules_score = min(rules / 30.0, 1.0)
    coverage_score = min(distinct_symbols / 8.0, 1.0)  # 8+ distinct symbols
    recency_score = (
        1.0 if recency_days <= 1
        else 0.7 if recency_days <= 3
        else 0.4 if recency_days <= 7
        else 0.1 if recency_days <= 14
        else 0.0
    )
    config_score = 1.0 if claude_is_configured() else 0.0
    process_total = (
        0.35 * process_score
        + 0.25 * rules_score
        + 0.20 * coverage_score
        + 0.10 * recency_score
        + 0.10 * config_score
    )

    composite = (
        0.25 * sample_score
        + 0.25 * profitability_score
        + 0.20 * edge_score
        + 0.15 * discipline_score
        + 0.15 * process_total
    )

    # Hard gates: even if composite is high, a tiny sample or unconfigured
    # Claude must keep the bot off live trading.
    if closed_count < 30:
        composite = min(composite, 0.45)
    if not claude_is_configured():
        composite = min(composite, 0.50)

    def _f(x: float) -> float:
        return float("inf") if x == float("inf") else round(x, 3)

    return {
        "score": round(composite * 100.0, 1),
        # Headline counts (kept for backward compatibility with existing UI).
        "closed_trades": closed_count,
        "open_trades": len(open_trades),
        "wins": win_count,
        "losses": loss_count,
        "flats": len(flats),
        "win_rate": round(win_rate, 4),
        "rules": rules,
        "reflections": reflections_total,
        "avg_reflection_score": round(avg_score, 3),
        "claude_configured": claude_is_configured(),
        # Full simulation metrics (the new "AI training is calculating everything" payload).
        "metrics": {
            "starting_bankroll": round(starting_bankroll, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(return_pct, 3),
            "gross_win": round(gross_win, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": _f(profit_factor),
            "payoff_ratio": _f(payoff_ratio),
            "expectancy": round(expectancy, 4),
            "expectancy_pct": round(expectancy_pct * 100.0, 4),
            "max_drawdown_abs": round(max_dd_abs, 2),
            "max_drawdown_pct": round(max_dd_pct * 100.0, 3),
            "consistency": round(consistency, 3),
            "avg_hold_minutes": round(avg_hold_min, 1),
            "distinct_symbols": distinct_symbols,
            "recency_days": recency_days,
        },
        # 5-axis composite. Each is 0..1.
        "components": {
            "sample": round(sample_score, 3),
            "profitability": round(profitability_score, 3),
            "edge": round(edge_score, 3),
            "discipline": round(discipline_score, 3),
            "process": round(process_total, 3),
            # Legacy keys preserved so older templates still render.
            "win": round(win_rate_score, 3),
            "rules": round(rules_score, 3),
            "config": round(config_score, 3),
        },
        # Detail breakdown for tooltips / "why this score?" UI.
        "subcomponents": {
            "win_rate": round(win_rate_score, 3),
            "profit_factor": round(pf_score, 3),
            "drawdown": round(dd_score, 3),
            "payoff": round(payoff_score, 3),
            "consistency": round(consistency_score, 3),
            "reflection": round(process_score, 3),
            "playbook": round(rules_score, 3),
            "coverage": round(coverage_score, 3),
            "recency": round(recency_score, 3),
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
