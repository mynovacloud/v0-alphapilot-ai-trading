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


REFLECTION_SYSTEM_PROMPT = """You are AlphaPilot's reflection coach and trading mentor. Your job is to extract
DEEP, DURABLE, and ACTIONABLE trading lessons from every paper-trade outcome.

You will receive: the original decision (action, rationale, key factors, risk flags),
the live market context, and the realized P&L. You must return a JSON object:

{
  "verdict": "good_call" | "bad_call" | "lucky" | "unlucky" | "neutral",
  "score": -1.0 .. 1.0,
  "process_analysis": {
    "entry_quality": -1.0 .. 1.0,
    "exit_quality": -1.0 .. 1.0,
    "sizing_quality": -1.0 .. 1.0,
    "timing_quality": -1.0 .. 1.0,
    "risk_management_quality": -1.0 .. 1.0
  },
  "pattern_recognition": {
    "patterns_present": ["list of market patterns at entry"],
    "patterns_missed": ["patterns that should have been recognized"],
    "false_patterns": ["patterns that were noise, not signal"]
  },
  "lessons": [
    {"category": "lesson"|"mistake"|"rule"|"edge"|"anti_pattern", "content": "...", "weight": 0.1 .. 2.0, "applies_to": "entry|exit|sizing|timing|risk"}
  ],
  "meta_learning": {
    "strategy_fitness": "how well did the chosen strategy fit the market regime?",
    "confidence_calibration": "was confidence too high, too low, or well-calibrated?",
    "indicator_reliability": "which indicators were reliable vs misleading?",
    "time_of_day_factor": "did time of day impact the outcome?",
    "correlation_with_btc": "did BTC movement affect this altcoin trade?"
  },
  "improvement_suggestions": [
    "specific actionable improvement for next similar trade"
  ],
  "similar_past_mistakes": "brief note if this repeats a past error pattern",
  "summary": "two-sentence postmortem"
}

=== DEEP REFLECTION GUIDELINES ===

1. ENTRY ANALYSIS:
   - Was the entry triggered by genuine edge or noise?
   - Did multiple indicators confirm, or was it a single-indicator gamble?
   - Was there adequate separation between buy/sell signals?
   - Did volume confirm the move?

2. EXIT ANALYSIS:
   - Did we exit too early (left money on table)?
   - Did we exit too late (gave back profits)?
   - Was the stop-loss placed correctly?
   - Should we have trailed the stop differently?

3. TIMING ANALYSIS:
   - Did we enter at a good price within the move?
   - Did momentum support immediate entry or should we have waited?
   - How long did we hold vs optimal hold time for this setup?

4. PATTERN RECOGNITION:
   - What recurring market patterns were present?
   - Are there patterns we keep missing?
   - Are there patterns that consistently fail?

5. META-LEARNING:
   - Is this the 2nd, 3rd, or Nth time we made this mistake?
   - What conditions keep leading to the same error?
   - What rule would have prevented this loss?

Rules:
  - Generate 2-5 lessons. Prioritize ACTIONABLE, SPECIFIC rules.
  - "lucky" = positive PnL but the decision was unjustified by evidence.
  - "unlucky" = negative PnL but the decision was sound.
  - "score" must reflect PROCESS quality, not outcome alone.
  - Lessons must be ACTIONABLE generalizations ("when X, do Y"), not narration.
  - Flag if this is a REPEAT of a past mistake pattern.
  - No prose outside the JSON.
"""


CONSOLIDATION_SYSTEM_PROMPT = """You are AlphaPilot's playbook editor and trading rule optimizer. You will receive the
current list of learned rules with weights. Your job is to create a SHARP, HIGH-SIGNAL playbook.

=== CONSOLIDATION PRINCIPLES ===

1. MERGE SIMILAR RULES:
   - Combine rules that say the same thing differently
   - Create sharper, more specific rules from vague ones
   - Preserve the BEST phrasing, not the first one

2. STRENGTHEN PROVEN RULES:
   - Rules confirmed by 3+ trades should get weight boost
   - Rules that prevented losses should get HIGH weight (1.5+)
   - Rules that generated profits consistently should be canonical

3. DEMOTE/DELETE WEAK RULES:
   - Remove rules contradicted by recent outcomes
   - Demote rules that are too vague to be actionable
   - Delete rules that are subsets of better rules

4. CREATE SYNTHESIS RULES:
   - When multiple rules point to the same insight, create ONE master rule
   - Look for patterns across rules and extract meta-rules
   - Create "killer rules" that capture the most important insights

5. CATEGORY MANAGEMENT:
   - "rule": General trading principles (weight 1.0-2.0)
   - "mistake": Errors to avoid (weight 1.5-2.5) - these are CRITICAL
   - "edge": Proven profitable patterns (weight 1.5-2.0)
   - "anti_pattern": Patterns that look good but fail (weight 1.5-2.0)
   - "lesson": Learning from specific trades (weight 0.5-1.5)

Return JSON:
{
  "keep": [{"id": <int>, "new_weight": float, "new_content": "rewritten"|null, "reason": "why keep/modify"}],
  "delete": [{"id": <int>, "reason": "why delete"}],
  "create": [{"category": "rule"|"mistake"|"edge"|"anti_pattern", "content": "...", "weight": float, "source": "merged from X,Y,Z"|"synthesized from pattern"}],
  "playbook_health": {
    "total_rules": int,
    "high_confidence_rules": int,
    "needs_more_data": ["areas where we need more trades to learn"],
    "strongest_edges": ["our best 3 edges"],
    "biggest_leaks": ["our worst 3 recurring mistakes"]
  }
}

No prose outside the JSON. Be aggressive about consolidation - a playbook with 30 sharp rules beats 100 fuzzy ones.
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
        # The reflection schema is nested (process_analysis, pattern_recognition,
        # meta_learning, lessons[], improvement_suggestions[], ...). At 900 we
        # were getting silent truncation that landed the response in the
        # "Could not parse" branch and produced empty reflections. 2000 gives
        # Claude room to complete the JSON; cost is bounded by daily budget.
        max_tokens=2000,
        temperature=0.2,
        # 30s default was timing out on every reflection because the read
        # phase scales with output size and a 2000-token nested JSON takes
        # longer than a 700-token decision response. 120s gives a comfortable
        # margin for the longest plausible reflection. Errors are still
        # caught and persisted as empty reflections with the cause logged.
        timeout=120.0,
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

    result = _save_reflection(
        trade_id=trade_id,
        verdict=str(parsed.get("verdict", "neutral"))[:40],
        score=_float(parsed.get("score"), 0.0, lo=-1.0, hi=1.0),
        summary=str(parsed.get("summary", ""))[:1500],
        lessons=_clean_lessons(parsed.get("lessons", [])),
        decision_payload=decision_payload,
        trade_payload=trade_payload,
        raw=result.get("text", ""),
    )
    
    # Also update the adaptive learning engine
    try:
        from ai.adaptive_learning_engine import learn_from_trade as adaptive_learn
        # Extract strategy and regime from decision payload if available
        strategy_name = "Momentum"  # Default
        regime = "UNKNOWN"
        patterns = []
        
        if decision_payload:
            key_factors = decision_payload.get("key_factors", [])
            for factor in key_factors:
                if isinstance(factor, str):
                    if factor.startswith("strategy="):
                        strategy_name = factor.split("=", 1)[1]
                    elif factor.startswith("patterns="):
                        # Parse pattern list
                        try:
                            import ast
                            patterns = ast.literal_eval(factor.split("=", 1)[1])
                        except Exception:
                            pass
        
        adaptive_learn(
            trade_id=trade_id,
            patterns_at_entry=patterns,
            strategy_name=strategy_name,
            regime=regime,
        )
    except Exception:
        # Bumped from logger.warning (no traceback) -> logger.exception
        # so any future shape mismatch surfaces with a stack trace.
        # The adaptive-learning hook is a side path off reflection —
        # failure here doesn't break the reflection save above.
        logger.exception("[REFLECTION] Failed to update adaptive learning side-path")
    
    return result


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
        # Per-item error handling in the consolidation loops below:
        # one bad rule must not abort the entire pass. Previously these
        # were `except Exception: continue` with no log at all — if 5 of
        # 50 rules silently dropped, no one would know. Now each failure
        # logs at warning level (no stack trace per item to keep the log
        # readable, since these are expected to be data-shape errors)
        # and tracks the count in `applied["errors"]` so a summary
        # appears at the end.
        applied["errors"] = 0
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
            except Exception as e:
                logger.warning("[CONSOLIDATE] keep item dropped (id=%s): %s",
                               item.get("id"), e)
                applied["errors"] += 1
                continue
        for did in delete:
            try:
                row = s.get(AILearningMemory, int(did))
                if row:
                    s.delete(row)
                    applied["deleted"] += 1
            except Exception as e:
                logger.warning("[CONSOLIDATE] delete item dropped (id=%s): %s", did, e)
                applied["errors"] += 1
                continue
        for c in create:
            try:
                s.add(AILearningMemory(
                    category=str(c.get("category", "rule"))[:60],
                    content=str(c.get("content", ""))[:2000],
                    weight=max(0.0, min(_float(c.get("weight"), 1.0, lo=0.0, hi=5.0), 5.0)),
                ))
                applied["created"] += 1
            except Exception as e:
                logger.warning("[CONSOLIDATE] create item dropped: %s — content=%r",
                               e, str(c.get("content", ""))[:80])
                applied["errors"] += 1
                continue

        s.add(ActivityLog(
            category="ai",
            level="info",
            message=f"Playbook consolidated: {applied}",
        ))

    return {"ok": True, "applied": applied}


def compact_playbook_offline() -> dict[str, Any]:
    """Collapse near-duplicate playbook entries WITHOUT calling Claude.

    Walks every AILearningMemory row, groups them by normalized-token Jaccard
    similarity (same threshold used at save-time), and within each group
    keeps the highest-weight phrasing while accumulating sibling weights into
    it (capped at 5.0). Used to clean up an already-bloated playbook — e.g.
    the user had 100 rules that were really ~15 distinct insights repeated
    with different wording. Returns counts of merged/kept rows.

    This is intentionally cheaper and more deterministic than
    `consolidate_lessons`, which round-trips to Claude and has been observed
    to leave duplicate clusters intact.
    """
    with session_scope() as s:
        rows = (
            s.query(AILearningMemory)
            .order_by(AILearningMemory.weight.desc(), AILearningMemory.id.asc())
            .all()
        )
        if len(rows) < 2:
            return {"ok": True, "skipped": True, "kept": len(rows), "merged": 0}

        # Greedy union-find by Jaccard. We seed clusters with the
        # highest-weight rule first, so smaller paraphrases get folded INTO
        # the canonical phrasing.
        normalized = [_normalize_lesson(r.content or "") for r in rows]
        cluster_of: dict[int, int] = {}  # row.id -> canonical row.id
        for i, row_i in enumerate(rows):
            if row_i.id in cluster_of:
                continue
            cluster_of[row_i.id] = row_i.id
            for j in range(i + 1, len(rows)):
                row_j = rows[j]
                if row_j.id in cluster_of:
                    continue
                if _jaccard(normalized[i], normalized[j]) >= _DEDUP_SIMILARITY_THRESHOLD:
                    cluster_of[row_j.id] = row_i.id

        # Group siblings.
        siblings: dict[int, list[Any]] = {}
        for row in rows:
            canon_id = cluster_of[row.id]
            siblings.setdefault(canon_id, []).append(row)

        merged = 0
        for canon_id, group in siblings.items():
            if len(group) <= 1:
                continue
            canonical = next(r for r in group if r.id == canon_id)
            # Sum weights with damping (sqrt) so a 30-row pileup doesn't
            # immediately peg the rule at the cap; we still want repeated
            # firings to push it up, but proportionally.
            total = float(canonical.weight or 0.0)
            for r in group:
                if r.id == canon_id:
                    continue
                total += float(r.weight or 0.0) * 0.5
                s.delete(r)
                merged += 1
            canonical.weight = max(0.05, min(total, 5.0))

        s.add(ActivityLog(
            category="ai",
            level="info",
            message=f"Playbook compacted offline: merged {merged} duplicates, kept {len(rows) - merged}.",
        ))

    return {"ok": True, "merged": merged, "kept": len(rows) - merged}


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


def _normalize_lesson(text: str) -> str:
    """Normalize lesson text for fuzzy duplicate detection.

    Strips punctuation, collapses whitespace, lowercases, and drops common
    filler tokens so two phrasings of the same insight collapse to the same
    bag-of-words key. Used to detect when a "new" lesson is really a
    restatement of one already in the playbook.
    """
    import re
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    # Drop tiny/uninformative tokens; the structural words like "do", "not",
    # "when" are what keeps two opposite rules from collapsing to the same
    # signature, so we keep them. Numbers are stripped because exact thresholds
    # ("0.5x", "0.1x", "30%", etc.) shouldn't make an otherwise-duplicate
    # lesson look novel.
    tokens = [t for t in s.split() if t and not t.isdigit() and len(t) > 1]
    return " ".join(sorted(set(tokens)))


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity over normalized token sets."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# Rules whose normalized token sets overlap by >= this threshold are treated
# as the SAME rule. 0.72 is empirically tight enough to keep "do not buy when
# RSI < 30" distinct from "do not sell when RSI < 30" while collapsing the
# 30+ paraphrases of "contradictory divergence patterns = noise" into one.
_DEDUP_SIMILARITY_THRESHOLD = 0.72

# Per-repeat weight increment when a duplicate fires. Capped at 5.0 globally
# so a loud insight crowds out noise but can't infinitely dominate.
_DEDUP_WEIGHT_BOOST = 0.15


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

        # Promote each lesson to AILearningMemory — but DEDUP first.
        # Claude generates 2-5 lessons per losing trade and tends to repeat
        # the same insight with slightly different wording each time. Without
        # dedup the playbook bloats to hundreds of near-duplicates (we had
        # 30+ rules about "contradictory divergence patterns = noise" before
        # this fix). Instead, when a near-duplicate fires we reinforce the
        # existing rule by bumping its weight — so "we keep getting burned
        # by this" becomes a strong, single rule rather than 30 weak ones.
        existing_rules: list[tuple[Any, str]] = []
        for row in s.query(AILearningMemory).all():
            if row.content:
                existing_rules.append((row, _normalize_lesson(row.content)))

        added = 0
        reinforced = 0
        for lesson in lessons:
            content = str(lesson.get("content", "")).strip()
            if not content:
                continue
            new_norm = _normalize_lesson(content)
            new_weight = max(0.05, min(float(lesson.get("weight", 1.0) or 1.0), 5.0))

            # Find best fuzzy match against existing playbook.
            best_row = None
            best_sim = 0.0
            for row, row_norm in existing_rules:
                sim = _jaccard(new_norm, row_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row

            if best_row is not None and best_sim >= _DEDUP_SIMILARITY_THRESHOLD:
                # Same insight — reinforce instead of cloning. Bump weight
                # toward the higher of (existing, lesson_weight + boost) so a
                # repeatedly-fired rule converges to the cap quickly.
                target = max(float(best_row.weight or 0.0), new_weight) + _DEDUP_WEIGHT_BOOST
                best_row.weight = max(0.05, min(target, 5.0))
                # Replace the stored content if Claude's new phrasing is
                # meaningfully shorter and the categories agree — keeps the
                # playbook from drifting toward verbose paraphrases.
                if (
                    str(lesson.get("category", "lesson")) == (best_row.category or "")
                    and 0 < len(content) < len(best_row.content or "") * 0.8
                ):
                    best_row.content = content[:2000]
                reinforced += 1
            else:
                row = AILearningMemory(
                    category=str(lesson.get("category", "lesson"))[:60],
                    content=content[:2000],
                    weight=new_weight,
                )
                s.add(row)
                # Add to in-memory list so subsequent lessons in the SAME
                # reflection also dedup against it.
                existing_rules.append((row, new_norm))
                added += 1

        # When a reflection lands with zero lessons it's almost always a
        # failure path (Claude not configured, API call failed, JSON parse
        # failed, max_tokens truncation). The cause lives in `summary` but the
        # UI console only renders the ActivityLog message — so without
        # surfacing summary here, every empty reflection looks identical and
        # the operator can't tell which failure mode is biting. Promote those
        # to level=warn AND include a slice of summary so the cause is visible
        # in the training console in real time.
        empty = not lessons
        s.add(ActivityLog(
            category="ai",
            level="warn" if empty else "info",
            message=(
                f"Reflection saved for trade #{trade_id}: verdict={verdict}, "
                f"score={score:.2f}, lessons={len(lessons)} "
                f"(new={added}, reinforced={reinforced})"
                + (f" — {summary[:200]}" if empty and summary else "")
            ),
        ))

    return {
        "ok": True,
        "reflection_id": ref_id,
        "lessons_added": added,
        "lessons_reinforced": reinforced,
    }


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
    for item in raw[:5]:  # Allow up to 5 lessons now
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        out.append({
            "category": str(item.get("category", "lesson"))[:60],
            "content": content[:2000],
            "weight": _float(item.get("weight"), 1.0, lo=0.05, hi=5.0),
            "applies_to": str(item.get("applies_to", "general"))[:30],
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


# ============================================================================
# Advanced Meta-Learning Functions
# ============================================================================

META_ANALYSIS_SYSTEM_PROMPT = """You are AlphaPilot's meta-learning analyst. You analyze patterns ACROSS multiple trades
to discover higher-order insights that aren't visible from single trade reflections.

You will receive a batch of recent trades with their outcomes and reflections.
Your job is to find PATTERNS, CORRELATIONS, and SYSTEMATIC ISSUES.

Return JSON:
{
  "recurring_mistakes": [
    {"pattern": "description of recurring error", "frequency": int, "avg_loss": float, "fix_rule": "rule to prevent this"}
  ],
  "winning_patterns": [
    {"pattern": "description of winning setup", "frequency": int, "avg_profit": float, "conditions": "when this works best"}
  ],
  "strategy_performance": {
    "best_performing": {"strategy": "name", "win_rate": float, "avg_return": float},
    "worst_performing": {"strategy": "name", "win_rate": float, "avg_return": float},
    "recommendation": "which strategy to favor/avoid"
  },
  "timing_insights": {
    "optimal_hold_time": "X-Y minutes for this trading style",
    "early_exits": {"count": int, "avg_missed_profit": float},
    "late_exits": {"count": int, "avg_given_back": float}
  },
  "indicator_reliability": [
    {"indicator": "name", "reliability": float, "notes": "when it works/fails"}
  ],
  "market_condition_insights": {
    "trending_performance": {"win_rate": float, "best_strategy": "name"},
    "ranging_performance": {"win_rate": float, "best_strategy": "name"},
    "volatile_performance": {"win_rate": float, "notes": "observations"}
  },
  "new_rules_to_add": [
    {"category": "rule"|"mistake"|"edge", "content": "...", "weight": float, "rationale": "why this rule"}
  ],
  "rules_to_strengthen": [
    {"rule_id": int, "new_weight": float, "reason": "why strengthen"}
  ],
  "confidence_calibration": {
    "overconfident_trades": int,
    "underconfident_trades": int,
    "calibration_adjustment": "recommendation for future confidence"
  },
  "summary": "3-sentence meta-analysis summary"
}

Focus on ACTIONABLE insights that will improve future trading decisions.
No prose outside the JSON.
"""


def analyze_trade_patterns(lookback_trades: int = 50) -> dict[str, Any]:
    """
    Perform meta-analysis across recent trades to discover patterns.
    
    This is a higher-order learning function that looks at GROUPS of trades
    to find systematic issues and opportunities.
    """
    if not claude_is_configured():
        return {"ok": False, "error": "claude_not_configured"}
    
    with session_scope() as s:
        # Get recent closed trades with their reflections
        trades = (
            s.query(PaperTrade)
            .filter(PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc())
            .limit(lookback_trades)
            .all()
        )
        
        if len(trades) < 10:
            return {"ok": False, "error": "insufficient_trades", "count": len(trades)}
        
        # Build trade data with reflections
        trade_data = []
        for t in trades:
            reflection = (
                s.query(TradeReflection)
                .filter(TradeReflection.trade_id == t.id)
                .first()
            )
            
            trade_info = {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": float(t.entry_price or 0),
                "exit_price": float(t.exit_price or 0),
                "realized_pnl": float(t.realized_pnl or 0),
                "pnl_pct": round((float(t.realized_pnl or 0) / (float(t.entry_price or 1) * float(t.qty or 1))) * 100, 2),
                "held_minutes": _minutes_between(t.opened_at, t.closed_at),
                "confidence": float(t.confidence or 0),
                "exit_reason": t.exit_reason,
                "notes": t.notes,
            }
            
            if reflection:
                trade_info["verdict"] = reflection.verdict
                trade_info["score"] = float(reflection.score or 0)
                trade_info["summary"] = reflection.summary
            
            trade_data.append(trade_info)
        
        # Get current playbook rules for context
        rules = get_playbook_with_metadata(limit=50)
    
    # Call Claude for meta-analysis
    payload = {
        "trades": trade_data,
        "current_rules_count": len(rules),
        "top_rules": [r["content"][:200] for r in rules[:10]],  # Truncate for context
    }
    
    result = claude_chat(
        prompt=(
            "Perform meta-analysis on these recent trades to find patterns and systematic issues.\n\n"
            f"{json.dumps(payload, default=str)}"
        ),
        system=META_ANALYSIS_SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.3,
    )
    
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}
    
    parsed = _parse_json_loose(result.get("text", ""))
    if not parsed:
        return {"ok": False, "error": "parse_failed", "raw": result.get("text", "")[:500]}
    
    # Apply any recommended rule changes
    applied = {"rules_added": 0, "rules_strengthened": 0}
    
    with session_scope() as s:
        # Add new rules from meta-analysis
        new_rules = parsed.get("new_rules_to_add", [])
        for rule in new_rules[:5]:  # Limit to 5 new rules per analysis
            content = rule.get("content", "").strip()
            if content:
                s.add(AILearningMemory(
                    category=str(rule.get("category", "rule"))[:60],
                    content=content[:2000],
                    weight=_float(rule.get("weight"), 1.5, lo=0.5, hi=3.0),
                ))
                applied["rules_added"] += 1
        
        # Strengthen existing rules
        strengthen = parsed.get("rules_to_strengthen", [])
        for item in strengthen[:10]:
            try:
                rule_id = int(item.get("rule_id", 0))
                new_weight = _float(item.get("new_weight"), 1.5, lo=0.5, hi=3.0)
                row = s.get(AILearningMemory, rule_id)
                if row:
                    row.weight = new_weight
                    applied["rules_strengthened"] += 1
            except Exception:
                continue
        
        # Log the analysis
        s.add(ActivityLog(
            category="ai",
            level="info",
            message=f"Meta-analysis complete: {applied}. Summary: {parsed.get('summary', 'N/A')[:200]}",
        ))
    
    return {
        "ok": True,
        "analysis": parsed,
        "applied": applied,
        "trades_analyzed": len(trade_data),
    }


def get_learning_stats() -> dict[str, Any]:
    """
    Get comprehensive learning statistics for the Training UI.
    """
    with session_scope() as s:
        # Count rules by category
        rules = s.query(AILearningMemory).all()
        rule_stats = {
            "total": len(rules),
            "by_category": {},
            "avg_weight": 0,
            "high_weight_count": 0,  # weight > 1.5
        }
        
        weight_sum = 0
        for r in rules:
            cat = r.category or "unknown"
            rule_stats["by_category"][cat] = rule_stats["by_category"].get(cat, 0) + 1
            weight_sum += float(r.weight or 0)
            if (r.weight or 0) > 1.5:
                rule_stats["high_weight_count"] += 1
        
        if rules:
            rule_stats["avg_weight"] = round(weight_sum / len(rules), 2)
        
        # Reflection stats
        reflections = s.query(TradeReflection).all()
        reflection_stats = {
            "total": len(reflections),
            "by_verdict": {},
            "avg_score": 0,
        }
        
        score_sum = 0
        for r in reflections:
            verdict = r.verdict or "unknown"
            reflection_stats["by_verdict"][verdict] = reflection_stats["by_verdict"].get(verdict, 0) + 1
            score_sum += float(r.score or 0)
        
        if reflections:
            reflection_stats["avg_score"] = round(score_sum / len(reflections), 3)
        
        # Trade outcome stats (last 100)
        recent_trades = (
            s.query(PaperTrade)
            .filter(PaperTrade.status == "closed")
            .order_by(PaperTrade.closed_at.desc())
            .limit(100)
            .all()
        )
        
        trade_stats = {
            "count": len(recent_trades),
            "wins": 0,
            "losses": 0,
            "total_pnl": 0,
            "avg_hold_minutes": 0,
        }
        
        hold_minutes_sum = 0
        for t in recent_trades:
            pnl = float(t.realized_pnl or 0)
            trade_stats["total_pnl"] += pnl
            if pnl > 0:
                trade_stats["wins"] += 1
            elif pnl < 0:
                trade_stats["losses"] += 1
            hold_minutes_sum += _minutes_between(t.opened_at, t.closed_at)
        
        if recent_trades:
            trade_stats["avg_hold_minutes"] = round(hold_minutes_sum / len(recent_trades), 1)
            trade_stats["win_rate"] = round(trade_stats["wins"] / len(recent_trades), 3)
        else:
            trade_stats["win_rate"] = 0
        
        trade_stats["total_pnl"] = round(trade_stats["total_pnl"], 2)
    
    return {
        "rules": rule_stats,
        "reflections": reflection_stats,
        "recent_trades": trade_stats,
    }
