"""Market scanner generates mocked opportunities and persists them for the UI."""
from __future__ import annotations

import random
from typing import Any

from database.db import session_scope
from database.models import MarketOpportunity
from mock_data.market_data import (
    crypto_snapshot,
    prediction_snapshot,
    stock_snapshot,
)
from utils.constants import SUGGESTED_ACTIONS
from utils.logger import get_logger

logger = get_logger(__name__)


def _classify(edge_pct: float, liquidity: float, volatility: float) -> tuple[str, str]:
    if liquidity < 0.3:
        return "Low Liquidity", "Liquidity too low to trade safely."
    if abs(edge_pct) > 0.15 and liquidity > 0.5:
        return "Strong Opportunity", f"Edge {edge_pct:+.1%} with healthy liquidity."
    if volatility > 0.8:
        return "High Risk", "Volatility extremely high, expect wide swings."
    if abs(edge_pct) > 0.05:
        return "Paper Trade", "Moderate edge — good candidate for paper trade."
    if abs(edge_pct) < 0.02:
        return "Watch", "No meaningful edge yet."
    return random.choice(SUGGESTED_ACTIONS), "Heuristic classification."


def scan_markets(n: int = 20) -> list[dict[str, Any]]:
    """Generate `n` mock market opportunities and persist them to SQLite."""
    rows: list[dict[str, Any]] = []
    with session_scope() as s:
        for _ in range(n):
            snap = random.choice([crypto_snapshot, stock_snapshot, prediction_snapshot])()
            ai_prob = snap.get("ai_probability") or random.uniform(0.3, 0.7)
            mkt_prob = snap.get("market_probability") or random.uniform(0.3, 0.7)
            edge = ai_prob - mkt_prob
            confidence = round(random.uniform(0.4, 0.95), 2)
            liq = snap.get("liquidity", random.uniform(0.3, 0.95))
            vol = snap.get("volatility", random.uniform(0.1, 0.9))
            action, reason = _classify(edge, liq, vol)
            risk = "High" if vol > 0.7 else ("Medium" if vol > 0.3 else "Low")

            row = MarketOpportunity(
                platform=snap["platform"],
                symbol=snap["symbol"],
                market_type=snap["market_type"],
                current_price=float(snap["current_price"]),
                fair_value=float(snap.get("fair_value", snap["current_price"])),
                ai_probability=float(ai_prob),
                market_probability=float(mkt_prob),
                edge_pct=float(edge),
                confidence=confidence,
                liquidity=float(liq),
                volatility=float(vol),
                risk_rating=risk,
                suggested_action=action,
                reasoning=reason,
            )
            s.add(row)
            s.flush()
            rows.append(
                {
                    "id": row.id,
                    "platform": row.platform,
                    "symbol": row.symbol,
                    "market_type": row.market_type,
                    "current_price": row.current_price,
                    "fair_value": row.fair_value,
                    "ai_probability": row.ai_probability,
                    "market_probability": row.market_probability,
                    "edge_pct": row.edge_pct,
                    "confidence": row.confidence,
                    "liquidity": row.liquidity,
                    "volatility": row.volatility,
                    "risk_rating": row.risk_rating,
                    "suggested_action": row.suggested_action,
                    "reasoning": row.reasoning,
                }
            )
    return rows
