"""Strategy CRUD helpers."""
from __future__ import annotations

from typing import Any

from database.db import session_scope
from database.models import Strategy


def list_strategies() -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.query(Strategy).order_by(Strategy.created_at.desc()).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "wallet_id": r.wallet_id,
                "market_type": r.market_type,
                "strategy_type": r.strategy_type,
                "max_position_size": r.max_position_size,
                "max_daily_loss": r.max_daily_loss,
                "stop_loss_pct": r.stop_loss_pct,
                "take_profit_pct": r.take_profit_pct,
                "min_confidence": r.min_confidence,
                "max_trades_per_day": r.max_trades_per_day,
                "max_open_trades": r.max_open_trades,
                "risk_level": r.risk_level,
                "paper_trading_only": r.paper_trading_only,
                "allow_ai_adjustments": r.allow_ai_adjustments,
            }
            for r in rows
        ]


def create_strategy(data: dict[str, Any]) -> int:
    with session_scope() as s:
        strat = Strategy(**data)
        s.add(strat)
        s.flush()
        return strat.id


def delete_strategy(strategy_id: int) -> bool:
    with session_scope() as s:
        strat = s.get(Strategy, strategy_id)
        if not strat:
            return False
        s.delete(strat)
        return True
