"""
AI Learning Memory.

Stores 'lessons learned' from trading sessions. The AI uses these to bias
future decisions (e.g., avoid low-liquidity markets after slippage losses).
"""
from __future__ import annotations

from typing import Any

from database.db import session_scope
from database.models import AILearningMemory
from utils.logger import get_logger

logger = get_logger(__name__)


class LearningMemory:
    def add_lesson(self, content: str, category: str = "lesson", weight: float = 1.0) -> int:
        with session_scope() as s:
            row = AILearningMemory(category=category, content=content, weight=weight)
            s.add(row)
            s.flush()
            logger.info("AI lesson added: %s", content)
            return row.id

    def list_lessons(self, limit: int = 100) -> list[dict[str, Any]]:
        with session_scope() as s:
            rows = (
                s.query(AILearningMemory)
                .order_by(AILearningMemory.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "category": r.category,
                    "content": r.content,
                    "weight": r.weight,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def reset(self) -> int:
        with session_scope() as s:
            n = s.query(AILearningMemory).delete()
            return int(n or 0)

    def export(self) -> list[dict[str, Any]]:
        return self.list_lessons(limit=10_000)
