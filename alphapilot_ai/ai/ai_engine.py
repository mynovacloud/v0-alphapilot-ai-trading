"""
AI Engine — the brain that orchestrates training sessions and decisions.

Coordinates:
- DecisionEngine (signal -> action)
- MistakeAnalyzer (post-trade lessons)
- StrategyOptimizer (adjust strategy params)
- LearningMemory (persistent lessons store)
- FutureMLModel (placeholder)

Used by the AI Training Lab and the Paper Trading Engine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from ai.decision_engine import Decision, DecisionEngine
from ai.learning_memory import LearningMemory
from ai.mistake_analyzer import MistakeAnalyzer
from ai.model_placeholder import FutureMLModel
from ai.strategy_optimizer import StrategyOptimizer
from database.db import session_scope
from database.models import (
    ActivityLog,
    AITrainingSession,
    PaperTrade,
    Strategy,
    Wallet,
)
from mock_data.market_data import (
    crypto_snapshot,
    prediction_snapshot,
    stock_snapshot,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    session_id: int
    starting_balance: float
    ending_balance: float
    pnl: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_confidence: float
    max_drawdown: float
    decisions: list[dict[str, Any]]
    lessons: list[str]


def _snapshot_for_market(market_type: str) -> dict[str, Any]:
    if market_type == "Crypto":
        return crypto_snapshot()
    if market_type == "Stocks":
        return stock_snapshot()
    if market_type == "Prediction Markets":
        return prediction_snapshot()
    # Default: pick something reasonable
    return random.choice([crypto_snapshot, stock_snapshot, prediction_snapshot])()


class AIEngine:
    def __init__(self) -> None:
        self.decision = DecisionEngine()
        self.analyzer = MistakeAnalyzer()
        self.optimizer = StrategyOptimizer()
        self.memory = LearningMemory()
        self.model = FutureMLModel()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_training_session(
        self,
        wallet_id: int | None = None,
        strategy_id: int | None = None,
        market_type: str = "Crypto",
        risk_level: str = "Moderate",
        num_trades: int = 50,
        starting_balance: float = 10_000.0,
    ) -> TrainingResult:
        """
        Run a simulated training session. Each iteration:
        1. Pull a mock market snapshot
        2. Ask DecisionEngine for a decision
        3. Simulate the trade outcome (rule-based + stochastic)
        4. Update equity, record decision
        5. After all trades, analyze mistakes and store lessons
        """
        rng = np.random.default_rng()
        balance = float(starting_balance)
        equity_curve = [balance]
        decisions: list[dict[str, Any]] = []
        wins = 0
        losses = 0
        confidences: list[float] = []

        # Risk-level affects trade size and outcome variance
        size_pct = {"Conservative": 0.02, "Moderate": 0.05, "Aggressive": 0.1, "Degenerate": 0.2}.get(
            risk_level, 0.05
        )
        edge_bias = {
            "Conservative": 0.02,
            "Moderate": 0.0,
            "Aggressive": -0.02,
            "Degenerate": -0.05,
        }.get(risk_level, 0.0)

        # Pull strategy info if available
        strategy_type = "Momentum"
        if strategy_id:
            with session_scope() as s:
                strat = s.get(Strategy, strategy_id)
                if strat:
                    strategy_type = strat.strategy_type

        # Open a training session row up front so the UI can poll it
        with session_scope() as s:
            session = AITrainingSession(
                wallet_id=wallet_id,
                strategy_id=strategy_id,
                market_type=market_type,
                risk_level=risk_level,
                starting_balance=starting_balance,
                ending_balance=starting_balance,
                trades_simulated=0,
                wins=0,
                losses=0,
                avg_confidence=0.0,
                max_drawdown=0.0,
                status="running",
            )
            s.add(session)
            s.flush()
            session_id = session.id

        for i in range(num_trades):
            snap = _snapshot_for_market(market_type)
            decision: Decision = self.decision.decide(snap, strategy_type)
            confidences.append(decision.confidence)

            if decision.side == "HOLD":
                decisions.append(
                    {
                        "i": i,
                        "symbol": snap["symbol"],
                        "side": "HOLD",
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "pnl": 0.0,
                        "balance": balance,
                    }
                )
                equity_curve.append(balance)
                continue

            # Simulate outcome: probability of success scales with confidence + edge bias
            p_win = max(0.1, min(0.9, decision.confidence + edge_bias))
            won = rng.random() < p_win
            stake = balance * size_pct
            # Win/loss magnitudes scale with volatility
            vol = float(snap.get("volatility", 0.2))
            win_amt = stake * rng.uniform(0.4, 1.6) * (0.5 + vol)
            loss_amt = stake * rng.uniform(0.3, 1.2) * (0.5 + vol)
            pnl = win_amt if won else -loss_amt
            balance += pnl
            equity_curve.append(balance)
            if won:
                wins += 1
            else:
                losses += 1

            decisions.append(
                {
                    "i": i,
                    "symbol": snap["symbol"],
                    "platform": snap.get("platform", ""),
                    "side": decision.side,
                    "confidence": round(decision.confidence, 3),
                    "reasoning": decision.reasoning,
                    "pnl": round(pnl, 2),
                    "balance": round(balance, 2),
                    "liquidity": float(snap.get("liquidity", 0.5)),
                    "volatility": vol,
                    "realized_pnl": round(pnl, 2),
                    "duration_hours": rng.integers(1, 72),
                }
            )

        # Drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = float(((peak - equity) / np.maximum(peak, 1)).max())

        win_rate = wins / max(1, wins + losses)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Mistake analysis -> lessons
        lessons = self.analyzer.analyze_session(decisions)
        for l in lessons:
            self.memory.add_lesson(l, category="lesson", weight=1.0)

        # Persist final session result
        with session_scope() as s:
            session = s.get(AITrainingSession, session_id)
            if session:
                session.ending_balance = round(balance, 2)
                session.trades_simulated = num_trades
                session.wins = wins
                session.losses = losses
                session.avg_confidence = round(avg_conf, 3)
                session.max_drawdown = round(drawdown, 4)
                session.status = "completed"
                session.ended_at = utcnow()
                session.notes = f"Strategy={strategy_type}, market={market_type}"

            s.add(
                ActivityLog(
                    category="ai",
                    level="info",
                    wallet_id=wallet_id,
                    message=(
                        f"AI training completed: trades={num_trades}, "
                        f"win_rate={win_rate:.0%}, pnl={balance - starting_balance:+.2f}"
                    ),
                )
            )

        return TrainingResult(
            session_id=session_id,
            starting_balance=round(starting_balance, 2),
            ending_balance=round(balance, 2),
            pnl=round(balance - starting_balance, 2),
            trades=num_trades,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 3),
            avg_confidence=round(avg_conf, 3),
            max_drawdown=round(drawdown, 4),
            decisions=decisions,
            lessons=lessons,
        )

    # ------------------------------------------------------------------
    # Single-shot decision (used by Market Scanner / Wallet Detail)
    # ------------------------------------------------------------------

    def evaluate_opportunity(
        self, snapshot: dict[str, Any], strategy_type: str = "Momentum"
    ) -> dict[str, Any]:
        decision = self.decision.decide(snapshot, strategy_type)
        return {
            "side": decision.side,
            "confidence": round(decision.confidence, 3),
            "reasoning": decision.reasoning,
        }
