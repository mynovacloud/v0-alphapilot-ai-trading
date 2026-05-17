"""
Seed mock wallets, strategies, and trades so the app feels alive on first launch.

Idempotent: only seeds when the wallets table is empty.
"""
from __future__ import annotations

import random
from datetime import timedelta

from database.db import session_scope
from database.models import (
    ActivityLog,
    AILearningMemory,
    PaperTrade,
    Position,
    Strategy,
    Wallet,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

_MOCK_WALLETS = [
    ("Polymarket Main", "Polymarket", 8500.0, "Moderate"),
    ("Crypto.com Spot", "Crypto.com", 12_400.0, "Aggressive"),
    ("Webull Stocks", "Webull", 15_000.0, "Conservative"),
    ("Robinhood Options", "Robinhood", 6_750.0, "Moderate"),
    ("E*TRADE Long-term", "E*TRADE", 22_000.0, "Conservative"),
]

_MOCK_SYMBOLS = {
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    "Stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"],
    "Prediction Markets": ["ELECTION-2028", "FED-RATE-CUT", "OSCAR-WINNER"],
}

_LESSONS = [
    "Avoid low-liquidity markets after repeated slippage losses.",
    "Reduce position size after 3 consecutive losses.",
    "Increase confidence weighting for strategies with consistent profitability.",
    "Avoid trading during historically weak time windows (e.g. illiquid hours).",
    "Tighten exits when profit targets are reached faster than expected.",
    "Flag markets where AI probability differs significantly from market price.",
    "Penalize strategies that produce excessive drawdown.",
    "Reward strategies with consistent risk-adjusted returns.",
]


def seed_if_empty() -> None:
    """
    Seed mock data ONLY if explicitly requested via SEED_DEMO_DATA=true env var.

    By default this is a no-op so the app starts empty and the user can
    create their own real wallets. To re-enable demos for development,
    set SEED_DEMO_DATA=true in your .env.
    """
    import os
    if os.getenv("SEED_DEMO_DATA", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    with session_scope() as s:
        if s.query(Wallet).count() > 0:
            return

        logger.info("Seeding mock data...")

        # Wallets
        wallets: list[Wallet] = []
        for name, platform, bal, risk in _MOCK_WALLETS:
            w = Wallet(
                name=name,
                platform=platform,
                paper_balance=bal,
                real_balance_placeholder=0.0,
                risk_profile=risk,
                connection_status="connected (mock)",
                api_status="mock",
            )
            s.add(w)
            wallets.append(w)
        s.flush()

        # Strategies
        strategies = []
        for name, stype, market in [
            ("Momentum BTC", "Momentum", "Crypto"),
            ("Mean Reversion SPY", "Mean Reversion", "Stocks"),
            ("Probability Edge", "Probability Edge", "Prediction Markets"),
            ("Vol Breakout", "Volatility Breakout", "Crypto"),
        ]:
            strat = Strategy(name=name, strategy_type=stype, market_type=market, description=f"Default {name} strategy")
            s.add(strat)
            strategies.append(strat)
        s.flush()

        # Paper trades + positions
        now = utcnow()
        for w in wallets:
            market_type = (
                "Prediction Markets"
                if w.platform in {"Polymarket", "Kalshi"}
                else ("Crypto" if w.platform in {"Crypto.com", "Coinbase", "Binance", "Kraken"} else "Stocks")
            )
            symbols = _MOCK_SYMBOLS.get(market_type, ["AAPL"])
            for i in range(random.randint(8, 14)):
                sym = random.choice(symbols)
                side = random.choice(["BUY", "SELL"])
                entry = round(random.uniform(20, 400), 2)
                qty = round(random.uniform(0.5, 10), 2)
                is_closed = random.random() > 0.35
                pnl = round(random.uniform(-250, 400), 2) if is_closed else 0.0
                opened = now - timedelta(days=random.randint(0, 25), hours=random.randint(0, 23))
                trade = PaperTrade(
                    wallet_id=w.id,
                    strategy_id=random.choice(strategies).id,
                    symbol=sym,
                    market_type=market_type,
                    side=side,
                    qty=qty,
                    entry_price=entry,
                    exit_price=round(entry + pnl / max(qty, 0.01), 2) if is_closed else None,
                    fees=round(qty * entry * 0.001, 2),
                    slippage=round(random.uniform(0, 1.5), 2),
                    realized_pnl=pnl if is_closed else 0.0,
                    unrealized_pnl=0.0 if is_closed else round(random.uniform(-50, 80), 2),
                    confidence=round(random.uniform(0.4, 0.9), 2),
                    status="closed" if is_closed else "open",
                    opened_at=opened,
                    closed_at=opened + timedelta(hours=random.randint(1, 72)) if is_closed else None,
                )
                s.add(trade)

            # A couple of open positions
            for _ in range(random.randint(1, 3)):
                sym = random.choice(symbols)
                avg = round(random.uniform(20, 400), 2)
                cur = round(avg * random.uniform(0.92, 1.12), 2)
                qty = round(random.uniform(1, 8), 2)
                s.add(
                    Position(
                        wallet_id=w.id,
                        symbol=sym,
                        qty=qty,
                        avg_entry=avg,
                        current_price=cur,
                        unrealized_pnl=round((cur - avg) * qty, 2),
                    )
                )

        # AI learning memory
        for lesson in _LESSONS:
            s.add(AILearningMemory(category="lesson", content=lesson, weight=round(random.uniform(0.5, 1.2), 2)))

        # Activity log starter entries
        s.add(ActivityLog(category="system", level="info", message="AlphaPilot AI initialized with mock data."))
        s.add(ActivityLog(category="system", level="info", message="Live trading is locked by default."))

        logger.info("Seed complete.")
