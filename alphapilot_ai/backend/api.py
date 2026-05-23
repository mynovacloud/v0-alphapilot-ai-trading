"""
FastAPI backend for AlphaPilot AI.

Exposes a clean JSON API over the same SQLite/SQLAlchemy state the Streamlit
UI uses. The Streamlit dashboard works WITHOUT the API (it talks to the DB
directly), but the API is here for:
- programmatic access
- future integrations / external tools
- mobile / cloud roadmap

Run separately:
    uvicorn backend.api:app --reload --port 8000
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ai.learning_memory import LearningMemory
from analytics.performance import performance_metrics
from analytics.portfolio import portfolio_summary
from config.settings import settings
from connectors.registry import CONNECTOR_REGISTRY, get_connector
from database.db import session_scope
from database.models import (
    ActivityLog,
    ApiCredentialPlaceholder,
    PaperTrade,
    Position,
    Strategy,
    Wallet,
)
from database.schemas import (
    PaperTradeCreate,
    StrategyCreate,
    StrategyOut,
    WalletCreate,
    WalletOut,
)
from trading.backtester import run_backtest
from trading.market_scanner import scan_markets
from trading.paper_trading_engine import PaperTradingEngine
from trading.strategy_manager import (
    create_strategy,
    delete_strategy,
    list_strategies,
)
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="AlphaPilot AI backend (paper trading only — live trading is locked).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = PaperTradingEngine()
_memory = LearningMemory()


# ----------------------------------------------------------------------
# Health / meta
# ----------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "live_trading_enabled": settings.live_trading_enabled,
        "warning": "Paper trading only. Live trading is locked.",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ts": utcnow().isoformat()}


@app.get("/platforms")
def platforms() -> list[str]:
    return list(CONNECTOR_REGISTRY.keys())


# ----------------------------------------------------------------------
# Wallets
# ----------------------------------------------------------------------

@app.get("/wallets", response_model=list[WalletOut])
def list_wallets() -> list[WalletOut]:
    with session_scope() as s:
        rows = s.query(Wallet).order_by(Wallet.created_at.asc()).all()
        return [WalletOut.model_validate(w) for w in rows]


@app.post("/wallets", response_model=WalletOut)
def create_wallet(payload: WalletCreate) -> WalletOut:
    with session_scope() as s:
        w = Wallet(
            name=payload.name,
            platform=payload.platform,
            paper_balance=payload.paper_balance,
            risk_profile=payload.risk_profile,
            sandbox_mode=payload.sandbox_mode,
            paper_trading_only=payload.paper_trading_only,
            connection_status="connected (mock)",
            api_status="mock",
        )
        s.add(w)
        s.flush()
        # Save credential placeholders
        s.add(
            ApiCredentialPlaceholder(
                wallet_id=w.id,
                api_key=payload.api_key,
                api_secret=payload.api_secret,
                api_passphrase=payload.api_passphrase,
                account_id=payload.account_id,
            )
        )
        s.add(
            ActivityLog(
                category="wallet",
                level="info",
                wallet_id=w.id,
                message=f"Wallet '{w.name}' ({w.platform}) created with paper balance ${payload.paper_balance:.2f}",
            )
        )
        s.flush()
        s.refresh(w)
        return WalletOut.model_validate(w)


@app.get("/wallets/{wallet_id}", response_model=WalletOut)
def get_wallet(wallet_id: int) -> WalletOut:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            raise HTTPException(404, "Wallet not found")
        return WalletOut.model_validate(w)


@app.delete("/wallets/{wallet_id}")
def delete_wallet(wallet_id: int) -> dict[str, Any]:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            raise HTTPException(404, "Wallet not found")
        s.delete(w)
        s.add(ActivityLog(category="wallet", level="warn", message=f"Wallet {wallet_id} deleted."))
    return {"ok": True}


@app.post("/wallets/{wallet_id}/test_connection")
def test_connection(wallet_id: int) -> dict[str, Any]:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            raise HTTPException(404, "Wallet not found")
        connector = get_connector(w.platform, sandbox=w.sandbox_mode)
        result = connector.connect()
        s.add(
            ActivityLog(
                category="api",
                level="info",
                wallet_id=wallet_id,
                message=f"Mock connection test for {w.platform}: ok",
            )
        )
        return result


# ----------------------------------------------------------------------
# Paper trades
# ----------------------------------------------------------------------

@app.get("/trades")
def list_trades(wallet_id: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with session_scope() as s:
        q = s.query(PaperTrade)
        if wallet_id:
            q = q.filter(PaperTrade.wallet_id == wallet_id)
        rows = q.order_by(PaperTrade.opened_at.desc()).limit(limit).all()
        return [
            {
                "id": t.id,
                "wallet_id": t.wallet_id,
                "strategy_id": t.strategy_id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "realized_pnl": t.realized_pnl,
                "unrealized_pnl": t.unrealized_pnl,
                "status": t.status,
                "confidence": t.confidence,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in rows
        ]


@app.post("/trades/open")
def open_trade(payload: PaperTradeCreate) -> dict[str, Any]:
    return _engine.open_trade(
        wallet_id=payload.wallet_id,
        symbol=payload.symbol,
        side=payload.side,
        qty=payload.qty,
        entry_price=payload.entry_price,
        confidence=payload.confidence,
        market_type=payload.market_type,
        strategy_id=payload.strategy_id,
        notes=payload.notes,
    )


@app.post("/trades/{trade_id}/close")
def close_trade(trade_id: int, exit_price: float) -> dict[str, Any]:
    return _engine.close_trade(trade_id, exit_price)


# ----------------------------------------------------------------------
# Strategies + backtests
# ----------------------------------------------------------------------

@app.get("/strategies")
def get_strategies() -> list[dict[str, Any]]:
    return list_strategies()


@app.post("/strategies", response_model=StrategyOut)
def add_strategy(payload: StrategyCreate) -> StrategyOut:
    sid = create_strategy(payload.model_dump())
    with session_scope() as s:
        return StrategyOut.model_validate(s.get(Strategy, sid))


@app.delete("/strategies/{strategy_id}")
def remove_strategy(strategy_id: int) -> dict[str, Any]:
    ok = delete_strategy(strategy_id)
    if not ok:
        raise HTTPException(404, "Strategy not found")
    return {"ok": True}


@app.post("/strategies/{strategy_id}/backtest")
def backtest(strategy_id: int, n_trades: int = 200) -> dict[str, Any]:
    return run_backtest(strategy_id, n_trades=n_trades)


# ----------------------------------------------------------------------
# Market scanner
# ----------------------------------------------------------------------

@app.get("/scan")
def scan(n: int = 20) -> list[dict[str, Any]]:
    return scan_markets(n=n)


# ----------------------------------------------------------------------
# AI memory (admin)
# ----------------------------------------------------------------------

@app.get("/ai/memory")
def ai_memory(limit: int = 100) -> list[dict[str, Any]]:
    return _memory.list_lessons(limit=limit)


@app.delete("/ai/memory")
def ai_memory_reset() -> dict[str, Any]:
    n = _memory.reset()
    return {"deleted": n}


# ----------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------

@app.get("/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    return portfolio_summary()


@app.get("/analytics/performance")
def analytics_performance() -> dict[str, Any]:
    return performance_metrics()


# ----------------------------------------------------------------------
# Activity logs
# ----------------------------------------------------------------------

@app.get("/logs")
def logs(limit: int = 200, category: str | None = None) -> list[dict[str, Any]]:
    with session_scope() as s:
        q = s.query(ActivityLog)
        if category:
            q = q.filter(ActivityLog.category == category)
        rows = q.order_by(ActivityLog.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "category": r.category,
                "level": r.level,
                "message": r.message,
                "wallet_id": r.wallet_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]
