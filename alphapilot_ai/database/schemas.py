"""Pydantic schemas used by the FastAPI layer."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WalletCreate(BaseModel):
    name: str
    platform: str
    paper_balance: float = 10_000.0
    risk_profile: str = "Moderate"
    sandbox_mode: bool = True
    paper_trading_only: bool = True
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    account_id: str = ""


class WalletOut(BaseModel):
    id: int
    name: str
    platform: str
    paper_balance: float
    real_balance_placeholder: float
    risk_profile: str
    sandbox_mode: bool
    paper_trading_only: bool
    connection_status: str
    api_status: str
    last_synced: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class PaperTradeCreate(BaseModel):
    wallet_id: int
    symbol: str
    market_type: str = "Crypto"
    side: str = Field(pattern="^(BUY|SELL)$")
    qty: float
    entry_price: float
    confidence: float = 0.5
    strategy_id: Optional[int] = None
    notes: str = ""


class PaperTradeOut(BaseModel):
    id: int
    wallet_id: int
    symbol: str
    market_type: str
    side: str
    qty: float
    entry_price: float
    exit_price: Optional[float]
    fees: float
    slippage: float
    realized_pnl: float
    unrealized_pnl: float
    confidence: float
    status: str
    opened_at: datetime
    closed_at: Optional[datetime]
    notes: str

    class Config:
        from_attributes = True


class StrategyCreate(BaseModel):
    name: str
    description: str = ""
    wallet_id: Optional[int] = None
    market_type: str = "Crypto"
    strategy_type: str = "Momentum"
    max_position_size: float = 1000.0
    max_daily_loss: float = 500.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    min_confidence: float = 0.6
    max_trades_per_day: int = 20
    max_open_trades: int = 5
    risk_level: str = "Moderate"
    paper_trading_only: bool = True
    allow_ai_adjustments: bool = True


class StrategyOut(StrategyCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


