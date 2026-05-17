"""SQLAlchemy ORM models for AlphaPilot AI."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from utils.helpers import utcnow


class Base(DeclarativeBase):
    pass


class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    platform = Column(String(80), nullable=False)
    paper_balance = Column(Float, default=10_000.0, nullable=False)
    real_balance_placeholder = Column(Float, default=0.0)
    risk_profile = Column(String(40), default="Moderate")
    sandbox_mode = Column(Boolean, default=True)
    paper_trading_only = Column(Boolean, default=True)
    connection_status = Column(String(40), default="disconnected")
    api_status = Column(String(40), default="mock")
    last_synced = Column(DateTime, default=utcnow)
    created_at = Column(DateTime, default=utcnow)

    trades = relationship("PaperTrade", back_populates="wallet", cascade="all,delete")
    positions = relationship("Position", back_populates="wallet", cascade="all,delete")
    credentials = relationship(
        "ApiCredentialPlaceholder", back_populates="wallet", cascade="all,delete"
    )


class ApiCredentialPlaceholder(Base):
    """
    Placeholder credential storage for FUTURE real-API integration.

    NOTE: In a real deployment these fields MUST be encrypted at rest
    (e.g. with `cryptography.fernet`) and ideally stored in an OS keychain
    or secrets vault. The current implementation is for mocked testing only.
    """

    __tablename__ = "api_credentials_placeholder"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False)
    api_key = Column(String(255), default="")          # TODO: encrypt
    api_secret = Column(String(255), default="")       # TODO: encrypt
    api_passphrase = Column(String(255), default="")   # TODO: encrypt
    account_id = Column(String(120), default="")
    created_at = Column(DateTime, default=utcnow)

    wallet = relationship("Wallet", back_populates="credentials")


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    symbol = Column(String(80), nullable=False)
    market_type = Column(String(40), default="Crypto")
    side = Column(String(10), nullable=False)  # BUY / SELL
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    fees = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    confidence = Column(Float, default=0.5)
    status = Column(String(20), default="open")  # open / closed / cancelled
    opened_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime, nullable=True)
    notes = Column(Text, default="")

    wallet = relationship("Wallet", back_populates="trades")
    strategy = relationship("Strategy")


class LiveTradePlaceholder(Base):
    """Placeholder table for future LIVE trading. Empty by default."""

    __tablename__ = "live_trades_placeholder"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False)
    payload = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False)
    symbol = Column(String(80), nullable=False)
    qty = Column(Float, nullable=False)
    avg_entry = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=utcnow)

    wallet = relationship("Wallet", back_populates="positions")


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    description = Column(Text, default="")
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    market_type = Column(String(40), default="Crypto")
    strategy_type = Column(String(60), default="Momentum")
    max_position_size = Column(Float, default=1000.0)
    max_daily_loss = Column(Float, default=500.0)
    stop_loss_pct = Column(Float, default=0.05)
    take_profit_pct = Column(Float, default=0.10)
    min_confidence = Column(Float, default=0.6)
    max_trades_per_day = Column(Integer, default=20)
    max_open_trades = Column(Integer, default=5)
    risk_level = Column(String(40), default="Moderate")
    paper_trading_only = Column(Boolean, default=True)
    allow_ai_adjustments = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


class AITrainingSession(Base):
    __tablename__ = "ai_training_sessions"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    market_type = Column(String(40), default="Crypto")
    risk_level = Column(String(40), default="Moderate")
    starting_balance = Column(Float, default=10_000.0)
    ending_balance = Column(Float, default=10_000.0)
    trades_simulated = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.5)
    max_drawdown = Column(Float, default=0.0)
    status = Column(String(20), default="completed")  # running / completed / stopped
    started_at = Column(DateTime, default=utcnow)
    ended_at = Column(DateTime, nullable=True)
    notes = Column(Text, default="")


class AILearningMemory(Base):
    __tablename__ = "ai_learning_memory"

    id = Column(Integer, primary_key=True)
    category = Column(String(60), default="lesson")  # lesson / mistake / rule
    content = Column(Text, nullable=False)
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utcnow)


class MarketOpportunity(Base):
    __tablename__ = "market_opportunities"

    id = Column(Integer, primary_key=True)
    platform = Column(String(80))
    symbol = Column(String(80))
    market_type = Column(String(40))
    current_price = Column(Float)
    fair_value = Column(Float)
    ai_probability = Column(Float)
    market_probability = Column(Float)
    edge_pct = Column(Float)
    confidence = Column(Float)
    liquidity = Column(Float)
    volatility = Column(Float)
    risk_rating = Column(String(40))
    suggested_action = Column(String(60))
    reasoning = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True)
    category = Column(String(60))  # api / paper_trade / ai / risk / settings / etc
    level = Column(String(20), default="info")
    message = Column(Text, nullable=False)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)
    snapshot_date = Column(DateTime, default=utcnow)
    portfolio_value = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    ai_confidence = Column(Float, default=0.0)


class AppSetting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(80), unique=True, nullable=False)
    value = Column(String(255), default="")
    updated_at = Column(DateTime, default=utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    recommendation = Column(String(40), default="moderate")
    created_at = Column(DateTime, default=utcnow)
