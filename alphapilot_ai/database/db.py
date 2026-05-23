"""SQLAlchemy engine, session, and DB lifecycle helpers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# SQLite needs check_same_thread=False so FastAPI + Streamlit threads can share it.
_is_sqlite = settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

# Use connection pooling for better performance
engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args=_connect_args,
    # Connection pool settings for better concurrency
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
)

# Enable WAL mode for SQLite (much better concurrent read/write performance)
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't exist, then run lightweight column migrations."""
    from database import models  # noqa: F401  (registers models on Base)
    from database.models import Base

    logger.info("Initializing database at %s", settings.database_url)
    Base.metadata.create_all(bind=engine)
    _migrate_schema()


def _migrate_schema() -> None:
    """
    SQLite-friendly forward-only schema migration.

    `create_all` only creates missing tables — it never alters existing ones.
    When we add a new column to a model we have to ALTER manually here.
    Idempotent: safe to call on every startup.
    """
    from sqlalchemy import inspect, text

    desired = {
        "wallets": [
            ("trading_mode", "VARCHAR(20) DEFAULT 'paper'"),
            ("bot_paused", "BOOLEAN DEFAULT 0"),
            ("max_position_usd", "FLOAT DEFAULT 500.0"),
            ("max_open_positions", "INTEGER DEFAULT 3"),
            ("max_daily_loss_usd", "FLOAT DEFAULT 200.0"),
            ("max_daily_trades", "INTEGER DEFAULT 10"),
            # Scalper mode settings
            ("trading_style", "VARCHAR(20) DEFAULT 'hybrid'"),
            ("micro_profit_target_usd", "FLOAT DEFAULT 0.25"),
            ("min_profit_pct", "FLOAT DEFAULT 0.003"),
            ("auto_reinvest", "BOOLEAN DEFAULT 1"),
            # Futures settings
            ("futures_enabled", "BOOLEAN DEFAULT 0"),
            ("max_leverage", "FLOAT DEFAULT 1.0"),
            ("default_leverage", "FLOAT DEFAULT 1.0"),
            ("margin_mode", "VARCHAR(20) DEFAULT 'isolated'"),
            ("liquidation_buffer_pct", "FLOAT DEFAULT 0.10"),
            # Metadata column for session settings backup/restore
            ("meta", "JSON DEFAULT '{}'"),
            # Bankroll-reset cursor: stamped when the operator hits "Reset
            # Paper Balance" in Settings. The training-page money strip uses
            # these to scope "this session" P&L without ever touching the
            # underlying trade history. NULL on wallets that have never been
            # reset.
            ("bankroll_reset_at", "DATETIME"),
            ("session_starting_bankroll", "FLOAT"),
        ],
        "paper_trades": [
            ("is_perp", "BOOLEAN DEFAULT 0"),
            ("leverage", "FLOAT DEFAULT 1.0"),
            ("margin_used", "FLOAT DEFAULT 0.0"),
            ("liquidation_price", "FLOAT"),
            ("funding_paid", "FLOAT DEFAULT 0.0"),
            # Position management columns (added for SL/TP/trailing/DCA support)
            ("stop_loss_price", "FLOAT"),
            ("take_profit_price", "FLOAT"),
            ("trailing_stop_pct", "FLOAT"),
            ("trailing_stop_price", "FLOAT"),
            ("high_water_price", "FLOAT"),
            ("max_loss_pct", "FLOAT DEFAULT 0.10"),
            ("time_limit_hours", "FLOAT"),
            # Holding profile resolved at entry (see holding_profiles.py).
            ("holding_profile", "VARCHAR(20)"),
            ("dca_count", "INTEGER DEFAULT 0"),
            # Scale-in (pyramiding) tracker — separate from `dca_count`.
            # DCA = "average down on a loser to lower cost basis"
            # Scale-in = "pyramid into a winner to ride the trend"
            # These need separate counters because a position that
            # pyramided 3 times during an uptrend should still be allowed
            # to DCA on the eventual pullback.
            ("scale_in_count", "INTEGER DEFAULT 0"),
            # When the most recent scale-in occurred — used for cooldowns
            # so we don't pyramid 3x in 30 seconds during a vertical spike.
            ("last_scale_in_at", "DATETIME"),
            ("original_entry", "FLOAT"),
            ("exit_reason", "VARCHAR(30)"),
            # Breakeven stop columns - move stop to lock in small profit
            ("breakeven_trigger_pct", "FLOAT"),
            ("breakeven_stop_pct", "FLOAT"),
            ("breakeven_activated", "BOOLEAN DEFAULT 0"),
            # Link to the ClaudeDecision that produced this entry, so the
            # autonomous learning engine can rebuild entry-time market context
            # at close time. Nullable for trades that don't originate from
            # a Claude decision.
            ("claude_decision_id", "INTEGER"),
            # Phase B calibration audit (see PaperTrade model docstring).
            # Lets the training-page scorecard show how many of today's
            # trades were backed by measured pattern data vs raw confidence.
            ("calibration_source", "VARCHAR(20)"),
            ("calibration_sample_size", "INTEGER"),
        ],
        "claude_decisions": [
            # JSON snapshot of indicators + regime at decision time. Consumed
            # by autonomous_learning_engine._build_context_from_trade so closed
            # trades produce real fingerprints instead of degenerate defaults.
            ("market_snapshot", "TEXT DEFAULT '{}'"),
        ],
    }

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in desired.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in existing:
                    try:
                        conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}'))
                        logger.info("Migrated: added %s.%s", table, name)
                    except Exception as e:
                        logger.warning("Could not add %s.%s: %s", table, name, e)


def reset_db() -> None:
    """Drop all tables and recreate. Useful for resetting mock data."""
    from database.models import Base

    logger.warning("Resetting database!")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context-managed DB session that commits on success, rolls back on error."""
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_session() -> Session:
    """Plain session factory (caller is responsible for closing)."""
    return SessionLocal()
