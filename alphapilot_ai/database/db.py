"""SQLAlchemy engine, session, and DB lifecycle helpers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# SQLite needs check_same_thread=False so FastAPI + Streamlit threads can share it.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't exist."""
    from database import models  # noqa: F401  (registers models on Base)
    from database.models import Base

    logger.info("Initializing database at %s", settings.database_url)
    Base.metadata.create_all(bind=engine)


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
