"""Shared pytest configuration.

The project's import root is ``alphapilot_ai/`` (see run.py / main.py which
both prepend it to sys.path before booting). Tests live one level deeper at
``alphapilot_ai/tests/`` so we need to do the same prepend here for imports
like ``from ai.autonomous_learning_engine import TradeContext`` to resolve
without packaging the project.

We also point DATABASE_URL at a transient SQLite file before any module is
imported, so any test that touches database/db.py gets a clean isolated DB
instead of clobbering the dev one. Pure-logic tests don't trigger DB import
at all and pay no cost for this.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Path setup: tests/ → alphapilot_ai/ (the import root used at runtime).
_TEST_DIR = Path(__file__).resolve().parent
_ROOT = _TEST_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Test database: a temp file per session, deleted at exit. We do this BEFORE
# any test imports project modules, since config.settings reads DATABASE_URL
# at import time.
_TMP_DB = Path(tempfile.gettempdir()) / "alphapilot_test.db"
# Use a fresh DB every session (deletion happens at the bottom in
# pytest_sessionstart so a previous run's residue can't poison the new one).
if _TMP_DB.exists():
    try:
        _TMP_DB.unlink()
    except OSError:
        pass
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP_DB}")


def pytest_sessionstart(session):  # noqa: ARG001
    """Initialize the empty schema once before any test runs.

    Tests that touch the database (anything reaching session_scope or a
    SQLAlchemy query) need the tables to exist. Pure-logic tests pay zero
    cost — init_db is idempotent and fast on SQLite.
    """
    try:
        from database.db import init_db
        init_db()
    except Exception as e:  # noqa: BLE001
        # init_db failure is loud — without a schema most DB-touching
        # tests will fail downstream anyway, so we want this visible.
        print(f"[conftest] init_db failed during session start: {e}")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Best-effort cleanup of the per-session SQLite file."""
    try:
        if _TMP_DB.exists():
            _TMP_DB.unlink()
    except OSError:
        pass
