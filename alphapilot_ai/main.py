"""
AlphaPilot AI - Main launcher.

Starts the FastAPI backend (uvicorn) in a background thread and the Streamlit
dashboard in the foreground. Initializes the SQLite database on first run and
seeds mock data so the app feels alive immediately.

Run:
    python main.py

You can also run pieces separately:
    uvicorn backend.api:app --reload --port 8000
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
import threading
import subprocess
from pathlib import Path

# Make the project root importable regardless of where main.py is invoked from.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from database.db import init_db  # noqa: E402
from database.seed import seed_if_empty  # noqa: E402

logger = get_logger(__name__)


def _start_backend() -> None:
    """Start the FastAPI backend with uvicorn in this process (background thread)."""
    import uvicorn  # local import so Streamlit-only runs don't require it eagerly

    logger.info("Starting FastAPI backend on %s:%s", settings.api_host, settings.api_port)
    uvicorn.run(
        "backend.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        reload=False,
    )


def _start_streamlit() -> int:
    """Spawn streamlit as a subprocess so we can keep main.py simple."""
    streamlit_app = ROOT / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        str(settings.streamlit_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    logger.info("Starting Streamlit dashboard: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    print("=" * 64)
    print(" AlphaPilot AI  -  AI Trading Intelligence (Paper Trading)")
    print(" Live trading is LOCKED by default. This is not financial advice.")
    print("=" * 64)

    # 1. Initialize DB + seed mock data
    init_db()
    seed_if_empty()

    # 2. Start backend in background thread (daemon -> dies with main process)
    backend_thread = threading.Thread(target=_start_backend, daemon=True)
    backend_thread.start()

    # Give uvicorn a moment to bind
    time.sleep(1.5)
    logger.info("Backend ready at http://%s:%s/docs", settings.api_host, settings.api_port)

    # 3. Start Streamlit in foreground (blocks)
    try:
        rc = _start_streamlit()
        sys.exit(rc)
    except KeyboardInterrupt:
        logger.info("Shutting down AlphaPilot AI...")
        sys.exit(0)


if __name__ == "__main__":
    main()
