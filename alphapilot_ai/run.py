"""
AlphaPilot AI - One-click launcher.

Just run:   python run.py

This starts the backend API and the Streamlit UI together,
initializes the database, seeds demo data, and opens your browser.
Press Ctrl+C once to shut everything down cleanly.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

API_HOST = "127.0.0.1"
API_PORT = 8000
UI_PORT = 8501


def _banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {msg}\n{line}")


def _check_python() -> None:
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required. You have:", sys.version)
        sys.exit(1)


def _ensure_dependencies() -> None:
    """Install requirements.txt if any required package is missing."""
    required = [
        "streamlit",
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pydantic",
        "pandas",
        "numpy",
        "plotly",
        "httpx",
        "loguru",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    _banner(f"Installing missing packages: {', '.join(missing)}")
    req_file = ROOT / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("ERROR: pip install failed:", e)
        sys.exit(1)


def _init_database() -> None:
    from database.db import init_db
    from database.seed import seed_if_empty

    _banner("Initializing database")
    init_db()
    seed_if_empty()
    print("  Database ready at data/alphapilot.db")


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.3)
    return False


def main() -> None:
    _check_python()
    _ensure_dependencies()
    _init_database()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["ALPHAPILOT_API_URL"] = f"http://{API_HOST}:{API_PORT}"

    procs: list[subprocess.Popen] = []

    def cleanup() -> None:
        for p in procs:
            if p.poll() is None:
                try:
                    if os.name == "nt":
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    atexit.register(cleanup)

    popen_kwargs = {"env": env, "cwd": str(ROOT)}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    _banner(f"Starting backend API on http://{API_HOST}:{API_PORT}")
    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.api:app",
        "--host", API_HOST,
        "--port", str(API_PORT),
        "--log-level", "warning",
    ]
    procs.append(subprocess.Popen(api_cmd, **popen_kwargs))

    if not _wait_for_port(API_HOST, API_PORT, timeout=25):
        print("ERROR: Backend API failed to start.")
        cleanup()
        sys.exit(1)
    print("  Backend ready.  Docs: http://%s:%d/docs" % (API_HOST, API_PORT))

    _banner(f"Starting UI on http://localhost:{UI_PORT}")
    ui_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "app" / "streamlit_app.py"),
        "--server.port", str(UI_PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    procs.append(subprocess.Popen(ui_cmd, **popen_kwargs))

    if _wait_for_port("127.0.0.1", UI_PORT, timeout=30):
        url = f"http://localhost:{UI_PORT}"
        print(f"  UI ready.  Opening {url} ...")
        try:
            webbrowser.open(url)
        except Exception:
            pass
    else:
        print("  UI did not respond in time. You can still try http://localhost:%d" % UI_PORT)

    _banner("AlphaPilot AI is running. Press Ctrl+C to stop.")
    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    print("\nA process exited unexpectedly. Shutting down...")
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
