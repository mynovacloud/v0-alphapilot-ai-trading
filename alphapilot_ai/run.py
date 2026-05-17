"""AlphaPilot AI — one-command launcher.

No Streamlit, no two processes. Just runs the unified FastAPI server
that serves both the HTML web UI and the JSON API on one port.

Usage:
    python run.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 8000


def _banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {msg}\n{line}")


def _check_python() -> None:
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required. You have:", sys.version)
        sys.exit(1)


def _ensure_dependencies() -> None:
    required = ["fastapi", "uvicorn", "jinja2", "sqlalchemy", "pydantic", "pandas", "numpy", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _banner(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]
        )


def _wait_for_port(host: str, port: int, timeout: float = 25.0) -> bool:
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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    _banner(f"Starting AlphaPilot AI on http://{HOST}:{PORT}")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--log-level", "info",
    ]
    proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT))

    try:
        if _wait_for_port(HOST, PORT, timeout=25):
            url = f"http://{HOST}:{PORT}"
            print(f"  Ready. Opening {url} ...")
            try:
                webbrowser.open(url)
            except Exception:
                pass
        else:
            print(f"  Server did not respond in time. Try {url} manually.")

        _banner("AlphaPilot AI is running. Press Ctrl+C to stop.")
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
