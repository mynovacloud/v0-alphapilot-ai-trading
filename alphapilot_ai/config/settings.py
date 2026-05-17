"""
Centralized application settings, loaded from environment variables (.env)
with sane defaults for local development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    # dotenv is optional at runtime; missing .env should not crash the app.
    pass


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AlphaPilot AI")
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = _bool(os.getenv("APP_DEBUG"), True)

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./alphapilot.db")

    # Backend
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Streamlit
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # Safety
    live_trading_enabled: bool = _bool(os.getenv("LIVE_TRADING_ENABLED"), False)


settings = Settings()
