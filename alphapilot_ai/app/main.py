"""
AlphaPilot AI — single-process web app.

Serves:
  - HTML pages at  /, /wallets, /scanner, /strategies, /training, /analytics, /activity, /settings
  - JSON API at    /api/* (the original FastAPI endpoints)
  - Static files   /static/*

Everything runs on one port from one command:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.web import router as web_router
from backend.api import app as api_app
from config.settings import settings
from database.db import init_db
from database.seed import seed_if_empty
from utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="AlphaPilot AI — paper-trading dashboard with built-in JSON API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    seed_if_empty()
    logger.info("AlphaPilot AI ready.")


# Static
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# JSON API at /api/*
app.mount("/api", api_app)

# HTML web UI at /
app.include_router(web_router)
