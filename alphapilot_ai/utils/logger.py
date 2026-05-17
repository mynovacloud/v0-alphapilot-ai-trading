"""
Centralized logging.

- Console + rotating-style file logger (single file, append mode for simplicity).
- Every module should call get_logger(__name__).
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "alphapilot.log"

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_initialized = False


def _init_root() -> None:
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(_FMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    fh = RotatingFileHandler(_LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    _init_root()
    return logging.getLogger(name)
