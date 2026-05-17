"""
APScheduler wrapper around the autonomous BotEngine.

Owns the wake-up timer. The interval comes from the AppSetting
`bot_tick_seconds` (configurable from the Settings page). Calling
`reload()` after a settings change replaces the running job with a new
interval — no app restart needed.
"""
from __future__ import annotations

import threading
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.bot_config import BotConfig
from trading.bot_engine import bot_engine
from trading.reconciler import reconciler
from services.daily_summary import maybe_send_daily_summary
from utils.logger import get_logger

logger = get_logger(__name__)

_JOB_ID = "alphapilot_bot_tick"
_RECON_JOB_ID = "alphapilot_reconciler"
_SUMMARY_JOB_ID = "alphapilot_daily_summary"
_RECON_INTERVAL_SECONDS = 30  # reconcile open orders every 30s
_SUMMARY_INTERVAL_SECONDS = 600  # check the daily-summary trigger every 10 min


class BotScheduler:
    def __init__(self) -> None:
        self._scheduler: BackgroundScheduler | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        with self._lock:
            if self._scheduler and self._scheduler.running:
                return
            self._scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
            self._install_job(self._scheduler)
            self._scheduler.start()
            logger.info("Bot scheduler started.")

    def shutdown(self) -> None:
        with self._lock:
            if self._scheduler and self._scheduler.running:
                self._scheduler.shutdown(wait=False)
                logger.info("Bot scheduler shut down.")
            self._scheduler = None

    def reload(self) -> None:
        """
        Re-read configuration and re-install the tick job with the latest
        interval. Safe to call after every settings save.
        """
        with self._lock:
            if not self._scheduler:
                # If the scheduler was never started, calling reload should also start it.
                self._scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
                self._install_job(self._scheduler)
                self._scheduler.start()
                logger.info("Bot scheduler started via reload().")
                return

            # Remove the existing jobs (if any) and re-create with current interval.
            for jid in (_JOB_ID, _RECON_JOB_ID, _SUMMARY_JOB_ID):
                try:
                    self._scheduler.remove_job(jid)
                except Exception:
                    pass
            self._install_job(self._scheduler)
            logger.info("Bot scheduler reloaded.")

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, Any]:
        cfg = BotConfig.load()
        running = bool(self._scheduler and self._scheduler.running)
        next_run = None
        if running:
            try:
                job = self._scheduler.get_job(_JOB_ID)
                if job and job.next_run_time:
                    next_run = job.next_run_time.isoformat()
            except Exception:
                next_run = None
        return {
            "scheduler_running": running,
            "bot_enabled": cfg.bot_enabled,
            "tick_seconds": cfg.tick_seconds,
            "dry_run": cfg.dry_run,
            "next_tick": next_run,
            "universe": cfg.universe,
            "universe_limit": cfg.universe_limit,
            "min_confidence": cfg.min_confidence,
            "position_size_usd": cfg.position_size_usd,
        }

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _install_job(scheduler: BackgroundScheduler) -> None:
        cfg = BotConfig.load()
        scheduler.add_job(
            _safe_tick,
            trigger=IntervalTrigger(seconds=cfg.tick_seconds),
            id=_JOB_ID,
            name="AlphaPilot autonomous bot tick",
            max_instances=1,         # never overlap
            coalesce=True,           # collapse missed runs into one
            misfire_grace_time=30,
            replace_existing=True,
        )
        # Reconciler runs on its own fixed interval, independent of bot tick.
        scheduler.add_job(
            _safe_reconcile,
            trigger=IntervalTrigger(seconds=_RECON_INTERVAL_SECONDS),
            id=_RECON_JOB_ID,
            name="AlphaPilot LiveOrder reconciler",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
            replace_existing=True,
        )
        # Daily summary checker — fires often, but only sends once per day at the configured hour.
        scheduler.add_job(
            _safe_daily_summary,
            trigger=IntervalTrigger(seconds=_SUMMARY_INTERVAL_SECONDS),
            id=_SUMMARY_JOB_ID,
            name="AlphaPilot daily summary trigger",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
            replace_existing=True,
        )


def _safe_tick() -> None:
    """Top-level callable for APScheduler — never let an exception kill the worker."""
    try:
        bot_engine.tick()
    except Exception:
        logger.exception("Bot tick raised; swallowed to keep scheduler alive.")


def _safe_reconcile() -> None:
    """Top-level reconciler callable. Always isolate exceptions."""
    try:
        reconciler.reconcile()
    except Exception:
        logger.exception("Reconciler raised; swallowed to keep scheduler alive.")


def _safe_daily_summary() -> None:
    """Top-level daily summary trigger. Idempotent inside maybe_send_daily_summary."""
    try:
        maybe_send_daily_summary()
    except Exception:
        logger.exception("Daily summary raised; swallowed to keep scheduler alive.")


# Process-wide singleton.
bot_scheduler = BotScheduler()
