"""Phase A: trading-hours filter.

Crypto is 24/7, but real volume is concentrated in the London-NY
overlap (12:00-22:00 UTC). Trading dead hours just bleeds fees on chop.
The bot now skips ticks outside the configured window — manual ticks
bypass the check so an operator can still kick one off any time.
"""
from __future__ import annotations

import datetime
import inspect
from dataclasses import dataclass

from trading.bot_engine import BotEngine, _within_trading_hours
from config.bot_config import BotConfig
from config import bot_config as _bc


@dataclass
class _MockCfg:
    trading_hours_start_utc: int = 12
    trading_hours_end_utc: int = 22


def _at(hour: int) -> datetime.datetime:
    return datetime.datetime(2026, 1, 15, hour, 30, 0, tzinfo=datetime.timezone.utc)


# --------------------------------------------------------------------------
# _within_trading_hours — the core check
# --------------------------------------------------------------------------

def test_inside_same_day_window_passes():
    cfg = _MockCfg(12, 22)
    for h in (12, 15, 18, 21):
        assert _within_trading_hours(cfg, _at(h)) is True, f"hour {h} should be in window"


def test_outside_same_day_window_blocks():
    cfg = _MockCfg(12, 22)
    for h in (0, 3, 11, 22, 23):
        assert _within_trading_hours(cfg, _at(h)) is False, f"hour {h} should be blocked"


def test_wrap_around_window_includes_both_sides_of_midnight():
    """A 22:00 -> 04:00 window must accept 23:00 AND 02:00, reject 12:00."""
    cfg = _MockCfg(22, 4)
    for h in (22, 23, 0, 1, 3):
        assert _within_trading_hours(cfg, _at(h)) is True, f"wrap hour {h} should pass"
    for h in (4, 6, 12, 18, 21):
        assert _within_trading_hours(cfg, _at(h)) is False, f"wrap hour {h} should block"


def test_equal_bounds_disable_the_filter():
    """0/0 (or any equal pair) means 'trade 24/7' — every hour passes."""
    cfg = _MockCfg(0, 0)
    for h in range(24):
        assert _within_trading_hours(cfg, _at(h)) is True


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------

def test_bot_config_default_hours_are_peak_overlap():
    cfg = BotConfig.load()
    assert cfg.trading_hours_start_utc == 12
    assert cfg.trading_hours_end_utc == 22


def test_bot_config_overrides_take_effect():
    _bc.set_many({
        "bot_trading_hours_start_utc": "9",
        "bot_trading_hours_end_utc": "17",
    })
    try:
        cfg = BotConfig.load()
        assert cfg.trading_hours_start_utc == 9
        assert cfg.trading_hours_end_utc == 17
    finally:
        _bc.set_many({
            "bot_trading_hours_start_utc": "12",
            "bot_trading_hours_end_utc": "22",
        })


def test_bot_config_clamps_garbage_to_default():
    """Bad strings shouldn't crash the bot — they fall back to default."""
    _bc.set_many({"bot_trading_hours_start_utc": "not-a-number"})
    try:
        cfg = BotConfig.load()
        assert cfg.trading_hours_start_utc == 12   # default
    finally:
        _bc.set_many({"bot_trading_hours_start_utc": "12"})


# --------------------------------------------------------------------------
# Engine integration — the guard must actually skip ticks
# --------------------------------------------------------------------------

def test_run_tick_carries_the_trading_hours_guard():
    """Source-inspection regression: the guard must be present in
    _run_tick. Without it the filter is a no-op even if config is set."""
    src = inspect.getsource(BotEngine._run_tick)
    assert "_within_trading_hours" in src, "_run_tick lost its trading-hours guard"
    # Must skip (return) inside the guard, not just log. Widen the
    # window enough to span a multi-line log message + the return.
    lines = src.splitlines()
    guard_idx = next(i for i, ln in enumerate(lines) if "_within_trading_hours" in ln)
    window = " ".join(lines[guard_idx:guard_idx + 25])
    assert "return" in window, "trading-hours guard does not return — would not actually skip"


def test_run_tick_lets_manual_ticks_bypass():
    """Operators must be able to fire a manual tick any time —
    `manual=True` bypasses the time gate."""
    src = inspect.getsource(BotEngine._run_tick)
    # The guard line should require `not manual` before the time check.
    assert "not manual" in src, (
        "manual ticks must bypass the trading-hours filter; "
        "no `not manual` check found near the guard"
    )
