"""Regression: long-only policy.

The signal-edge harness measured every strategy's SELL signals at -127
to -196 bps forward return — shorting a structurally up-drifting asset
class loses, and a spot account cannot really short anyway. The bot now
refuses to OPEN short positions when `long_only` is set (the default).

These tests cover the config plumbing and assert the engine guard exists.
"""
from __future__ import annotations

import inspect

from config.bot_config import BotConfig
from config import bot_config as _bc
from trading.bot_engine import BotEngine


def test_long_only_defaults_on():
    """A fresh config must be long-only — that is the safe default."""
    assert BotConfig.load().long_only is True


def test_long_only_can_be_disabled():
    """Operators running futures can turn it off via the setting."""
    _bc.set_many({"bot_long_only": "false"})
    try:
        assert BotConfig.load().long_only is False
    finally:
        _bc.set_many({"bot_long_only": "true"})
    assert BotConfig.load().long_only is True


def test_engine_skips_short_entries_when_long_only():
    """The _evaluate_wallet guard must skip SELL entries under long_only.

    Verified by source inspection — a full tick is a heavy integration
    setup, but the guard is a few specific lines and regressions are
    obvious in the source text."""
    src = inspect.getsource(BotEngine._evaluate_wallet)
    assert "long_only" in src, "_evaluate_wallet lost its long-only guard"
    # The guard must skip (continue) on a SELL, not just log.
    assert 'side == "SELL"' in src
    lines = [ln.strip() for ln in src.splitlines()]
    guard_idx = next(i for i, ln in enumerate(lines) if "cfg.long_only" in ln)
    window = " ".join(lines[guard_idx:guard_idx + 12])
    assert "continue" in window, "long-only guard does not skip the SELL entry"
