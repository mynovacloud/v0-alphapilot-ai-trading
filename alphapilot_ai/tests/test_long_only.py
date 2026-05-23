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


# --------------------------------------------------------------------------
# Engine-level deadbolt — every path that bypasses bot_engine must still
# be blocked when long_only is on. This is the fix the playbook spent an
# overnight run shouting about ("a gate that can be routed around is not
# a gate"): portfolio_intelligence offsets, scale-ins, manual tickets,
# and the autonomous learner all call paper_engine.open_trade directly.
# --------------------------------------------------------------------------

def _make_wallet(name: str) -> int:
    from database.db import session_scope
    from database.models import Wallet
    with session_scope() as s:
        w = Wallet(name=name, platform="paper")
        s.add(w); s.flush()
        return w.id


def test_open_trade_blocks_sell_directly_when_long_only():
    """Calling open_trade with side=SELL must be refused at the engine
    boundary — even if the caller never went through bot_engine's guard."""
    from trading.paper_trading_engine import PaperTradingEngine

    _bc.set_many({"bot_long_only": "true"})
    try:
        wallet_id = _make_wallet("engine-deadbolt-on")
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="SELL",
            qty=0.01, entry_price=50000.0,
        )
        assert outcome["ok"] is False
        assert outcome["code"] == "long_only_block"
    finally:
        _bc.set_many({"bot_long_only": "true"})


def test_open_trade_allows_sell_when_long_only_disabled():
    """When the operator disables long_only (e.g. for futures), SELL must
    reach the rest of the open_trade pipeline. Any rejection from this
    point on is for a different reason — never long_only_block."""
    from trading.paper_trading_engine import PaperTradingEngine

    _bc.set_many({"bot_long_only": "false"})
    try:
        wallet_id = _make_wallet("engine-deadbolt-off")
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="SELL",
            qty=0.01, entry_price=50000.0,
        )
        assert outcome.get("code") != "long_only_block"
    finally:
        _bc.set_many({"bot_long_only": "true"})


def test_open_trade_buys_pass_long_only_check():
    """BUY entries must never trip the long-only deadbolt — it's a
    short-blocker, not a no-trade-at-all switch."""
    from trading.paper_trading_engine import PaperTradingEngine

    _bc.set_many({"bot_long_only": "true"})
    try:
        wallet_id = _make_wallet("engine-deadbolt-buy")
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="BUY",
            qty=0.01, entry_price=50000.0,
        )
        assert outcome.get("code") != "long_only_block"
    finally:
        _bc.set_many({"bot_long_only": "true"})


# --------------------------------------------------------------------------
# Position-size ceiling — make `bot_position_size_usd` mean it.
# An overnight run opened $1,192 trades on a $80 config because every
# upstream caller (autonomous size_multiplier, portfolio_intelligence,
# scale-in) does its own sizing math. The engine-level clamp is the
# single source of truth for "how much money per trade".
# --------------------------------------------------------------------------

def test_open_trade_clamps_bloated_size_to_config():
    """A $500-notional request on a $100 config must come out the other
    side at ~$100. Clamped, not rejected — the trade still happens."""
    from trading.paper_trading_engine import PaperTradingEngine
    from database.db import session_scope
    from database.models import PaperTrade

    _bc.set_many({"bot_position_size_usd": "100", "bot_long_only": "false"})
    try:
        wallet_id = _make_wallet("size-cap-bloated")
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="BUY",
            qty=0.01, entry_price=50000.0,  # $500 requested
        )
        assert outcome.get("ok") is True, outcome
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == outcome["trade_id"]).first()
            notional = float(trade.qty) * float(trade.entry_price)
            assert 95.0 <= notional <= 105.0, f"notional {notional} not near $100 cap"
    finally:
        _bc.set_many({"bot_position_size_usd": "80", "bot_long_only": "true"})


def test_open_trade_leaves_in_budget_size_alone():
    """A request below the cap must pass through untouched — no surprise
    shrinkage when the caller already sized correctly."""
    from trading.paper_trading_engine import PaperTradingEngine
    from database.db import session_scope
    from database.models import PaperTrade

    _bc.set_many({"bot_position_size_usd": "100", "bot_long_only": "false"})
    try:
        wallet_id = _make_wallet("size-cap-passthrough")
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="BUY",
            qty=0.001, entry_price=50000.0,  # $50 requested, well under cap
        )
        assert outcome.get("ok") is True, outcome
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == outcome["trade_id"]).first()
            assert float(trade.qty) == 0.001
    finally:
        _bc.set_many({"bot_position_size_usd": "80", "bot_long_only": "true"})


def test_open_trade_allows_5pct_slack_above_cap():
    """A request 3% above the cap is float-rounding, not a real breach.
    Don't trip the clamp on noise — only on real over-sizing."""
    from trading.paper_trading_engine import PaperTradingEngine
    from database.db import session_scope
    from database.models import PaperTrade

    _bc.set_many({"bot_position_size_usd": "100", "bot_long_only": "false"})
    try:
        wallet_id = _make_wallet("size-cap-slack")
        # $103 notional — within the 5% slack on a $100 cap.
        outcome = PaperTradingEngine().open_trade(
            wallet_id=wallet_id, symbol="BTC-USD", side="BUY",
            qty=0.00206, entry_price=50000.0,
        )
        assert outcome.get("ok") is True, outcome
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == outcome["trade_id"]).first()
            assert float(trade.qty) == 0.00206  # unchanged — under the slack ceiling
    finally:
        _bc.set_many({"bot_position_size_usd": "80", "bot_long_only": "true"})
