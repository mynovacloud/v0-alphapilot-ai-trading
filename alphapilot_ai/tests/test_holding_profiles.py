"""Regression: holding profiles + the universal hard time cap.

The bug this guards against
---------------------------
position_monitor previously branched on a wallet `trading_style` and,
for any trade with entry confidence >= 0.70, set a `high_conviction`
flag that skipped the ENTIRE scalper exit block — including both of its
time-stops. The swing time-limit only applied to `trading_style ==
"swing"`, which the monitor auto-switched away from. Net effect: a
high-confidence trade had NO time-based exit at all and could sit dead
for 40-60+ minutes until a 3-9% price move resolved it.

The fix: every trade is stamped with a HoldingProfile at entry, and the
monitor enforces `profile.hard_cap_minutes` for EVERY trade with no
escape hatch. These tests lock that in.
"""
from __future__ import annotations

from datetime import timedelta

from database.db import session_scope
from database.models import Wallet, PaperTrade
from utils.helpers import utcnow
from trading.position_monitor import PositionMonitor
from trading.holding_profiles import (
    BASE_PROFILES,
    resolve_profile_name,
    profile_from_claude_targets,
    get_profile,
)


# --------------------------------------------------------------------------
# Pure-logic: profile definitions + resolution
# --------------------------------------------------------------------------

def test_every_profile_is_mathematically_winnable():
    """max_loss_pct < target_pct for every base profile — the invariant
    that keeps the system winnable below a 50% win rate."""
    for name, p in BASE_PROFILES.items():
        assert p.max_loss_pct < p.target_pct, f"{name} violates the payoff invariant"
        assert p.hard_cap_minutes > 0, f"{name} has no hard time cap"
        assert p.stale_minutes <= p.hard_cap_minutes, f"{name} stale > hard cap"


def test_mixed_mode_tiers_by_confidence():
    """`mixed` must escalate the profile as confidence rises."""
    assert resolve_profile_name("mixed", 0.50) == "scalp"
    assert resolve_profile_name("mixed", 0.60) == "short_hold"
    assert resolve_profile_name("mixed", 0.70) == "short_swing"
    assert resolve_profile_name("mixed", 0.90) == "long_hold"


def test_fixed_modes_resolve_to_themselves():
    for name in BASE_PROFILES:
        assert resolve_profile_name(name, 0.5) == name


def test_ai_decide_reads_claude_take_profit():
    """`ai_decide` derives the profile from the TP magnitude Claude chose."""
    assert profile_from_claude_targets(0.004) == "scalp"
    assert profile_from_claude_targets(0.012) == "short_hold"
    assert profile_from_claude_targets(0.030) == "short_swing"
    assert profile_from_claude_targets(0.090) == "long_hold"
    # ai_decide consults that choice.
    assert resolve_profile_name("ai_decide", 0.8, "long_hold") == "long_hold"


def test_legacy_and_unknown_modes_fall_back_safely():
    """Old wallet.trading_style values and garbage both resolve, never raise."""
    assert resolve_profile_name("hybrid", 0.5) == "short_swing"
    assert resolve_profile_name("scalper", 0.5) == "scalp"
    assert resolve_profile_name("swing", 0.5) == "long_hold"
    assert resolve_profile_name("not-a-real-mode", 0.5) == "short_swing"
    assert get_profile(None).name == "short_swing"


# --------------------------------------------------------------------------
# Integration: the monitor enforces the hard time cap
# --------------------------------------------------------------------------

def _open_trade(session, *, age_minutes, profile, confidence=0.92,
                entry=100.0, side="BUY"):
    """Seed a wallet + one open PaperTrade aged `age_minutes` in the past."""
    wallet = Wallet(name=f"hp-test-{age_minutes}-{profile}", platform="paper")
    session.add(wallet)
    session.flush()
    trade = PaperTrade(
        wallet_id=wallet.id,
        symbol="BTC-USD",
        side=side,
        qty=1.0,
        entry_price=entry,
        status="open",
        confidence=confidence,
        holding_profile=profile,
        high_water_price=entry,
        opened_at=utcnow() - timedelta(minutes=age_minutes),
    )
    session.add(trade)
    session.flush()
    return trade


def test_high_conviction_trade_is_force_exited_by_hard_cap():
    """THE bug fix: a high-confidence ('high_conviction' under the old
    code), flat trade held well past its cap MUST be force-closed.

    Under the old logic this trade had no time-based exit and would hold
    indefinitely. The scalp profile caps the hold at 10 minutes; a
    60-minute-old trade must come back as a `time_cap` exit."""
    mon = PositionMonitor()
    with session_scope() as s:
        trade = _open_trade(s, age_minutes=60, profile="scalp", confidence=0.95)
        # current price == entry: flat, so no profit/loss/trailing exit
        # could fire. The ONLY thing that can close this is the time cap.
        signal = mon._check_single_position(s, trade, current_price=100.0)
        assert signal is not None, (
            "a 60-min-old scalp trade was not force-exited — the hard "
            "time cap is not being enforced"
        )
        assert signal.reason == "time_cap"


def test_fresh_trade_is_held():
    """A trade still inside every threshold must NOT be exited."""
    mon = PositionMonitor()
    with session_scope() as s:
        trade = _open_trade(s, age_minutes=2, profile="short_swing")
        signal = mon._check_single_position(s, trade, current_price=100.0)
        assert signal is None


def test_take_profit_fires_at_profile_target():
    mon = PositionMonitor()
    with session_scope() as s:
        trade = _open_trade(s, age_minutes=1, profile="scalp")
        # scalp target is 0.3%; +0.4% must take profit.
        signal = mon._check_single_position(s, trade, current_price=100.4)
        assert signal is not None and signal.reason == "take_profit"


def test_hard_max_loss_fires_with_no_min_hold_delay():
    """The profile's max_loss must fire immediately — no 15-minute
    minimum-hold gate (that gate was part of the original problem)."""
    mon = PositionMonitor()
    with session_scope() as s:
        trade = _open_trade(s, age_minutes=1, profile="scalp")
        # scalp max_loss is 0.15%; -0.2% on a 1-minute-old trade must cut.
        signal = mon._check_single_position(s, trade, current_price=99.8)
        assert signal is not None and signal.reason == "max_loss"
