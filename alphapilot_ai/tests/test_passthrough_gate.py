"""Regression: the passthrough confluence gate.

The training_passthrough / technical_strong decision paths execute a raw
technical signal WITHOUT Claude review. With no quality gate, an overnight
training run fired 45 trades at an 11% win rate — including textbook-bad
entries the learned playbook already had rules against:

    ARPA-USD SELL  RSI 30   vol 0.6x   (shorting into oversold)
    FUN1-USD BUY   RSI 83.6 vol 0.1x   (buying a blow-off on no volume)
    FOX-USD  SELL  vol 0.0x            (entering with zero participation)
    FLOCK-USD BUY  EMA cross 1 bar old (trading noise)

`_confluence_gate` is the enforcement layer. These tests reconstruct the
real losing signals and assert the gate now blocks them, while a genuine
multi-indicator signal still passes.
"""
from __future__ import annotations

from trading.strategy_engine import Signal
from ai.claude_decision_engine import _confluence_gate, _PASSTHROUGH_MIN_CONFLUENCE


def _sig(side: str, **indicators) -> Signal:
    return Signal(side, 0.79, "test signal", "Momentum", dict(indicators))


# --------------------------------------------------------------------------
# Hard vetoes — the egregious entries from the overnight run
# --------------------------------------------------------------------------

def test_blocks_overbought_buy():
    """FUN1-USD: buying RSI 83.6 is a blow-off, not momentum."""
    ok, reason = _confluence_gate(
        _sig("BUY", rsi=83.6, macd_histogram=0.0002, relative_volume=0.1,
             velocity_3bar=0.0128, body_direction=0.86),
        "BUY",
    )
    assert not ok
    assert "overbought" in reason


def test_blocks_oversold_sell():
    """AXS-USD: shorting RSI 28.6 is exhaustion, not momentum."""
    ok, reason = _confluence_gate(
        _sig("SELL", rsi=28.6, macd_histogram=-0.0009, relative_volume=0.1,
             velocity_3bar=-0.0008, body_direction=-1.0),
        "SELL",
    )
    assert not ok
    assert "oversold" in reason


def test_blocks_zero_volume_entry():
    """FOX-USD: 0.0x volume means no participation behind the move."""
    ok, reason = _confluence_gate(
        _sig("SELL", rsi=51.7, macd_histogram=-0.00001, relative_volume=0.0,
             velocity_3bar=-0.0164, body_direction=-2.0),
        "SELL",
    )
    assert not ok
    assert "volume" in reason


def test_blocks_one_bar_cross():
    """FLOCK-USD: a 1-bar-old EMA cross is noise, not a signal."""
    ok, reason = _confluence_gate(
        _sig("BUY", rsi=68.2, macd_histogram=0.0001, relative_volume=4.6,
             velocity_3bar=0.0088, body_direction=1.0, cross_age_bars=1),
        "BUY",
    )
    assert not ok
    assert "cross" in reason


# --------------------------------------------------------------------------
# Confluence — weak signals fail, genuine signals pass
# --------------------------------------------------------------------------

def test_blocks_weak_confluence_sell():
    """A SELL that clears every hard veto (RSI 50, volume 0.7x, fresh cross)
    but agrees on too few indicators — MACD up, velocity up, flat body — is
    blocked for weak confluence rather than vetoed outright."""
    ok, reason = _confluence_gate(
        _sig("SELL", rsi=50.0, macd_histogram=0.0008, relative_volume=0.7,
             velocity_3bar=0.0009, body_direction=0.0, cross_age_bars=6),
        "SELL",
    )
    assert not ok
    assert "confluence" in reason


def test_passes_genuine_momentum_buy():
    """A real momentum BUY — RSI with room, MACD up, volume, velocity,
    bullish body — hits all five checks and must pass."""
    ok, reason = _confluence_gate(
        _sig("BUY", rsi=58.0, macd_histogram=0.0021, relative_volume=1.6,
             velocity_3bar=0.009, body_direction=1.0, cross_age_bars=5),
        "BUY",
    )
    assert ok, reason


def test_passes_genuine_momentum_sell():
    ok, reason = _confluence_gate(
        _sig("SELL", rsi=44.0, macd_histogram=-0.0021, relative_volume=1.4,
             velocity_3bar=-0.009, body_direction=-1.0, cross_age_bars=6),
        "SELL",
    )
    assert ok, reason


def test_exactly_three_checks_is_enough():
    """The threshold is >= 3 of 5. A signal that clears exactly 3 passes."""
    # BUY: rsi_room (Y), macd_up (Y), volume (Y), velocity (0 -> N), body (0 -> N)
    ok, reason = _confluence_gate(
        _sig("BUY", rsi=55.0, macd_histogram=0.001, relative_volume=1.2,
             velocity_3bar=0.0, body_direction=0.0),
        "BUY",
    )
    assert ok, reason
    assert _PASSTHROUGH_MIN_CONFLUENCE == 3
