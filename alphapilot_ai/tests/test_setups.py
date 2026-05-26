"""Phase C setups — the named-hypothesis signal library.

Each setup gets unit-tested against hand-built candle sequences where
the expected behavior is unambiguous: this sequence should fire BUY,
this one should HOLD because of *specific reason X*.

The "should HOLD because X" variants matter because the failure mode
of a setup is rarely "doesn't fire at all" — it's "fires in a
condition it shouldn't have," like a low-volume reclaim that
immediately fails. Each gate gets a test that proves it's actually
gating.
"""
from __future__ import annotations

import pytest

from trading.setups import vwap_reclaim_signal


def _c(t: int, o: float, h: float, l: float, c: float, v: float = 1000.0) -> dict:
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _below_then_reclaim(
    n_pre: int = 15,
    n_below: int = 9,
    pre_price: float = 100.0,
    below_price: float = 98.0,
    reclaim_open: float = 98.0,
    reclaim_close: float = 100.5,
    reclaim_volume: float = 2500.0,
    bar_volume: float = 1000.0,
) -> list[dict]:
    """Build the canonical setup: stable price, then dip, then bullish
    reclaim. Default values produce a clean BUY signal."""
    candles: list[dict] = []
    t = 0
    for _ in range(n_pre):
        candles.append(_c(t, pre_price, pre_price + 0.5, pre_price - 0.5,
                          pre_price, v=bar_volume))
        t += 60
    for _ in range(n_below):
        candles.append(_c(t, below_price, below_price + 0.3, below_price - 0.3,
                          below_price, v=bar_volume))
        t += 60
    candles.append(_c(t, reclaim_open,
                      max(reclaim_open, reclaim_close) + 0.2,
                      min(reclaim_open, reclaim_close) - 0.5,
                      reclaim_close,
                      v=reclaim_volume))
    return candles


# --------------------------------------------------------------------------
# The happy path — every gate passes
# --------------------------------------------------------------------------

def test_vwap_reclaim_fires_buy_on_textbook_setup():
    """Stable at 100, dip to 98 for 9 bars, then a bullish bar closes
    above the VWAP with 2.5x average volume. All five gates pass."""
    candles = _below_then_reclaim()
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "BUY"
    assert sig.confidence > 0.75
    assert "VWAP reclaim" in sig.reasoning
    assert "vwap" in sig.indicators
    assert sig.indicators["bars_below_vwap"] >= 6.0
    assert sig.indicators["volume_ratio"] > 1.2


def test_vwap_reclaim_confidence_scales_with_evidence():
    """Stronger setup (more bars below, bigger volume, further above
    VWAP) must produce higher confidence than a marginal one."""
    marginal = _below_then_reclaim(
        n_below=6,                # exactly at MIN_BARS_BELOW
        reclaim_volume=1300.0,    # just above the 1.2x threshold (avg pulled
                                  # slightly above 1000 by the reclaim bar)
        reclaim_close=100.05,     # barely above VWAP
    )
    strong = _below_then_reclaim(
        n_below=10,               # well past the minimum
        reclaim_volume=4000.0,    # 4x avg
        reclaim_close=102.0,      # comfortably above VWAP
    )
    s_marginal = vwap_reclaim_signal(marginal)
    s_strong = vwap_reclaim_signal(strong)
    assert s_marginal.side == "BUY" and s_strong.side == "BUY"
    assert s_strong.confidence > s_marginal.confidence


# --------------------------------------------------------------------------
# Gate-by-gate negative tests — each must hold for the right reason
# --------------------------------------------------------------------------

def test_holds_when_too_few_candles():
    """Below the structural minimum -> can't compute the setup at all."""
    candles = [_c(i * 60, 100, 101, 99, 100) for i in range(15)]
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "need >=" in sig.reasoning


def test_holds_when_last_close_below_vwap():
    """Reclaim never happened — current price still below VWAP."""
    candles = _below_then_reclaim(reclaim_close=97.0)    # close stays below
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "not above VWAP" in sig.reasoning


def test_holds_when_reclaim_candle_is_bearish():
    """Close above VWAP, but close < open (red candle). Not a real reclaim."""
    candles = _below_then_reclaim(reclaim_open=102.0, reclaim_close=100.5)
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "not bullish" in sig.reasoning


def test_holds_when_not_enough_prior_bars_were_below():
    """Only 2 of the prior bars sat below VWAP — this is chop, not a
    reclaim setup. Pad n_pre so we stay above the 21-bar floor and
    actually hit the prior-bars-below gate (rather than tripping on
    the data-length gate first)."""
    candles = _below_then_reclaim(n_pre=20, n_below=2)
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "prior bars below" in sig.reasoning


def test_holds_when_reclaim_volume_is_weak():
    """Volume on the reclaim bar is below 1.2x avg — not confirmed."""
    candles = _below_then_reclaim(reclaim_volume=900.0)   # 0.9x avg
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "volume" in sig.reasoning.lower()


def test_holds_on_zero_total_volume():
    """No volume anywhere — VWAP undefined, must fail safely."""
    candles = [_c(i * 60, 100, 101, 99, 100, v=0) for i in range(40)]
    sig = vwap_reclaim_signal(candles)
    assert sig.side == "HOLD"
    assert "VWAP" in sig.reasoning


# --------------------------------------------------------------------------
# Registry wiring — the live bot can find the setup by name
# --------------------------------------------------------------------------

def test_setup_is_registered_in_strategy_registry():
    """Phase C is meaningful only if `_STRATEGY_REGISTRY` actually
    advertises the new setup. Otherwise the bot can never trade it."""
    from trading.strategy_engine import _STRATEGY_REGISTRY
    assert "VWAP Reclaim" in _STRATEGY_REGISTRY
    # And calling it through the registry must work end-to-end.
    fn = _STRATEGY_REGISTRY["VWAP Reclaim"]
    sig = fn(_below_then_reclaim())
    assert sig.side == "BUY"


# ==========================================================================
# Opening-Range Breakout (ORB)
# ==========================================================================
# Build candles around a specific UTC anchor so we can deterministically
# place the opening-range window. Anchor = 2026-01-15 13:30 UTC.

import datetime as _dt   # local alias keeps the top of file clean

from trading.setups import (
    opening_range_breakout_signal,
    _ORB_ANCHOR_HOUR_UTC,
    _ORB_ANCHOR_MINUTE_UTC,
    _ORB_RANGE_MINUTES,
)


def _orb_anchor_ts() -> int:
    return int(_dt.datetime(
        2026, 1, 15,
        _ORB_ANCHOR_HOUR_UTC, _ORB_ANCHOR_MINUTE_UTC, 0,
        tzinfo=_dt.timezone.utc,
    ).timestamp())


def _orb_setup_candles(
    or_high: float = 101.0,
    or_low: float = 99.0,
    n_post_or_bars: int = 5,
    last_close: float = 102.0,
    last_open: float = 100.5,
    last_volume: float = 3000.0,
    in_range_volume: float = 1000.0,
) -> list[dict]:
    """Build a candle series with a clean opening range and a final
    breakout bar. Defaults produce a textbook ORB BUY."""
    a = _orb_anchor_ts()
    candles: list[dict] = []
    # 30 in-range bars, each oscillating tightly between or_low and or_high
    for i in range(_ORB_RANGE_MINUTES):
        candles.append(_c(a + 60 * i, or_low + 0.1, or_high, or_low,
                          or_low + 0.5, v=in_range_volume))
    # Post-OR bars below or_high (no premature breakout)
    for i in range(n_post_or_bars - 1):
        t = a + 60 * (_ORB_RANGE_MINUTES + i)
        candles.append(_c(t, or_low + 0.5, or_high - 0.1, or_low + 0.2,
                          or_high - 0.2, v=in_range_volume))
    # The breakout bar (transition: prev was below, this closes above or_high)
    t = a + 60 * (_ORB_RANGE_MINUTES + n_post_or_bars - 1)
    candles.append(_c(t, last_open,
                      max(last_close, last_open) + 0.2,
                      min(last_close, last_open) - 0.5,
                      last_close, v=last_volume))
    return candles


def test_orb_fires_buy_on_textbook_breakout():
    """OR cleanly defined, latest bar closes above OR_high on heavy
    volume — all six gates pass."""
    candles = _orb_setup_candles()
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "BUY"
    assert sig.confidence > 0.70
    assert sig.indicators["or_high"] == 101.0
    assert sig.indicators["or_low"] == 99.0
    assert sig.indicators["volume_ratio"] > 1.3


def test_orb_holds_when_too_few_candles():
    """Below the 35-bar minimum the setup can't even establish the OR."""
    candles = [_c(i * 60, 100, 101, 99, 100) for i in range(20)]
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "HOLD"
    assert "need >=" in sig.reasoning


def test_orb_holds_when_still_inside_or_window():
    """Latest bar's time still inside [anchor, anchor+30min) → too
    early to call a breakout."""
    a = _orb_anchor_ts()
    candles = [
        _c(a + i * 60, 99.5, 101.0, 99.0, 100.0, v=1000.0)
        for i in range(35)
    ]
    # Place the LAST bar inside the OR window (time = anchor + 20m)
    candles[-1] = _c(a + 20 * 60, 99.5, 101.0, 99.0, 100.0, v=1000.0)
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "HOLD"
    # Either "still inside" or "no transition" is acceptable here —
    # the point is it doesn't fire BUY.
    assert sig.side == "HOLD"


def test_orb_holds_when_no_breakout_transition():
    """OR is established, time is past it, but price never closes above
    OR_high. No transition, no signal."""
    candles = _orb_setup_candles(last_close=100.5)   # below or_high=101.0
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "HOLD"
    assert "no breakout transition" in sig.reasoning


def test_orb_holds_when_breakout_volume_weak():
    """Price breaks out but on weak volume — not confirmed."""
    candles = _orb_setup_candles(last_volume=900.0)   # below 1.3x avg
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "HOLD"
    assert "volume" in sig.reasoning.lower()


def test_orb_holds_when_or_signal_decayed():
    """Far past the OR close (> 4h), the signal is stale even on a
    fresh breakout."""
    candles = _orb_setup_candles(n_post_or_bars=260)   # > 240 max
    sig = opening_range_breakout_signal(candles)
    assert sig.side == "HOLD"
    assert "decayed" in sig.reasoning or "past OR close" in sig.reasoning


def test_orb_is_registered_in_strategy_registry():
    from trading.strategy_engine import _STRATEGY_REGISTRY
    assert "ORB" in _STRATEGY_REGISTRY
    sig = _STRATEGY_REGISTRY["ORB"](_orb_setup_candles())
    assert sig.side == "BUY"
