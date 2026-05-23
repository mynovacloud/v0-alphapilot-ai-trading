"""Phase B: session-anchored TA primitives.

These are pure math on candle data. Tests verify the arithmetic against
hand-computed expected values plus boundary cases (empty data, single
bar, zero volume). The whole point of pure-function design is that
these are trivially testable without DB, network, or time mocks.
"""
from __future__ import annotations

import datetime

import pytest

from trading.levels import (
    SessionLevels,
    compute_opening_range,
    compute_pivots,
    compute_prior_day,
    compute_session_levels,
    compute_vwap,
)


def _c(t: int, o: float, h: float, l: float, c: float, v: float = 1000.0) -> dict:
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


# --------------------------------------------------------------------------
# VWAP
# --------------------------------------------------------------------------

def test_vwap_basic_arithmetic():
    """Two equal-volume bars with typical-price midpoints 100 and 110.
    Equal weighting means the VWAP must be the midpoint, 105."""
    candles = [_c(0, 99, 101, 99, 100, v=1000),
               _c(60, 109, 111, 109, 110, v=1000)]
    vwap, _hi, _lo = compute_vwap(candles)
    assert vwap == pytest.approx(105.0, abs=0.01)


def test_vwap_is_volume_weighted_not_just_averaged():
    """A bar with 10x more volume must dominate the VWAP."""
    candles = [_c(0, 99, 101, 99, 100, v=100),
               _c(60, 109, 111, 109, 110, v=1000)]
    vwap, _, _ = compute_vwap(candles)
    # Heavy weight on the 110 bar pulls VWAP up toward it.
    assert vwap > 108.0


def test_vwap_bands_are_symmetric_around_vwap():
    candles = [_c(0, 99, 105, 95, 100, v=500),
               _c(60, 109, 115, 105, 110, v=500)]
    vwap, hi, lo = compute_vwap(candles)
    assert (hi - vwap) == pytest.approx(vwap - lo, abs=0.01)


def test_vwap_returns_none_on_zero_volume():
    """If every bar reports zero volume, VWAP is undefined."""
    candles = [_c(0, 99, 101, 99, 100, v=0)]
    vwap, hi, lo = compute_vwap(candles)
    assert vwap is None and hi is None and lo is None


def test_vwap_empty_input_safe():
    assert compute_vwap([]) == (None, None, None)


def test_vwap_skips_zero_volume_bars_but_uses_the_rest():
    """A bar with zero volume should not crash the computation —
    it just doesn't contribute, but the others still produce VWAP."""
    candles = [_c(0, 99, 101, 99, 100, v=0),
               _c(60, 109, 111, 109, 110, v=1000)]
    vwap, _, _ = compute_vwap(candles)
    # Only the second bar contributes. Its typical price is 110.
    assert vwap == pytest.approx(110.0, abs=0.01)


# --------------------------------------------------------------------------
# Prior-day HLC
# --------------------------------------------------------------------------

def test_prior_day_uses_second_to_last_bar():
    """The latest daily bar is 'in progress today' — yesterday is the
    one before it."""
    yesterday = _c(-86400, 95, 102, 88, 99)
    today = _c(0, 100, 110, 95, 105)
    h, l, c = compute_prior_day([yesterday, today])
    assert h == 102.0 and l == 88.0 and c == 99.0


def test_prior_day_needs_at_least_two_bars():
    h, l, c = compute_prior_day([_c(0, 100, 110, 95, 105)])
    assert h is None and l is None and c is None


def test_prior_day_empty_safe():
    assert compute_prior_day([]) == (None, None, None)


# --------------------------------------------------------------------------
# Pivot points
# --------------------------------------------------------------------------

def test_pivot_formula_matches_textbook():
    """H=110, L=90, C=100  =>  P=100, R1=110, S1=90, R2=120, S2=80."""
    p, r1, r2, s1, s2 = compute_pivots(110, 90, 100)
    assert p == pytest.approx(100.0)
    assert r1 == pytest.approx(110.0)
    assert s1 == pytest.approx(90.0)
    assert r2 == pytest.approx(120.0)
    assert s2 == pytest.approx(80.0)


def test_pivot_ordering_invariant():
    """Across realistic BTC-scale inputs, the ordering S2 < S1 < P < R1 < R2
    must hold — this is a property of the classic formula and any bug that
    breaks it would be obvious here."""
    p, r1, r2, s1, s2 = compute_pivots(78500, 76800, 77200)
    assert s2 < s1 < p < r1 < r2


# --------------------------------------------------------------------------
# Opening range
# --------------------------------------------------------------------------

def test_opening_range_captures_only_first_n_minutes():
    """A 30-minute opening range with bars at t=0, +10m, +25m, +30m, +60m
    must include the first three (window is half-open [0, 30))."""
    anchor = datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    a = int(anchor.timestamp())
    candles = [
        _c(a,        100, 105,  99, 102),   # 0m  — IN
        _c(a +  600, 102, 108, 101, 106),   # 10m — IN
        _c(a + 1500, 106, 110, 105, 109),   # 25m — IN
        _c(a + 1800, 109, 112, 108, 111),   # 30m — OUT (exclusive boundary)
        _c(a + 3600, 111, 115, 110, 114),   # 60m — OUT
    ]
    h, l = compute_opening_range(candles, anchor, minutes=30)
    assert h == 110.0     # max of the first 3 bars' highs
    assert l == 99.0      # min of the first 3 bars' lows


def test_opening_range_none_when_no_bars_in_window():
    anchor = datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    a = int(anchor.timestamp())
    only_before = [_c(a - 3600, 100, 105, 99, 102)]
    assert compute_opening_range(only_before, anchor, minutes=30) == (None, None)


def test_opening_range_empty_input_safe():
    anchor = datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert compute_opening_range([], anchor) == (None, None)


# --------------------------------------------------------------------------
# End-to-end SessionLevels
# --------------------------------------------------------------------------

def test_compute_session_levels_combines_every_primitive():
    """Smoke test that the top-level helper wires every primitive together
    correctly given realistic inputs."""
    anchor = datetime.datetime(2026, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)
    a = int(anchor.timestamp())
    # 5 intraday bars at 5-min spacing starting right at the anchor.
    intraday = [_c(a + 300 * i, 100 + i, 102 + i, 99 + i, 101 + i, v=1000)
                for i in range(5)]
    daily = [
        _c(a - 86400 * 2, 90, 100, 85, 95),
        _c(a - 86400,     95, 108, 90, 100),   # the prior day (completed)
        _c(a,             100, 103, 98, 101),  # "today, in progress" — must be ignored
    ]
    levels = compute_session_levels(
        "BTC-USD", intraday, daily,
        session_start_utc=anchor,
        opening_range_minutes=30,
    )
    assert isinstance(levels, SessionLevels)
    assert levels.symbol == "BTC-USD"
    assert levels.bars_in_session == 5
    assert levels.vwap is not None
    assert levels.prior_day_high == 108.0
    assert levels.prior_day_low == 90.0
    assert levels.prior_day_close == 100.0
    # Pivot = (108 + 90 + 100) / 3 = 99.333…
    assert levels.pivot == pytest.approx((108 + 90 + 100) / 3.0)
    # All 5 bars sit inside the 30-minute opening range.
    assert levels.opening_range_high == 106.0   # max(102..106)
    assert levels.opening_range_low == 99.0     # min(99..103)


def test_compute_session_levels_handles_missing_daily_gracefully():
    """No daily candles -> the daily-derived fields stay None, but VWAP
    and bars_in_session still compute. Should never raise."""
    anchor = datetime.datetime(2026, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)
    intraday = [_c(int(anchor.timestamp()), 100, 102, 99, 101, v=1000)]
    levels = compute_session_levels("BTC-USD", intraday, daily_candles=None,
                                    session_start_utc=anchor)
    assert levels.vwap is not None
    assert levels.prior_day_high is None
    assert levels.pivot is None
    assert levels.r1 is None and levels.s1 is None


def test_compute_session_levels_filters_out_pre_session_bars_from_vwap():
    """Bars before the anchor must NOT contribute to VWAP — the whole
    point of an 'anchored' VWAP is that it resets at session start."""
    anchor = datetime.datetime(2026, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)
    a = int(anchor.timestamp())
    intraday = [
        _c(a - 3600, 200, 200, 200, 200, v=10_000),   # massive pre-session bar at 200
        _c(a,         100, 100, 100, 100, v=1000),    # in-session at 100
    ]
    levels = compute_session_levels("BTC-USD", intraday, None, session_start_utc=anchor)
    # If filtering works, VWAP is 100 (only the in-session bar counts).
    assert levels.vwap == pytest.approx(100.0, abs=0.01)
    assert levels.bars_in_session == 1
