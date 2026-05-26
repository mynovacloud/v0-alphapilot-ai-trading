"""Phase B Batch 2: structural S/R zones from swing-pivot clustering.

Tests verify pivot detection on hand-built candle sequences with
known-correct swing locations, then check the clustering and ranking
logic against expected behavior. Pure-math module, so no mocks.
"""
from __future__ import annotations

import pytest

from trading.structure import (
    SRZone,
    detect_swing_pivots,
    detect_sr_zones,
    find_nearest_zones,
)


def _c(t: int, h: float, l: float) -> dict:
    """Minimal candle. Open/close don't matter for swing detection on
    the structure side — only high and low do."""
    mid = (h + l) / 2.0
    return {"time": t, "open": mid, "high": h, "low": l, "close": mid, "volume": 1000.0}


# --------------------------------------------------------------------------
# Swing pivot detection
# --------------------------------------------------------------------------

def test_detects_one_obvious_swing_high():
    """A bar whose high beats the previous 3 AND the next 3 highs
    must be flagged as a swing high. Lows monotonically lower in this
    series so there should be no swing low."""
    candles = [
        _c(0,  101, 99),
        _c(1,  102, 99),
        _c(2,  104, 100),
        _c(3,  110, 102),   # <- swing high
        _c(4,  107, 105),
        _c(5,  105, 103),
        _c(6,  103, 101),
    ]
    highs, lows = detect_swing_pivots(candles, lookback=3)
    assert highs == [3]
    assert lows == []


def test_detects_one_obvious_swing_low():
    """Mirror of the high test — a clear V at index 3."""
    candles = [
        _c(0, 110, 105),
        _c(1, 109, 104),
        _c(2, 108, 102),
        _c(3, 105,  95),    # <- swing low
        _c(4, 107,  99),
        _c(5, 109, 101),
        _c(6, 110, 103),
    ]
    highs, lows = detect_swing_pivots(candles, lookback=3)
    assert highs == []
    assert lows == [3]


def test_no_pivots_on_monotonic_series():
    """A strictly rising or falling sequence has no internal extremum."""
    rising = [_c(i, 100 + i, 99 + i) for i in range(10)]
    highs, lows = detect_swing_pivots(rising, lookback=3)
    assert highs == [] and lows == []


def test_no_pivots_when_series_too_short():
    """With lookback=3 we need at least 7 bars (3+1+3). Anything less
    can't confirm a pivot."""
    candles = [_c(i, 100 + i, 99 + i) for i in range(6)]   # 6 < 7
    highs, lows = detect_swing_pivots(candles, lookback=3)
    assert highs == [] and lows == []


def test_strict_inequality_rejects_plateau_duplicates():
    """If a high is TIED with a neighbor, it's not a swing — prevents
    flat tops from producing multiple pivots at the same price."""
    candles = [
        _c(0, 100, 95),
        _c(1, 100, 95),
        _c(2, 100, 95),
        _c(3, 100, 95),     # tied with all neighbors — NOT a swing
        _c(4, 100, 95),
        _c(5, 100, 95),
        _c(6, 100, 95),
    ]
    highs, lows = detect_swing_pivots(candles, lookback=3)
    assert highs == [] and lows == []


def test_bar_cannot_be_both_high_and_low():
    """A bar marked as a swing high should not ALSO be marked as a
    swing low even if its low happens to clear both neighbor checks."""
    candles = [
        _c(0, 95, 90),
        _c(1, 96, 91),
        _c(2, 98, 93),
        _c(3, 110, 95),     # swing high; its low (95) is also above neighbors' lows
        _c(4, 99, 93),
        _c(5, 97, 91),
        _c(6, 95, 89),
    ]
    highs, lows = detect_swing_pivots(candles, lookback=3)
    assert 3 in highs
    assert 3 not in lows


# --------------------------------------------------------------------------
# Zone clustering & strength
# --------------------------------------------------------------------------

def test_two_close_pivots_merge_into_one_zone():
    """Two swing highs at $100 and $100.20 (20 bps apart) with default
    tolerance 30 bps must merge into a single zone with touches=2."""
    candles = [
        # First swing high at ~100
        _c(0, 95, 92), _c(1, 97, 93), _c(2, 99, 95),
        _c(3, 100, 97),
        _c(4, 99, 95), _c(5, 97, 93), _c(6, 95, 91),
        # Second swing high at ~100.2 — within tolerance
        _c(7, 95, 91), _c(8, 97, 93), _c(9, 99, 95),
        _c(10, 100.2, 97),
        _c(11, 99, 95), _c(12, 97, 93), _c(13, 95, 91),
    ]
    zones = detect_sr_zones(candles, lookback=3, tolerance_pct=0.005)
    assert len(zones) == 1
    assert zones[0].touches == 2
    assert zones[0].zone_type == "resistance"
    assert 100.0 <= zones[0].price <= 100.21


def test_two_distant_pivots_stay_separate():
    """Two swing highs at $100 and $110 are 10% apart — way outside
    a 30 bps tolerance. Two distinct zones."""
    candles = [
        _c(0, 95, 92), _c(1, 97, 93), _c(2, 99, 95),
        _c(3, 100, 97),
        _c(4, 99, 95), _c(5, 97, 93), _c(6, 95, 91),
        _c(7, 95, 91), _c(8, 97, 93), _c(9, 105, 100),
        _c(10, 110, 105),
        _c(11, 105, 100), _c(12, 100, 95), _c(13, 95, 91),
    ]
    zones = detect_sr_zones(candles, lookback=3, tolerance_pct=0.003)
    assert len(zones) == 2


def test_both_sided_zone_is_classified_as_both():
    """A level the market has rejected from above AND bounced from below
    is the strongest kind. Construct it explicitly: swing HIGH at price
    100 (price rose to 100, rejected, fell), then later a swing LOW
    also at 100 (price climbed back up, dipped to 100, bounced)."""
    candles = [
        # Swing high at index 3, high=100. Neighbors all < 100.
        _c(0, 90, 85), _c(1, 92, 87), _c(2, 95, 90),
        _c(3, 100, 98),
        _c(4, 95, 90), _c(5, 92, 87), _c(6, 90, 85),
        # Price runs up to ~115-120, then dips back to 100, then climbs
        # again — index 10 has low=100 with all neighbor lows strictly above.
        _c(7, 110, 105), _c(8, 115, 110), _c(9, 120, 115),
        _c(10, 115, 100),    # <- swing low at 100
        _c(11, 120, 115), _c(12, 125, 120), _c(13, 130, 125),
    ]
    zones = detect_sr_zones(candles, lookback=3, tolerance_pct=0.005)
    # Expected: one cluster at ~100 with both a high-pivot and a low-pivot.
    both_zones = [z for z in zones if z.zone_type == "both"]
    assert both_zones, "expected a 'both' zone where price reacted from both sides"
    z = both_zones[0]
    assert z.touches == 2
    assert 99.5 <= z.price <= 100.5


def test_min_touches_filter_drops_solo_pivots():
    """min_touches=2 should drop zones that only have a single pivot."""
    candles = [
        # One isolated swing high at 100
        _c(0, 95, 92), _c(1, 97, 93), _c(2, 99, 95),
        _c(3, 100, 97),
        _c(4, 99, 95), _c(5, 97, 93), _c(6, 95, 91),
    ]
    with_solo = detect_sr_zones(candles, lookback=3, min_touches=1)
    without_solo = detect_sr_zones(candles, lookback=3, min_touches=2)
    assert len(with_solo) >= 1
    assert len(without_solo) == 0


def test_zones_returned_strongest_first():
    """The output must be sorted by strength descending."""
    # Two zones — one with 3 touches (strong), one with 1 touch (weak)
    candles = [
        # Triple touch at ~100 (resistance)
        _c(0, 95, 92), _c(1, 97, 93), _c(2, 99, 95),
        _c(3, 100, 97),
        _c(4, 99, 95), _c(5, 97, 93), _c(6, 95, 91),
        _c(7, 95, 91), _c(8, 97, 93), _c(9, 99, 95),
        _c(10, 100.1, 97),
        _c(11, 99, 95), _c(12, 97, 93), _c(13, 95, 91),
        _c(14, 95, 91), _c(15, 97, 93), _c(16, 99, 95),
        _c(17, 100.2, 97),
        _c(18, 99, 95), _c(19, 97, 93), _c(20, 95, 91),
        # Single touch at ~120 (weak)
        _c(21, 95, 91), _c(22, 100, 95), _c(23, 110, 100),
        _c(24, 120, 110),
        _c(25, 110, 100), _c(26, 100, 95), _c(27, 95, 91),
    ]
    zones = detect_sr_zones(candles, lookback=3, tolerance_pct=0.005)
    assert len(zones) >= 2
    # The first (strongest) zone should be the triple-touch one near 100.
    assert zones[0].touches >= zones[1].touches
    assert abs(zones[0].price - 100) < 1.0


# --------------------------------------------------------------------------
# find_nearest_zones convenience
# --------------------------------------------------------------------------

def test_find_nearest_zones_resistance_above_support_below():
    zones = [
        SRZone(price=90.0, low=89.5, high=90.5, touches=2,
               last_touch_bar=5, last_touch_unix=0,
               zone_type="support", strength=0.7),
        SRZone(price=110.0, low=109.5, high=110.5, touches=2,
               last_touch_bar=10, last_touch_unix=0,
               zone_type="resistance", strength=0.7),
        SRZone(price=130.0, low=129.5, high=130.5, touches=1,
               last_touch_bar=15, last_touch_unix=0,
               zone_type="resistance", strength=0.4),
    ]
    res, sup = find_nearest_zones(zones, current_price=100.0)
    assert res is not None and res.price == 110.0   # 110 closer than 130
    assert sup is not None and sup.price == 90.0


def test_find_nearest_zones_handles_no_resistance_above():
    """At a price above every zone, there's nothing to call resistance."""
    zones = [
        SRZone(price=90.0, low=89.5, high=90.5, touches=2,
               last_touch_bar=5, last_touch_unix=0,
               zone_type="support", strength=0.7),
    ]
    res, sup = find_nearest_zones(zones, current_price=200.0)
    assert res is None
    assert sup is not None and sup.price == 90.0


def test_find_nearest_zones_on_empty_input():
    res, sup = find_nearest_zones([], current_price=100.0)
    assert res is None and sup is None
