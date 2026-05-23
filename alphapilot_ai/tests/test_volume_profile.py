"""Phase B Batch 3: volume profile (POC, VAH, VAL).

Pure-math module — tests verify the histogram allocation and the
value-area expansion against hand-built candle sequences with known
expected POC / VAH / VAL.
"""
from __future__ import annotations

import pytest

from trading.volume_profile import (
    VolumeProfile,
    compute_volume_profile,
    is_high_volume_node,
    is_inside_value_area,
)


def _c(low: float, high: float, vol: float) -> dict:
    """Minimal candle. Open/close/time don't affect volume-profile math."""
    mid = (high + low) / 2.0
    return {"time": 0, "open": mid, "high": high, "low": low,
            "close": mid, "volume": vol}


# --------------------------------------------------------------------------
# POC — point of control
# --------------------------------------------------------------------------

def test_poc_lands_in_the_bin_with_the_most_volume():
    """One heavy candle at $100-101, several thin ones at $90-95.
    POC must sit in the heavy band."""
    candles = [
        _c(90, 91, 100),
        _c(91, 92, 100),
        _c(93, 94, 100),
        _c(94, 95, 100),
        _c(100, 101, 5000),   # the heavyweight
    ]
    p = compute_volume_profile(candles, bins=20)
    assert p is not None
    assert 100.0 <= p.poc <= 101.0


def test_poc_when_every_bar_is_at_same_price_is_that_price():
    """All bars at exactly $100 — degenerate case, profile collapses to
    a single point. POC = VAH = VAL = $100."""
    candles = [_c(100, 100, 500) for _ in range(5)]
    p = compute_volume_profile(candles)
    assert p is not None
    assert p.poc == 100.0
    assert p.vah == 100.0 and p.val == 100.0


def test_returns_none_on_empty_input():
    assert compute_volume_profile([]) is None


def test_returns_none_on_zero_total_volume():
    """A profile with no volume can't tell you where capital sat."""
    candles = [_c(100, 110, 0), _c(105, 115, 0)]
    assert compute_volume_profile(candles) is None


# --------------------------------------------------------------------------
# Value area
# --------------------------------------------------------------------------

def test_value_area_brackets_the_target_percent_of_volume():
    """Default target is 70%. The reported actual % must be >= 70%
    (we always expand UNTIL the threshold, so we can land at-or-above)."""
    candles = [_c(100 + i * 0.5, 101 + i * 0.5, 1000 + i * 50) for i in range(10)]
    p = compute_volume_profile(candles, bins=20, value_area_pct=0.70)
    assert p is not None
    assert p.value_area_pct_actual >= 0.70 - 0.01  # tiny float slack


def test_value_area_brackets_the_poc():
    """VAL <= POC <= VAH must always hold — the value area is
    BUILT around the POC."""
    candles = [_c(100, 105, 500), _c(102, 107, 800), _c(104, 109, 600)]
    p = compute_volume_profile(candles)
    assert p is not None
    assert p.val <= p.poc <= p.vah


def test_value_area_tighter_with_higher_target():
    """The 95% value area must be at least as wide as the 70% one —
    you need more of the volume, so you need a wider price range."""
    candles = [_c(100 + i, 101 + i, 1000) for i in range(20)]
    p70 = compute_volume_profile(candles, bins=20, value_area_pct=0.70)
    p95 = compute_volume_profile(candles, bins=20, value_area_pct=0.95)
    assert p70 is not None and p95 is not None
    assert (p95.vah - p95.val) >= (p70.vah - p70.val)


# --------------------------------------------------------------------------
# Histogram allocation — the per-bin volumes
# --------------------------------------------------------------------------

def test_bins_sum_to_total_volume():
    """Allocation must be conservation-of-volume: every unit of
    candle volume ends up in exactly one bin (no leaks)."""
    candles = [
        _c(100, 105, 1000),
        _c(102, 108, 500),
        _c(95, 110, 2000),
    ]
    p = compute_volume_profile(candles, bins=30)
    assert p is not None
    assert sum(v for _, v in p.bins) == pytest.approx(p.total_volume, rel=1e-9)
    assert p.total_volume == pytest.approx(3500.0, rel=1e-9)


def test_doji_candle_volume_goes_to_one_bin():
    """A candle with low == high (no range) drops everything into
    the bin containing its single price."""
    candles = [
        _c(100, 110, 1000),     # spread out — gives us a price range to bin in
        _c(105, 105, 5000),     # doji at 105 — should dump 5000 into one bin
    ]
    p = compute_volume_profile(candles, bins=20)
    assert p is not None
    # The bin containing $105 must hold at least the 5000 from the doji
    # (plus whatever overlap the wide candle gave it).
    idx = int((105 - p.overall_low) / p.bin_size)
    idx = max(0, min(p.bin_count - 1, idx))
    assert p.bins[idx][1] >= 5000.0


def test_uniform_distribution_for_a_wide_candle():
    """A single candle from $100 to $110 with 1000 volume on a 10-bin
    range should spread roughly evenly — each bin gets ~100."""
    candles = [_c(100, 110, 1000)]
    p = compute_volume_profile(candles, bins=10)
    assert p is not None
    assert p.bin_count == 10
    for _, v in p.bins:
        assert v == pytest.approx(100.0, rel=1e-6)


# --------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------

def test_is_inside_value_area():
    candles = [_c(100 + i, 101 + i, 1000) for i in range(10)]
    p = compute_volume_profile(candles, bins=20)
    assert p is not None
    assert is_inside_value_area(p, p.poc) is True
    assert is_inside_value_area(p, p.val) is True
    assert is_inside_value_area(p, p.vah) is True
    assert is_inside_value_area(p, p.vah + 100) is False
    assert is_inside_value_area(p, p.val - 100) is False


def test_is_high_volume_node_flags_the_poc():
    """The POC bin is by definition the heaviest — it must be flagged
    as a high-volume node under any reasonable multiplier."""
    candles = [
        _c(95, 96, 100), _c(96, 97, 100), _c(97, 98, 100),
        _c(100, 101, 10_000),     # massive volume here
        _c(102, 103, 100), _c(103, 104, 100),
    ]
    p = compute_volume_profile(candles, bins=15)
    assert p is not None
    assert is_high_volume_node(p, p.poc, multiplier=1.5) is True


def test_is_high_volume_node_rejects_a_quiet_price():
    candles = [
        _c(95, 96, 100), _c(96, 97, 100), _c(97, 98, 100),
        _c(100, 101, 10_000),
        _c(102, 103, 100), _c(103, 104, 100),
    ]
    p = compute_volume_profile(candles, bins=15)
    assert p is not None
    # A price in the thin region (95-97) should not register as HVN.
    assert is_high_volume_node(p, 95.5, multiplier=1.5) is False


# --------------------------------------------------------------------------
# End-to-end sanity
# --------------------------------------------------------------------------

def test_realistic_bimodal_distribution_picks_the_heavier_mode():
    """Two clusters: 10x at $100-101 (light) and 50x at $110-111 (heavy).
    POC must land in the heavy cluster."""
    candles = []
    for _ in range(10):
        candles.append(_c(100, 101, 100))
    for _ in range(50):
        candles.append(_c(110, 111, 100))
    p = compute_volume_profile(candles, bins=20)
    assert p is not None
    assert 110.0 <= p.poc <= 111.0
