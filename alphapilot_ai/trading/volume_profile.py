"""
Volume profile — Phase B Batch 3 of the signal overhaul.

Levels (Batch 1) tells you WHERE the chart anchors are. Structure
(Batch 2) tells you WHERE price has been reacting. This module tells
you WHERE the volume actually was — which is what makes a level
respected vs. just incidentally visited.

A volume profile is a histogram of traded volume by price level over
some window of candles. The key metrics:

  POC (Point of Control)
      The single price level with the most traded volume. The market's
      "fair price" anchor for the window.

  Value Area (VAL, VAH)
      The contiguous price range, centered on the POC, that contains a
      configurable percentage of total volume (typically 70%). VAL is
      the lower bound; VAH is the upper bound. Outside the value area
      is "low-volume territory" — price moves there but capital didn't
      sit there for long.

Volume profile is what tells the bot the difference between:
  - "Resistance at $76,500" — pivot says so, but maybe nobody traded
    much around that level, so it's a soft zone the next move can
    blow through.
  - "Resistance at $76,500 AND 18% of the last day's volume happened
    in the $76,400-76,600 band" — that's a high-volume node, a real
    wall.

Phase C setups will combine this with structural zones from Batch 2:
a structural resistance that ALSO coincides with a HVN is a much
higher-conviction reject point than one that doesn't.

THE COMPUTATION
Each candle's volume is allocated across the price bins it overlaps,
weighted by the overlap fraction (uniform-distribution model). The
bin with the highest accumulated volume is the POC. The value area
expands outward from POC, greedily adding the higher-volume neighbor
on each side, until the cumulative volume crosses the threshold.

PURE MATH — same design as Batches 1 and 2. No DB, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VolumeProfile:
    """Snapshot of where volume sat over a window of candles."""
    poc: float                                # the single highest-volume price level
    vah: float                                # value-area high
    val: float                                # value-area low
    value_area_pct_target: float              # configured target (e.g. 0.70)
    value_area_pct_actual: float              # actual fraction of volume the [VAL, VAH] range holds
    total_volume: float
    overall_low: float                        # lowest bar low in the window
    overall_high: float                       # highest bar high in the window
    bin_count: int
    bin_size: float
    # The full histogram, low -> high. Each tuple is (bin_low_price, volume).
    # The bin spans [bin_low_price, bin_low_price + bin_size). The last bin
    # is inclusive on the high end. Exposed so setups can ask things like
    # "is current price inside a high-volume node, or in a vacuum?"
    bins: list[tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Allocation: spread a candle's volume across the bins it overlaps.
# --------------------------------------------------------------------------

def _allocate_candle_volume(
    candle_low: float,
    candle_high: float,
    candle_volume: float,
    overall_low: float,
    bin_size: float,
    n_bins: int,
) -> list[float]:
    """Return per-bin volume contribution for a single candle.

    Uniform-distribution model: the candle's volume is spread evenly
    across its [low, high] range, then each bin's share is proportional
    to its overlap with that range. For a doji (low == high), the
    whole volume lands in the bin containing the single price.
    """
    out = [0.0] * n_bins
    if candle_volume <= 0 or bin_size <= 0 or n_bins <= 0:
        return out

    if candle_low >= candle_high:
        # Doji — assign everything to the bin containing this price.
        idx = int((candle_low - overall_low) / bin_size)
        idx = max(0, min(n_bins - 1, idx))
        out[idx] = candle_volume
        return out

    candle_range = candle_high - candle_low
    # Restrict iteration to the bins that COULD overlap, instead of all N.
    first_bin = max(0, int((candle_low - overall_low) / bin_size))
    last_bin = min(n_bins - 1, int((candle_high - overall_low) / bin_size))
    for i in range(first_bin, last_bin + 1):
        bin_lo = overall_low + i * bin_size
        bin_hi = bin_lo + bin_size
        overlap_lo = max(candle_low, bin_lo)
        overlap_hi = min(candle_high, bin_hi)
        if overlap_hi > overlap_lo:
            out[i] = candle_volume * ((overlap_hi - overlap_lo) / candle_range)
    return out


# --------------------------------------------------------------------------
# Value area expansion: grow outward from the POC.
# --------------------------------------------------------------------------

def _expand_value_area(
    bin_volumes: list[float],
    poc_idx: int,
    total_volume: float,
    target_pct: float,
) -> tuple[int, int, float]:
    """Greedy expansion: pick the heavier neighbor each step.

    Returns (val_bin_idx, vah_bin_idx, achieved_pct) — the inclusive
    bin range that holds `>= target_pct * total_volume`.
    """
    n = len(bin_volumes)
    lo = hi = poc_idx
    acc = bin_volumes[poc_idx]
    target = target_pct * total_volume

    while acc < target:
        above = bin_volumes[hi + 1] if hi + 1 < n else None
        below = bin_volumes[lo - 1] if lo - 1 >= 0 else None
        if above is None and below is None:
            break
        if above is None:
            lo -= 1
            acc += below       # type: ignore[operator]
        elif below is None:
            hi += 1
            acc += above
        elif above >= below:
            hi += 1
            acc += above
        else:
            lo -= 1
            acc += below
    actual_pct = acc / total_volume if total_volume > 0 else 0.0
    return lo, hi, actual_pct


# --------------------------------------------------------------------------
# Top-level: build a profile from candles.
# --------------------------------------------------------------------------

def compute_volume_profile(
    candles: list[dict],
    bins: int = 50,
    value_area_pct: float = 0.70,
) -> Optional[VolumeProfile]:
    """Build a volume-by-price histogram and find POC, VAH, VAL.

    Returns None if the input has no candles, zero total volume, or
    all candles share the same price (no range to bin). Callers
    treat None as "no profile available" and fall back accordingly.

    Args:
        candles: oldest -> newest. Each needs low / high / volume.
        bins: histogram resolution. 50 is a good default — fine enough
            to find a tight POC, coarse enough that single bars don't
            drive noise. Bump to 100 for tighter zones on big-range
            symbols.
        value_area_pct: target fraction of volume to bracket inside
            [VAL, VAH]. The industry-standard 70% is the default.
    """
    if not candles or bins <= 0:
        return None

    lows = [float(c["low"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    overall_low = min(lows)
    overall_high = max(highs)

    if overall_high <= overall_low:
        # Degenerate: every bar at the same price.
        total = sum(float(c.get("volume", 0.0) or 0.0) for c in candles)
        if total <= 0:
            return None
        return VolumeProfile(
            poc=overall_low, vah=overall_low, val=overall_low,
            value_area_pct_target=value_area_pct,
            value_area_pct_actual=1.0,
            total_volume=total,
            overall_low=overall_low, overall_high=overall_high,
            bin_count=1, bin_size=0.0,
            bins=[(overall_low, total)],
        )

    bin_size = (overall_high - overall_low) / bins
    bin_volumes = [0.0] * bins
    for c in candles:
        allocs = _allocate_candle_volume(
            float(c["low"]), float(c["high"]),
            float(c.get("volume", 0.0) or 0.0),
            overall_low, bin_size, bins,
        )
        for i, v in enumerate(allocs):
            bin_volumes[i] += v

    total_volume = sum(bin_volumes)
    if total_volume <= 0:
        return None

    # POC: highest-volume bin. Tie-break by earliest bin (lower price).
    poc_idx = max(range(bins), key=lambda i: (bin_volumes[i], -i))
    poc_price = overall_low + (poc_idx + 0.5) * bin_size

    val_idx, vah_idx, actual_pct = _expand_value_area(
        bin_volumes, poc_idx, total_volume, value_area_pct,
    )
    val_price = overall_low + val_idx * bin_size
    vah_price = overall_low + (vah_idx + 1) * bin_size

    histogram = [
        (overall_low + i * bin_size, bin_volumes[i])
        for i in range(bins)
    ]
    return VolumeProfile(
        poc=poc_price, vah=vah_price, val=val_price,
        value_area_pct_target=value_area_pct,
        value_area_pct_actual=actual_pct,
        total_volume=total_volume,
        overall_low=overall_low, overall_high=overall_high,
        bin_count=bins, bin_size=bin_size,
        bins=histogram,
    )


def is_inside_value_area(profile: VolumeProfile, price: float) -> bool:
    """True iff `price` falls within [VAL, VAH]. Convenience for setups
    that want to ask 'are we in the middle of the volume cloud or out
    at the edges?'"""
    return profile.val <= price <= profile.vah


def is_high_volume_node(
    profile: VolumeProfile,
    price: float,
    multiplier: float = 1.5,
) -> bool:
    """True iff the bin containing `price` has at least `multiplier`x
    the average bin volume. Marks the price as sitting in a region
    where capital actually concentrated. Default 1.5x is a moderate
    threshold; raise to 2.0 for stricter HVN definition.
    """
    if profile.bin_size <= 0 or not profile.bins:
        return False
    avg = profile.total_volume / profile.bin_count if profile.bin_count else 0.0
    if avg <= 0:
        return False
    idx = int((price - profile.overall_low) / profile.bin_size)
    idx = max(0, min(profile.bin_count - 1, idx))
    return profile.bins[idx][1] >= multiplier * avg
