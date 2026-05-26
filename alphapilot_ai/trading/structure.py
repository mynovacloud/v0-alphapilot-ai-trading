"""
Structural support/resistance zones — Phase B Batch 2 of the signal overhaul.

The first batch (trading/levels.py) computed *anchor* lines — VWAP,
prior-day HLC, classic pivots, opening range. Those are mechanical:
given today's session, you get today's numbers. They don't change as
the market trades.

This module computes *structural* levels — the price zones where the
market has actually been reacting. Same idea a discretionary trader
uses when they draw a horizontal line at "the place we keep bouncing
from." The math:

  1. Detect swing pivots (a bar's high beats all N neighbors on
     both sides => confirmed swing high; same in reverse for lows).
  2. Cluster pivots that sit within a small tolerance of each other —
     three touches of $76,500 in different sessions all describe the
     same zone, not three zones.
  3. Score each zone by touch count + recency. The strongest zones
     are returned first.

Zones get a `zone_type` tag: pure resistance (only swing highs), pure
support (only swing lows), or "both" — a level the market has bounced
off AND rejected at, which is the strongest kind. Both-sided zones
get a small strength bonus.

PHASE B SCOPE
This is the second of three modules under Phase B. Batch 1 was
session anchors; Batch 3 will be volume profile (POC/VAH/VAL).
Together they give Phase C's setup library the level data it needs.
No setup logic, no live wiring — those layers consume these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SRZone:
    """A structural support / resistance zone built from clustered pivots."""
    price: float            # midpoint of the zone (cluster mean)
    low: float              # lowest pivot price in the cluster
    high: float             # highest pivot price in the cluster
    touches: int            # number of pivots that formed this zone
    last_touch_bar: int     # index in the input candle list (recency)
    last_touch_unix: int    # absolute timestamp of the last touch
    zone_type: str          # "support" / "resistance" / "both"
    strength: float         # composite 0..1, used for ranking


def detect_swing_pivots(
    candles: list[dict],
    lookback: int = 3,
) -> tuple[list[int], list[int]]:
    """Find confirmed swing-high and swing-low bar indices.

    A bar at index i is a swing high if `candle[i].high` is strictly
    greater than the highs of the `lookback` bars before AND after it.
    Same in reverse for swing lows (using lows, strict-less).

    Strict inequality means flat plateaus don't produce duplicate
    pivots. Bars within `lookback` of either end are skipped — they
    don't have enough neighbors to confirm.
    """
    highs: list[int] = []
    lows: list[int] = []
    n = len(candles)
    if n < 2 * lookback + 1 or lookback < 1:
        return highs, lows

    for i in range(lookback, n - lookback):
        h_i = float(candles[i]["high"])
        l_i = float(candles[i]["low"])

        is_high = True
        for j in range(1, lookback + 1):
            if float(candles[i - j]["high"]) >= h_i or float(candles[i + j]["high"]) >= h_i:
                is_high = False
                break
        if is_high:
            highs.append(i)
            continue   # a bar can be a swing high OR low, not both

        is_low = True
        for j in range(1, lookback + 1):
            if float(candles[i - j]["low"]) <= l_i or float(candles[i + j]["low"]) <= l_i:
                is_low = False
                break
        if is_low:
            lows.append(i)
    return highs, lows


def _cluster_pivots(
    pivots: list[tuple[int, float, str]],
    tolerance_pct: float,
) -> list[list[tuple[int, float, str]]]:
    """Greedy 1-D clustering by price, with a running-mean center.

    `pivots` is a list of (bar_idx, price, kind) tuples where `kind`
    is "high" or "low". Returns a list of clusters; each cluster is
    a list of the original pivot tuples.
    """
    if not pivots:
        return []
    sorted_pivots = sorted(pivots, key=lambda p: p[1])

    clusters: list[list[tuple[int, float, str]]] = []
    current: list[tuple[int, float, str]] = [sorted_pivots[0]]
    current_mean = sorted_pivots[0][1]

    for piv in sorted_pivots[1:]:
        if abs(piv[1] - current_mean) / max(current_mean, 1e-9) <= tolerance_pct:
            current.append(piv)
            current_mean = sum(p[1] for p in current) / len(current)
        else:
            clusters.append(current)
            current = [piv]
            current_mean = piv[1]
    clusters.append(current)
    return clusters


def _cluster_to_zone(
    cluster: list[tuple[int, float, str]],
    candles: list[dict],
) -> SRZone:
    """Turn a list of pivot tuples into a scored SRZone."""
    prices = [p[1] for p in cluster]
    bar_indices = [p[0] for p in cluster]
    kinds = [p[2] for p in cluster]

    last_bar = max(bar_indices)
    last_unix = int(candles[last_bar]["time"]) if 0 <= last_bar < len(candles) else 0

    n_highs = sum(1 for k in kinds if k == "high")
    n_lows = sum(1 for k in kinds if k == "low")
    if n_highs and n_lows:
        zone_type = "both"
    elif n_highs > n_lows:
        zone_type = "resistance"
    else:
        zone_type = "support"

    # Composite strength: touches saturate around 5, recency over the
    # window length. Both-sided zones get a small bonus — a level
    # respected from both sides is structurally stronger than a level
    # only ever rejected once-direction.
    touch_score = min(1.0, len(cluster) / 5.0)
    recency = last_bar / (len(candles) - 1) if len(candles) > 1 else 0.0
    strength = 0.6 * touch_score + 0.4 * recency
    if zone_type == "both":
        strength = min(1.0, strength * 1.2)

    return SRZone(
        price=sum(prices) / len(prices),
        low=min(prices),
        high=max(prices),
        touches=len(cluster),
        last_touch_bar=last_bar,
        last_touch_unix=last_unix,
        zone_type=zone_type,
        strength=round(strength, 4),
    )


def detect_sr_zones(
    candles: list[dict],
    lookback: int = 3,
    tolerance_pct: float = 0.003,
    min_touches: int = 1,
    max_zones: int = 8,
) -> list[SRZone]:
    """Top-level: pivots -> clusters -> scored zones, strongest first.

    Args:
        candles: oldest -> newest, each with time/open/high/low/close/volume.
        lookback: swing-pivot confirmation window. 3 is sensitive (more
            zones, faster to react); 5 is conservative.
        tolerance_pct: maximum gap (as a fraction) between pivots to
            still be in the same zone. 0.003 = 30 bps on a $50k name.
        min_touches: drop zones formed by fewer than this many pivots.
            Default 1 keeps everything; bump to 2 to filter solo pivots.
        max_zones: cap on the number of zones returned. Sorted by
            strength descending, so the cap keeps the strongest.
    """
    if not candles:
        return []

    highs, lows = detect_swing_pivots(candles, lookback=lookback)
    pivots: list[tuple[int, float, str]] = []
    for i in highs:
        pivots.append((i, float(candles[i]["high"]), "high"))
    for i in lows:
        pivots.append((i, float(candles[i]["low"]), "low"))
    if not pivots:
        return []

    zones = [_cluster_to_zone(c, candles) for c in _cluster_pivots(pivots, tolerance_pct)]
    zones = [z for z in zones if z.touches >= min_touches]
    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:max_zones]


def find_nearest_zones(
    zones: list[SRZone],
    current_price: float,
) -> tuple[Optional[SRZone], Optional[SRZone]]:
    """Return (nearest_resistance, nearest_support) — the zones closest
    above and below current_price. Either is None if no zone exists on
    that side. Useful for setups: "we're approaching resistance at X."

    "Above" / "below" is based on zone.price (the cluster midpoint),
    so a wide zone straddling the current price will be classified by
    whichever side its midpoint sits on.
    """
    above = [z for z in zones if z.price > current_price]
    below = [z for z in zones if z.price < current_price]
    nearest_resistance = min(above, key=lambda z: z.price - current_price) if above else None
    nearest_support = max(below, key=lambda z: z.price) if below else None
    return nearest_resistance, nearest_support
