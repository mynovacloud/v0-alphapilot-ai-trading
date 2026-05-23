"""
Session-anchored trading levels — Phase B of the signal overhaul.

Every successful day trader reads the same handful of lines on a chart:
- VWAP  (where the average market participant is positioned today)
- Prior-day high / low / close (recent reaction zones)
- Classic pivot points (mechanical S/R)
- Opening range (the first push of the session)

This module computes all of them. Pure: given candle data and a
session anchor, you get the levels. No DB, no network, no clocks
inside the computation. Higher layers (signal setups, the training
UI) consume `SessionLevels` and decide what to do with them.

PHASE B SCOPE — Batch 1.
This file is for primitives that need only OHLC/volume plus a session
anchor. Things that need pivot detection or histogram analysis
(structural S/R zones, volume profile, market structure) get their
own module in a follow-up commit. Keeps each module testable and the
review surface small.

Pivot formula (classic floor-trader):
    P  = (H + L + C) / 3
    R1 = 2P - L
    S1 = 2P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SessionLevels:
    """Snapshot of the lines a trader would draw on a chart right now.

    Any field can be None — fewer candles in, fewer values out. Callers
    must handle missing values gracefully (e.g. no daily history yet =>
    no pivot point => fall back to the next-best reference).
    """
    symbol: str
    session_start_utc: datetime.datetime
    bars_in_session: int

    # VWAP — cumulative (typical_price × volume) / cumulative volume from anchor.
    # The single most-watched intraday line.
    vwap: Optional[float] = None
    vwap_upper_band: Optional[float] = None    # +1 volume-weighted sigma
    vwap_lower_band: Optional[float] = None    # -1 volume-weighted sigma

    # Prior-day reactions (computed from daily candles).
    prior_day_high: Optional[float] = None
    prior_day_low: Optional[float] = None
    prior_day_close: Optional[float] = None

    # Classic floor-trader pivot points from prior-day HLC.
    pivot: Optional[float] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    s1: Optional[float] = None
    s2: Optional[float] = None

    # Opening range — high/low of the first N minutes after the session anchor.
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None


# --------------------------------------------------------------------------
# Tiny helpers — kept private; only the public computes are exported.
# --------------------------------------------------------------------------

def _typical_price(candle: dict) -> float:
    """Standard HLC/3, the price each bar 'represents' for VWAP purposes."""
    return (float(candle["high"]) + float(candle["low"]) + float(candle["close"])) / 3.0


def _default_anchor_utc(now_utc: datetime.datetime | None = None) -> datetime.datetime:
    """Default session anchor: 00:00 UTC of the current day."""
    now = now_utc or datetime.datetime.now(datetime.timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------
# Public primitives — each pure, each separately testable.
# --------------------------------------------------------------------------

def compute_vwap(candles: list[dict]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Volume-weighted average price + ±1 sigma envelope.

    Returns (vwap, upper_band, lower_band). All three are None if total
    volume is zero or no candles were supplied. The sigma is the
    volume-weighted standard deviation of typical price around VWAP —
    a natural "wide vs. tight" indicator.
    """
    if not candles:
        return None, None, None
    pv_sum = 0.0
    v_sum = 0.0
    for c in candles:
        v = float(c.get("volume", 0.0) or 0.0)
        if v <= 0:
            continue
        pv_sum += _typical_price(c) * v
        v_sum += v
    if v_sum <= 0:
        return None, None, None
    vwap = pv_sum / v_sum
    sq_diff_sum = 0.0
    for c in candles:
        v = float(c.get("volume", 0.0) or 0.0)
        if v <= 0:
            continue
        sq_diff_sum += v * (_typical_price(c) - vwap) ** 2
    sd = math.sqrt(sq_diff_sum / v_sum)
    return vwap, vwap + sd, vwap - sd


def compute_prior_day(daily_candles: list[dict]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """High/low/close of the most recently COMPLETED day.

    `daily_candles` is oldest->newest. We use the second-to-last bar
    because the last one is "today, still in progress". Returns
    (high, low, close) or (None, None, None) if only one bar exists.
    """
    if len(daily_candles) < 2:
        return None, None, None
    prev = daily_candles[-2]
    return float(prev["high"]), float(prev["low"]), float(prev["close"])


def compute_pivots(
    prior_high: float, prior_low: float, prior_close: float,
) -> tuple[float, float, float, float, float]:
    """Classic floor-trader pivot points. See module docstring for the formula.

    Returns (P, R1, R2, S1, S2). Always returns concrete numbers — caller
    is responsible for not passing it None inputs.
    """
    p = (prior_high + prior_low + prior_close) / 3.0
    r1 = 2.0 * p - prior_low
    s1 = 2.0 * p - prior_high
    r2 = p + (prior_high - prior_low)
    s2 = p - (prior_high - prior_low)
    return p, r1, r2, s1, s2


def compute_opening_range(
    candles: list[dict],
    session_start_utc: datetime.datetime,
    minutes: int = 30,
) -> tuple[Optional[float], Optional[float]]:
    """High/low of bars whose timestamp falls in [anchor, anchor+minutes).

    The window is half-open — a bar exactly at anchor+minutes belongs
    to the post-opening session, not the opening range.
    """
    if not candles:
        return None, None
    anchor_ts = int(session_start_utc.timestamp())
    end_ts = anchor_ts + minutes * 60
    in_range = [c for c in candles if anchor_ts <= int(c["time"]) < end_ts]
    if not in_range:
        return None, None
    return (
        max(float(c["high"]) for c in in_range),
        min(float(c["low"]) for c in in_range),
    )


# --------------------------------------------------------------------------
# Top-level convenience — combine everything into one SessionLevels.
# --------------------------------------------------------------------------

def compute_session_levels(
    symbol: str,
    intraday_candles: list[dict],
    daily_candles: list[dict] | None = None,
    session_start_utc: datetime.datetime | None = None,
    opening_range_minutes: int = 30,
) -> SessionLevels:
    """Combine every primitive into one snapshot. Pure: no I/O.

    `intraday_candles` should be 60s or 300s bars covering the current
    session (and ideally a few hours before, for context). `daily_candles`
    should include at least 2 daily bars so we can read yesterday's
    HLC. Either can be empty — the result will just have more Nones.
    """
    anchor = session_start_utc or _default_anchor_utc()
    anchor_ts = int(anchor.timestamp())

    session_bars = [c for c in intraday_candles if int(c["time"]) >= anchor_ts]
    vwap, vw_upper, vw_lower = compute_vwap(session_bars)
    or_h, or_l = compute_opening_range(intraday_candles, anchor, opening_range_minutes)

    pd_h = pd_l = pd_c = None
    p = r1 = r2 = s1 = s2 = None
    if daily_candles:
        pd_h, pd_l, pd_c = compute_prior_day(daily_candles)
        if pd_h is not None and pd_l is not None and pd_c is not None:
            p, r1, r2, s1, s2 = compute_pivots(pd_h, pd_l, pd_c)

    return SessionLevels(
        symbol=symbol,
        session_start_utc=anchor,
        bars_in_session=len(session_bars),
        vwap=vwap,
        vwap_upper_band=vw_upper,
        vwap_lower_band=vw_lower,
        prior_day_high=pd_h,
        prior_day_low=pd_l,
        prior_day_close=pd_c,
        pivot=p, r1=r1, r2=r2, s1=s1, s2=s2,
        opening_range_high=or_h,
        opening_range_low=or_l,
    )


def get_session_levels(
    symbol: str,
    session_start_utc: datetime.datetime | None = None,
    opening_range_minutes: int = 30,
) -> SessionLevels:
    """Convenience wrapper: fetch the candles, compute the levels.

    Intraday: 60s bars (300 max — gives 5 hours of context). Daily:
    daily bars (10 max — plenty for prior-day + pivots). Both go
    through the shared candle cache, so repeated calls within a few
    minutes are free.
    """
    from connectors.candles import get_candles

    intraday = get_candles(symbol, granularity=60, limit=300)
    daily = get_candles(symbol, granularity=86400, limit=10)
    return compute_session_levels(
        symbol, intraday, daily,
        session_start_utc=session_start_utc,
        opening_range_minutes=opening_range_minutes,
    )
