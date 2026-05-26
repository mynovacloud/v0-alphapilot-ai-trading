"""
Setup library — Phase C of the signal overhaul.

Each function here is one specific, testable trading hypothesis. Not
"Momentum" (a muddled blend of conflicting signals), not "Mean
Reversion" (same problem in reverse) — a *named setup* with explicit
conditions that mirror how a discretionary day trader thinks:

    "If price reclaims VWAP after sitting below it, on volume, then I
    want to be long."

Phase C ships these one at a time. Each gets:
  - implemented as a function that takes candles and returns a Signal
  - tested in isolation (deterministic unit tests on hand-built candles)
  - measured in the harness against the Phase A baseline (+10 bps gross
    alpha @ 30b on Momentum)
  - shipped to the live registry ONLY if its harness result beats the
    baseline net of fees, with t > 2

This file starts with one setup. As more are added (opening-range
breakout, pivot rejection, range-break with retest, etc.) it stays
one-file until 3+ — at which point we'll split into a `setups/`
package, one file per hypothesis.

THE FIRST SETUP: VWAP Reclaim
The market spent a meaningful stretch below the volume-weighted
average price (the "below" state), then a bar closes back above VWAP
with bullish characteristics AND above-average volume (the "reclaim").
The hypothesis is that the volume + reclaim together signal a
short-term bottoming pattern with room to run toward the upper VWAP
band or yesterday's POC.

Why this should* work*:
  - VWAP is the most-watched intraday line by every desk that uses
    technicals. Bots, market makers, and discretionary traders all
    react to VWAP touches. That makes it a self-fulfilling level.
  - The "below for N bars then back above on volume" pattern filters
    out chop — when price oscillates around VWAP every bar, no setup
    fires. Only the genuine regime-change reclaims trigger.

Why it might NOT work:
  - VWAP is most relevant during high-volume sessions. The trading-
    hours filter helps, but doesn't fully solve it.
  - On low-volatility days the reclaim might immediately fail back
    below — and we'd be flat too quickly to capture the win or
    structurally short the loss.

The harness will tell us. If it does work, we have our first real
setup. If it doesn't, we kill it and try the next hypothesis.
"""
from __future__ import annotations

import datetime

from trading.levels import compute_opening_range, compute_vwap

# `Signal` is defined in trading.strategy_engine. strategy_engine ALSO
# registers our setups at the bottom of its module, which means a top-
# of-file import here would form a cycle. We import Signal lazily
# inside each setup function — the cost is one cached attr lookup per
# call, the benefit is no cycle and no extra types module.


# --------------------------------------------------------------------------
# VWAP Reclaim — config constants kept module-level so they're tunable
# without code surgery and visible in tests / debugger.
# --------------------------------------------------------------------------

_MIN_CANDLES_NEEDED = 21      # 10 pre-context + 10 lookback + 1 reclaim bar
_LOOKBACK_BARS = 10           # window for the "was price below VWAP recently?" check
_MIN_BARS_BELOW = 6           # at least this many of the lookback bars must be below
_VOL_MULTIPLIER = 1.2         # reclaim bar volume >= this many x average


def vwap_reclaim_signal(candles: list[dict]):
    from trading.strategy_engine import Signal
    """One-bar VWAP reclaim with volume confirmation.

    Returns BUY when ALL of these hold simultaneously:
      1. We have at least 30 bars to compute a stable VWAP.
      2. The latest bar's close is above VWAP.
      3. The latest bar is bullish (close > open).
      4. At least 6 of the prior 10 bars closed below the current VWAP
         (the "below" state we're reclaiming from).
      5. The latest bar's volume is at least 1.2x the rolling average.

    Otherwise returns HOLD with a low confidence and a reasoning string
    that says WHY the setup didn't qualify — useful for the audit trail.

    The Signal's `indicators` carries the diagnostic numbers (VWAP, bars
    below, volume ratio, distance above VWAP) so a downstream caller
    can reason about them or surface them in the UI.
    """
    # --- Gate 1: enough data ------------------------------------------------
    if len(candles) < _MIN_CANDLES_NEEDED:
        return Signal(
            "HOLD", 0.0,
            f"VWAP Reclaim: need >= {_MIN_CANDLES_NEEDED} bars (have {len(candles)})",
            "VWAP Reclaim", {},
        )

    vwap, vw_upper, vw_lower = compute_vwap(candles)
    if vwap is None:
        return Signal(
            "HOLD", 0.0,
            "VWAP Reclaim: cannot compute VWAP (zero or missing volume)",
            "VWAP Reclaim", {},
        )

    last = candles[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    base_indicators = {"vwap": vwap}

    # --- Gate 2: latest bar closes above VWAP ------------------------------
    if last_close <= vwap:
        return Signal(
            "HOLD", 0.05,
            f"VWAP Reclaim: last close ${last_close:.4f} not above VWAP ${vwap:.4f}",
            "VWAP Reclaim", base_indicators,
        )

    # --- Gate 3: reclaim candle must be bullish ----------------------------
    if last_close <= last_open:
        return Signal(
            "HOLD", 0.10,
            f"VWAP Reclaim: reclaim bar is not bullish (close {last_close:.4f} "
            f"<= open {last_open:.4f})",
            "VWAP Reclaim", base_indicators,
        )

    # --- Gate 4: enough prior bars were below VWAP -------------------------
    prior = candles[-_LOOKBACK_BARS - 1:-1]   # the N bars BEFORE the reclaim bar
    bars_below = sum(1 for c in prior if float(c["close"]) < vwap)
    if bars_below < _MIN_BARS_BELOW:
        return Signal(
            "HOLD", 0.15,
            f"VWAP Reclaim: only {bars_below}/{_LOOKBACK_BARS} prior bars below "
            f"VWAP (need >= {_MIN_BARS_BELOW})",
            "VWAP Reclaim", {**base_indicators, "bars_below_vwap": float(bars_below)},
        )

    # --- Gate 5: reclaim bar volume confirms -------------------------------
    volumes = [float(c.get("volume", 0.0) or 0.0) for c in candles]
    avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
    last_vol = float(last.get("volume", 0.0) or 0.0)
    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0.0
    if vol_ratio < _VOL_MULTIPLIER:
        return Signal(
            "HOLD", 0.20,
            f"VWAP Reclaim: reclaim bar volume {vol_ratio:.2f}x avg "
            f"(need >= {_VOL_MULTIPLIER:.1f}x)",
            "VWAP Reclaim",
            {**base_indicators, "bars_below_vwap": float(bars_below),
             "volume_ratio": vol_ratio},
        )

    # --- All gates passed: build the BUY signal ----------------------------
    # Confidence: 0.65 base + small bonuses for the strength of each gate.
    # The base sits just above a typical 0.60 min-confidence floor, so a
    # qualifying-but-marginal reclaim makes it past min_conf but won't
    # outrank a high-conviction one.
    base = 0.65
    below_bonus = 0.10 * min(1.0, (bars_below - _MIN_BARS_BELOW)
                              / max(_LOOKBACK_BARS - _MIN_BARS_BELOW, 1))
    vol_bonus = 0.10 * min(1.0, (vol_ratio - _VOL_MULTIPLIER) / 1.0)
    dist_pct = (last_close - vwap) / vwap
    dist_bonus = 0.10 * min(1.0, dist_pct / 0.003)   # full bonus by +30 bps above
    confidence = min(0.95, base + below_bonus + vol_bonus + dist_bonus)

    indicators = {
        "vwap": vwap,
        "vwap_upper_band": vw_upper if vw_upper is not None else 0.0,
        "vwap_lower_band": vw_lower if vw_lower is not None else 0.0,
        "bars_below_vwap": float(bars_below),
        "volume_ratio": vol_ratio,
        "distance_above_vwap_pct": dist_pct * 100.0,
    }
    reasoning = (
        f"VWAP reclaim @ ${vwap:.4f}: {bars_below}/{_LOOKBACK_BARS} prior bars below, "
        f"reclaim bullish on {vol_ratio:.1f}x avg volume, "
        f"close {dist_pct * 100:+.2f}% above VWAP."
    )
    return Signal("BUY", confidence, reasoning, "VWAP Reclaim", indicators)


# ==========================================================================
# Opening-Range Breakout (ORB)
# ==========================================================================
# The classic intraday day-trader setup. After the session opens, the
# first 30 minutes of price action defines an "opening range" — the
# initial battle between buyers and sellers. When price subsequently
# closes ABOVE that range high on confirming volume, that's the signal:
# the buyers won, breakout established, ride it for the next move.
#
# Why this might work in addition to VWAP Reclaim:
#   - Completely independent firing logic. VWAP reclaim wants price
#     below VWAP first. ORB wants price inside the opening range. The
#     two can't both miss for the same reason — so they add real
#     samples to the combined edge measurement.
#   - The opening-range concept is one of the most-back-tested intraday
#     setups in stock-trading literature. There's external evidence it
#     works at least in some regimes.
#   - Crypto's continuous market still has session structure: the
#     London-NY overlap (12:00-22:00 UTC) carries most volume and the
#     NY-equities open (13:30 UTC) is a real volume event.
#
# Crypto session anchor: 13:30 UTC = NY equities open. Best alignment
# with the bot's existing trading-hours window (12-22 UTC) and the
# moment when overlapping liquidity from US institutional desks arrives.
# Could alternatively use 00:00 UTC for a "daily" anchor — defer the
# tradeoff to a future experiment if needed.
#
# Transition-only firing: we only fire on the bar where the breakout
# JUST HAPPENED — previous bar closed <= OR_high, current bar closes
# strictly above. Without this, the harness would record the same
# breakout 60+ times (once per subsequent tick), inflating the sample
# with non-independent observations. In production, the bot's
# held-symbol skip handles the equivalent (once we're in, we don't
# re-enter the same trade).

_ORB_ANCHOR_HOUR_UTC = 13          # NY equities open
_ORB_ANCHOR_MINUTE_UTC = 30
_ORB_RANGE_MINUTES = 30
_ORB_MIN_CANDLES = 35              # OR window (30) + a few post-OR bars
_ORB_MAX_BARS_AFTER_OR = 240       # don't trade ORB more than 4h past OR close
_ORB_VOL_MULTIPLIER = 1.3          # breakout bar volume vs avg


def _most_recent_session_anchor(
    now_utc: datetime.datetime,
    hour: int,
    minute: int,
) -> datetime.datetime:
    """Return the most recent UTC datetime at hour:minute that's <= now_utc.

    If today's anchor hasn't happened yet (it's still morning UTC), roll
    back to yesterday's. Used by ORB to find "today's opening range"
    from whatever the harness slid the candle window to.
    """
    anchor = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if anchor > now_utc:
        anchor -= datetime.timedelta(days=1)
    return anchor


def opening_range_breakout_signal(candles: list[dict]):
    """First-close-above the opening-range high, on confirming volume.

    Gates (each must pass; HOLDs cite which one failed):
      1. >= 35 candles (OR window + post-OR room).
      2. Opening range successfully established (>= 1 bar inside the
         [anchor, anchor+30min) window).
      3. The latest bar is AFTER the OR window closes.
      4. The OR window closed no more than 4 hours ago (signal decays).
      5. Transition: previous close <= OR_high AND current close > OR_high.
      6. Breakout bar volume >= 1.3x average.

    Returns BUY at gate 6 pass; HOLD otherwise with a specific reason.
    """
    from trading.strategy_engine import Signal

    # --- Gate 1: enough data ----------------------------------------------
    if len(candles) < _ORB_MIN_CANDLES:
        return Signal(
            "HOLD", 0.0,
            f"ORB: need >= {_ORB_MIN_CANDLES} bars (have {len(candles)})",
            "ORB", {},
        )

    last_unix = int(candles[-1]["time"])
    now_utc = datetime.datetime.fromtimestamp(last_unix, datetime.timezone.utc)
    anchor = _most_recent_session_anchor(now_utc, _ORB_ANCHOR_HOUR_UTC, _ORB_ANCHOR_MINUTE_UTC)
    anchor_ts = int(anchor.timestamp())
    or_end_ts = anchor_ts + _ORB_RANGE_MINUTES * 60

    # --- Gate 2: opening range established? --------------------------------
    or_high, or_low = compute_opening_range(candles, anchor, minutes=_ORB_RANGE_MINUTES)
    if or_high is None or or_low is None:
        return Signal(
            "HOLD", 0.0,
            "ORB: opening range not in this candle window",
            "ORB", {},
        )

    base_ind = {"or_high": or_high, "or_low": or_low}

    # --- Gate 3: are we past the OR window? -------------------------------
    if last_unix < or_end_ts:
        return Signal(
            "HOLD", 0.05,
            "ORB: still inside the opening-range window",
            "ORB", base_ind,
        )

    # --- Gate 4: OR signal decays past 4h ---------------------------------
    minutes_since_or_close = (last_unix - or_end_ts) // 60
    if minutes_since_or_close > _ORB_MAX_BARS_AFTER_OR:
        return Signal(
            "HOLD", 0.05,
            f"ORB: {minutes_since_or_close}m past OR close — signal decayed",
            "ORB", base_ind,
        )

    # --- Gate 5: transition breakout (prev <= high < now) ------------------
    if len(candles) < 2:
        return Signal("HOLD", 0.0, "ORB: need >= 2 bars", "ORB", base_ind)

    prev_close = float(candles[-2]["close"])
    last_close = float(candles[-1]["close"])

    if not (prev_close <= or_high and last_close > or_high):
        return Signal(
            "HOLD", 0.10,
            f"ORB: no breakout transition (prev ${prev_close:.4f} -> last "
            f"${last_close:.4f} vs OR_high ${or_high:.4f})",
            "ORB", base_ind,
        )

    # --- Gate 6: volume confirms ------------------------------------------
    volumes = [float(c.get("volume", 0.0) or 0.0) for c in candles]
    avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
    last_vol = float(candles[-1].get("volume", 0.0) or 0.0)
    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0.0
    if vol_ratio < _ORB_VOL_MULTIPLIER:
        return Signal(
            "HOLD", 0.20,
            f"ORB: breakout volume {vol_ratio:.2f}x avg (need >= "
            f"{_ORB_VOL_MULTIPLIER:.1f}x)",
            "ORB",
            {**base_ind, "volume_ratio": vol_ratio},
        )

    # --- All gates passed --------------------------------------------------
    base = 0.65
    vol_bonus = 0.10 * min(1.0, (vol_ratio - _ORB_VOL_MULTIPLIER) / 1.0)
    # Fresher break = higher conviction. Full bonus right after OR closes,
    # decays to zero at the 4h cutoff.
    fresh_bonus = 0.10 * max(0.0, 1.0 - minutes_since_or_close / _ORB_MAX_BARS_AFTER_OR)
    # Wider OR = bigger meaningful range, more conviction on the break.
    or_width_pct = (or_high - or_low) / or_low if or_low > 0 else 0.0
    width_bonus = 0.10 * min(1.0, or_width_pct / 0.01)   # full bonus at >= 1% wide
    confidence = min(0.95, base + vol_bonus + fresh_bonus + width_bonus)

    indicators = {
        "or_high": or_high,
        "or_low": or_low,
        "or_width_pct": or_width_pct * 100.0,
        "minutes_since_or_close": float(minutes_since_or_close),
        "volume_ratio": vol_ratio,
    }
    reasoning = (
        f"ORB: closed ${last_close:.4f} above OR_high ${or_high:.4f} "
        f"(OR was ${or_low:.4f}-${or_high:.4f}, width {or_width_pct * 100:.2f}%); "
        f"breakout on {vol_ratio:.1f}x avg volume, "
        f"{minutes_since_or_close}m past OR close."
    )
    return Signal("BUY", confidence, reasoning, "ORB", indicators)
