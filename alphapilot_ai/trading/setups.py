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

from trading.levels import compute_vwap

# `Signal` is defined in trading.strategy_engine. strategy_engine ALSO
# registers our setups at the bottom of its module, which means a top-
# of-file import here would form a cycle. We import Signal lazily
# inside the setup function — the cost is one cached attr lookup per
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
