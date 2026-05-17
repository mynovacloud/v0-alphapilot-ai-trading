"""
Live strategy engine.

Implements REAL signals computed from real Coinbase candle data:

  - Momentum:        EMA(fast) vs EMA(slow) cross + recent return slope
  - Mean Reversion:  Z-score of close vs rolling SMA, with volatility floor

Both strategies emit a `Signal(side, confidence, reasoning, metadata)`.

The bot engine's old `_build_snapshot` produced static synthetic features.
This module replaces that with computed indicators per symbol per tick. The
existing `DecisionEngine` is still available for snapshot-style features
(used by the AI Training Lab); the bot loop now uses these signals directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from connectors.candles import get_candles
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------- #
# Indicator helpers (pure functions, no numpy dependency for hot path)
# ---------------------------------------------------------------------------- #


def _closes(candles: Iterable[dict[str, Any]]) -> list[float]:
    return [float(c["close"]) for c in candles if c.get("close") is not None]


def ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average. Returns a list aligned with `values`."""
    if not values or period <= 1:
        return list(values)
    k = 2.0 / (period + 1.0)
    out: list[float] = []
    prev = values[0]
    for v in values:
        prev = v * k + prev * (1.0 - k)
        out.append(prev)
    return out


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def stdev(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 1:
        return None
    window = values[-period:]
    m = sum(window) / period
    var = sum((x - m) ** 2 for x in window) / (period - 1)
    return math.sqrt(var) if var > 0 else 0.0


def pct_return(values: list[float], lookback: int) -> float | None:
    if len(values) <= lookback or values[-lookback - 1] == 0:
        return None
    return values[-1] / values[-lookback - 1] - 1.0


def atr(candles: list[dict[str, Any]], period: int = 14) -> float | None:
    """Average True Range — used to size stops in volatility units."""
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_close = float(candles[i - 1]["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


# ---------------------------------------------------------------------------- #
# Signal type
# ---------------------------------------------------------------------------- #


@dataclass
class Signal:
    side: str                # "BUY" / "SELL" / "HOLD"
    confidence: float        # 0..1
    reasoning: str
    strategy: str            # "Momentum" / "Mean Reversion" / etc.
    indicators: dict[str, float] = field(default_factory=dict)

    def is_actionable(self, min_confidence: float = 0.0) -> bool:
        return self.side in {"BUY", "SELL"} and self.confidence >= min_confidence


# ---------------------------------------------------------------------------- #
# Individual strategies
# ---------------------------------------------------------------------------- #


def momentum_signal(
    candles: list[dict[str, Any]],
    *,
    fast: int = 12,
    slow: int = 26,
    lookback: int = 6,
) -> Signal:
    """
    EMA-cross momentum:
      BUY  when EMA_fast > EMA_slow AND short-term return is positive
      SELL when EMA_fast < EMA_slow AND short-term return is negative

    Confidence scales with the magnitude of the EMA gap (normalized by price)
    and the magnitude of the recent return.
    """
    closes = _closes(candles)
    if len(closes) < max(slow, lookback + 2):
        return Signal("HOLD", 0.0, "insufficient data for momentum", "Momentum")

    ema_fast = ema(closes, fast)[-1]
    ema_slow = ema(closes, slow)[-1]
    ret = pct_return(closes, lookback) or 0.0
    last = closes[-1] or 1e-9
    gap = (ema_fast - ema_slow) / last  # normalized

    indicators = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "gap_pct": gap,
        "return_lb": ret,
    }

    # Decide side. Loosened: previously required BOTH ema_cross AND positive
    # return slope, which kept confidence at 0 on most ticks. Now we let either
    # condition emit a directional signal at moderate confidence; Claude (or
    # the user's confidence floor) is the final arbiter.
    if ema_fast > ema_slow and ret >= 0:
        side = "BUY"
    elif ema_fast < ema_slow and ret <= 0:
        side = "SELL"
    elif ema_fast > ema_slow:        # cross says up, return says down -> weak BUY
        side = "BUY"
    elif ema_fast < ema_slow:        # cross says down, return says up -> weak SELL
        side = "SELL"
    else:
        return Signal("HOLD", 0.15, "EMAs perfectly equal", "Momentum", indicators)

    # Confidence: blend of normalized EMA gap and lookback return,
    # squashed into 0..1 with a soft cap.
    raw = min(1.0, abs(gap) * 25.0 + abs(ret) * 8.0)
    confidence = 0.5 + 0.5 * raw  # always >= 0.5 once we've decided to act
    confidence = max(0.0, min(0.99, confidence))

    reasoning = (
        f"EMA{fast}={ema_fast:.4f} vs EMA{slow}={ema_slow:.4f} "
        f"({gap:+.2%}); {lookback}-bar return {ret:+.2%}"
    )
    return Signal(side, confidence, reasoning, "Momentum", indicators)


def mean_reversion_signal(
    candles: list[dict[str, Any]],
    *,
    period: int = 20,
    z_entry: float = 1.5,
    min_vol_pct: float = 0.001,
) -> Signal:
    """
    Z-score mean reversion:
      BUY  when close is z_entry standard deviations BELOW the SMA
      SELL when close is z_entry standard deviations ABOVE the SMA

    Skipped when realized volatility is below `min_vol_pct` (price is flat,
    Z-scores are noise).

    Confidence scales with |z| above the entry threshold.
    """
    closes = _closes(candles)
    if len(closes) < period + 2:
        return Signal("HOLD", 0.0, "insufficient data for mean reversion", "Mean Reversion")

    mean = sma(closes, period)
    sd = stdev(closes, period)
    if mean is None or sd is None or mean <= 0:
        return Signal("HOLD", 0.0, "stats unavailable", "Mean Reversion")

    last = closes[-1]
    vol_pct = sd / mean
    if vol_pct < min_vol_pct:
        return Signal(
            "HOLD",
            0.0,
            f"vol {vol_pct:.3%} below floor {min_vol_pct:.3%}",
            "Mean Reversion",
            {"sma": mean, "stdev": sd, "vol_pct": vol_pct},
        )

    z = (last - mean) / sd if sd > 0 else 0.0
    indicators = {"sma": mean, "stdev": sd, "z": z, "vol_pct": vol_pct}

    if z <= -z_entry:
        side = "BUY"
    elif z >= z_entry:
        side = "SELL"
    else:
        return Signal("HOLD", 0.0, f"|z|={abs(z):.2f} below entry {z_entry}", "Mean Reversion", indicators)

    # Confidence: how far past the threshold are we?
    excess = abs(z) - z_entry  # >= 0
    raw = min(1.0, excess / 1.5)  # full confidence at z = entry + 1.5
    confidence = max(0.0, min(0.99, 0.55 + 0.4 * raw))

    reasoning = f"Z={z:+.2f} vs SMA{period}; vol={vol_pct:.2%}"
    return Signal(side, confidence, reasoning, "Mean Reversion", indicators)


# ---------------------------------------------------------------------------- #
# Public API used by the bot loop
# ---------------------------------------------------------------------------- #


def evaluate_symbol(
    product_id: str,
    strategy_type: str,
    *,
    granularity: int | None = None,
    lookback_bars: int = 200,
    tick_seconds: int | None = None,
) -> Signal:
    """
    Pull candles for `product_id` and compute the signal for `strategy_type`.

    `granularity` defaults to a value that matches the bot's tick cadence:
      - tick <=  10s  ->  60s bars   (high-frequency training)
      - tick <=  60s  -> 300s bars   (default)
      - tick >  60s   -> 900s bars
    Passing the candle granularity in lock-step with the tick is what makes
    a real-time training session actually produce fresh signals every tick
    instead of replaying the same 5-minute bar 150 times in a row.

    Falls back to HOLD on empty candles or unsupported strategy.
    """
    if granularity is None:
        if tick_seconds is None or tick_seconds <= 0:
            granularity = 300
        elif tick_seconds <= 10:
            granularity = 60
        elif tick_seconds <= 60:
            granularity = 300
        else:
            granularity = 900

    candles = get_candles(product_id, granularity=granularity, limit=lookback_bars)
    if not candles or len(candles) < 30:
        # Always include the price + bar count so Claude can still reason
        # about a thinly-traded symbol instead of seeing "no indicators".
        last_price = float(candles[-1]["close"]) if candles else 0.0
        return Signal(
            "HOLD",
            0.0,
            f"only {len(candles)} bars at {granularity}s — insufficient history",
            strategy_type or "Momentum",
            {"bars": len(candles), "granularity_s": granularity, "last_price": last_price},
        )

    st = (strategy_type or "Momentum").strip()
    if st == "Momentum":
        return momentum_signal(candles)
    if st == "Mean Reversion":
        return mean_reversion_signal(candles)
    if st == "Volatility Breakout":
        # Reuse mean reversion math but invert: trade WITH big moves.
        sig = mean_reversion_signal(candles, z_entry=2.0)
        if sig.side == "HOLD":
            return Signal("HOLD", 0.0, sig.reasoning, "Volatility Breakout", sig.indicators)
        # Flip side: a high Z is a BUY (chase the breakout), low Z is SELL.
        flipped = "BUY" if sig.side == "SELL" else "SELL"
        return Signal(
            flipped,
            sig.confidence,
            f"Breakout: {sig.reasoning}",
            "Volatility Breakout",
            sig.indicators,
        )
    if st == "Probability Edge":
        # No external probability feed yet — fall back to momentum-style alignment
        # but with a stricter confidence floor so it acts less often.
        sig = momentum_signal(candles)
        if sig.confidence < 0.7:
            return Signal("HOLD", 0.0, "no probability edge", "Probability Edge", sig.indicators)
        return Signal(sig.side, sig.confidence, sig.reasoning, "Probability Edge", sig.indicators)

    # Unknown strategy: default to momentum.
    return momentum_signal(candles)


def stop_take_levels(
    candles: list[dict[str, Any]],
    side: str,
    *,
    atr_period: int = 14,
    stop_atr_mult: float = 1.5,
    take_atr_mult: float = 3.0,
) -> dict[str, float]:
    """
    Compute volatility-aware stop-loss and take-profit prices using ATR.
    Returns {} if not enough data.
    """
    if not candles:
        return {}
    a = atr(candles, period=atr_period)
    if a is None or a <= 0:
        return {}
    last = float(candles[-1]["close"])
    if side == "BUY":
        return {
            "stop_loss": last - stop_atr_mult * a,
            "take_profit": last + take_atr_mult * a,
            "atr": a,
        }
    if side == "SELL":
        return {
            "stop_loss": last + stop_atr_mult * a,
            "take_profit": last - take_atr_mult * a,
            "atr": a,
        }
    return {}
