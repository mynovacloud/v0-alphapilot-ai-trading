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


def sma_list(values: list[float], period: int) -> list[float]:
    """SMA that returns a full list aligned with input."""
    if len(values) < period or period <= 0:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Calculate RSI (Relative Strength Index)."""
    if len(closes) < period + 1:
        return None
    
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_indicator(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD line, signal line, and histogram."""
    if len(closes) < slow + signal:
        return {"macd": 0, "signal": 0, "histogram": 0}
    
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = macd_line[-1] - signal_line[-1] if macd_line and signal_line else 0
    
    return {
        "macd": macd_line[-1] if macd_line else 0,
        "signal": signal_line[-1] if signal_line else 0,
        "histogram": histogram
    }


def bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0) -> dict:
    """Calculate Bollinger Bands."""
    if len(closes) < period:
        return {"upper": 0, "middle": 0, "lower": 0, "percent_b": 0.5}
    
    middle = sum(closes[-period:]) / period
    variance = sum((x - middle) ** 2 for x in closes[-period:]) / period
    std = math.sqrt(variance) if variance > 0 else 0
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    current = closes[-1]
    band_width = upper - lower
    percent_b = (current - lower) / band_width if band_width > 0 else 0.5
    
    return {"upper": upper, "middle": middle, "lower": lower, "percent_b": percent_b}


def adx_indicator(candles: list[dict], period: int = 14) -> dict:
    """Calculate ADX (Average Directional Index) for trend strength."""
    if len(candles) < period + 1:
        return {"adx": 0, "plus_di": 0, "minus_di": 0}
    
    plus_dm = []
    minus_dm = []
    tr_list = []
    
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_high = float(candles[i - 1]["high"])
        prev_low = float(candles[i - 1]["low"])
        prev_close = float(candles[i - 1]["close"])
        
        up_move = high - prev_high
        down_move = prev_low - low
        
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return {"adx": 0, "plus_di": 0, "minus_di": 0}
    
    # Smoothed averages
    smoothed_tr = sum(tr_list[-period:])
    smoothed_plus_dm = sum(plus_dm[-period:])
    smoothed_minus_dm = sum(minus_dm[-period:])
    
    if smoothed_tr == 0:
        return {"adx": 0, "plus_di": 0, "minus_di": 0}
    
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
    
    return {"adx": dx, "plus_di": plus_di, "minus_di": minus_di}


def volume_analysis(candles: list[dict], period: int = 20) -> dict:
    """Analyze volume patterns."""
    if len(candles) < period:
        return {"relative_volume": 1.0, "volume_trend": "NEUTRAL", "buying_pressure": 0.5}
    
    volumes = [float(c.get("volume", 0)) for c in candles]
    avg_volume = sum(volumes[-period:]) / period if period > 0 else 1
    current_volume = volumes[-1] if volumes else 0
    
    relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Volume trend
    recent_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_volume
    if recent_avg > avg_volume * 1.3:
        volume_trend = "INCREASING"
    elif recent_avg < avg_volume * 0.7:
        volume_trend = "DECREASING"
    else:
        volume_trend = "NEUTRAL"
    
    # Buying pressure (simplified)
    up_volume = 0
    down_volume = 0
    closes = [float(c["close"]) for c in candles[-10:]]
    vols = volumes[-10:]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            up_volume += vols[i]
        else:
            down_volume += vols[i]
    
    total = up_volume + down_volume
    buying_pressure = up_volume / total if total > 0 else 0.5
    
    return {
        "relative_volume": relative_volume,
        "volume_trend": volume_trend,
        "buying_pressure": buying_pressure
    }


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


# ---------------------------------------------------------------------------- #
# Velocity / freshness helpers
# ---------------------------------------------------------------------------- #
# These guards exist because the bot was firing low-confidence BUYs on stale
# signals (e.g. an EMA crossover that happened 50 bars ago, or a price sitting
# at a Bollinger lower band without bouncing). The result was many trades that
# went nowhere then drifted slightly negative before exiting on stop. We now
# require the price to actually be moving in our direction RIGHT NOW before
# committing capital.


def _short_velocity(closes: list[float], bars: int = 3) -> float:
    """Percent change over the last `bars` closes — our 'is it moving now' check.

    Returns 0.0 if we don't have enough data. Positive = price is rising in
    the last few bars, negative = falling. We use this as a freshness gate
    on every entry: a BUY signal that doesn't have positive short-velocity
    is buying into stagnation or a falling knife.
    """
    if not closes or len(closes) < bars + 1:
        return 0.0
    base = closes[-bars - 1]
    if not base:
        return 0.0
    return (closes[-1] - base) / base


def _ema_cross_freshness(closes: list[float], fast_n: int, slow_n: int) -> int:
    """How many bars ago did EMA_fast cross EMA_slow?

    Returns the number of bars since the last sign-flip of (fast - slow).
    Higher numbers mean the cross is stale — and a stale cross is exactly
    the kind of signal that produces "buy then sit at 0.00". We cap the
    search at 30 bars; anything older is essentially "no fresh cross".
    """
    if len(closes) < slow_n + 5:
        return 99
    fast = ema(closes, fast_n)
    slow = ema(closes, slow_n)
    if not fast or not slow:
        return 99
    n = min(len(fast), len(slow))
    if n < 3:
        return 99
    cur_sign = 1 if (fast[-1] - slow[-1]) >= 0 else -1
    for i in range(2, min(31, n)):
        prev_sign = 1 if (fast[-i] - slow[-i]) >= 0 else -1
        if prev_sign != cur_sign:
            return i - 1
    return 30


def _bar_body_direction(candles: list[dict[str, Any]]) -> float:
    """Sum of bullish-body fraction over the last 3 bars.

    Returns a value in [-3, 3]. Positive = recent bars are closing near their
    highs (buyers in control), negative = closing near their lows. Used to
    confirm that the most recent price action agrees with the entry side.
    """
    if not candles or len(candles) < 3:
        return 0.0
    total = 0.0
    for c in candles[-3:]:
        try:
            o = float(c.get("open", 0))
            h = float(c.get("high", 0))
            l = float(c.get("low", 0))
            cl = float(c.get("close", 0))
            rng = h - l
            if rng <= 0:
                continue
            # +1 if close > open (bullish body), scaled by where in the range
            # the close lands. Closes near the high count more than mid-range.
            pos = (cl - l) / rng  # 0..1
            total += (pos - 0.5) * 2.0  # -1..1
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return total


def momentum_signal(
    candles: list[dict[str, Any]],
    *,
    fast: int = 12,
    slow: int = 26,
    lookback: int = 6,
) -> Signal:
    """
    Enhanced EMA-cross momentum with RSI, MACD, volume AND velocity confirmation.

    A trade only fires when:
      * EMA / MACD / RSI / volume factors agree on a direction (>=3 of 5)
      * The price is ACTUALLY moving in that direction over the last few bars
        (short-velocity check — eliminates "buy then drift sideways" trades)
      * The recent bars confirm with body direction (closes near highs for BUY,
        closes near lows for SELL)
      * The EMA cross is reasonably fresh (<= 12 bars old) — we don't chase
        ancient crosses
    """
    closes = _closes(candles)
    if len(closes) < max(slow, lookback + 2, 26):
        return Signal("HOLD", 0.0, "insufficient data for momentum", "Momentum")

    # Core EMAs
    ema_fast = ema(closes, fast)[-1]
    ema_slow = ema(closes, slow)[-1]
    ret = pct_return(closes, lookback) or 0.0
    last = closes[-1] or 1e-9
    gap = (ema_fast - ema_slow) / last  # normalized

    # Additional indicators
    rsi_val = rsi(closes, 14) or 50.0
    macd_data = macd_indicator(closes)
    vol_data = volume_analysis(candles)

    # Live-velocity guards — these are what separate "the EMA crossed
    # weeks ago" signals from "price is moving up RIGHT NOW" signals.
    velocity_3 = _short_velocity(closes, 3)
    velocity_1 = _short_velocity(closes, 1)
    cross_age = _ema_cross_freshness(closes, fast, slow)
    body_dir = _bar_body_direction(candles)

    indicators = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "gap_pct": gap,
        "return_lb": ret,
        "rsi": rsi_val,
        "macd_histogram": macd_data["histogram"],
        "relative_volume": vol_data["relative_volume"],
        "buying_pressure": vol_data["buying_pressure"],
        "velocity_3bar": velocity_3,
        "velocity_1bar": velocity_1,
        "cross_age_bars": cross_age,
        "body_direction": body_dir,
    }

    # Multi-factor signal generation
    bullish_factors = 0
    bearish_factors = 0

    # EMA cross — only counts if it's fresh (<= 12 bars old). A 50-bar-old
    # cross is not actionable; the move it predicted has already happened.
    if ema_fast > ema_slow and cross_age <= 12:
        bullish_factors += 1
    elif ema_fast < ema_slow and cross_age <= 12:
        bearish_factors += 1

    # Lookback return direction
    if ret > 0.002:
        bullish_factors += 1
    elif ret < -0.002:
        bearish_factors += 1

    # RSI — but ONLY count oversold-bounce when velocity confirms a bounce,
    # not when price is still falling. This was a major source of bad
    # trades: buying RSI<30 mid-collapse.
    if rsi_val < 30 and velocity_3 > 0:
        bullish_factors += 1
    elif rsi_val > 70 and velocity_3 < 0:
        bearish_factors += 1
    elif rsi_val < 45:
        bullish_factors += 0.5
    elif rsi_val > 55:
        bearish_factors += 0.5

    # MACD histogram
    if macd_data["histogram"] > 0:
        bullish_factors += 1
    else:
        bearish_factors += 1

    # Volume confirmation
    if vol_data["relative_volume"] > 1.2:
        if vol_data["buying_pressure"] > 0.6:
            bullish_factors += 1
        elif vol_data["buying_pressure"] < 0.4:
            bearish_factors += 1

    # ----- Decision logic -----
    # Tightened: require >=3 factors AND a clear edge (>= 1.5 over the
    # opposite side). The previous 2-factor / 0.5-edge thresholds were the
    # main reason the bot kept buying coins that then went sideways.
    if bullish_factors >= 3 and bullish_factors >= bearish_factors + 1.5:
        side = "BUY"
        strength = bullish_factors
    elif bearish_factors >= 3 and bearish_factors >= bullish_factors + 1.5:
        side = "SELL"
        strength = bearish_factors
    else:
        return Signal(
            "HOLD",
            0.20,
            (
                f"No edge: bull={bullish_factors:.1f}, bear={bearish_factors:.1f}, "
                f"v3={velocity_3:+.2%}, cross_age={cross_age}b"
            ),
            "Momentum",
            indicators,
        )

    # ----- Velocity confirmation gate -----
    # The trade direction must agree with what price is ACTUALLY doing in
    # the last 1-3 bars. If we want to BUY but price is flat or falling
    # right now, this is a stale signal — bail out. This is the single
    # most important fix for "buy and watch nothing happen".
    MIN_VEL_3 = 0.0008  # 0.08% over 3 bars — small but non-zero
    if side == "BUY":
        if velocity_3 < MIN_VEL_3:
            return Signal(
                "HOLD",
                0.20,
                f"BUY signal rejected — price not moving up (v3={velocity_3:+.2%})",
                "Momentum",
                indicators,
            )
        if body_dir < -0.5:
            return Signal(
                "HOLD",
                0.20,
                f"BUY signal rejected — recent bars closing weak (body={body_dir:+.2f})",
                "Momentum",
                indicators,
            )
    else:  # SELL
        if velocity_3 > -MIN_VEL_3:
            return Signal(
                "HOLD",
                0.20,
                f"SELL signal rejected — price not moving down (v3={velocity_3:+.2%})",
                "Momentum",
                indicators,
            )
        if body_dir > 0.5:
            return Signal(
                "HOLD",
                0.20,
                f"SELL signal rejected — recent bars closing strong (body={body_dir:+.2f})",
                "Momentum",
                indicators,
            )

    # ----- Confidence -----
    # Base from factor alignment (0.50 .. 0.85), then bonuses for fresh
    # velocity, fresh cross, and volume confirmation.
    confidence = 0.50 + (strength / 5.0) * 0.35

    # Velocity bonus — strong recent move in our direction adds conviction.
    vel_aligned = velocity_3 if side == "BUY" else -velocity_3
    if vel_aligned > 0.005:
        confidence += 0.08
    elif vel_aligned > 0.002:
        confidence += 0.04

    # Fresh cross bonus
    if cross_age <= 3:
        confidence += 0.04

    # Volume bonus
    if vol_data["relative_volume"] > 1.5:
        confidence += 0.04

    confidence = min(0.95, confidence)

    reasoning = (
        f"EMA{fast}={ema_fast:.4f} vs EMA{slow}={ema_slow:.4f} ({gap:+.2%}, "
        f"cross_age={cross_age}b); RSI={rsi_val:.1f}; "
        f"MACD_h={macd_data['histogram']:.4f}; Vol={vol_data['relative_volume']:.1f}x; "
        f"v3={velocity_3:+.2%}, body={body_dir:+.2f}"
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
    velocity_2 = _short_velocity(closes, 2)
    body_dir = _bar_body_direction(candles)
    indicators = {
        "sma": mean, "stdev": sd, "z": z, "vol_pct": vol_pct,
        "velocity_2bar": velocity_2, "body_direction": body_dir,
    }

    if z <= -z_entry:
        side = "BUY"
    elif z >= z_entry:
        side = "SELL"
    else:
        return Signal("HOLD", 0.0, f"|z|={abs(z):.2f} below entry {z_entry}", "Mean Reversion", indicators)

    # Reversal-confirmation gate. A 2-sigma stretch is meaningless if price
    # is still racing in that direction — we'd be catching a knife. Require
    # the most recent bars to be turning back toward the mean before
    # committing to a reversion trade.
    if side == "BUY" and (velocity_2 <= 0 or body_dir < 0):
        return Signal(
            "HOLD",
            0.20,
            f"MR BUY rejected — no reversal yet (z={z:+.2f}, v2={velocity_2:+.2%}, body={body_dir:+.2f})",
            "Mean Reversion",
            indicators,
        )
    if side == "SELL" and (velocity_2 >= 0 or body_dir > 0):
        return Signal(
            "HOLD",
            0.20,
            f"MR SELL rejected — no reversal yet (z={z:+.2f}, v2={velocity_2:+.2%}, body={body_dir:+.2f})",
            "Mean Reversion",
            indicators,
        )

    # Confidence: how far past the threshold are we?
    excess = abs(z) - z_entry  # >= 0
    raw = min(1.0, excess / 1.5)  # full confidence at z = entry + 1.5
    confidence = max(0.0, min(0.99, 0.55 + 0.4 * raw))

    reasoning = f"Z={z:+.2f} vs SMA{period}; vol={vol_pct:.2%}; v2={velocity_2:+.2%}, body={body_dir:+.2f}"
    return Signal(side, confidence, reasoning, "Mean Reversion", indicators)


def scalping_signal(
    candles: list[dict[str, Any]],
    *,
    bb_period: int = 20,
    rsi_period: int = 7,  # Shorter RSI for scalping
) -> Signal:
    """
    Scalping strategy using Bollinger Bands and fast RSI:
      BUY  when price near lower band AND RSI oversold (<25) AND BOUNCING
      SELL when price near upper band AND RSI overbought (>75) AND ROLLING OVER
    
    The "bouncing / rolling over" check is critical — without it, the bot
    buys falling knives at the lower band and watches them keep falling.
    """
    closes = _closes(candles)
    if len(closes) < max(bb_period, rsi_period + 1):
        return Signal("HOLD", 0.0, "insufficient data for scalping", "Scalping")
    
    bb = bollinger_bands(closes, bb_period)
    rsi_val = rsi(closes, rsi_period) or 50.0
    vol_data = volume_analysis(candles, 10)  # Shorter volume lookback
    velocity_2 = _short_velocity(closes, 2)
    velocity_1 = _short_velocity(closes, 1)
    body_dir = _bar_body_direction(candles)
    
    indicators = {
        "bb_percent_b": bb["percent_b"],
        "bb_upper": bb["upper"],
        "bb_lower": bb["lower"],
        "rsi_fast": rsi_val,
        "relative_volume": vol_data["relative_volume"],
        "velocity_2bar": velocity_2,
        "body_direction": body_dir,
    }
    
    # Scalp BUY: price near lower band + oversold RSI + ACTIVE BOUNCE
    # The bounce gate is the difference between catching a reversal and
    # catching a falling knife. We require the very last bar to be up
    # AND the recent body direction to lean bullish.
    if bb["percent_b"] < 0.15 and rsi_val < 25:
        if velocity_1 <= 0 or velocity_2 <= 0:
            return Signal(
                "HOLD",
                0.20,
                f"Scalp BUY setup but no bounce yet: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v1={velocity_1:+.2%}",
                "Scalping",
                indicators,
            )
        if body_dir < 0:
            return Signal(
                "HOLD",
                0.20,
                f"Scalp BUY setup but bars closing weak: body={body_dir:+.2f}",
                "Scalping",
                indicators,
            )
        confidence = 0.70 + min(0.20, (25 - rsi_val) / 50)
        if vol_data["relative_volume"] > 1.3:
            confidence = min(0.95, confidence + 0.05)
        return Signal(
            "BUY",
            confidence,
            f"Scalp BUY: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v2={velocity_2:+.2%}, Vol={vol_data['relative_volume']:.1f}x",
            "Scalping",
            indicators
        )
    
    # Scalp SELL: price near upper band + overbought RSI + ACTIVE ROLL-OVER
    if bb["percent_b"] > 0.85 and rsi_val > 75:
        if velocity_1 >= 0 or velocity_2 >= 0:
            return Signal(
                "HOLD",
                0.20,
                f"Scalp SELL setup but no roll-over yet: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v1={velocity_1:+.2%}",
                "Scalping",
                indicators,
            )
        if body_dir > 0:
            return Signal(
                "HOLD",
                0.20,
                f"Scalp SELL setup but bars closing strong: body={body_dir:+.2f}",
                "Scalping",
                indicators,
            )
        confidence = 0.70 + min(0.20, (rsi_val - 75) / 50)
        if vol_data["relative_volume"] > 1.3:
            confidence = min(0.95, confidence + 0.05)
        return Signal(
            "SELL",
            confidence,
            f"Scalp SELL: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v2={velocity_2:+.2%}, Vol={vol_data['relative_volume']:.1f}x",
            "Scalping",
            indicators
        )
    
    return Signal("HOLD", 0.10, f"No scalp setup: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}", "Scalping", indicators)


def trend_following_signal(
    candles: list[dict[str, Any]],
    *,
    adx_threshold: float = 25.0,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> Signal:
    """
    Trend following strategy using ADX for trend strength confirmation:
      BUY  when ADX > threshold AND +DI > -DI AND price > EMA
      SELL when ADX > threshold AND -DI > +DI AND price < EMA
    
    Only trades when there's a confirmed strong trend.
    """
    closes = _closes(candles)
    if len(closes) < max(ema_slow, 30):
        return Signal("HOLD", 0.0, "insufficient data for trend following", "Trend Following")
    
    adx_data = adx_indicator(candles)
    ema_fast_val = ema(closes, ema_fast)[-1]
    ema_slow_val = ema(closes, ema_slow)[-1]
    vol_data = volume_analysis(candles)
    
    current_price = closes[-1]
    indicators = {
        "adx": adx_data["adx"],
        "plus_di": adx_data["plus_di"],
        "minus_di": adx_data["minus_di"],
        "ema_fast": ema_fast_val,
        "ema_slow": ema_slow_val,
        "relative_volume": vol_data["relative_volume"],
    }
    
    # Check for strong trend
    if adx_data["adx"] < adx_threshold:
        return Signal(
            "HOLD",
            0.15,
            f"Weak trend: ADX={adx_data['adx']:.1f} < {adx_threshold}",
            "Trend Following",
            indicators
        )
    
    # Bullish trend: +DI > -DI and price above EMAs
    if adx_data["plus_di"] > adx_data["minus_di"] and current_price > ema_fast_val > ema_slow_val:
        strength = (adx_data["adx"] - adx_threshold) / 30  # Normalize strength
        confidence = min(0.90, 0.60 + strength * 0.3)
        
        if vol_data["relative_volume"] > 1.2:
            confidence = min(0.95, confidence + 0.05)
        
        return Signal(
            "BUY",
            confidence,
            f"Bullish trend: ADX={adx_data['adx']:.1f}, +DI={adx_data['plus_di']:.1f} > -DI={adx_data['minus_di']:.1f}",
            "Trend Following",
            indicators
        )
    
    # Bearish trend: -DI > +DI and price below EMAs
    if adx_data["minus_di"] > adx_data["plus_di"] and current_price < ema_fast_val < ema_slow_val:
        strength = (adx_data["adx"] - adx_threshold) / 30
        confidence = min(0.90, 0.60 + strength * 0.3)
        
        if vol_data["relative_volume"] > 1.2:
            confidence = min(0.95, confidence + 0.05)
        
        return Signal(
            "SELL",
            confidence,
            f"Bearish trend: ADX={adx_data['adx']:.1f}, -DI={adx_data['minus_di']:.1f} > +DI={adx_data['plus_di']:.1f}",
            "Trend Following",
            indicators
        )
    
    return Signal(
        "HOLD",
        0.20,
        f"Trend conflict: ADX={adx_data['adx']:.1f}, +DI={adx_data['plus_di']:.1f}, -DI={adx_data['minus_di']:.1f}",
        "Trend Following",
        indicators
    )


def breakout_signal(
    candles: list[dict[str, Any]],
    *,
    lookback: int = 20,
    volume_threshold: float = 1.5,
) -> Signal:
    """
    Breakout strategy detecting price breaking out of recent range:
      BUY  when price breaks above recent high with volume confirmation
      SELL when price breaks below recent low with volume confirmation
    """
    if len(candles) < lookback + 5:
        return Signal("HOLD", 0.0, "insufficient data for breakout", "Breakout")
    
    closes = _closes(candles)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    
    # Recent range (excluding last 2 bars)
    recent_high = max(highs[-lookback:-2])
    recent_low = min(lows[-lookback:-2])
    current_price = closes[-1]
    
    vol_data = volume_analysis(candles, lookback)
    atr_val = atr(candles, 14) or (recent_high - recent_low) / 10
    
    indicators = {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "current_price": current_price,
        "relative_volume": vol_data["relative_volume"],
        "atr": atr_val,
    }
    
    # Bullish breakout
    if current_price > recent_high:
        breakout_strength = (current_price - recent_high) / atr_val if atr_val > 0 else 0
        confidence = min(0.90, 0.55 + breakout_strength * 0.2)
        
        # Volume confirmation is crucial for breakouts
        if vol_data["relative_volume"] >= volume_threshold:
            confidence = min(0.95, confidence + 0.15)
        elif vol_data["relative_volume"] < 1.0:
            confidence = max(0.40, confidence - 0.15)  # Weak volume = suspicious
        
        return Signal(
            "BUY",
            confidence,
            f"Bullish breakout: ${current_price:.4f} > ${recent_high:.4f}, Vol={vol_data['relative_volume']:.1f}x",
            "Breakout",
            indicators
        )
    
    # Bearish breakout
    if current_price < recent_low:
        breakout_strength = (recent_low - current_price) / atr_val if atr_val > 0 else 0
        confidence = min(0.90, 0.55 + breakout_strength * 0.2)
        
        if vol_data["relative_volume"] >= volume_threshold:
            confidence = min(0.95, confidence + 0.15)
        elif vol_data["relative_volume"] < 1.0:
            confidence = max(0.40, confidence - 0.15)
        
        return Signal(
            "SELL",
            confidence,
            f"Bearish breakout: ${current_price:.4f} < ${recent_low:.4f}, Vol={vol_data['relative_volume']:.1f}x",
            "Breakout",
            indicators
        )
    
    # Price within range
    range_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    return Signal(
        "HOLD",
        0.10,
        f"In range: {range_position:.0%} between ${recent_low:.4f} and ${recent_high:.4f}",
        "Breakout",
        indicators
    )


# ---------------------------------------------------------------------------- #
# Public API used by the bot loop
# ---------------------------------------------------------------------------- #


def evaluate_entry_quality(
    candles: list[dict[str, Any]],
    side: str,
) -> dict[str, Any]:
    """
    Reject-or-accept gate based on where current price sits relative to
    market structure. Used to stop the bot from BUYing right under a wall
    of resistance or SELLing right above support — both of which produce
    "trade goes nowhere then drifts the wrong way" outcomes.

    Beyond the binary accept/reject, this also returns a `quality_score` in
    [0, 1] that the caller should use as a CONFIDENCE MULTIPLIER. The score
    rewards setups where there is plenty of room to the next structural
    level relative to the structural risk (large potential R:R), and
    penalises setups where headroom barely clears the minimum.

    Returns:
        {
          "accept": bool,
          "reason": str,
          "headroom_pct": float,   # % distance to next resistance (BUY) or support (SELL)
          "risk_pct": float,       # % distance to the structural stop
          "potential_rr": float,   # headroom / risk — naive max R:R
          "quality_score": float,  # 0..1 multiplier for confidence
          "swing_high": float,
          "swing_low": float,
          "atr_pct": float,
        }
    """
    if not candles or len(candles) < 30:
        return {
            "accept": True,
            "reason": "insufficient bars for S/R check",
            "headroom_pct": 0.0,
            "risk_pct": 0.0,
            "potential_rr": 0.0,
            "quality_score": 1.0,
        }

    swings = swing_levels(candles, lookback=30)
    if not swings:
        return {
            "accept": True,
            "reason": "no swings detected",
            "headroom_pct": 0.0,
            "risk_pct": 0.0,
            "potential_rr": 0.0,
            "quality_score": 1.0,
        }

    last = float(candles[-1]["close"])
    a = atr(candles, period=14) or 0.0
    atr_pct = (a / last) if last > 0 else 0.0

    swing_high = swings["swing_high"]
    swing_low = swings["swing_low"]

    if side == "BUY":
        headroom = (swing_high - last) / last if last > 0 else 0.0
        # Risk = distance to swing low (where our structural stop lives) + ATR buffer.
        risk = ((last - swing_low) / last + atr_pct * 0.2) if last > 0 else 0.0
        # Reject BUYs with less headroom than 1.0x ATR — that's barely room
        # to make 1R before hitting resistance. We need the target to be
        # reachable WITHIN visible market structure.
        min_headroom = max(0.008, atr_pct * 1.0)
        if headroom < min_headroom:
            return {
                "accept": False,
                "reason": (
                    f"BUY rejected: only {headroom:.2%} headroom to swing high "
                    f"${swing_high:.4f} (need {min_headroom:.2%})"
                ),
                "headroom_pct": headroom,
                "risk_pct": risk,
                "potential_rr": (headroom / risk) if risk > 0 else 0.0,
                "quality_score": 0.0,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "atr_pct": atr_pct,
            }
    elif side == "SELL":
        headroom = (last - swing_low) / last if last > 0 else 0.0
        risk = ((swing_high - last) / last + atr_pct * 0.2) if last > 0 else 0.0
        min_headroom = max(0.008, atr_pct * 1.0)
        if headroom < min_headroom:
            return {
                "accept": False,
                "reason": (
                    f"SELL rejected: only {headroom:.2%} headroom to swing low "
                    f"${swing_low:.4f} (need {min_headroom:.2%})"
                ),
                "headroom_pct": headroom,
                "risk_pct": risk,
                "potential_rr": (headroom / risk) if risk > 0 else 0.0,
                "quality_score": 0.0,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "atr_pct": atr_pct,
            }
    else:
        return {
            "accept": True,
            "reason": "non-directional side",
            "headroom_pct": 0.0,
            "risk_pct": 0.0,
            "potential_rr": 0.0,
            "quality_score": 1.0,
        }

    # ----- Quality score -----
    # Two factors: (a) potential R:R and (b) absolute headroom in ATR units.
    # Both must be decent to score high. Floor at 0.65 (don't kill a valid
    # signal too aggressively) and cap at 1.20 (reward genuinely great
    # setups but don't let them bypass other safeguards).
    potential_rr = (headroom / risk) if risk > 0 else 0.0
    headroom_atrs = (headroom / atr_pct) if atr_pct > 0 else 0.0

    # R:R component: 1.5 R:R -> 0.85, 2.0 -> 1.00, 3.0+ -> 1.15
    if potential_rr >= 3.0:
        rr_factor = 1.15
    elif potential_rr >= 2.0:
        rr_factor = 1.00 + (potential_rr - 2.0) * 0.15
    elif potential_rr >= 1.5:
        rr_factor = 0.85 + (potential_rr - 1.5) * 0.30
    elif potential_rr >= 1.0:
        rr_factor = 0.70 + (potential_rr - 1.0) * 0.30
    else:
        rr_factor = 0.65

    # Headroom-in-ATRs component: 1 ATR -> 0.80, 2 ATR -> 1.00, 4+ ATR -> 1.10
    if headroom_atrs >= 4.0:
        room_factor = 1.10
    elif headroom_atrs >= 2.0:
        room_factor = 1.00 + (headroom_atrs - 2.0) * 0.05
    elif headroom_atrs >= 1.0:
        room_factor = 0.80 + (headroom_atrs - 1.0) * 0.20
    else:
        room_factor = 0.70

    quality_score = max(0.65, min(1.20, rr_factor * room_factor))

    return {
        "accept": True,
        "reason": (
            f"S/R OK: headroom={headroom:.2%} ({headroom_atrs:.1f}xATR), "
            f"risk={risk:.2%}, potential R:R={potential_rr:.1f}, q={quality_score:.2f}"
        ),
        "headroom_pct": headroom,
        "risk_pct": risk,
        "potential_rr": potential_rr,
        "quality_score": quality_score,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "atr_pct": atr_pct,
    }


def get_entry_candles(
    product_id: str,
    *,
    tick_seconds: int | None = None,
    lookback_bars: int = 100,
) -> list[dict[str, Any]]:
    """Convenience helper: fetch candles at the same granularity used by
    `evaluate_symbol`, so the bot engine can run S/R + stop calculations on
    the same data the signal saw."""
    if tick_seconds is None:
        granularity = 300
    else:
        if tick_seconds <= 10:
            granularity = 60
        elif tick_seconds <= 60:
            granularity = 300
        else:
            granularity = 900
    return get_candles(product_id, granularity=granularity, limit=lookback_bars)


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
    if st == "Scalping":
        return scalping_signal(candles)
    if st == "Trend Following":
        return trend_following_signal(candles)
    if st == "Breakout":
        return breakout_signal(candles)
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


# ---------------------------------------------------------------------------- #
# Swing-anchored stop / take-profit
# ---------------------------------------------------------------------------- #
# A stop placed at "round percentage X% below entry" is meaningless to the
# market — price has no reason to respect it, so the trade frequently grinds
# through it on noise. A stop placed just past the most recent swing low (for
# longs) or swing high (for shorts) is anchored to a level where the market
# has actually pivoted before; if price breaks that level our directional
# thesis is genuinely wrong.


def swing_levels(candles: list[dict[str, Any]], lookback: int = 30) -> dict[str, float]:
    """Return the most recent swing high / swing low over `lookback` bars.

    A "swing" here is the conventional 3-bar pattern: a bar whose high (or
    low) is more extreme than its two neighbours on each side. Returns dict
    with `swing_low` / `swing_high` (latest within lookback) and falls back
    to absolute high/low if no clean swing was found.
    """
    if len(candles) < 7:
        return {}
    window = candles[-min(lookback, len(candles)):]
    swing_high = max(float(c["high"]) for c in window)
    swing_low = min(float(c["low"]) for c in window)
    # Look for a more recent (closer to current bar) confirmed pivot. We
    # walk backward from bar n-3 toward the start, checking 2-bar neighbours.
    n = len(window)
    for i in range(n - 3, 1, -1):
        h = float(window[i]["high"])
        l = float(window[i]["low"])
        is_swing_high = (
            h > float(window[i - 1]["high"])
            and h > float(window[i - 2]["high"])
            and h > float(window[i + 1]["high"])
            and h > float(window[i + 2]["high"])
        )
        is_swing_low = (
            l < float(window[i - 1]["low"])
            and l < float(window[i - 2]["low"])
            and l < float(window[i + 1]["low"])
            and l < float(window[i + 2]["low"])
        )
        if is_swing_high and h < swing_high * 1.0001:
            swing_high = h
        if is_swing_low and l > swing_low * 0.9999:
            swing_low = l
    return {"swing_high": swing_high, "swing_low": swing_low}


def smart_stops(
    candles: list[dict[str, Any]],
    side: str,
    confidence: float,
    *,
    min_stop_pct: float = 0.012,
    max_stop_pct: float = 0.06,
    atr_period: int = 14,
) -> dict[str, float]:
    """
    Compute volatility-AND-structure-aware stops for a fresh entry.

    The stop is the WIDER of (a) `1.2x ATR` (so noise can't tag us out) and
    (b) `entry - swing_low - 0.2x ATR buffer` (anchored to actual market
    structure). Then clamped to [min_stop_pct, max_stop_pct] of entry so we
    never bet the farm on a single trade.

    Confidence does NOT widen the stop — high-confidence trades get the
    SAME tight invalidation level. Confidence instead widens the take-profit
    target so we let winners run further when conviction is strong.

    Returns {} if data is insufficient.
    """
    if not candles or len(candles) < atr_period + 5:
        return {}
    a = atr(candles, period=atr_period)
    if a is None or a <= 0:
        return {}
    swings = swing_levels(candles, lookback=30)
    if not swings:
        return {}
    last = float(candles[-1]["close"])
    if last <= 0:
        return {}

    atr_pct = a / last
    atr_buffer = a * 0.20  # tiny buffer past the swing so we don't sit on it

    if side == "BUY":
        structural_stop = swings["swing_low"] - atr_buffer
        atr_stop = last - 1.2 * a
        # Use the WIDER of the two — whichever is further from price — but
        # never wider than max_stop_pct.
        stop_price = min(structural_stop, atr_stop)
        stop_pct = (last - stop_price) / last
    elif side == "SELL":
        structural_stop = swings["swing_high"] + atr_buffer
        atr_stop = last + 1.2 * a
        stop_price = max(structural_stop, atr_stop)
        stop_pct = (stop_price - last) / last
    else:
        return {}

    stop_pct = max(min_stop_pct, min(max_stop_pct, stop_pct))

    # ----- Take-profit: confidence-scaled R:R -----
    # Base 2.0R, scaling up to 3.5R for high-conviction trades. Capped so
    # the target stays inside reach of typical intraday moves.
    rr = 2.0 + max(0.0, min(1.0, confidence)) * 1.5
    tp_pct = stop_pct * rr

    # Also cap TP at 1.5x recent range (high - low over lookback) — chasing
    # a 12% target on a coin that hasn't moved 4% in a week is a phantom.
    window = candles[-30:]
    rng = max(float(c["high"]) for c in window) - min(float(c["low"]) for c in window)
    if rng > 0:
        max_tp_pct = (rng * 1.5) / last
        tp_pct = min(tp_pct, max_tp_pct)

    if side == "BUY":
        sl_price = last * (1 - stop_pct)
        tp_price = last * (1 + tp_pct)
    else:
        sl_price = last * (1 + stop_pct)
        tp_price = last * (1 - tp_pct)

    return {
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "atr": a,
        "atr_pct": atr_pct,
        "swing_high": swings["swing_high"],
        "swing_low": swings["swing_low"],
        "rr_ratio": rr,
    }
