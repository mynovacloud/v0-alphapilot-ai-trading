"""
Live strategy engine.  (v2 — enhanced)

Pure-math technical analysis. Computes indicators from real candle data and
emits a Signal(side, confidence, reasoning, strategy, indicators) per symbol.
No AI, no learning, no database — just candles in, a directional signal out.
(The only I/O is candle fetching inside the orchestrators evaluate_symbol /
get_entry_candles; the indicator primitives and the *_signal strategies are
pure functions and stay testable/fast.)

STRATEGIES (selected by strategy_type in evaluate_symbol):
  - Momentum            EMA stack + RSI + MACD + volume + live-velocity gates
  - Mean Reversion      z-score fade of extremes, with a reversal-confirmation gate
  - Scalping            Bollinger + fast RSI, requires an active bounce/roll-over
  - Trend Following     ADX-confirmed trend riding (now using REAL smoothed ADX)
  - Breakout            range-exit with volume confirmation
  - Volatility Breakout trade WITH range expansion (rewritten — see below)
  - Probability Edge    momentum with a stricter confidence floor

WHAT CHANGED FROM v1 (full CHANGELOG at the bottom):
  * FIXED: adx_indicator() now returns a properly smoothed ADX, not the raw
    single-bar DX it returned before. trend_following_signal's "ADX>25" gate is
    now a real sustained-trend filter. (Behavior change — backtest it.)
  * FIXED: "Volatility Breakout" no longer reuses reversal-gated mean reversion
    and flips the side (which produced incoherent SELLs into bounces). It is now
    a real volatility-expansion breakout strategy.
  * NEW:   every Signal from evaluate_symbol carries an enriched, consistent core
    indicator set (rsi, macd_histogram, adx, relative_volume, atr_pct,
    atr_expansion, velocity, body_direction, ...). This pairs with the decision
    engine now actually reading signal.indicators.
  * NEW:   optional multi-timeframe confirmation (mtf_confirm, default OFF) and
    volatility-regime dampener (vol_gate, default OFF) in evaluate_symbol.
  * FIXED: module docstring (v1 claimed only two strategies existed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from connectors.candles import get_candles
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CENTRALIZED TUNABLES
# (kept as module constants so thresholds are documented in one place; each
#  strategy still accepts keyword overrides for backtest sweeps)
# =============================================================================

# Momentum
MOM_MIN_FACTORS = 3.0          # need at least this many aligned factors
MOM_EDGE = 1.5                 # and this much edge over the opposite side
MOM_CROSS_MAX_AGE = 12         # ignore EMA crosses older than this many bars
MOM_MIN_VELOCITY_3 = 0.0008    # require >=0.08% move over 3 bars in trade dir

# Mean reversion
MR_Z_ENTRY = 1.5
MR_MIN_VOL_PCT = 0.001

# Trend following
TF_ADX_THRESHOLD = 25.0        # now a REAL ADX threshold (smoothed)

# Volatility regime
VOL_EXPANSION_SHORT = 5        # short ATR window
VOL_EXPANSION_LONG = 20        # long ATR window
VOL_GATE_EXTREME = 2.5         # atr_expansion above this = volatility spike
VOL_GATE_DAMPEN = 0.90         # confidence multiplier applied in a spike (opt-in)

# Multi-timeframe
MTF_GRANULARITY_MULT = 4       # higher TF is this multiple of the base granularity
MTF_AGREE_BOOST = 1.05
MTF_CONFLICT_DAMPEN = 0.85


# =============================================================================
# Indicator primitives (pure functions; no numpy on the hot path)
# =============================================================================

def _closes(candles: Iterable[dict[str, Any]]) -> list[float]:
    return [float(c["close"]) for c in candles if c.get("close") is not None]


def ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average, aligned with `values`."""
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
    """SMA returned as a full list aligned with input."""
    if len(values) < period or period <= 0:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(0.0, change))
        losses.append(max(0.0, -change))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_indicator(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line, signal line, histogram."""
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
        "histogram": histogram,
    }


def bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands + %B."""
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


def _directional_movement(candles: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Return (+DM, -DM, TR) series for ADX/DI computation."""
    plus_dm, minus_dm, tr = [], [], []
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_high = float(candles[i - 1]["high"])
        prev_low = float(candles[i - 1]["low"])
        prev_close = float(candles[i - 1]["close"])
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return plus_dm, minus_dm, tr


def adx_indicator(candles: list[dict], period: int = 14) -> dict:
    """ADX (Average Directional Index) with +DI / -DI.

    *** v2 FIX ***
    v1 returned the raw single-period DX under the "adx" key with no smoothing,
    so trend_following_signal was gating on a noisy single-bar reading rather
    than a sustained trend. This implementation uses Wilder's smoothing for TR
    and DM, builds a DX series, and Wilder-smooths it into a true ADX — matching
    the (correct) approach advanced_signal_engine already used, so the two
    engines now agree on what "ADX" means.

    Returns {"adx", "plus_di", "minus_di"} (same keys as before).
    """
    if len(candles) < period + 1:
        return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    plus_dm, minus_dm, tr = _directional_movement(candles)
    n = len(tr)
    if n < period:
        return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    # Wilder-smoothed initial sums over the first `period` bars.
    atr_s = sum(tr[:period])
    plus_s = sum(plus_dm[:period])
    minus_s = sum(minus_dm[:period])

    def _di(plus_smoothed: float, minus_smoothed: float, atr_smoothed: float) -> tuple[float, float]:
        if atr_smoothed <= 0:
            return 0.0, 0.0
        return 100.0 * plus_smoothed / atr_smoothed, 100.0 * minus_smoothed / atr_smoothed

    plus_di, minus_di = _di(plus_s, minus_s, atr_s)
    dx_list: list[float] = []
    di_sum = plus_di + minus_di
    dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0)

    # Wilder smoothing forward through the remaining bars.
    for i in range(period, n):
        atr_s = atr_s - (atr_s / period) + tr[i]
        plus_s = plus_s - (plus_s / period) + plus_dm[i]
        minus_s = minus_s - (minus_s / period) + minus_dm[i]
        plus_di, minus_di = _di(plus_s, minus_s, atr_s)
        di_sum = plus_di + minus_di
        dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0)

    # ADX = Wilder average of the DX series.
    if len(dx_list) >= period:
        adx = sum(dx_list[:period]) / period
        for i in range(period, len(dx_list)):
            adx = (adx * (period - 1) + dx_list[i]) / period
    else:
        adx = sum(dx_list) / len(dx_list) if dx_list else 0.0

    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def volume_analysis(candles: list[dict], period: int = 20) -> dict:
    """Relative volume, volume trend, buying pressure."""
    if len(candles) < period:
        return {"relative_volume": 1.0, "volume_trend": "NEUTRAL", "buying_pressure": 0.5}
    volumes = [float(c.get("volume", 0)) for c in candles]
    avg_volume = sum(volumes[-period:]) / period if period > 0 else 1
    current_volume = volumes[-1] if volumes else 0
    relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

    recent_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_volume
    if recent_avg > avg_volume * 1.3:
        volume_trend = "INCREASING"
    elif recent_avg < avg_volume * 0.7:
        volume_trend = "DECREASING"
    else:
        volume_trend = "NEUTRAL"

    up_volume = down_volume = 0.0
    closes = [float(c["close"]) for c in candles[-10:]]
    vols = volumes[-10:]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            up_volume += vols[i]
        else:
            down_volume += vols[i]
    total = up_volume + down_volume
    buying_pressure = up_volume / total if total > 0 else 0.5

    return {"relative_volume": relative_volume, "volume_trend": volume_trend, "buying_pressure": buying_pressure}


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
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def _atr_expansion(candles: list[dict[str, Any]]) -> float | None:
    """Cheap volatility-regime proxy: short-window ATR / long-window ATR.
    >1 = volatility expanding (breakout-friendly), <1 = contracting (squeeze)."""
    short = atr(candles, VOL_EXPANSION_SHORT)
    long = atr(candles, VOL_EXPANSION_LONG)
    if short is not None and long and long > 0:
        return short / long
    return None


# =============================================================================
# Signal type  (UNCHANGED contract)
# =============================================================================

@dataclass
class Signal:
    side: str                # "BUY" / "SELL" / "HOLD"
    confidence: float        # 0..1
    reasoning: str
    strategy: str            # "Momentum" / "Mean Reversion" / etc.
    indicators: dict[str, float] = field(default_factory=dict)

    def is_actionable(self, min_confidence: float = 0.0) -> bool:
        return self.side in {"BUY", "SELL"} and self.confidence >= min_confidence


# =============================================================================
# Velocity / freshness helpers
# =============================================================================
# These guards exist because the bot was firing low-confidence BUYs on stale
# signals. We require the price to actually be moving in our direction NOW
# before committing capital.


def _short_velocity(closes: list[float], bars: int = 3) -> float:
    """Percent change over the last `bars` closes — the 'is it moving now' check."""
    if not closes or len(closes) < bars + 1:
        return 0.0
    base = closes[-bars - 1]
    if not base:
        return 0.0
    return (closes[-1] - base) / base


def _ema_cross_freshness(closes: list[float], fast_n: int, slow_n: int) -> int:
    """How many bars ago EMA_fast crossed EMA_slow (capped at 30; 99 = unknown)."""
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
    """Sum of bullish-body fraction over the last 3 bars, in [-3, 3]."""
    if not candles or len(candles) < 3:
        return 0.0
    total = 0.0
    for c in candles[-3:]:
        try:
            h = float(c.get("high", 0))
            l = float(c.get("low", 0))
            cl = float(c.get("close", 0))
            rng = h - l
            if rng <= 0:
                continue
            pos = (cl - l) / rng
            total += (pos - 0.5) * 2.0
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return total


def _common_indicators(candles: list[dict[str, Any]]) -> dict[str, float]:
    """Compute a consistent CORE indicator set once, so every Signal from
    evaluate_symbol carries the same rich context regardless of strategy.

    This is the complement to the decision engine now reading signal.indicators:
    mean-reversion signals (which previously carried only z/sma) now also expose
    rsi/adx/macd/volume so Claude and the adaptive engine see a full picture.
    """
    closes = _closes(candles)
    out: dict[str, float] = {}
    if len(closes) < 2:
        return out

    if len(closes) >= 26:
        out["ema_fast"] = ema(closes, 12)[-1]
        out["ema_slow"] = ema(closes, 26)[-1]
    r = rsi(closes, 14)
    if r is not None:
        out["rsi"] = r
    macd = macd_indicator(closes)
    out["macd_histogram"] = macd["histogram"]
    vol = volume_analysis(candles)
    out["relative_volume"] = vol["relative_volume"]
    out["buying_pressure"] = vol["buying_pressure"]
    a = atr(candles, 14)
    last = closes[-1] if closes else 0
    if a is not None and last:
        out["atr_pct"] = a / last
    adxd = adx_indicator(candles, 14)
    out["adx"] = adxd["adx"]
    out["plus_di"] = adxd["plus_di"]
    out["minus_di"] = adxd["minus_di"]
    out["velocity_3bar"] = _short_velocity(closes, 3)
    out["velocity_1bar"] = _short_velocity(closes, 1)
    out["body_direction"] = _bar_body_direction(candles)
    exp = _atr_expansion(candles)
    if exp is not None:
        out["atr_expansion"] = exp
    return {k: v for k, v in out.items() if v is not None}


# =============================================================================
# Individual strategies
# =============================================================================


def momentum_signal(candles: list[dict[str, Any]], *, fast: int = 12, slow: int = 26, lookback: int = 6) -> Signal:
    """EMA-cross momentum with RSI, MACD, volume AND live-velocity confirmation."""
    closes = _closes(candles)
    if len(closes) < max(slow, lookback + 2, 26):
        return Signal("HOLD", 0.0, "insufficient data for momentum", "Momentum")

    ema_fast = ema(closes, fast)[-1]
    ema_slow = ema(closes, slow)[-1]
    ret = pct_return(closes, lookback) or 0.0
    last = closes[-1] or 1e-9
    gap = (ema_fast - ema_slow) / last

    rsi_val = rsi(closes, 14) or 50.0
    macd_data = macd_indicator(closes)
    vol_data = volume_analysis(candles)

    velocity_3 = _short_velocity(closes, 3)
    velocity_1 = _short_velocity(closes, 1)
    cross_age = _ema_cross_freshness(closes, fast, slow)
    body_dir = _bar_body_direction(candles)

    indicators = {
        "ema_fast": ema_fast, "ema_slow": ema_slow, "gap_pct": gap, "return_lb": ret,
        "rsi": rsi_val, "macd_histogram": macd_data["histogram"],
        "relative_volume": vol_data["relative_volume"], "buying_pressure": vol_data["buying_pressure"],
        "velocity_3bar": velocity_3, "velocity_1bar": velocity_1,
        "cross_age_bars": cross_age, "body_direction": body_dir,
    }

    bullish = bearish = 0.0
    if ema_fast > ema_slow and cross_age <= MOM_CROSS_MAX_AGE:
        bullish += 1
    elif ema_fast < ema_slow and cross_age <= MOM_CROSS_MAX_AGE:
        bearish += 1

    if ret > 0.002:
        bullish += 1
    elif ret < -0.002:
        bearish += 1

    if rsi_val < 30 and velocity_3 > 0:
        bullish += 1
    elif rsi_val > 70 and velocity_3 < 0:
        bearish += 1
    elif rsi_val < 45:
        bullish += 0.5
    elif rsi_val > 55:
        bearish += 0.5

    if macd_data["histogram"] > 0:
        bullish += 1
    else:
        bearish += 1

    if vol_data["relative_volume"] > 1.2:
        if vol_data["buying_pressure"] > 0.6:
            bullish += 1
        elif vol_data["buying_pressure"] < 0.4:
            bearish += 1

    if bullish >= MOM_MIN_FACTORS and bullish >= bearish + MOM_EDGE:
        side, strength = "BUY", bullish
    elif bearish >= MOM_MIN_FACTORS and bearish >= bullish + MOM_EDGE:
        side, strength = "SELL", bearish
    else:
        return Signal(
            "HOLD", 0.20,
            f"No edge: bull={bullish:.1f}, bear={bearish:.1f}, v3={velocity_3:+.2%}, cross_age={cross_age}b",
            "Momentum", indicators,
        )

    # Velocity confirmation gate.
    if side == "BUY":
        if velocity_3 < MOM_MIN_VELOCITY_3:
            return Signal("HOLD", 0.20, f"BUY rejected — price not moving up (v3={velocity_3:+.2%})", "Momentum", indicators)
        if body_dir < -0.5:
            return Signal("HOLD", 0.20, f"BUY rejected — bars closing weak (body={body_dir:+.2f})", "Momentum", indicators)
    else:
        if velocity_3 > -MOM_MIN_VELOCITY_3:
            return Signal("HOLD", 0.20, f"SELL rejected — price not moving down (v3={velocity_3:+.2%})", "Momentum", indicators)
        if body_dir > 0.5:
            return Signal("HOLD", 0.20, f"SELL rejected — bars closing strong (body={body_dir:+.2f})", "Momentum", indicators)

    confidence = 0.50 + (strength / 5.0) * 0.35
    vel_aligned = velocity_3 if side == "BUY" else -velocity_3
    if vel_aligned > 0.005:
        confidence += 0.08
    elif vel_aligned > 0.002:
        confidence += 0.04
    if cross_age <= 3:
        confidence += 0.04
    if vol_data["relative_volume"] > 1.5:
        confidence += 0.04
    confidence = min(0.95, confidence)

    reasoning = (
        f"EMA{fast}={ema_fast:.4f} vs EMA{slow}={ema_slow:.4f} ({gap:+.2%}, cross_age={cross_age}b); "
        f"RSI={rsi_val:.1f}; MACD_h={macd_data['histogram']:.4f}; "
        f"Vol={vol_data['relative_volume']:.1f}x; v3={velocity_3:+.2%}, body={body_dir:+.2f}"
    )
    return Signal(side, confidence, reasoning, "Momentum", indicators)


def mean_reversion_signal(candles: list[dict[str, Any]], *, period: int = 20, z_entry: float = MR_Z_ENTRY, min_vol_pct: float = MR_MIN_VOL_PCT) -> Signal:
    """Z-score mean reversion with a reversal-confirmation gate."""
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
        return Signal("HOLD", 0.0, f"vol {vol_pct:.3%} below floor {min_vol_pct:.3%}", "Mean Reversion",
                      {"sma": mean, "stdev": sd, "vol_pct": vol_pct})

    z = (last - mean) / sd if sd > 0 else 0.0
    velocity_2 = _short_velocity(closes, 2)
    body_dir = _bar_body_direction(candles)
    indicators = {"sma": mean, "stdev": sd, "z": z, "vol_pct": vol_pct,
                  "velocity_2bar": velocity_2, "body_direction": body_dir}

    if z <= -z_entry:
        side = "BUY"
    elif z >= z_entry:
        side = "SELL"
    else:
        return Signal("HOLD", 0.0, f"|z|={abs(z):.2f} below entry {z_entry}", "Mean Reversion", indicators)

    if side == "BUY" and (velocity_2 <= 0 or body_dir < 0):
        return Signal("HOLD", 0.20, f"MR BUY rejected — no reversal yet (z={z:+.2f}, v2={velocity_2:+.2%}, body={body_dir:+.2f})", "Mean Reversion", indicators)
    if side == "SELL" and (velocity_2 >= 0 or body_dir > 0):
        return Signal("HOLD", 0.20, f"MR SELL rejected — no reversal yet (z={z:+.2f}, v2={velocity_2:+.2%}, body={body_dir:+.2f})", "Mean Reversion", indicators)

    excess = abs(z) - z_entry
    raw = min(1.0, excess / 1.5)
    confidence = max(0.0, min(0.99, 0.55 + 0.4 * raw))
    reasoning = f"Z={z:+.2f} vs SMA{period}; vol={vol_pct:.2%}; v2={velocity_2:+.2%}, body={body_dir:+.2f}"
    return Signal(side, confidence, reasoning, "Mean Reversion", indicators)


def scalping_signal(candles: list[dict[str, Any]], *, bb_period: int = 20, rsi_period: int = 7) -> Signal:
    """Bollinger + fast-RSI scalp, requiring an active bounce / roll-over."""
    closes = _closes(candles)
    if len(closes) < max(bb_period, rsi_period + 1):
        return Signal("HOLD", 0.0, "insufficient data for scalping", "Scalping")

    bb = bollinger_bands(closes, bb_period)
    rsi_val = rsi(closes, rsi_period) or 50.0
    vol_data = volume_analysis(candles, 10)
    velocity_2 = _short_velocity(closes, 2)
    velocity_1 = _short_velocity(closes, 1)
    body_dir = _bar_body_direction(candles)

    indicators = {
        "bb_percent_b": bb["percent_b"], "bb_upper": bb["upper"], "bb_lower": bb["lower"],
        "rsi_fast": rsi_val, "relative_volume": vol_data["relative_volume"],
        "velocity_2bar": velocity_2, "body_direction": body_dir,
    }

    if bb["percent_b"] < 0.15 and rsi_val < 25:
        if velocity_1 <= 0 or velocity_2 <= 0:
            return Signal("HOLD", 0.20, f"Scalp BUY setup but no bounce yet: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v1={velocity_1:+.2%}", "Scalping", indicators)
        if body_dir < 0:
            return Signal("HOLD", 0.20, f"Scalp BUY setup but bars closing weak: body={body_dir:+.2f}", "Scalping", indicators)
        confidence = 0.70 + min(0.20, (25 - rsi_val) / 50)
        if vol_data["relative_volume"] > 1.3:
            confidence = min(0.95, confidence + 0.05)
        return Signal("BUY", confidence, f"Scalp BUY: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v2={velocity_2:+.2%}, Vol={vol_data['relative_volume']:.1f}x", "Scalping", indicators)

    if bb["percent_b"] > 0.85 and rsi_val > 75:
        if velocity_1 >= 0 or velocity_2 >= 0:
            return Signal("HOLD", 0.20, f"Scalp SELL setup but no roll-over yet: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v1={velocity_1:+.2%}", "Scalping", indicators)
        if body_dir > 0:
            return Signal("HOLD", 0.20, f"Scalp SELL setup but bars closing strong: body={body_dir:+.2f}", "Scalping", indicators)
        confidence = 0.70 + min(0.20, (rsi_val - 75) / 50)
        if vol_data["relative_volume"] > 1.3:
            confidence = min(0.95, confidence + 0.05)
        return Signal("SELL", confidence, f"Scalp SELL: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}, v2={velocity_2:+.2%}, Vol={vol_data['relative_volume']:.1f}x", "Scalping", indicators)

    return Signal("HOLD", 0.10, f"No scalp setup: %B={bb['percent_b']:.2f}, RSI={rsi_val:.1f}", "Scalping", indicators)


def trend_following_signal(candles: list[dict[str, Any]], *, adx_threshold: float = TF_ADX_THRESHOLD, ema_fast: int = 20, ema_slow: int = 50) -> Signal:
    """ADX-confirmed trend riding. Now using the corrected smoothed ADX."""
    closes = _closes(candles)
    if len(closes) < max(ema_slow, 30):
        return Signal("HOLD", 0.0, "insufficient data for trend following", "Trend Following")

    adx_data = adx_indicator(candles)
    ema_fast_val = ema(closes, ema_fast)[-1]
    ema_slow_val = ema(closes, ema_slow)[-1]
    vol_data = volume_analysis(candles)
    current_price = closes[-1]

    indicators = {
        "adx": adx_data["adx"], "plus_di": adx_data["plus_di"], "minus_di": adx_data["minus_di"],
        "ema_fast": ema_fast_val, "ema_slow": ema_slow_val, "relative_volume": vol_data["relative_volume"],
    }

    if adx_data["adx"] < adx_threshold:
        return Signal("HOLD", 0.15, f"Weak trend: ADX={adx_data['adx']:.1f} < {adx_threshold}", "Trend Following", indicators)

    if adx_data["plus_di"] > adx_data["minus_di"] and current_price > ema_fast_val > ema_slow_val:
        strength = (adx_data["adx"] - adx_threshold) / 30
        confidence = min(0.90, 0.60 + strength * 0.3)
        if vol_data["relative_volume"] > 1.2:
            confidence = min(0.95, confidence + 0.05)
        return Signal("BUY", confidence, f"Bullish trend: ADX={adx_data['adx']:.1f}, +DI={adx_data['plus_di']:.1f} > -DI={adx_data['minus_di']:.1f}", "Trend Following", indicators)

    if adx_data["minus_di"] > adx_data["plus_di"] and current_price < ema_fast_val < ema_slow_val:
        strength = (adx_data["adx"] - adx_threshold) / 30
        confidence = min(0.90, 0.60 + strength * 0.3)
        if vol_data["relative_volume"] > 1.2:
            confidence = min(0.95, confidence + 0.05)
        return Signal("SELL", confidence, f"Bearish trend: ADX={adx_data['adx']:.1f}, -DI={adx_data['minus_di']:.1f} > +DI={adx_data['plus_di']:.1f}", "Trend Following", indicators)

    return Signal("HOLD", 0.20, f"Trend conflict: ADX={adx_data['adx']:.1f}, +DI={adx_data['plus_di']:.1f}, -DI={adx_data['minus_di']:.1f}", "Trend Following", indicators)


def breakout_signal(candles: list[dict[str, Any]], *, lookback: int = 20, volume_threshold: float = 1.5) -> Signal:
    """Range-exit breakout with volume confirmation."""
    if len(candles) < lookback + 5:
        return Signal("HOLD", 0.0, "insufficient data for breakout", "Breakout")

    closes = _closes(candles)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    recent_high = max(highs[-lookback:-2])
    recent_low = min(lows[-lookback:-2])
    current_price = closes[-1]

    vol_data = volume_analysis(candles, lookback)
    atr_val = atr(candles, 14) or (recent_high - recent_low) / 10

    indicators = {
        "recent_high": recent_high, "recent_low": recent_low, "current_price": current_price,
        "relative_volume": vol_data["relative_volume"], "atr": atr_val,
    }

    if current_price > recent_high:
        strength = (current_price - recent_high) / atr_val if atr_val > 0 else 0
        confidence = min(0.90, 0.55 + strength * 0.2)
        if vol_data["relative_volume"] >= volume_threshold:
            confidence = min(0.95, confidence + 0.15)
        elif vol_data["relative_volume"] < 1.0:
            confidence = max(0.40, confidence - 0.15)
        return Signal("BUY", confidence, f"Bullish breakout: ${current_price:.4f} > ${recent_high:.4f}, Vol={vol_data['relative_volume']:.1f}x", "Breakout", indicators)

    if current_price < recent_low:
        strength = (recent_low - current_price) / atr_val if atr_val > 0 else 0
        confidence = min(0.90, 0.55 + strength * 0.2)
        if vol_data["relative_volume"] >= volume_threshold:
            confidence = min(0.95, confidence + 0.15)
        elif vol_data["relative_volume"] < 1.0:
            confidence = max(0.40, confidence - 0.15)
        return Signal("SELL", confidence, f"Bearish breakout: ${current_price:.4f} < ${recent_low:.4f}, Vol={vol_data['relative_volume']:.1f}x", "Breakout", indicators)

    range_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    return Signal("HOLD", 0.10, f"In range: {range_position:.0%} between ${recent_low:.4f} and ${recent_high:.4f}", "Breakout", indicators)


def volatility_breakout_signal(
    candles: list[dict[str, Any]],
    *,
    lookback: int = 20,
    volume_threshold: float = 1.3,
    min_expansion: float = 1.2,
) -> Signal:
    """Trade WITH a volatility-expansion breakout.

    *** v2 REWRITE ***
    v1's "Volatility Breakout" called mean_reversion_signal (which only fires
    AFTER a reversal is confirmed) and then flipped the side — so it could emit a
    SELL when price was 2σ low and bouncing UP. Incoherent. This is now a real
    breakout-with-the-move strategy:

      BUY  when price breaks the recent high, ATR is EXPANDING (short/long ATR >=
           min_expansion), volume confirms, and short-velocity is positive.
      SELL is the symmetric breakdown.

    Confidence scales with breakout distance (in ATRs), volatility expansion,
    and volume.
    """
    if len(candles) < lookback + 5:
        return Signal("HOLD", 0.0, "insufficient data for volatility breakout", "Volatility Breakout")

    closes = _closes(candles)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    recent_high = max(highs[-lookback:-2])
    recent_low = min(lows[-lookback:-2])
    current_price = closes[-1]

    vol_data = volume_analysis(candles, lookback)
    atr_val = atr(candles, 14) or (recent_high - recent_low) / 10
    expansion = _atr_expansion(candles) or 1.0
    velocity_2 = _short_velocity(closes, 2)
    body_dir = _bar_body_direction(candles)

    indicators = {
        "recent_high": recent_high, "recent_low": recent_low, "current_price": current_price,
        "relative_volume": vol_data["relative_volume"], "atr": atr_val,
        "atr_expansion": expansion, "velocity_2bar": velocity_2, "body_direction": body_dir,
    }

    # Volatility must actually be expanding — otherwise this is just a quiet
    # range nudge, not a breakout worth chasing.
    if expansion < min_expansion:
        return Signal("HOLD", 0.15, f"No volatility expansion (atr_exp={expansion:.2f} < {min_expansion})", "Volatility Breakout", indicators)

    def _confidence(break_atrs: float) -> float:
        c = 0.55 + min(0.20, break_atrs * 0.15)          # breakout distance
        c += min(0.12, (expansion - 1.0) * 0.20)         # expansion bonus
        if vol_data["relative_volume"] >= volume_threshold:
            c += 0.08
        elif vol_data["relative_volume"] < 1.0:
            c -= 0.10
        return max(0.40, min(0.95, c))

    # Bullish volatility breakout.
    if current_price > recent_high and velocity_2 > 0 and body_dir >= 0:
        break_atrs = (current_price - recent_high) / atr_val if atr_val > 0 else 0.0
        conf = _confidence(break_atrs)
        return Signal("BUY", conf, f"Vol breakout UP: ${current_price:.4f} > ${recent_high:.4f}, exp={expansion:.2f}x, Vol={vol_data['relative_volume']:.1f}x, v2={velocity_2:+.2%}", "Volatility Breakout", indicators)

    # Bearish volatility breakdown.
    if current_price < recent_low and velocity_2 < 0 and body_dir <= 0:
        break_atrs = (recent_low - current_price) / atr_val if atr_val > 0 else 0.0
        conf = _confidence(break_atrs)
        return Signal("SELL", conf, f"Vol breakdown DOWN: ${current_price:.4f} < ${recent_low:.4f}, exp={expansion:.2f}x, Vol={vol_data['relative_volume']:.1f}x, v2={velocity_2:+.2%}", "Volatility Breakout", indicators)

    return Signal("HOLD", 0.12, f"Expansion present (exp={expansion:.2f}) but no confirmed range break", "Volatility Breakout", indicators)


# =============================================================================
# Public API used by the bot loop
# =============================================================================

# Strategy registry — extend here to add a new strategy without touching the
# evaluate_symbol control flow.
_STRATEGY_REGISTRY = {
    "Momentum": momentum_signal,
    "Mean Reversion": mean_reversion_signal,
    "Scalping": scalping_signal,
    "Trend Following": trend_following_signal,
    "Breakout": breakout_signal,
    "Volatility Breakout": volatility_breakout_signal,
}

# Phase C setups — registered after the base strategies via a late
# import. The setups module imports Signal from THIS module, so a
# top-of-file import here would be a cycle. Doing the import after
# the class + registry are both defined breaks the cycle cleanly.
from trading.setups import vwap_reclaim_signal, opening_range_breakout_signal       # noqa: E402
_STRATEGY_REGISTRY["VWAP Reclaim"] = vwap_reclaim_signal
_STRATEGY_REGISTRY["ORB"] = opening_range_breakout_signal


def evaluate_entry_quality(candles: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """Accept/reject gate on where price sits vs market structure, plus a
    quality_score in [0.65, 1.20] the caller uses as a confidence multiplier.
    (Return-key contract unchanged from v1.)"""
    if not candles or len(candles) < 30:
        return {"accept": True, "reason": "insufficient bars for S/R check",
                "headroom_pct": 0.0, "risk_pct": 0.0, "potential_rr": 0.0, "quality_score": 1.0}

    swings = swing_levels(candles, lookback=30)
    if not swings:
        return {"accept": True, "reason": "no swings detected",
                "headroom_pct": 0.0, "risk_pct": 0.0, "potential_rr": 0.0, "quality_score": 1.0}

    last = float(candles[-1]["close"])
    a = atr(candles, period=14) or 0.0
    atr_pct = (a / last) if last > 0 else 0.0
    swing_high = swings["swing_high"]
    swing_low = swings["swing_low"]

    if side == "BUY":
        if last >= swing_high:
            headroom = max(atr_pct * 3.0, 0.02)
            risk = ((last - swing_low) / last + atr_pct * 0.2) if last > 0 else 0.0
        else:
            headroom = (swing_high - last) / last if last > 0 else 0.0
            risk = ((last - swing_low) / last + atr_pct * 0.2) if last > 0 else 0.0
        min_headroom = max(0.008, atr_pct * 1.0)
        if headroom < min_headroom:
            return {"accept": False,
                    "reason": f"BUY rejected: only {headroom:.2%} headroom to swing high ${swing_high:.4f} (need {min_headroom:.2%})",
                    "headroom_pct": headroom, "risk_pct": risk,
                    "potential_rr": (headroom / risk) if risk > 0 else 0.0, "quality_score": 0.0,
                    "swing_high": swing_high, "swing_low": swing_low, "atr_pct": atr_pct}
    elif side == "SELL":
        if last <= swing_low:
            headroom = max(atr_pct * 3.0, 0.02)
            risk = ((swing_high - last) / last + atr_pct * 0.2) if last > 0 else 0.0
        else:
            headroom = (last - swing_low) / last if last > 0 else 0.0
            risk = ((swing_high - last) / last + atr_pct * 0.2) if last > 0 else 0.0
        min_headroom = max(0.008, atr_pct * 1.0)
        if headroom < min_headroom:
            return {"accept": False,
                    "reason": f"SELL rejected: only {headroom:.2%} headroom to swing low ${swing_low:.4f} (need {min_headroom:.2%})",
                    "headroom_pct": headroom, "risk_pct": risk,
                    "potential_rr": (headroom / risk) if risk > 0 else 0.0, "quality_score": 0.0,
                    "swing_high": swing_high, "swing_low": swing_low, "atr_pct": atr_pct}
    else:
        return {"accept": True, "reason": "non-directional side",
                "headroom_pct": 0.0, "risk_pct": 0.0, "potential_rr": 0.0, "quality_score": 1.0}

    potential_rr = (headroom / risk) if risk > 0 else 0.0
    headroom_atrs = (headroom / atr_pct) if atr_pct > 0 else 0.0

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

    if headroom_atrs >= 4.0:
        room_factor = 1.10
    elif headroom_atrs >= 2.0:
        room_factor = 1.00 + (headroom_atrs - 2.0) * 0.05
    elif headroom_atrs >= 1.0:
        room_factor = 0.80 + (headroom_atrs - 1.0) * 0.20
    else:
        room_factor = 0.70

    quality_score = max(0.65, min(1.20, rr_factor * room_factor))
    return {"accept": True,
            "reason": f"S/R OK: headroom={headroom:.2%} ({headroom_atrs:.1f}xATR), risk={risk:.2%}, potential R:R={potential_rr:.1f}, q={quality_score:.2f}",
            "headroom_pct": headroom, "risk_pct": risk, "potential_rr": potential_rr,
            "quality_score": quality_score, "swing_high": swing_high, "swing_low": swing_low, "atr_pct": atr_pct}


def _granularity_for_tick(tick_seconds: int | None) -> int:
    if tick_seconds is None or tick_seconds <= 0:
        return 300
    if tick_seconds <= 10:
        return 60
    if tick_seconds <= 60:
        return 300
    return 900


def get_entry_candles(product_id: str, *, tick_seconds: int | None = None, lookback_bars: int = 100) -> list[dict[str, Any]]:
    """Fetch candles at the same granularity evaluate_symbol uses, so the bot can
    run S/R + stop calculations on the same data the signal saw."""
    granularity = _granularity_for_tick(tick_seconds) if tick_seconds is not None else 300
    return get_candles(product_id, granularity=granularity, limit=lookback_bars)


def _higher_tf_bias(product_id: str, base_granularity: int, lookback_bars: int = 120) -> str:
    """Lightweight higher-timeframe trend bias (BULLISH/BEARISH/NEUTRAL) used by
    the optional mtf_confirm gate. Returns NEUTRAL on any data shortfall."""
    try:
        htf_gran = base_granularity * MTF_GRANULARITY_MULT
        candles = get_candles(product_id, granularity=htf_gran, limit=lookback_bars)
        closes = _closes(candles)
        if len(closes) < 55:
            return "NEUTRAL"
        ema_fast_v = ema(closes, 21)[-1]
        ema_slow_v = ema(closes, 55)[-1]
        last = closes[-1]
        if last > ema_fast_v > ema_slow_v:
            return "BULLISH"
        if last < ema_fast_v < ema_slow_v:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def evaluate_symbol(
    product_id: str,
    strategy_type: str,
    *,
    granularity: int | None = None,
    lookback_bars: int = 200,
    tick_seconds: int | None = None,
    mtf_confirm: bool = False,
    vol_gate: bool = False,
) -> Signal:
    """Pull candles for `product_id` and compute the signal for `strategy_type`.

    Signature is backward-compatible: bot_engine calls
    ``evaluate_symbol(symbol, strategy_type, tick_seconds=...)`` and gets exactly
    the prior behavior plus a RICHER indicator dict. Two opt-in refinements
    (default OFF, so current behavior is unchanged unless enabled):

      mtf_confirm: confirm the entry against a higher timeframe; dampen confidence
                   on conflict, small boost on agreement. Costs one extra fetch.
      vol_gate:    dampen confidence during extreme volatility expansion
                   (atr_expansion >= VOL_GATE_EXTREME).

    Falls back to HOLD on empty candles or unsupported strategy.
    """
    if granularity is None:
        granularity = _granularity_for_tick(tick_seconds)

    candles = get_candles(product_id, granularity=granularity, limit=lookback_bars)
    if not candles or len(candles) < 30:
        last_price = float(candles[-1]["close"]) if candles else 0.0
        return Signal("HOLD", 0.0,
                      f"only {len(candles)} bars at {granularity}s — insufficient history",
                      strategy_type or "Momentum",
                      {"bars": len(candles), "granularity_s": granularity, "last_price": last_price})

    st = (strategy_type or "Momentum").strip()

    # Resolve the strategy (registry + the two derived variants).
    if st in _STRATEGY_REGISTRY:
        signal = _STRATEGY_REGISTRY[st](candles)
    elif st == "Probability Edge":
        # Stricter momentum: only acts on high-conviction momentum.
        sig = momentum_signal(candles)
        if sig.confidence < 0.7:
            signal = Signal("HOLD", 0.0, "no probability edge", "Probability Edge", sig.indicators)
        else:
            signal = Signal(sig.side, sig.confidence, sig.reasoning, "Probability Edge", sig.indicators)
    else:
        signal = momentum_signal(candles)  # unknown strategy -> momentum

    # ----- Enrich indicators (additive; pairs with the prompt fix) --------- #
    common = _common_indicators(candles)
    signal.indicators = {**common, **(signal.indicators or {})}

    # ----- Optional volatility-regime dampener (opt-in) -------------------- #
    if vol_gate and signal.side in {"BUY", "SELL"}:
        exp = signal.indicators.get("atr_expansion")
        if exp is not None and exp >= VOL_GATE_EXTREME:
            before = signal.confidence
            signal.confidence = max(0.0, min(1.0, signal.confidence * VOL_GATE_DAMPEN))
            signal.indicators["vol_gate_dampened"] = 1.0
            signal.reasoning += f" | vol-gate: spike exp={exp:.2f}, conf {before:.2f}->{signal.confidence:.2f}"

    # ----- Optional multi-timeframe confirmation (opt-in) ------------------ #
    if mtf_confirm and signal.side in {"BUY", "SELL"}:
        bias = _higher_tf_bias(product_id, granularity)
        signal.indicators["htf_bias"] = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}[bias]
        agrees = (signal.side == "BUY" and bias == "BULLISH") or (signal.side == "SELL" and bias == "BEARISH")
        conflicts = (signal.side == "BUY" and bias == "BEARISH") or (signal.side == "SELL" and bias == "BULLISH")
        if agrees:
            signal.confidence = min(1.0, signal.confidence * MTF_AGREE_BOOST)
            signal.reasoning += f" | HTF {bias} agrees"
        elif conflicts:
            signal.confidence = max(0.0, signal.confidence * MTF_CONFLICT_DAMPEN)
            signal.reasoning += f" | HTF {bias} conflicts"

    return signal


# =============================================================================
# Stops / structure
# =============================================================================

def stop_take_levels(candles: list[dict[str, Any]], side: str, *, atr_period: int = 14, stop_atr_mult: float = 1.5, take_atr_mult: float = 3.0) -> dict[str, float]:
    """Volatility-aware SL/TP prices from ATR. Returns {} if insufficient data."""
    if not candles:
        return {}
    a = atr(candles, period=atr_period)
    if a is None or a <= 0:
        return {}
    last = float(candles[-1]["close"])
    if side == "BUY":
        return {"stop_loss": last - stop_atr_mult * a, "take_profit": last + take_atr_mult * a, "atr": a}
    if side == "SELL":
        return {"stop_loss": last + stop_atr_mult * a, "take_profit": last - take_atr_mult * a, "atr": a}
    return {}


def swing_levels(candles: list[dict[str, Any]], lookback: int = 30) -> dict[str, float]:
    """Relevant swing high / low over `lookback` bars (5-bar pivots, falling back
    to the window extreme). Never returns a level below price as 'resistance'."""
    if len(candles) < 7:
        return {}
    window = candles[-min(lookback, len(candles)):]
    abs_high = max(float(c["high"]) for c in window)
    abs_low = min(float(c["low"]) for c in window)
    last_close = float(window[-1]["close"])

    proximate_high: float | None = None
    proximate_low: float | None = None
    n = len(window)
    for i in range(n - 3, 1, -1):
        h = float(window[i]["high"])
        l = float(window[i]["low"])
        is_swing_high = (
            h > float(window[i - 1]["high"]) and h > float(window[i - 2]["high"])
            and h > float(window[i + 1]["high"]) and h > float(window[i + 2]["high"])
        )
        is_swing_low = (
            l < float(window[i - 1]["low"]) and l < float(window[i - 2]["low"])
            and l < float(window[i + 1]["low"]) and l < float(window[i + 2]["low"])
        )
        if is_swing_high and proximate_high is None and h > last_close:
            proximate_high = h
        if is_swing_low and proximate_low is None and l < last_close:
            proximate_low = l
        if proximate_high is not None and proximate_low is not None:
            break

    swing_high = proximate_high if proximate_high is not None else abs_high
    swing_low = proximate_low if proximate_low is not None else abs_low
    return {"swing_high": swing_high, "swing_low": swing_low}


def smart_stops(candles: list[dict[str, Any]], side: str, confidence: float, *, min_stop_pct: float = 0.012, max_stop_pct: float = 0.06, atr_period: int = 14) -> dict[str, float]:
    """Volatility-AND-structure-aware stops. Stop is the WIDER of 1.2x ATR and a
    swing-anchored level (clamped to [min_stop_pct, max_stop_pct]). Confidence
    widens the take-profit, not the stop. (Return-key contract unchanged.)"""
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
    atr_buffer = a * 0.20

    if side == "BUY":
        structural_stop = swings["swing_low"] - atr_buffer
        atr_stop = last - 1.2 * a
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

    rr = 2.0 + max(0.0, min(1.0, confidence)) * 1.5
    tp_pct = stop_pct * rr

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

    return {"stop_loss": sl_price, "take_profit": tp_price, "stop_pct": stop_pct, "tp_pct": tp_pct,
            "atr": a, "atr_pct": atr_pct, "swing_high": swings["swing_high"], "swing_low": swings["swing_low"], "rr_ratio": rr}


# =============================================================================
# CHANGELOG (v1 -> v2)
# =============================================================================
# BUG FIXES
#   1. adx_indicator() returns a properly Wilder-smoothed ADX (was raw 1-bar DX).
#      trend_following_signal's ADX>25 gate is now a real sustained-trend filter.
#      *** BEHAVIOR CHANGE — backtest: trend-following fires LESS / later. ***
#   2. "Volatility Breakout" rewritten as a real volatility-expansion breakout
#      (volatility_breakout_signal). v1 reused reversal-gated mean reversion and
#      flipped the side, producing incoherent SELLs into bounces.
#      *** BEHAVIOR CHANGE — backtest. ***
#   3. Module docstring corrected (v1 claimed only Momentum + Mean Reversion).
#
# CAPABILITY UPGRADES (additive / opt-in — safe by default)
#   4. _common_indicators(): every Signal from evaluate_symbol now carries a
#      consistent core set (rsi, macd_histogram, adx/+DI/-DI, relative_volume,
#      buying_pressure, atr_pct, atr_expansion, velocity_1/3bar, body_direction).
#      Strategy-specific keys still win on conflict. This directly enriches the
#      context the (now-fixed) decision engine sends to Claude + the adaptive
#      engine. ADDITIVE — does not change any signal's side/confidence.
#   5. _atr_expansion(): cheap volatility-regime proxy (short ATR / long ATR).
#   6. evaluate_symbol(vol_gate=False): opt-in confidence dampener during extreme
#      volatility expansion. OFF by default.
#   7. evaluate_symbol(mtf_confirm=False): opt-in higher-timeframe confirmation
#      (boost on agreement, dampen on conflict). OFF by default; one extra fetch.
#   8. _STRATEGY_REGISTRY: add a strategy by registering a function, no control-
#      flow edits.
#
# CONTRACT PRESERVED
#   - Signal(side, confidence, reasoning, strategy, indicators) + is_actionable().
#   - All public functions kept with the same names and return-dict keys:
#     ema/sma/sma_list/rsi/macd_indicator/bollinger_bands/adx_indicator/
#     volume_analysis/stdev/pct_return/atr, evaluate_symbol, evaluate_entry_quality,
#     get_entry_candles, stop_take_levels, swing_levels, smart_stops.
#   - evaluate_symbol/get_entry_candles signatures are backward-compatible
#     (new params are keyword-only with safe defaults).
#   - All strategy_type strings still handled (incl. Volatility Breakout,
#     Probability Edge, unknown -> Momentum).
#
# NOTE
#   - The granularity passed to get_candles here is an INT (seconds), matching
#     the rest of the live path. advanced_signal_engine passes a STRING enum;
#     that mismatch lives there, not here, and is a batch-2 concern.