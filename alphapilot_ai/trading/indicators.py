"""
Advanced Technical Indicators Library
=====================================
Professional-grade indicators for trading signals.

All functions expect numpy arrays and return numpy arrays.
NaN values at the start are normal due to lookback periods.
"""

from __future__ import annotations
import numpy as np
from typing import TypedDict, Optional
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Core Moving Averages
# -----------------------------------------------------------------------------

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    alpha = 2 / (period + 1)
    result = np.zeros_like(data, dtype=float)
    result[:period-1] = np.nan
    result[period-1] = np.mean(data[:period])
    
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


def wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average (linear weights)."""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    weights = np.arange(1, period + 1)
    result = np.full_like(data, np.nan, dtype=float)
    
    for i in range(period - 1, len(data)):
        result[i] = np.sum(data[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    return result


# -----------------------------------------------------------------------------
# RSI (Relative Strength Index)
# -----------------------------------------------------------------------------

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.
    
    Interpretation:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI 40-60: Neutral zone
    """
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initial average gain/loss
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed averages using Wilder's method
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RSI
    result = np.full_like(close, np.nan, dtype=float)
    for i in range(period, len(close)):
        if avg_loss[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100 - (100 / (1 + rs))
    
    return result


# -----------------------------------------------------------------------------
# MACD (Moving Average Convergence Divergence)
# -----------------------------------------------------------------------------

@dataclass
class MACDResult:
    macd_line: np.ndarray      # MACD line (fast EMA - slow EMA)
    signal_line: np.ndarray    # Signal line (EMA of MACD)
    histogram: np.ndarray      # MACD - Signal (momentum)


def macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACDResult:
    """
    MACD Indicator.
    
    Interpretation:
    - MACD > Signal: Bullish momentum
    - MACD < Signal: Bearish momentum
    - Histogram growing: Momentum strengthening
    - Histogram shrinking: Momentum weakening
    - Zero line crossover: Trend change
    """
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram
    )


# -----------------------------------------------------------------------------
# Bollinger Bands
# -----------------------------------------------------------------------------

@dataclass
class BollingerResult:
    upper: np.ndarray          # Upper band (middle + 2*std)
    middle: np.ndarray         # Middle band (SMA)
    lower: np.ndarray          # Lower band (middle - 2*std)
    bandwidth: np.ndarray      # (upper - lower) / middle
    percent_b: np.ndarray      # (close - lower) / (upper - lower)


def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> BollingerResult:
    """
    Bollinger Bands.
    
    Interpretation:
    - Price near upper band: Overbought / strong uptrend
    - Price near lower band: Oversold / strong downtrend
    - Bandwidth squeeze: Low volatility, breakout incoming
    - Bandwidth expansion: High volatility, trend in progress
    - %B > 1: Price above upper band
    - %B < 0: Price below lower band
    """
    middle = sma(close, period)
    
    # Calculate rolling standard deviation
    std = np.full_like(close, np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1:i + 1], ddof=1)
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    bandwidth = np.where(middle != 0, (upper - lower) / middle, np.nan)
    band_width_raw = upper - lower
    percent_b = np.where(band_width_raw != 0, (close - lower) / band_width_raw, 0.5)
    
    return BollingerResult(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b
    )


# -----------------------------------------------------------------------------
# ATR (Average True Range) - Volatility
# -----------------------------------------------------------------------------

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range - measures volatility.
    
    Higher ATR = more volatile market
    Lower ATR = less volatile / consolidating
    """
    if len(close) < 2:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR is smoothed TR
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) >= period:
        result[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    
    return result


# -----------------------------------------------------------------------------
# Volume Indicators
# -----------------------------------------------------------------------------

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.
    
    Measures buying/selling pressure:
    - Rising OBV: Accumulation (buying pressure)
    - Falling OBV: Distribution (selling pressure)
    - OBV divergence from price: Potential reversal
    """
    if len(close) < 2:
        return volume.copy()
    
    result = np.zeros_like(volume, dtype=float)
    result[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            result[i] = result[i-1] + volume[i]
        elif close[i] < close[i-1]:
            result[i] = result[i-1] - volume[i]
        else:
            result[i] = result[i-1]
    
    return result


def volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume Simple Moving Average for relative volume calculation."""
    return sma(volume, period)


def relative_volume(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Relative Volume (RVOL).
    
    Current volume / average volume.
    - RVOL > 1.5: High volume (confirms moves)
    - RVOL < 0.5: Low volume (weak moves, potential fake breakout)
    """
    avg_vol = sma(volume, period)
    return np.where(avg_vol > 0, volume / avg_vol, 1.0)


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Volume Weighted Average Price.
    
    - Price > VWAP: Bullish intraday
    - Price < VWAP: Bearish intraday
    - Institutional benchmark for fair value
    """
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)
    
    return np.where(cumulative_vol > 0, cumulative_tp_vol / cumulative_vol, typical_price)


# -----------------------------------------------------------------------------
# Stochastic Oscillator
# -----------------------------------------------------------------------------

@dataclass
class StochasticResult:
    k: np.ndarray  # Fast stochastic
    d: np.ndarray  # Slow stochastic (signal line)


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> StochasticResult:
    """
    Stochastic Oscillator.
    
    Interpretation:
    - %K > 80: Overbought
    - %K < 20: Oversold
    - %K crosses above %D: Buy signal
    - %K crosses below %D: Sell signal
    """
    if len(close) < k_period:
        nan_arr = np.full_like(close, np.nan, dtype=float)
        return StochasticResult(k=nan_arr, d=nan_arr)
    
    k = np.full_like(close, np.nan, dtype=float)
    
    for i in range(k_period - 1, len(close)):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        
        if highest_high != lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k[i] = 50.0
    
    d = sma(k, d_period)
    
    return StochasticResult(k=k, d=d)


# -----------------------------------------------------------------------------
# ADX (Average Directional Index) - Trend Strength
# -----------------------------------------------------------------------------

@dataclass
class ADXResult:
    adx: np.ndarray     # Trend strength (0-100)
    plus_di: np.ndarray  # Positive directional indicator
    minus_di: np.ndarray # Negative directional indicator


def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> ADXResult:
    """
    Average Directional Index - measures trend strength.
    
    Interpretation:
    - ADX > 25: Strong trend
    - ADX < 20: Weak/no trend (ranging market)
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend
    """
    if len(close) < period + 1:
        nan_arr = np.full_like(close, np.nan, dtype=float)
        return ADXResult(adx=nan_arr, plus_di=nan_arr, minus_di=nan_arr)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    tr = np.zeros(len(close))
    
    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth the values
    smoothed_plus_dm = np.zeros(len(close))
    smoothed_minus_dm = np.zeros(len(close))
    smoothed_tr = np.zeros(len(close))
    
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])
    smoothed_tr[period] = np.sum(tr[1:period+1])
    
    for i in range(period + 1, len(close)):
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
    
    # Calculate +DI and -DI
    plus_di = np.full_like(close, np.nan, dtype=float)
    minus_di = np.full_like(close, np.nan, dtype=float)
    
    for i in range(period, len(close)):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(len(close))
    for i in range(period, len(close)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_result = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 2 * period:
        adx_result[2 * period - 1] = np.mean(dx[period:2*period])
        for i in range(2 * period, len(close)):
            adx_result[i] = (adx_result[i-1] * (period - 1) + dx[i]) / period
    
    return ADXResult(adx=adx_result, plus_di=plus_di, minus_di=minus_di)


# -----------------------------------------------------------------------------
# Support/Resistance Detection
# -----------------------------------------------------------------------------

def pivot_points(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    """
    Calculate pivot points from previous period.
    
    Returns dict with: pivot, r1, r2, r3, s1, s2, s3
    """
    # Use most recent complete period
    h = high[-1] if len(high) > 0 else 0
    l = low[-1] if len(low) > 0 else 0
    c = close[-1] if len(close) > 0 else 0
    
    pivot = (h + l + c) / 3
    
    return {
        "pivot": pivot,
        "r1": 2 * pivot - l,
        "r2": pivot + (h - l),
        "r3": h + 2 * (pivot - l),
        "s1": 2 * pivot - h,
        "s2": pivot - (h - l),
        "s3": l - 2 * (h - pivot),
    }


# -----------------------------------------------------------------------------
# Comprehensive Indicator Suite
# -----------------------------------------------------------------------------

@dataclass
class IndicatorSuite:
    """All indicators computed for a symbol."""
    # Price
    close: float
    
    # Trend
    ema_12: float
    ema_26: float
    ema_50: float
    ema_200: float
    sma_20: float
    
    # Momentum
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    stoch_k: float
    stoch_d: float
    
    # Volatility
    atr_14: float
    atr_pct: float  # ATR as % of price
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_bandwidth: float
    bb_percent_b: float
    
    # Volume
    volume: float
    volume_sma_20: float
    relative_volume: float
    obv: float
    vwap: float
    
    # Trend Strength
    adx: float
    plus_di: float
    minus_di: float
    
    # Derived Signals
    trend_direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    trend_strength: str   # "STRONG", "MODERATE", "WEAK", "NONE"
    momentum_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    volatility_state: str # "HIGH", "NORMAL", "LOW"
    volume_confirmation: bool


def compute_all_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> Optional[IndicatorSuite]:
    """
    Compute all indicators for the given OHLCV data.
    Returns None if insufficient data.
    """
    if len(close) < 50:
        return None
    
    # Moving Averages
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200) if len(close) >= 200 else np.full_like(close, np.nan)
    sma_20 = sma(close, 20)
    
    # Momentum
    rsi_14 = rsi(close, 14)
    macd_result = macd(close)
    stoch_result = stochastic(high, low, close)
    
    # Volatility
    atr_14 = atr(high, low, close, 14)
    bb_result = bollinger_bands(close)
    
    # Volume
    vol_sma = volume_sma(volume, 20)
    rvol = relative_volume(volume, 20)
    obv_line = obv(close, volume)
    vwap_line = vwap(high, low, close, volume)
    
    # Trend Strength
    adx_result = adx(high, low, close)
    
    # Get latest values
    latest = len(close) - 1
    current_close = close[latest]
    
    # Determine trend direction
    ema_12_val = ema_12[latest]
    ema_26_val = ema_26[latest]
    ema_50_val = ema_50[latest]
    
    if current_close > ema_12_val > ema_26_val > ema_50_val:
        trend_direction = "BULLISH"
    elif current_close < ema_12_val < ema_26_val < ema_50_val:
        trend_direction = "BEARISH"
    else:
        trend_direction = "NEUTRAL"
    
    # Determine trend strength from ADX
    adx_val = adx_result.adx[latest] if not np.isnan(adx_result.adx[latest]) else 0
    if adx_val > 40:
        trend_strength = "STRONG"
    elif adx_val > 25:
        trend_strength = "MODERATE"
    elif adx_val > 15:
        trend_strength = "WEAK"
    else:
        trend_strength = "NONE"
    
    # Momentum signal from RSI + MACD
    rsi_val = rsi_14[latest]
    macd_hist = macd_result.histogram[latest]
    
    if rsi_val > 50 and macd_hist > 0:
        momentum_signal = "BULLISH"
    elif rsi_val < 50 and macd_hist < 0:
        momentum_signal = "BEARISH"
    else:
        momentum_signal = "NEUTRAL"
    
    # Volatility state from bandwidth
    bw = bb_result.bandwidth[latest] if not np.isnan(bb_result.bandwidth[latest]) else 0
    avg_bw = np.nanmean(bb_result.bandwidth[-50:])
    
    if bw > avg_bw * 1.5:
        volatility_state = "HIGH"
    elif bw < avg_bw * 0.5:
        volatility_state = "LOW"
    else:
        volatility_state = "NORMAL"
    
    # Volume confirmation
    rvol_val = rvol[latest] if not np.isnan(rvol[latest]) else 1.0
    volume_confirmation = rvol_val > 1.2
    
    return IndicatorSuite(
        close=current_close,
        ema_12=ema_12_val,
        ema_26=ema_26_val,
        ema_50=ema_50_val,
        ema_200=ema_200[latest] if not np.isnan(ema_200[latest]) else 0,
        sma_20=sma_20[latest],
        rsi_14=rsi_val,
        macd_line=macd_result.macd_line[latest],
        macd_signal=macd_result.signal_line[latest],
        macd_histogram=macd_hist,
        stoch_k=stoch_result.k[latest],
        stoch_d=stoch_result.d[latest],
        atr_14=atr_14[latest],
        atr_pct=(atr_14[latest] / current_close * 100) if current_close > 0 else 0,
        bb_upper=bb_result.upper[latest],
        bb_middle=bb_result.middle[latest],
        bb_lower=bb_result.lower[latest],
        bb_bandwidth=bw,
        bb_percent_b=bb_result.percent_b[latest],
        volume=volume[latest],
        volume_sma_20=vol_sma[latest],
        relative_volume=rvol_val,
        obv=obv_line[latest],
        vwap=vwap_line[latest],
        adx=adx_val,
        plus_di=adx_result.plus_di[latest],
        minus_di=adx_result.minus_di[latest],
        trend_direction=trend_direction,
        trend_strength=trend_strength,
        momentum_signal=momentum_signal,
        volatility_state=volatility_state,
        volume_confirmation=volume_confirmation,
    )
