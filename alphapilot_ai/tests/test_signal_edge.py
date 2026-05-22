"""Tests for the signal-edge harness (signal_edge.py).

The harness measures whether the live technical signal predicts forward
returns. These tests exercise its pure core (no network): the signed-
return convention, the bar-walk over real signal code, and the verdict
logic that decides EDGE FOUND vs NO EDGE.
"""
from __future__ import annotations

from signal_edge import measure, summarize, _signed_bps, _Sample


def _candle(t: int, close: float, vol: float = 1000.0) -> dict:
    return {"time": t, "open": close, "high": close * 1.002,
            "low": close * 0.998, "close": close, "volume": vol}


def _series(closes: list[float]) -> list[dict]:
    return [_candle(i * 60, c) for i, c in enumerate(closes)]


# --------------------------------------------------------------------------
# Signed-return convention
# --------------------------------------------------------------------------

def test_signed_bps_buy_profits_on_rise():
    assert _signed_bps("BUY", 0.01) == 100.0
    assert _signed_bps("BUY", -0.01) == -100.0


def test_signed_bps_sell_profits_on_fall():
    assert _signed_bps("SELL", -0.01) == 100.0
    assert _signed_bps("SELL", 0.01) == -100.0


# --------------------------------------------------------------------------
# measure() runs the real strategy_engine signal over candles
# --------------------------------------------------------------------------

def test_measure_walks_real_signal_and_records_horizons():
    closes = [100.0 * (1.001 ** i) for i in range(220)]  # smooth uptrend
    samples = measure({"TEST-USD": _series(closes)}, "Momentum",
                      horizons=(5, 15, 30, 60))
    assert samples, "harness produced no samples on a 220-bar series"
    for s in samples:
        assert s.side in ("BUY", "SELL", "HOLD")
        assert set(s.fwd).issubset({5, 15, 30, 60})


def test_measure_skips_series_too_short():
    short = _series([100.0] * 50)  # < warmup + max horizon
    assert measure({"X-USD": short}, "Momentum") == []


def test_measure_rejects_unknown_strategy():
    try:
        measure({}, "NotAStrategy")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# summarize() verdict logic
# --------------------------------------------------------------------------

def test_summarize_reports_edge_when_signal_predicts():
    # BUYs rise ~0.5%, SELLs fall ~0.5% -> ~+50 bps signed (with realistic
    # spread so the t-stat is meaningful), beats a 30 bps cost.
    samples = []
    for i in range(600):
        noise = 0.002 if i % 2 == 0 else -0.002
        samples.append(_Sample("BUY", 0.7, {15: 0.005 + noise}))
        samples.append(_Sample("SELL", 0.7, {15: -0.005 + noise}))
    report = summarize(samples, (15,), cost_bps=30.0)
    assert "EDGE FOUND" in report


def test_summarize_reports_no_edge_on_noise():
    # Forward returns are tiny alternating noise -> mean ~0, net negative
    # after cost.
    samples = []
    for i in range(500):
        r = 0.0001 if i % 2 == 0 else -0.0001
        samples.append(_Sample("BUY", 0.7, {15: r}))
        samples.append(_Sample("SELL", 0.7, {15: r}))
    report = summarize(samples, (15,), cost_bps=30.0)
    assert "NO EDGE" in report


def test_summarize_flags_weak_edge_eaten_by_cost():
    # Real, significant direction (~+10 bps signed) but smaller than the
    # 30 bps round-trip cost.
    samples = []
    for i in range(800):
        noise = 0.002 if i % 2 == 0 else -0.002
        samples.append(_Sample("BUY", 0.7, {15: 0.0010 + noise}))
        samples.append(_Sample("SELL", 0.7, {15: -0.0010 + noise}))
    report = summarize(samples, (15,), cost_bps=30.0)
    assert "WEAK EDGE" in report
