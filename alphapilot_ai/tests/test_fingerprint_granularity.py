"""Regression: fingerprint granularity. The OLD 7-feature fingerprint
collapsed to `pattern=1tr` on every closed trade in a paper session —
patterns never accumulated, so Phase B's exact-pattern calibration tier
never fired. The new 5-feature fingerprint should produce meaningful
recurrence over a session.

Original incident: 200+ trades, zero patterns with sample_size >= 2.
After the coarsening to (side, regime, rsi/20-pt, macd_sign, adx_bucket)
the fingerprint space drops from ~3,600 cells to ~240. With a few
hundred trades, patterns should recur enough that the exact-pattern
calibration tier engages.

These tests are empirical/statistical — they probe the fingerprint
function with a synthetic distribution of contexts that approximates a
session's flow and assert the resulting fingerprint distribution has
acceptable density.
"""
from __future__ import annotations

import random

from ai.autonomous_learning_engine import TradeContext


def _build_realistic_context(rng: random.Random) -> TradeContext:
    """Synthesize a TradeContext from a distribution that approximates
    the in-the-wild distribution we see in a paper session.

    These shapes are deliberately chosen from observed log data:
      - sides split ~roughly 50/50 BUY vs SELL
      - rsi clusters around 30-70 (the active-signal band)
      - macd_histogram is ~50% positive
      - adx clusters around 15-35
      - regime mostly RANGING / TRENDING_UP / TRENDING_DOWN / VOLATILE
    """
    ctx = TradeContext(
        symbol="BTC-USD",  # symbol isn't in the fingerprint; doesn't matter
        side=rng.choice(["BUY", "SELL"]),
    )
    ctx.rsi = rng.uniform(20.0, 80.0)  # active band
    ctx.macd_histogram = rng.uniform(-0.05, 0.05)
    ctx.adx = rng.uniform(10.0, 40.0)
    ctx.regime = rng.choice(["RANGING", "TRENDING_UP", "TRENDING_DOWN", "VOLATILE"])
    # Dimensions that USED to be in the fingerprint and are now dropped
    # are still set on the context (other code reads them); the test just
    # verifies they no longer affect fingerprint identity.
    ctx.volume_ratio = rng.uniform(0.4, 2.0)
    ctx.hour_utc = rng.randint(0, 23)
    return ctx


def test_fingerprint_space_is_meaningfully_smaller_than_old():
    """With 200 plausible random contexts, the new fingerprint should
    produce significantly fewer unique values than the old one would
    have. We don't pin an exact number (the distribution depends on the
    RNG seed and the realistic-context approximation), but the count
    needs to be well below "everything is unique"."""
    rng = random.Random(42)
    unique = set()
    n_samples = 200
    for _ in range(n_samples):
        ctx = _build_realistic_context(rng)
        unique.add(ctx.to_fingerprint())
    # If fingerprints are still essentially-unique-per-trade, the
    # coarsening didn't take. Hard cap: 200 samples should land in
    # significantly fewer than 200 distinct cells.
    assert len(unique) < n_samples * 0.4, (
        f"fingerprint space too large: {n_samples} contexts produced "
        f"{len(unique)} unique fingerprints (expected < {n_samples * 0.4:.0f}). "
        f"Phase B's exact-pattern calibration tier can't fire if "
        f"patterns rarely recur."
    )
    # Sanity: also not absurdly small (everything collapsing to one cell
    # would mean the coarsening went TOO far).
    assert len(unique) > 10, (
        f"fingerprint space too small: only {len(unique)} unique cells "
        f"across 200 contexts. The coarsening may have dropped a feature "
        f"that carries real signal."
    )


def test_fingerprint_is_stable_for_same_inputs():
    """Calling to_fingerprint() repeatedly on the same context must
    return the same hash. Without this we'd have a hidden non-determinism
    that fragments patterns invisibly."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY")
    ctx.rsi = 55.0
    ctx.macd_histogram = 0.01
    ctx.adx = 28.0
    ctx.regime = "TRENDING_UP"

    fp1 = ctx.to_fingerprint()
    fp2 = ctx.to_fingerprint()
    fp3 = ctx.to_fingerprint()
    assert fp1 == fp2 == fp3


def test_fingerprint_ignores_volume_ratio():
    """volume_ratio used to be in the fingerprint. With the coarsening
    it isn't. Same trade across different volume regimes should now
    bucket the same — this is what enables pattern accumulation."""
    ctx_a = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_a.rsi = 55.0; ctx_a.macd_histogram = 0.01
    ctx_a.adx = 28.0; ctx_a.regime = "TRENDING_UP"
    ctx_a.volume_ratio = 0.5

    ctx_b = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_b.rsi = 55.0; ctx_b.macd_histogram = 0.01
    ctx_b.adx = 28.0; ctx_b.regime = "TRENDING_UP"
    ctx_b.volume_ratio = 2.0

    assert ctx_a.to_fingerprint() == ctx_b.to_fingerprint(), (
        "volume_ratio should no longer fragment the fingerprint"
    )


def test_fingerprint_ignores_hour_utc():
    """hour_utc used to be in the fingerprint. With the coarsening it
    isn't. Same trade in different sessions (Asian vs US hours) should
    now bucket the same."""
    ctx_a = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_a.rsi = 55.0; ctx_a.macd_histogram = 0.01
    ctx_a.adx = 28.0; ctx_a.regime = "TRENDING_UP"
    ctx_a.hour_utc = 3   # Asian

    ctx_b = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_b.rsi = 55.0; ctx_b.macd_histogram = 0.01
    ctx_b.adx = 28.0; ctx_b.regime = "TRENDING_UP"
    ctx_b.hour_utc = 18  # US

    assert ctx_a.to_fingerprint() == ctx_b.to_fingerprint()


def test_fingerprint_still_distinguishes_sides():
    """side IS still in the fingerprint. BUY and SELL of an otherwise
    identical setup must hash differently — they're opposite trades."""
    ctx_buy = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_buy.rsi = 55.0; ctx_buy.macd_histogram = 0.01
    ctx_buy.adx = 28.0; ctx_buy.regime = "TRENDING_UP"

    ctx_sell = TradeContext(symbol="BTC-USD", side="SELL")
    ctx_sell.rsi = 55.0; ctx_sell.macd_histogram = 0.01
    ctx_sell.adx = 28.0; ctx_sell.regime = "TRENDING_UP"

    assert ctx_buy.to_fingerprint() != ctx_sell.to_fingerprint()


def test_fingerprint_still_distinguishes_regimes():
    """regime IS still in the fingerprint. Same indicators in
    TRENDING_UP vs RANGING are NOT the same setup."""
    ctx_trend = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_trend.rsi = 55.0; ctx_trend.macd_histogram = 0.01
    ctx_trend.adx = 28.0; ctx_trend.regime = "TRENDING_UP"

    ctx_range = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_range.rsi = 55.0; ctx_range.macd_histogram = 0.01
    ctx_range.adx = 28.0; ctx_range.regime = "RANGING"

    assert ctx_trend.to_fingerprint() != ctx_range.to_fingerprint()


def test_fingerprint_widened_rsi_buckets():
    """RSI moved from 10-pt to 20-pt buckets. Two RSI values that USED
    to fragment (e.g. 41 vs 49 fell in different 10-pt buckets) should
    now share a fingerprint under 20-pt bucketing — both round to 40.

    NOTE: boundary cases still exist at 50/51 (round() goes 2.5→2 for
    banker's rounding but 50/20=2.5 → 2*20=40, 51/20=2.55 → 3*20=60).
    The point isn't to eliminate fragmentation, just to substantially
    reduce it. This test pins one concrete improvement."""
    ctx_41 = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_41.rsi = 41.0; ctx_41.macd_histogram = 0.01
    ctx_41.adx = 28.0; ctx_41.regime = "TRENDING_UP"

    ctx_49 = TradeContext(symbol="BTC-USD", side="BUY")
    ctx_49.rsi = 49.0; ctx_49.macd_histogram = 0.01
    ctx_49.adx = 28.0; ctx_49.regime = "TRENDING_UP"

    # Under OLD 10-pt buckets: 41→40, 49→50 (DIFFERENT cells)
    # Under NEW 20-pt buckets: 41→40, 49→40 (SAME cell)
    assert ctx_41.to_fingerprint() == ctx_49.to_fingerprint(), (
        "RSI 41 and 49 should bucket together under 20-pt rounding"
    )
