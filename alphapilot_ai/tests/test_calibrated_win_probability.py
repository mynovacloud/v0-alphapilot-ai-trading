"""Phase B: three-tier calibrated win-probability estimator.

These tests pin the new public API in autonomous_learning_engine:
  - get_pattern_stats(fingerprint) -> dict | None
  - get_calibrated_win_probability(context, fallback_confidence) -> dict

Estimator resolution order:
  1. exact_pattern    - exact fingerprint match, n >= MIN_EXACT_PATTERN_TRADES
  2. knn_neighbors    - kNN similar trades, n >= MIN_KNN_NEIGHBORS
  3. raw_confidence   - no historical data; return the caller's confidence

The original heuristic ("prob_win ≈ confidence") was the #1 piece of
structural debt in the project. This test file is the regression guard
against accidentally re-introducing it — every tier has a test that says
"in this situation, the estimator should NOT just be returning confidence".
"""
from __future__ import annotations

import pytest

from ai.autonomous_learning_engine import (
    AutonomousLearningEngine,
    LearnedPattern,
    MIN_EXACT_PATTERN_TRADES,
    MIN_KNN_NEIGHBORS,
    TradeContext,
    get_calibrated_win_probability,
    get_pattern_stats,
    get_autonomous_engine,
)


@pytest.fixture(autouse=True)
def _isolated_engine(monkeypatch):
    """Replace the module-level singleton with a fresh per-test engine.

    Without this, tests would share the on-disk persistence and step on
    each other. We construct via __new__ so __init__'s persistence load
    is skipped — every test gets a clean blank-slate engine."""
    eng = AutonomousLearningEngine.__new__(AutonomousLearningEngine)
    eng._loaded = True
    eng._patterns = {}
    eng._mistakes = {}
    eng._symbols = {}
    eng._trade_vectors = []
    eng._recent_trades = []
    monkeypatch.setattr("ai.autonomous_learning_engine._engine", eng)
    yield eng


def _pattern_with(n: int, win_rate: float, avg_win: float = 0.02, avg_loss: float = 0.01) -> LearnedPattern:
    """Build a LearnedPattern with known stats. We set the fields directly
    rather than calling .update() N times — faster and the exact stat
    values are what we want to assert against, not the update math."""
    p = LearnedPattern(fingerprint="test_fp", side="BUY")
    p.total_trades = n
    p.winning_trades = int(round(n * win_rate))
    p.win_rate = win_rate
    p.avg_win = avg_win
    p.avg_loss = avg_loss
    p.expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return p


# =============================================================================
# get_pattern_stats — exact-pattern tier
# =============================================================================

def test_get_pattern_stats_returns_none_when_no_pattern(_isolated_engine):
    """No data on this fingerprint -> None. Caller falls back to next tier."""
    assert get_pattern_stats("never_seen_fp") is None


def test_get_pattern_stats_returns_none_below_min_samples(_isolated_engine):
    """Even an existing pattern with insufficient samples must return None.
    Below MIN_EXACT_PATTERN_TRADES the measured win_rate is too noisy to
    be a better estimator than confidence — the whole point of the
    threshold is to avoid replacing a guess with a noisier guess."""
    p = _pattern_with(n=MIN_EXACT_PATTERN_TRADES - 1, win_rate=0.7)
    _isolated_engine._patterns["fp_too_few"] = p
    assert get_pattern_stats("fp_too_few") is None


def test_get_pattern_stats_returns_full_dict_at_threshold(_isolated_engine):
    """At exactly MIN_EXACT_PATTERN_TRADES we cross the threshold and
    return the measured stats."""
    p = _pattern_with(
        n=MIN_EXACT_PATTERN_TRADES, win_rate=0.6, avg_win=0.03, avg_loss=0.015,
    )
    _isolated_engine._patterns["fp_ok"] = p

    stats = get_pattern_stats("fp_ok")
    assert stats is not None
    assert stats["sample_size"] == MIN_EXACT_PATTERN_TRADES
    assert stats["win_rate"] == pytest.approx(0.6)
    assert stats["avg_win"] == pytest.approx(0.03)
    assert stats["avg_loss"] == pytest.approx(0.015)
    # expectancy = win_rate*avg_win - loss_rate*avg_loss
    assert stats["expectancy"] == pytest.approx(0.6 * 0.03 - 0.4 * 0.015, abs=1e-6)


# =============================================================================
# get_calibrated_win_probability — full three-tier resolution
# =============================================================================

def test_calibrated_falls_back_to_raw_confidence_with_no_data(_isolated_engine):
    """No exact pattern, no neighbors -> raw_confidence with sample_size=0
    and meta-confidence=0. Caller's blend ends up returning raw confidence."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY", signal_confidence=0.65)
    result = get_calibrated_win_probability(ctx, fallback_confidence=0.65)

    assert result["source"] == "raw_confidence"
    assert result["sample_size"] == 0
    assert result["confidence_in_estimate"] == 0.0
    assert result["win_probability"] == pytest.approx(0.65)


def test_calibrated_uses_exact_pattern_when_available(_isolated_engine):
    """When exact fingerprint has enough samples, return its win_rate
    REGARDLESS of the caller's confidence. The whole point of the
    calibration: ignore the guess when we have a measurement."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY", signal_confidence=0.95)
    fingerprint = ctx.to_fingerprint()

    # Pattern says 35% win rate over 12 trades — well below confidence=0.95.
    _isolated_engine._patterns[fingerprint] = _pattern_with(n=12, win_rate=0.35)

    result = get_calibrated_win_probability(ctx, fallback_confidence=0.95)
    assert result["source"] == "exact_pattern"
    assert result["sample_size"] == 12
    # Estimator returns the MEASURED win rate, not the confidence guess.
    assert result["win_probability"] == pytest.approx(0.35)
    # Meta-confidence grows with sample size but is < 1 at n=12.
    assert 0.5 < result["confidence_in_estimate"] < 0.7


def test_calibrated_meta_confidence_grows_with_sample_size(_isolated_engine):
    """The confidence_in_estimate must rise monotonically with n. At n=5
    we shouldn't trust the measurement much; at n=25 we should trust it a
    lot. This shape is what lets callers BLEND with raw confidence
    progressively as data accumulates."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY")
    fingerprint = ctx.to_fingerprint()

    meta_confs = []
    for n in [5, 10, 25, 100]:
        _isolated_engine._patterns[fingerprint] = _pattern_with(n=n, win_rate=0.6)
        r = get_calibrated_win_probability(ctx, fallback_confidence=0.6)
        meta_confs.append(r["confidence_in_estimate"])

    # Strictly increasing.
    for a, b in zip(meta_confs, meta_confs[1:]):
        assert b > a, f"meta_confidence should grow with n, got {meta_confs}"
    # Tight bounds at the extremes.
    assert meta_confs[0] < 0.5, "5 samples should not be trusted heavily"
    assert meta_confs[-1] > 0.9, "100 samples should be trusted nearly fully"


def test_calibrated_uses_knn_when_no_exact_pattern(_isolated_engine):
    """No exact-pattern match, but neighbors exist -> knn_neighbors source."""
    # Seed kNN vectors with a clear win-rate signal (8/10 wins). The exact
    # fingerprint for the context below won't match any of these because
    # we don't store it in _patterns — kNN is the fallback that fires.
    for pnl in [0.01, 0.02, 0.015, -0.008, 0.012, 0.018, -0.005, 0.022, 0.011, 0.007]:
        _isolated_engine._trade_vectors.append(([0.0] * 16, pnl, "some_other_fp"))

    ctx = TradeContext(symbol="ETH-USD", side="BUY", signal_confidence=0.5)
    result = get_calibrated_win_probability(ctx, fallback_confidence=0.5)

    assert result["source"] == "knn_neighbors"
    assert result["sample_size"] == 10
    # 8 of 10 positives -> ~0.8 win rate from kNN.
    assert result["win_probability"] == pytest.approx(0.8, abs=0.01)


def test_calibrated_skips_knn_below_min_neighbors(_isolated_engine):
    """If fewer than MIN_KNN_NEIGHBORS similar trades exist, kNN doesn't
    fire — we fall through to raw_confidence rather than basing the
    estimate on too small a sample."""
    # Only 2 neighbors, below MIN_KNN_NEIGHBORS=5.
    for pnl in [0.01, -0.02]:
        _isolated_engine._trade_vectors.append(([0.0] * 16, pnl, "fp"))

    ctx = TradeContext(symbol="XRP-USD", side="BUY")
    result = get_calibrated_win_probability(ctx, fallback_confidence=0.55)
    assert result["source"] == "raw_confidence"
    assert result["win_probability"] == pytest.approx(0.55)


def test_calibrated_prefers_exact_pattern_over_knn(_isolated_engine):
    """Both tiers have enough data -> exact_pattern wins. It's the more
    specific estimator (same fingerprint, not just similar)."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY")
    fp = ctx.to_fingerprint()

    # Seed kNN with a 0.9 win-rate signal.
    for _ in range(10):
        _isolated_engine._trade_vectors.append(([0.0] * 16, 0.01, "other"))
    # Seed an exact pattern at 0.4 win rate.
    _isolated_engine._patterns[fp] = _pattern_with(n=8, win_rate=0.4)

    result = get_calibrated_win_probability(ctx, fallback_confidence=0.95)
    assert result["source"] == "exact_pattern", (
        "exact_pattern must take precedence over kNN when both have data"
    )
    assert result["win_probability"] == pytest.approx(0.4)


def test_calibrated_never_raises_on_broken_context(_isolated_engine):
    """If anything in the lookup chain throws, the estimator must still
    return a sane raw_confidence dict. Calibration is best-effort — a
    broken estimator must not block trades."""

    class _ExplodingContext:
        def to_fingerprint(self):
            raise RuntimeError("context broken")

    result = get_calibrated_win_probability(
        _ExplodingContext(),  # type: ignore[arg-type]
        fallback_confidence=0.7,
    )
    assert result["source"] == "raw_confidence"
    assert result["win_probability"] == pytest.approx(0.7)


def test_calibrated_clamps_fallback_confidence(_isolated_engine):
    """A bad caller passing 1.5 or -0.3 as confidence should still get a
    valid 0..1 probability back. Defensive against upstream bugs."""
    ctx = TradeContext(symbol="BTC-USD", side="BUY")

    over = get_calibrated_win_probability(ctx, fallback_confidence=1.5)
    assert 0.0 <= over["win_probability"] <= 1.0

    under = get_calibrated_win_probability(ctx, fallback_confidence=-0.3)
    assert 0.0 <= under["win_probability"] <= 1.0
