"""Regression: decide-time vs persist-time regime classifier divergence.

Original incident (commit b8e3c1d):
    Two `_derive_regime` functions existed:
      * strategic_claude._derive_regime — used at decide-time when
        building the autonomous-engine TradeContext.
      * claude_decision_engine._derive_regime_hint — used at persist-time
        inside _extract_market_state when writing market_snapshot.
    Their fallback labels DIFFERED (strategic returned "RANGING",
    claude_decision_engine returned "DRIFT_UP" / "DRIFT_DOWN" / "UNKNOWN").
    `regime` is part of TradeContext.to_fingerprint(), so the same trade
    produced different fingerprints at decide-time vs learn-time. The
    autonomous engine's pattern/kNN/mistake tables silently fragmented;
    a pattern "learned" at close could never be re-recognized on entry.

    Fix: strategic_claude._derive_regime delegates to the canonical
    claude_decision_engine._derive_regime_hint. This test pins that
    consistency across a representative spread of indicator dicts.
"""
from __future__ import annotations

import pytest

from ai.claude_decision_engine import _derive_regime_hint
from trading.strategic_claude import _derive_regime


# Curated indicator dicts that exercise every branch in _derive_regime_hint.
# If anyone re-introduces a divergent classifier, at least one of these
# inputs will pin the disagreement.
_INDICATOR_CASES = [
    # name,                indicators dict
    ("trending_up",        {"adx": 30, "plus_di": 25, "minus_di": 10}),
    ("trending_down",      {"adx": 28, "plus_di": 8, "minus_di": 22}),
    ("volatile",           {"vol_pct": 0.06, "adx": 15}),
    ("ranging_low_adx",    {"adx": 12}),
    ("drift_up_fallback",  {"adx": 20, "gap_pct": 0.005, "velocity_3bar": 0.003}),
    ("drift_down_fallback",{"adx": 22, "gap_pct": -0.004, "velocity_3bar": -0.002}),
    ("unknown_empty",      {}),
    ("unknown_neutral",    {"adx": 20, "gap_pct": 0.0001, "velocity_3bar": -0.0001}),
    # Edge cases that previously caused different default behaviours.
    ("adx_borderline",     {"adx": 25, "plus_di": 14, "minus_di": 14}),
    ("vol_borderline",     {"vol_pct": 0.039}),  # just below VOLATILE threshold
]


@pytest.mark.parametrize("name,indicators", _INDICATOR_CASES, ids=[c[0] for c in _INDICATOR_CASES])
def test_decide_time_regime_matches_persist_time(name, indicators):
    """strategic_claude._derive_regime must return the same label as
    claude_decision_engine._derive_regime_hint(...)["regime"] for every
    indicator combination. Drift detection: if anyone re-forks the
    classifier, this test fails on at least one case."""
    decide_label = _derive_regime(indicators)
    persist_label = _derive_regime_hint(indicators).get("regime")
    assert decide_label == persist_label, (
        f"Regime classifier divergence on case {name!r}: "
        f"decide={decide_label!r} vs persist={persist_label!r}. "
        f"This means decide-time fingerprints won't match learn-time fingerprints "
        f"and the autonomous engine cannot recognize patterns it learned."
    )


def test_decide_time_regime_is_a_string():
    """Whatever the regime, the decide-side helper must always return a
    string. The autonomous engine puts the value directly into a JSON
    fingerprint key, so None/dict/int outputs would silently break
    fingerprint hashing across calls."""
    for _, indicators in _INDICATOR_CASES:
        label = _derive_regime(indicators)
        assert isinstance(label, str), f"non-string regime label: {label!r}"
        assert label, "regime label must be non-empty"


def test_fingerprint_consistency_decide_vs_learn():
    """End-to-end consistency check: a TradeContext built at decide-time
    from a signal's indicators must produce the same fingerprint as one
    rebuilt at learn-time from the persisted market_snapshot.

    This is the integration of all the consistency fixes in commit b8e3c1d.
    If decide-side reads `.metadata` instead of `.indicators`, if the regime
    classifiers diverge, or if _extract_market_state writes a key the
    learn-time builder doesn't read, the fingerprints will differ here."""
    from ai.autonomous_learning_engine import TradeContext
    from ai.claude_decision_engine import _extract_market_state

    # Synthesize the kind of indicators dict strategy_engine emits.
    indicators = {
        "rsi": 62.5,
        "rsi_14": 62.5,
        "macd_histogram": 0.018,
        "adx": 27.0,
        "adx_trend_strength": 27.0,
        "plus_di": 22.0,
        "minus_di": 14.0,
        "relative_volume": 1.4,
        "bollinger_percent_b": 0.62,
        "atr_pct": 0.018,
        "vol_pct": 0.018,
    }

    # ---- DECIDE-TIME PATH ----
    # Mirrors strategic_claude._build_autonomous_context (the live path
    # that feeds engine.decide).
    from trading.strategic_claude import _build_autonomous_context

    class _FakeSignal:
        side = "BUY"
        confidence = 0.72
        indicators = {}  # placeholder, overridden per-test
        strategy = "Momentum"
        metadata = {}
    sig = _FakeSignal()
    sig.indicators = indicators

    decide_ctx = _build_autonomous_context(
        symbol="BTC-USD", side="BUY",
        technical_signal=sig, strategy_type="Momentum",
        tech_confidence=0.72,
    )
    assert decide_ctx is not None, "decide-time ctx builder returned None"
    decide_fingerprint = decide_ctx.to_fingerprint()

    # ---- PERSIST-TIME PATH ----
    # _extract_market_state builds the market_snapshot JSON dict.
    snapshot = _extract_market_state(indicators, extra_context={})

    # ---- LEARN-TIME PATH ----
    # Rebuild a TradeContext from the persisted snapshot, the way
    # autonomous_learning_engine._build_context_from_trade does.
    learn_ctx = TradeContext(symbol="BTC-USD", side="BUY")
    learn_ctx.rsi = snapshot.get("rsi", 50)
    learn_ctx.macd_histogram = snapshot.get("macd_histogram", 0)
    learn_ctx.adx = snapshot.get("adx", 25)
    learn_ctx.volume_ratio = snapshot.get("volume_ratio", 1)
    learn_ctx.regime = snapshot.get("regime", "UNKNOWN")
    # hour_utc is set from trade.opened_at at learn-time; the decide-side
    # builder set it from now() — pin both to the same value so the test
    # exercises the FINGERPRINT logic, not clock drift.
    learn_ctx.hour_utc = decide_ctx.hour_utc

    learn_fingerprint = learn_ctx.to_fingerprint()

    assert decide_fingerprint == learn_fingerprint, (
        f"Fingerprint mismatch: decide={decide_fingerprint!r} "
        f"learn={learn_fingerprint!r}. "
        f"decide-ctx rsi/macd/adx/vol/regime = "
        f"{decide_ctx.rsi},{decide_ctx.macd_histogram},{decide_ctx.adx},"
        f"{decide_ctx.volume_ratio},{decide_ctx.regime}. "
        f"learn-ctx rsi/macd/adx/vol/regime = "
        f"{learn_ctx.rsi},{learn_ctx.macd_histogram},{learn_ctx.adx},"
        f"{learn_ctx.volume_ratio},{learn_ctx.regime}. "
        f"snapshot = {snapshot}"
    )
