"""Phase B integration: _compute_mission_inputs must use the calibrated
win probability instead of raw confidence when historical data exists.

This is the test that proves the wiring works end-to-end. The new fields
in the returned dict (win_probability_source, win_probability_sample_size,
win_probability_used) record which tier of the estimator was used so the
operator can audit decisions after the fact.
"""
from __future__ import annotations

import pytest

from ai.autonomous_learning_engine import (
    AutonomousLearningEngine,
    LearnedPattern,
    TradeContext,
)
from trading.bot_engine import _compute_mission_inputs


@pytest.fixture(autouse=True)
def _isolated_engine(monkeypatch):
    """Same shape as test_calibrated_win_probability's fixture. Mission-
    inputs reaches into the engine through get_calibrated_win_probability
    so we need the same per-test isolation."""
    eng = AutonomousLearningEngine.__new__(AutonomousLearningEngine)
    eng._loaded = True
    eng._patterns = {}
    eng._mistakes = {}
    eng._symbols = {}
    eng._trade_vectors = []
    eng._recent_trades = []
    monkeypatch.setattr("ai.autonomous_learning_engine._engine", eng)
    yield eng


class _FakeSignal:
    side = "BUY"
    confidence = 0.7
    strategy = "Momentum"
    indicators = {
        "rsi": 60.0, "rsi_14": 60.0,
        "macd_histogram": 0.01,
        "adx": 22.0,
        "relative_volume": 1.2,
        "atr_pct": 0.02,
    }


class _FakeDecision:
    stop_loss_pct = 0.02
    take_profit_pct = 0.04
    source = "technical"


def test_mission_inputs_falls_back_to_raw_confidence_with_no_data(_isolated_engine):
    """No historical patterns -> source is raw_confidence, sample_size 0.
    This preserves pre-Phase-B behavior on cold-start days."""
    out = _compute_mission_inputs(
        signal=_FakeSignal(),
        decision=_FakeDecision(),
        confidence=0.7,
        proposed_notional=200.0,
        symbol="BTC-USD",
        side="BUY",
        strategy_type="Momentum",
    )
    assert out["win_probability_source"] == "raw_confidence"
    assert out["win_probability_sample_size"] == 0
    # With raw_confidence, win_probability_used == confidence (no blend).
    assert out["win_probability_used"] == pytest.approx(0.7)


def test_mission_inputs_picks_up_exact_pattern_data(_isolated_engine):
    """When the autonomous engine has accumulated stats on the current
    fingerprint, _compute_mission_inputs must use them. This is the
    point of Phase B — replace the heuristic with measurement."""
    # Build the same context strategic_claude would build and seed an
    # exact-pattern stat under its fingerprint.
    from trading.strategic_claude import _build_autonomous_context

    sig = _FakeSignal()
    ctx = _build_autonomous_context(
        symbol="BTC-USD", side="BUY",
        technical_signal=sig, strategy_type="Momentum",
        tech_confidence=0.7,
    )
    assert ctx is not None
    fp = ctx.to_fingerprint()

    # Seed a pattern with measured low win rate to make the difference
    # from raw confidence (0.7) obvious in the assertions.
    pat = LearnedPattern(fingerprint=fp, side="BUY")
    pat.total_trades = 15
    pat.winning_trades = 6
    pat.win_rate = 0.4
    pat.avg_win = 0.025
    pat.avg_loss = 0.012
    pat.expectancy = 0.4 * 0.025 - 0.6 * 0.012
    _isolated_engine._patterns[fp] = pat

    out = _compute_mission_inputs(
        signal=sig,
        decision=_FakeDecision(),
        confidence=0.7,
        proposed_notional=200.0,
        symbol="BTC-USD",
        side="BUY",
        strategy_type="Momentum",
    )

    assert out["win_probability_source"] == "exact_pattern"
    assert out["win_probability_sample_size"] == 15
    # The blended probability lands between measured (0.4) and confidence (0.7).
    # With sample_size=15, meta_confidence ≈ 15/(15+8) ≈ 0.65, so blend is
    # 0.65 * 0.4 + 0.35 * 0.7 ≈ 0.50.
    assert 0.40 < out["win_probability_used"] < 0.70
    assert out["win_probability_used"] < 0.7  # must pull below raw confidence


def test_exact_pattern_uses_measured_magnitudes_for_edge(_isolated_engine):
    """When exact_pattern data has avg_win/avg_loss, _compute_mission_inputs
    must use those measured magnitudes in the edge calculation instead of
    the decision's tp_pct/sl_pct. The latter are forward targets; the
    former are what actually happens after partial fills / early exits."""
    from trading.strategic_claude import _build_autonomous_context

    sig = _FakeSignal()
    ctx = _build_autonomous_context(
        symbol="BTC-USD", side="BUY",
        technical_signal=sig, strategy_type="Momentum",
        tech_confidence=0.7,
    )
    fp = ctx.to_fingerprint()

    # Seed measured magnitudes that DIFFER from the decision's tp/sl
    # (decision has tp=0.04, sl=0.02). Measured: wins are smaller (0.015)
    # and losses are larger (0.018) — realistic for a marginal pattern.
    pat = LearnedPattern(fingerprint=fp, side="BUY")
    pat.total_trades = 20
    pat.winning_trades = 12
    pat.win_rate = 0.6
    pat.avg_win = 0.015
    pat.avg_loss = 0.018
    pat.expectancy = 0.6 * 0.015 - 0.4 * 0.018
    _isolated_engine._patterns[fp] = pat

    # Compare edge with measured magnitudes vs edge with optimistic targets:
    notional = 200.0
    fee_drag = 0.005 * notional  # 50 bps round-trip
    prob_blended = 0.6 * (20 / 28) + 0.7 * (8 / 28)  # = 0.629 (approx)
    edge_measured = (
        prob_blended * 0.015 * notional
        - (1 - prob_blended) * 0.018 * notional
        - fee_drag
    )
    edge_targets = (
        prob_blended * 0.04 * notional
        - (1 - prob_blended) * 0.02 * notional
        - fee_drag
    )

    out = _compute_mission_inputs(
        signal=sig, decision=_FakeDecision(),
        confidence=0.7, proposed_notional=notional,
        symbol="BTC-USD", side="BUY", strategy_type="Momentum",
    )

    # Should be much closer to the measured edge than to the targets edge.
    assert abs(out["expected_net_edge"] - edge_measured) < abs(
        out["expected_net_edge"] - edge_targets
    ), (
        f"edge should use measured win/loss magnitudes, not decision tp/sl. "
        f"got {out['expected_net_edge']:.4f}, measured-based={edge_measured:.4f}, "
        f"target-based={edge_targets:.4f}"
    )


def test_calibration_never_crashes_when_strategic_claude_unavailable(_isolated_engine):
    """If for some reason _build_autonomous_context fails (raised, missing
    import, etc), _compute_mission_inputs must still return a sane dict
    with raw_confidence fallback. Calibration is best-effort."""
    # Force the import lookup to fail by passing junk signal — but the
    # function must catch internally and degrade. This guards the contract
    # that calibration NEVER blocks the mission pipeline.

    class _BrokenSignal:
        # No indicators attribute, no side, no strategy
        pass

    out = _compute_mission_inputs(
        signal=_BrokenSignal(),
        decision=_FakeDecision(),
        confidence=0.65,
        proposed_notional=100.0,
        symbol="DOGE-USD",
        side="BUY",
        strategy_type="Momentum",
    )
    # Must still return all the expected fields without raising.
    assert "expected_net_edge" in out
    assert "win_probability_source" in out
    assert out["win_probability_used"] == pytest.approx(0.65)
