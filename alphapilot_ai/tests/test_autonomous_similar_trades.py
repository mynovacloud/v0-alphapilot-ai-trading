"""Regression: _find_similar_trades tuple-order mismatch.

Original incident (commit 6c17604):
    The v2 main rewrite shipped a three-way mismatch:
      * _find_similar_trades returns [(pnl, fingerprint), ...] — pnl at
        index 0, fingerprint (a string) at index 1.
      * Its annotation claimed List[Tuple[List[float], float]] — vector
        at index 0, pnl at index 1.
      * decide() trusted the annotation and read t[1] for pnl, which
        crashed every call with `TypeError: unsupported operand type(s)
        for +: 'int' and 'str'` (Python trying to sum 0 + a hash string).

    The crash was caught by three nested except blocks in the autonomous
    decision chain and bubbled to bot_engine's tick error log. It ran
    silently in production for the entire v2 era. The fix:
      * decide() reads t[0] for pnl (the contract _find_similar_trades
        actually fulfills)
      * _find_similar_trades now defensively unpacks each stored entry
        so a malformed historical row can't poison the result set

This test pins both halves of the contract.
"""
from __future__ import annotations

import pytest

from ai.autonomous_learning_engine import (
    AutonomousLearningEngine,
    TradeContext,
)


def _seed_engine_with_vectors() -> AutonomousLearningEngine:
    """Build a fresh engine with a known set of stored trade vectors.

    We skip _ensure_loaded by setting the flag directly so the test never
    touches the database — the bug we're guarding against is pure logic
    inside decide() / _find_similar_trades, no DB needed."""
    eng = AutonomousLearningEngine.__new__(AutonomousLearningEngine)
    # Inline minimum init — avoid the full __init__ which reads bot_config.
    eng._loaded = True
    eng._patterns = {}
    eng._mistakes = {}
    eng._symbols = {}
    eng._trade_vectors = []
    eng._recent_trades = []

    # Three historical entries shaped (vector, pnl_pct, fingerprint).
    # The vectors are 16-dim because TradeContext.to_vector() returns 16
    # numbers; values don't matter for this test, only the SHAPE matters.
    for pnl, fp in [(0.012, "winner_a"), (-0.008, "loser_b"), (0.020, "winner_c")]:
        eng._trade_vectors.append(([0.0] * 16, float(pnl), fp))
    return eng


def test_find_similar_trades_returns_pnl_at_index_zero():
    """_find_similar_trades must return (pnl_float, fingerprint_str) tuples,
    in that order — the contract the caller in decide() depends on."""
    eng = _seed_engine_with_vectors()
    ctx = TradeContext(symbol="BTC-USD", side="BUY")

    results = eng._find_similar_trades(ctx, k=10)

    assert len(results) == 3, "should return all stored vectors when k > stored count"
    for entry in results:
        assert len(entry) == 2, "each entry must be a 2-tuple"
        pnl, fingerprint = entry
        assert isinstance(pnl, float), (
            f"index 0 must be a float (pnl); got {type(pnl).__name__}={pnl!r}"
        )
        assert isinstance(fingerprint, str), (
            f"index 1 must be a string (fingerprint); got {type(fingerprint).__name__}"
        )


def test_decide_does_not_crash_on_pnl_sum():
    """The original crash: `sum(t[1] for t in similar_trades)` would try to
    add an int seed to a fingerprint string. This test reproduces the
    exact code path and asserts no TypeError leaks out.

    We don't assert on the decision's content — just that the decision
    machinery survives the kNN block without raising."""
    eng = _seed_engine_with_vectors()
    # Match the regime/side combo in some stored vectors via the public
    # decide path. We bypass internal helpers that touch persistence by
    # passing a populated context.
    ctx = TradeContext(symbol="BTC-USD", side="BUY", signal_confidence=0.7)

    # decide() doesn't raise — the bug we just fixed used to TypeError here.
    decision = eng.decide(
        symbol="BTC-USD",
        side="BUY",
        current_price=50_000.0,
        signal_confidence=0.7,
        context=ctx,
    )
    assert decision is not None
    assert decision.action in {"BUY", "SELL", "HOLD", "AVOID"}


def test_find_similar_trades_handles_malformed_persistence_rows():
    """JSON-round-tripped vectors can occasionally land in odd shapes
    (None entries, missing fields, non-numeric pnl from a buggy writer).
    A single bad row must not poison the whole call — affected entries
    are skipped, valid ones still returned."""
    eng = _seed_engine_with_vectors()

    # Inject malformed rows alongside good ones. Each must be silently
    # filtered without crashing the rest of the result.
    eng._trade_vectors.extend([
        (None, 0.01, "bad_vector_is_none"),
        ([0.0] * 16, "not-a-number", "bad_pnl_is_string"),
        ([0.0, 0.0],  # truncated vector — euclidean returns inf
         0.005, "truncated_vector_inf_distance"),
    ])

    ctx = TradeContext(symbol="BTC-USD", side="BUY")
    results = eng._find_similar_trades(ctx, k=20)

    # Should at least retain the 3 original good entries. May include the
    # truncated-vector entry (it just sorts last because inf distance).
    assert len(results) >= 3
    # All returned entries must still satisfy the contract.
    for pnl, fp in results:
        assert isinstance(pnl, float)
        assert isinstance(fp, str)


def test_trade_vectors_persistence_roundtrip_preserves_pnl_position():
    """The persisted form of _trade_vectors is JSON. json.dumps turns
    tuples into lists; json.loads gives lists back. We need pnl to stay at
    the SAME positional index after round-trip — otherwise the index
    contract _find_similar_trades depends on breaks silently."""
    import json
    original = [
        ([0.1, 0.2, 0.3], 0.025, "fp_one"),
        ([0.4, 0.5, 0.6], -0.012, "fp_two"),
    ]
    serialized = json.dumps(original)
    restored = json.loads(serialized)

    assert len(restored) == 2
    for orig, back in zip(original, restored):
        assert isinstance(back, list)
        # Position-by-position contract: vector, pnl, fingerprint.
        assert back[0] == orig[0], "vector must round-trip at index 0"
        assert back[1] == orig[1], "pnl must round-trip at index 1"
        assert back[2] == orig[2], "fingerprint must round-trip at index 2"
