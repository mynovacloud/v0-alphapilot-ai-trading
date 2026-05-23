"""Regression: LEARN_BLOCK has a STRONG tier that fires in training too.

The overnight playbook hammered this exact bypass at weight 2.00 —
training mode let known-losing fingerprints fall through "to gather
more evidence" even after they had already failed 5-of-5 times. Once
a pattern has crossed conclusive-loser territory, more samples will
not reverse it, and "gather evidence" becomes "pay for the same
mistake again". The STRONG tier closes that gap.
"""
from __future__ import annotations

import inspect

from ai.claude_decision_engine import (
    decide,
    _LEARN_BLOCK_STRONG_MIN_TRADES,
    _LEARN_BLOCK_STRONG_MAX_WR,
)


def test_strong_tier_thresholds_are_in_a_sensible_range():
    """Block-in-training requires real evidence, not a single sample.
    Below the minimum-trades floor the bot is still exploring and a
    block would over-fit early noise; above 30% WR it's a marginal
    pattern that might still recover. The values live in a sane band."""
    assert 3 <= _LEARN_BLOCK_STRONG_MIN_TRADES <= 10
    assert 0.10 <= _LEARN_BLOCK_STRONG_MAX_WR <= 0.30


def test_decide_contains_training_strong_block():
    """The `decide` function must carry the STRONG-evidence branch that
    refuses a trade in training mode. Source inspection because a real
    integration call needs wallet/signal/adaptive_rec fixtures that are
    fragile; the branch itself is a small block we can lock by text."""
    src = inspect.getsource(decide)
    assert "_LEARN_BLOCK_STRONG_MIN_TRADES" in src, (
        "decide() lost its strong-evidence threshold reference"
    )
    assert "_LEARN_BLOCK_STRONG_MAX_WR" in src
    assert "LEARN_BLOCK/STRONG" in src, (
        "decide() no longer logs the STRONG-tier learn-block"
    )
    # And it must return as a learn_block source — not get rerouted to
    # passthrough.
    assert 'source="learn_block"' in src


def test_strong_block_branch_returns_hold():
    """The STRONG-tier branch must exit via `return refused` — falling
    through after the branch would defeat the entire purpose."""
    src = inspect.getsource(decide)
    # Find the STRONG-tier block and ensure it contains a return.
    strong_idx = src.find("_LEARN_BLOCK_STRONG_MIN_TRADES")
    assert strong_idx > 0
    # Look ahead a generous window (the whole branch is < 50 lines).
    window = src[strong_idx:strong_idx + 2000]
    assert "return refused" in window, (
        "STRONG-tier branch does not return — would fall through to passthrough"
    )
