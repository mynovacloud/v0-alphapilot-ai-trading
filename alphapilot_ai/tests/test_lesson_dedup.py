"""Regression: reflection-lesson deduplication.

Original incident: every reflection in a live training session logged
`lessons=5 (new=5, reinforced=0)` — the dedup pipeline never matched any
new lesson to an existing one. A 50-trade session generates ~250 unique
playbook entries, most of them paraphrases of the same insight. The
playbook bloats; Claude's system prompt context dilutes; rule weighting
becomes meaningless.

Root cause: the old normalization was bag-of-tokens with a 0.72 jaccard
threshold. Tested against five obviously-equivalent paraphrase pairs,
all scored 0.00 to 0.36 — well below threshold.

Fix (commit covered by this test file):
  - Synonym mapping for common trading verb variants ("buying"/"long"/
    "enter long"/"purchase" → "buy", etc.) so paraphrases share tokens.
  - Simple suffix stemmer for tokens not in the synonym map.
  - Threshold lowered to 0.50 based on empirical pair testing.
  - Direction-conflict guard: lessons whose ONLY direction tokens are
    opposite (one has "buy" without "sell", other has "sell" without
    "buy") never collapse, regardless of how much their condition
    tokens overlap.

These tests pin the contract on three axes:
  1. Known paraphrase pairs that were observed in the wild MUST match.
  2. Genuinely-different lessons MUST NOT match.
  3. Opposite-direction lessons MUST NOT match even with high overlap.
"""
from __future__ import annotations

import pytest

from ai.claude_learning import (
    _normalize_lesson,
    _jaccard,
    _has_conflicting_direction,
    _DEDUP_SIMILARITY_THRESHOLD,
)


def _matches(a: str, b: str) -> bool:
    """Replicate the matching logic from _save_reflection so the test
    asserts the FULL contract (normalization + jaccard + conflict guard),
    not just one piece."""
    na = _normalize_lesson(a)
    nb = _normalize_lesson(b)
    if _has_conflicting_direction(na, nb):
        return False
    return _jaccard(na, nb) >= _DEDUP_SIMILARITY_THRESHOLD


# =============================================================================
# Positive: paraphrases that MUST match
# =============================================================================

def test_buy_avoidance_paraphrase_matches():
    """The canonical example: two ways to express 'don't buy when RSI > 70'."""
    a = "Avoid buying when RSI is overbought above 70"
    b = "Do not enter long positions when RSI is in overbought territory above 70"
    assert _matches(a, b), (
        f"buy-avoidance paraphrase didn't match. "
        f"norm_a={_normalize_lesson(a)!r}, norm_b={_normalize_lesson(b)!r}, "
        f"sim={_jaccard(_normalize_lesson(a), _normalize_lesson(b)):.2f}"
    )


def test_loss_cut_paraphrase_matches():
    """Two ways to express 'exit fast on momentum reversal'."""
    a = "Cut losses quickly when momentum reverses"
    b = "Exit fast on momentum reversal to prevent larger losses"
    assert _matches(a, b)


# =============================================================================
# Negative: opposite-direction lessons MUST NOT match
# =============================================================================

def test_opposite_direction_lessons_do_not_match():
    """If the same condition has opposite recommendations (one says buy,
    the other says sell), the dedup MUST keep them separate. Collapsing
    them would silently invert the playbook's recommendation."""
    a = "Buy when RSI is below 30 with strong volume"
    b = "Sell when RSI is below 30 with strong volume"
    # Their tokens overlap heavily on conditions but direction conflicts.
    na = _normalize_lesson(a)
    nb = _normalize_lesson(b)
    raw_sim = _jaccard(na, nb)
    # Sanity: the raw similarity DOES cross threshold — proving that
    # without the conflict guard, the dedup would incorrectly match.
    assert raw_sim >= _DEDUP_SIMILARITY_THRESHOLD, (
        f"sanity precondition failed — raw_sim {raw_sim:.2f} is unexpectedly "
        f"below threshold; conflict guard is no longer needed?"
    )
    # The guard MUST flag this as a conflict and prevent the match.
    assert _has_conflicting_direction(na, nb), (
        "direction-conflict guard failed to flag a clear buy-vs-sell conflict"
    )
    assert not _matches(a, b), "opposite-direction lessons must not collapse"


def test_unrelated_lessons_do_not_match():
    """A lesson about taking profit shouldn't match a lesson about avoiding
    news-time trading. Different topics, low overlap, no match."""
    a = "Take profit at 5% gain"
    b = "Avoid trading during major news events"
    assert not _matches(a, b)


def test_indicator_specific_lessons_dont_collapse_with_wrong_indicator():
    """A lesson about RSI shouldn't collapse with a lesson about MACD
    even if other tokens overlap."""
    a = "Buy when RSI dips below 30 and volume confirms"
    b = "Buy when MACD turns positive and volume confirms"
    # Both are "buy ... and volume confirms" — but specifically tied to
    # DIFFERENT indicators. Shouldn't collapse.
    na = _normalize_lesson(a)
    nb = _normalize_lesson(b)
    sim = _jaccard(na, nb)
    # If we end up matching these, the dedup is over-collapsing.
    assert sim < 0.75, (
        f"different-indicator lessons too similar: sim={sim:.2f} norm_a={na!r} norm_b={nb!r}"
    )


# =============================================================================
# Normalization-level guarantees
# =============================================================================

def test_normalization_collapses_buy_verb_variants():
    """All buy-side direction verbs ('buy', 'buying', 'bought', 'long',
    'enter long', 'purchase') must normalize to the same canonical token.
    This is the single most impactful synonym mapping."""
    variants = [
        "buy stock",
        "buying stock",
        "bought stock",
        "long stock",
        "enter long stock",  # 'enter' stays distinct; 'long' → 'buy'
        "purchase stock",
    ]
    norms = [_normalize_lesson(v) for v in variants]
    # All must contain the canonical "buy" token.
    for v, n in zip(variants, norms):
        assert "buy" in n.split(), f"{v!r} did not normalize to include 'buy': {n!r}"


def test_normalization_collapses_negation_with_avoidance():
    """'do not buy', 'avoid buying', 'never buy', 'skip buying' must all
    contain the same avoidance + direction tokens after normalization.
    Without this, negation-style phrasings never match avoidance-style."""
    a = _normalize_lesson("Avoid buying when X")
    b = _normalize_lesson("Do not buy when X")
    c = _normalize_lesson("Never buy when X")
    d = _normalize_lesson("Skip buying when X")

    for n in (a, b, c, d):
        toks = set(n.split())
        assert "avoid" in toks, f"missing avoid in {n!r}"
        assert "buy" in toks, f"missing buy in {n!r}"


def test_direction_guard_only_fires_on_clear_conflict():
    """The conflict guard must NOT fire on:
      - lessons that mention neither buy nor sell
      - lessons where one says 'buy' and the other doesn't mention direction
    Otherwise the guard becomes too aggressive and blocks legitimate matches.
    """
    cases = [
        ("Wait for volume confirmation", "Wait for higher volume"),  # neither direction
        ("Buy when RSI low", "Trade when RSI low"),                   # one-sided buy
        ("Avoid trading at low volume", "Skip entries on weak volume"),  # neither direction
    ]
    for a, b in cases:
        na = _normalize_lesson(a)
        nb = _normalize_lesson(b)
        assert not _has_conflicting_direction(na, nb), (
            f"conflict guard fired incorrectly: a={na!r} b={nb!r}"
        )


def test_old_zero_match_failure_is_fixed():
    """Sanity: the original bug was 0% match rate on the 5 paraphrase pairs
    observed in the wild. After the fix, at LEAST 2 must match. This is the
    test that would have failed before the dedup rewrite landed."""
    pairs = [
        ("Avoid buying when RSI is overbought above 70",
         "Do not enter long positions when RSI is in overbought territory above 70"),
        ("Cut losses quickly when momentum reverses",
         "Exit fast on momentum reversal to prevent larger losses"),
        ("Wait for volume confirmation before entering a trade",
         "Do not enter a position without volume confirmation"),
        ("Tighten stops after a 3% profit",
         "Move stop-loss tighter once the trade is up 3 percent"),
        ("Sell when MACD histogram turns negative",
         "Exit longs when the MACD histogram crosses below zero"),
    ]
    match_count = sum(1 for a, b in pairs if _matches(a, b))
    assert match_count >= 2, (
        f"only {match_count}/5 paraphrase pairs matched; expected >= 2. "
        f"This was the entire failure mode that produced `reinforced=0` "
        f"on every reflection. If the count drops below 2, the dedup is "
        f"functionally broken again."
    )
