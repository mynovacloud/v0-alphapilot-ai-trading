"""Regression: format-string bugs in console messages.

Observed in a live training session:

    [LOCK_IN] ASTER-USD BUY: peak +1.00% -> exit at +-0.29% (floor +0.50%)

The "+-0.29%" was produced by an `f"+{x:.2f}"` prefix-then-format where
`x` was negative. Python's format spec puts the sign INSIDE the format
specifier (`f"{x:+.2f}"` produces "+1.00" or "-0.29" correctly); putting
a literal `+` before a format that doesn't suppress the sign yields the
double-sign nonsense.

This test pins the fix at the source-string level so any regression of
the format-spec pattern fails immediately.
"""
from __future__ import annotations


def test_lock_in_message_uses_explicit_sign_format_for_pnl():
    """The LOCK_IN audit-log message must use `{x:+.2f}` (explicit-sign
    format) for any field that can be negative (pnl_pct, floor_pct).
    The peak_pct field stays at `+{x:.2f}` because it's always >= 0."""
    import inspect
    import trading.position_monitor as pm_mod
    src = inspect.getsource(pm_mod)
    # The bad pattern: prefix + then unsigned format on pnl_pct.
    bad = "exit at +{pnl_pct*100:.2f}%"
    assert bad not in src, (
        f"format-string regression: {bad!r} is back. This emits '+-0.29%' "
        f"for negative pnl. Use {{pnl_pct*100:+.2f}}% (explicit-sign in "
        f"the format spec) instead."
    )
    # Verify the GOOD pattern is what's there now.
    good = "exit at {pnl_pct*100:+.2f}%"
    assert good in src, (
        f"format-string fix missing: expected {good!r} in position_monitor.py"
    )


def test_format_spec_actually_handles_negatives():
    """Sanity: prove that f-string `{x:+.2f}` does the right thing for
    both signs. If this ever fails, Python's stdlib has changed."""
    pos = 1.23
    neg = -0.29
    assert f"{pos:+.2f}" == "+1.23"
    assert f"{neg:+.2f}" == "-0.29"
    # And the old bad pattern that caused the bug:
    assert f"+{neg:.2f}" == "+-0.29"  # This is the FAILURE mode we fixed.
