"""Phase C: source-level guards against re-introducing silent failures.

These tests are unusual — they inspect source text rather than runtime
behavior — but they're the right tool for the bug we keep falling into:
silent ``except Exception: pass`` or ``except Exception: logger.debug``
patterns that hide real failures for weeks.

Each assertion below picks a specific code location where Phase C bumped
a quiet swallow to a loud one, and pins it there. If a future maintainer
(human or AI) reverts the migration back to ``logger.debug`` or ``pass``,
the test fails with a pointer to the original incident.

Limitations of this approach:
  - Source inspection is fragile to refactors. Comments / line-number
    drift won't break the tests because we look for substrings.
  - We can't catch every silent swallow ever added. New silent paths
    introduced after Phase C aren't covered here — they need their own
    pin once we discover them.

The tests pay for themselves the first time someone tries to "clean up"
a noisy log by reverting it. That's the move that has put us into bug-
hiding loops three times this month.
"""
from __future__ import annotations

import inspect


def _source_of(callable_or_module) -> str:
    return inspect.getsource(callable_or_module)


def test_paper_trading_close_uses_exception_for_adaptive_learn_failure():
    """`paper_trading_engine.close_trade` must log adaptive-learning
    failures at exception level (with traceback). Previously this was
    logger.debug — which hid the symbol/duration_minutes NameError for
    weeks."""
    from trading.paper_trading_engine import PaperTradingEngine
    src = _source_of(PaperTradingEngine.close_trade)
    assert "[LEARN] Adaptive learning update failed" in src, (
        "adaptive-learn failure log message changed; verify it still "
        "exists and is at exception/warning level"
    )
    # The actual call must NOT be `logger.debug(...)`.
    bad_pattern = 'logger.debug("Adaptive learning update failed'
    assert bad_pattern not in src, (
        f"adaptive-learn failure has been reverted to debug-level. "
        f"This is the exact pattern that hid the symbol NameError for weeks. "
        f"Find {bad_pattern!r} in close_trade and bump it back to "
        f"logger.exception(...) per the Phase C migration."
    )


def test_paper_trading_close_uses_exception_for_autonomous_learn_failure():
    """Same shape as the adaptive-learn test, for the autonomous hook."""
    from trading.paper_trading_engine import PaperTradingEngine
    src = _source_of(PaperTradingEngine.close_trade)
    bad_pattern = 'logger.debug(f"Autonomous learning update failed'
    assert bad_pattern not in src, (
        f"autonomous-learn failure has been reverted to debug-level. "
        f"This hook depends on the autonomous engine seeing every closed "
        f"trade; silent failure here means fingerprints stop accumulating. "
        f"Find {bad_pattern!r} in close_trade and restore logger.exception."
    )


def test_paper_trading_close_uses_exception_for_mission_record_failure():
    """Mission Controller must learn every outcome. Silent failure here
    means the next mode transition is based on stale data."""
    from trading.paper_trading_engine import PaperTradingEngine
    src = _source_of(PaperTradingEngine.close_trade)
    bad_pattern = 'logger.debug(f"Mission-controller record_trade_result failed'
    assert bad_pattern not in src, (
        "mission record_trade_result failure has been reverted to debug. "
        "The boss layer's state machine relies on this hook firing on "
        "every close. Bump it back to logger.exception per Phase C."
    )


def test_autonomous_build_context_logs_loudly_on_failure():
    """`_build_context_from_trade` was the original hider of the
    metadata-vs-indicators bug. Must log at exception level (with
    traceback), not debug."""
    from ai.autonomous_learning_engine import AutonomousLearningEngine
    src = _source_of(AutonomousLearningEngine._build_context_from_trade)
    bad_pattern = 'logger.debug(f"[AUTONOMOUS] Failed to build context'
    assert bad_pattern not in src, (
        "context-build failure has been reverted to debug-level. "
        "This swallow hid the metadata/.indicators bug for weeks — "
        "every learn-time fingerprint silently defaulted to rsi=50 / "
        "regime=UNKNOWN. The bumped logger.exception was the only way "
        "future shape mismatches would surface."
    )


def test_autonomous_persist_logs_loudly_on_failure():
    """Persistence failure means learning this tick is lost. Must be
    at exception level so the operator sees it."""
    from ai.autonomous_learning_engine import AutonomousLearningEngine
    src = _source_of(AutonomousLearningEngine._persist)
    bad_pattern = "logger.error(f\"[AUTONOMOUS] Failed to persist memory"
    assert bad_pattern not in src, (
        "_persist failure has been reverted from logger.exception "
        "back to logger.error (no traceback). The whole point of Phase C "
        "is that the operator must see WHICH table / WHICH serialization "
        "step blew up — exception level is the bare minimum."
    )


def test_mission_reject_console_includes_calibration_source():
    """The [MISSION REJECT] console line must surface which calibration
    tier (raw_confidence / knn_neighbors / exact_pattern) backed the
    rejection. Without this, the operator can't tell from the live
    console whether the system is learning or still guessing.

    Pinned source-level so a future cleanup pass can't quietly drop the
    suffix to "reduce log noise" — visibility into calibration source is
    a feature, not noise."""
    import inspect
    import trading.bot_engine as be_mod
    src = inspect.getsource(be_mod)
    assert "[calib=" in src, (
        "calibration-source suffix missing from console logs. After "
        "fingerprint coarsening + Phase B, the operator needs to see "
        "[calib=exact_pattern n=N] in [MISSION REJECT] and [TRADE OPENED] "
        "lines to verify the learning loop is closing."
    )


def test_trade_opened_console_includes_calibration_source():
    """Same pin as above, applied to the [TRADE OPENED] line."""
    import inspect
    import trading.bot_engine as be_mod
    src = inspect.getsource(be_mod)
    # The [TRADE OPENED] line should include the calibration suffix.
    # We look for both substrings rather than the exact composed string
    # to be robust to whitespace/format edits.
    assert "[TRADE OPENED]" in src
    # The combined [calib=...] suffix is added in two places (reject +
    # open); both should be present.
    assert src.count("[calib=") >= 2, (
        f"expected calibration suffix in BOTH [MISSION REJECT] and "
        f"[TRADE OPENED] console lines; only {src.count('[calib=')} found"
    )


def test_calibration_uses_swallow_with_reason():
    """The two calibration fallbacks introduced in Phase B were given
    labeled swallows in Phase C. Re-checking those labels are present
    in source so the swallows don't become silent again."""
    import ai.autonomous_learning_engine as mod
    src = _source_of(mod)
    assert "calibration falls back to raw_confidence when context.to_fingerprint() raises" in src, (
        "calibration to_fingerprint swallow label missing — was the "
        "swallow_with_reason() call deleted or reverted to bare except?"
    )
    assert "calibration kNN lookup is best-effort" in src, (
        "calibration kNN swallow label missing"
    )
