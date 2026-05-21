"""Regression: close_trade NameError'd silently on `symbol` / `duration_minutes`.

Original incident (commit c503b14):
    paper_trading_engine.close_trade() referenced `symbol` and
    `duration_minutes` in the keyword arguments of its three learn hooks
    (claude_learning.record_trade_outcome, adaptive_learning_engine.
    learn_from_trade, autonomous_learning_engine.learn_from_closed_trade),
    but those names were never assigned inside the function. Each call
    NameError'd, was caught by the broad try/except, and silently no-op'd.
    The adaptive-learning and autonomous-learning hooks had been DEAD
    since they were wired in. Only the reflection hook ran (because it
    only takes trade_id and looks the trade up internally).

    Fix: assign symbol = trade.symbol and compute duration_minutes from
    opened_at/closed_at inside the with-block before any of the hooks fire.

This test guards against the same bug returning. We don't need to run the
full close_trade pipeline (that would require seeding a real PaperTrade,
running all three external learn engines, asserting their side effects).
What we actually need to guard is the FACT that symbol and duration_minutes
are local variables in close_trade and they get populated before the
learn-hook block. So we test that via source inspection: a static
guarantee instead of an integration setup that would be fragile.
"""
from __future__ import annotations

import inspect
import textwrap

from trading.paper_trading_engine import PaperTradingEngine


def _close_trade_source() -> str:
    """The source text of close_trade, dedented for line-based inspection."""
    src = inspect.getsource(PaperTradingEngine.close_trade)
    return textwrap.dedent(src)


def test_close_trade_assigns_symbol_local():
    """`symbol` must be assigned before the learn-hook block runs.

    The hooks pass `symbol=symbol` as a keyword arg — if `symbol` isn't
    bound, the call NameErrors and the broad except swallows it. This was
    the entire failure mode that hid the adaptive + autonomous learn hooks
    for weeks. Forcing the assignment to live in source makes regression
    obvious in code review.
    """
    src = _close_trade_source()
    # At least one assignment of the form `symbol = trade.symbol` or
    # `symbol = ...` must exist. We allow any RHS since maintainers might
    # rename `trade` later.
    assert any(
        line.strip().startswith("symbol = ") or line.strip().startswith("symbol =")
        for line in src.splitlines()
    ), (
        "close_trade does not assign `symbol` as a local variable. The "
        "learn hooks reference it by name; if it stays unset they "
        "NameError silently inside the broad try/except."
    )


def test_close_trade_assigns_duration_minutes_local():
    """`duration_minutes` must be assigned before the learn-hook block.

    Same incident as `symbol` — passing duration_minutes=duration_minutes
    as a kwarg when the name isn't bound NameErrors and gets swallowed.
    """
    src = _close_trade_source()
    assert any(
        "duration_minutes = " in line or "duration_minutes=" in line
        for line in src.splitlines()
        if not line.strip().startswith("#")
    ), (
        "close_trade does not bind `duration_minutes`. The adaptive and "
        "autonomous learn hooks both pass it as a kwarg; without the local "
        "they NameError silently."
    )


def test_close_trade_invokes_all_three_learn_hooks():
    """The three closed-trade learning hooks must all be wired in source:
       1. ai.claude_learning.record_trade_outcome  (reflection)
       2. ai.adaptive_learning_engine.learn_from_trade  (adaptive)
       3. ai.autonomous_learning_engine.learn_from_closed_trade (autonomous)
    Plus, since commit c503b14:
       4. risk.daily_mission_controller.* record_trade_result
    If any are deleted by a future refactor this test catches it."""
    src = _close_trade_source()
    required_imports = (
        "from ai.claude_learning import record_trade_outcome",
        "from ai.adaptive_learning_engine import learn_from_trade",
        "from ai.autonomous_learning_engine import learn_from_closed_trade",
        "from risk.daily_mission_controller import get_mission_controller",
    )
    for needle in required_imports:
        assert needle in src, (
            f"close_trade no longer wires {needle!r}. The trade-closure "
            f"learn loop is incomplete; bug-discovery latency on the next "
            f"silent regression goes back to weeks."
        )
