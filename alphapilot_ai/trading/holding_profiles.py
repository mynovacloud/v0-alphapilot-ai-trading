"""
Holding profiles — the single source of truth for how long a trade is
held and what its exit thresholds are.

This module replaces the scattered scalper/hybrid/swing branching that
previously lived (inconsistently) across paper_trading_engine and
position_monitor, plus the silent `high_conviction` override that gave
high-confidence trades no time limit at all.

A HoldingProfile is a bundle of exit parameters. The operator picks a
MODE in Settings (`bot_holding_mode`); a mode resolves to exactly one of
the four BASE profiles at the moment a trade opens, and that resolved
name is stamped on the trade (`PaperTrade.holding_profile`) so its rules
never change mid-flight even if the operator switches the global mode.

Modes:
  scalp / short_hold / short_swing / long_hold
      Fixed — always that base profile.
  mixed
      Conviction-tiered — entry confidence picks the base profile.
  ai_decide
      Claude picks — derived from the take-profit magnitude Claude
      already chose for the trade (a 10% TP is inherently a long hold,
      a 1% TP a scalp), so no fragile prompt-schema changes are needed.

INVARIANT enforced for every base profile: max_loss_pct < target_pct.
That alone keeps the system mathematically winnable below a 50% win
rate. A profile that violates it raises at import time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoldingProfile:
    name: str                    # one of the four BASE names
    label: str                   # human label for UI / logs
    target_pct: float            # take-profit, fraction of entry (0.02 = 2%)
    max_loss_pct: float          # hard stop, fraction of entry
    trailing_pct: float          # trailing-stop gap from the high-water mark
    hard_cap_minutes: float      # absolute max hold — NO escape hatch
    stale_minutes: float         # after this, exit unless meaningfully in profit
    stale_min_profit_pct: float  # the "meaningfully in profit" bar for the stale check

    def __post_init__(self) -> None:
        # The one invariant that makes the math work. Fail loud at import
        # time rather than silently shipping a money-losing profile.
        if self.max_loss_pct >= self.target_pct:
            raise ValueError(
                f"HoldingProfile {self.name}: max_loss_pct ({self.max_loss_pct}) "
                f"must be < target_pct ({self.target_pct})"
            )


# The four concrete profiles. `holding_profile` on a trade is always one
# of these names — `mixed` and `ai_decide` resolve down to one of them.
BASE_PROFILES: dict[str, HoldingProfile] = {
    "scalp": HoldingProfile(
        name="scalp", label="Scalp",
        target_pct=0.003, max_loss_pct=0.0015, trailing_pct=0.0015,
        hard_cap_minutes=10, stale_minutes=5, stale_min_profit_pct=0.001,
    ),
    "short_hold": HoldingProfile(
        name="short_hold", label="Short hold",
        target_pct=0.008, max_loss_pct=0.004, trailing_pct=0.0035,
        hard_cap_minutes=20, stale_minutes=10, stale_min_profit_pct=0.0025,
    ),
    "short_swing": HoldingProfile(
        name="short_swing", label="Short swing",
        target_pct=0.02, max_loss_pct=0.01, trailing_pct=0.009,
        hard_cap_minutes=45, stale_minutes=25, stale_min_profit_pct=0.006,
    ),
    "long_hold": HoldingProfile(
        name="long_hold", label="Long hold",
        target_pct=0.05, max_loss_pct=0.025, trailing_pct=0.02,
        hard_cap_minutes=360, stale_minutes=120, stale_min_profit_pct=0.015,
    ),
}

# Modes the operator can pick in Settings — (value, human label).
SELECTABLE_MODES: tuple[tuple[str, str], ...] = (
    ("scalp",       "Scalp — fastest in/out (~0.3% target, 10-min cap)"),
    ("short_hold",  "Short hold — quick (~0.8% target, 20-min cap)"),
    ("short_swing", "Short swing — balanced (~2% target, 45-min cap)"),
    ("long_hold",   "Long hold — patient (~5% target, 6-hour cap)"),
    ("mixed",       "Mixed — pick a profile per trade by AI confidence"),
    ("ai_decide",   "AI decides — Claude picks the profile per trade"),
)
VALID_MODES = frozenset(m for m, _ in SELECTABLE_MODES)
DEFAULT_MODE = "mixed"

# Old wallet.trading_style values map onto the new profiles so existing
# data and any un-migrated caller keeps working.
_LEGACY_ALIASES: dict[str, str] = {
    "scalper": "scalp",
    "hybrid": "short_swing",
    "swing": "long_hold",
}


def _mixed_tier(confidence: float) -> str:
    """Conviction -> base profile name for the `mixed` mode."""
    if confidence < 0.55:
        return "scalp"
    if confidence < 0.65:
        return "short_hold"
    if confidence < 0.78:
        return "short_swing"
    return "long_hold"


def profile_from_claude_targets(take_profit_pct: float | None) -> str:
    """Map the take-profit magnitude Claude chose to a holding profile.

    This is how `ai_decide` mode works: Claude already expresses its
    holding intent through the take-profit it wants (a wide TP means it
    expects a big, slow move; a tight TP means a quick scalp). Reading
    that intent avoids touching the Claude prompt/response schema.
    """
    tp = float(take_profit_pct or 0.0)
    if tp <= 0.0:
        return "short_swing"  # no signal — safe middle ground
    if tp <= 0.007:
        return "scalp"
    if tp <= 0.015:
        return "short_hold"
    if tp <= 0.035:
        return "short_swing"
    return "long_hold"


def resolve_profile_name(
    mode: str,
    confidence: float = 0.5,
    ai_choice: str | None = None,
) -> str:
    """Resolve a Settings mode to exactly one BASE profile name.

    Called once, when a trade opens. The result is stamped on the trade.

    `ai_choice` is only consulted for `ai_decide` mode — pass the name
    returned by `profile_from_claude_targets`.
    """
    mode = (mode or DEFAULT_MODE).strip().lower()
    if mode in BASE_PROFILES:
        return mode
    if mode == "mixed":
        return _mixed_tier(float(confidence or 0.5))
    if mode == "ai_decide":
        choice = (ai_choice or "").strip().lower()
        return choice if choice in BASE_PROFILES else "short_swing"
    # Unknown / legacy value (e.g. old "hybrid" / "scalper" wallets).
    return _LEGACY_ALIASES.get(mode, "short_swing")


def get_profile(name: str | None) -> HoldingProfile:
    """Look up a BASE profile by its stamped name. Falls back to a safe
    middle-ground profile for an unknown / missing name."""
    return BASE_PROFILES.get((name or "").strip().lower(), BASE_PROFILES["short_swing"])
