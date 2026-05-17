"""
Bot configuration — persisted in the AppSetting table.

These are user-tunable knobs that control the autonomous trading loop.
Stored as key/value rows so we don't need a schema migration each time we
add a new knob.
"""
from __future__ import annotations

from dataclasses import dataclass

from database.db import session_scope
from database.models import AppSetting

# Defaults
DEFAULTS: dict[str, str] = {
    "bot_enabled": "false",                # master kill switch (string bool)
    "bot_tick_seconds": "60",              # how often the loop wakes up
    "bot_universe": "coinbase_usd",        # which universe to trade
    "bot_universe_limit": "100",           # max symbols per tick (increased for diversity)
    "bot_min_confidence": "0.55",          # minimum AI confidence to act (lowered to trade more)
    "bot_default_strategy_type": "Momentum",
    "bot_position_size_usd": "80",         # default per-trade notional in USD
    "bot_max_open_per_wallet": "25",       # max concurrent positions (increased for scalping)
    "bot_max_ticks_log": "200",            # how many tick rows to keep visible
    "bot_dry_run": "true",                 # if true, decisions are logged only (paper layer ignored)
    # Aggressive trading settings
    "bot_auto_dca_enabled": "true",        # automatically DCA into losing positions
    "bot_auto_dca_threshold_pct": "0.03",  # DCA when position down 3%+
    "bot_auto_scale_in_enabled": "true",   # automatically add to winning positions
    "bot_diversification_target": "15",    # try to hold at least this many different assets
    "bot_recovery_mode_enabled": "true",   # enable aggressive recovery when portfolio is down
    # Notifier settings
    "notifier_provider": "none",           # "telegram" | "discord" | "none"
    "notifier_telegram_bot_token": "",
    "notifier_telegram_chat_id": "",
    "notifier_discord_webhook_url": "",
    "notifier_min_level": "info",          # "info" | "warn" | "error"
    "notifier_daily_summary": "true",      # send a daily P&L summary
    "notifier_daily_summary_hour_utc": "23",  # 0..23
    # Anthropic / Claude
    "anthropic_api_key": "",
    "anthropic_model": "claude-sonnet-4-6",
    # Live training session (Training Center "Start Live Session" button).
    # When active, the scheduler runs at session_tick_seconds with paper trading
    # ON. Stopping the session restores the previous bot_enabled / dry_run /
    # tick_seconds values so the user's normal config isn't disturbed.
    "training_session_active": "false",
    "training_session_started_at": "",
    "training_session_tick_seconds": "15",
    "training_session_prev_bot_enabled": "",
    "training_session_prev_dry_run": "",
    "training_session_prev_tick_seconds": "",
    "training_session_prev_min_confidence": "",
    "training_session_prev_position_size_usd": "",
    "training_session_prev_max_open_per_wallet": "",
    "training_session_prev_universe_limit": "",
}


@dataclass
class BotConfig:
    bot_enabled: bool
    tick_seconds: int
    universe: str
    universe_limit: int
    min_confidence: float
    default_strategy_type: str
    position_size_usd: float
    max_open_per_wallet: int
    dry_run: bool
    # Aggressive trading settings
    auto_dca_enabled: bool
    auto_dca_threshold_pct: float
    auto_scale_in_enabled: bool
    diversification_target: int
    recovery_mode_enabled: bool

    @classmethod
    def load(cls) -> "BotConfig":
        raw = _load_raw()
        return cls(
            bot_enabled=_b(raw.get("bot_enabled")),
            tick_seconds=max(2, int(float(raw.get("bot_tick_seconds") or 60))),
            universe=raw.get("bot_universe") or "coinbase_usd",
            universe_limit=max(1, int(float(raw.get("bot_universe_limit") or 100))),
            min_confidence=max(0.0, min(1.0, float(raw.get("bot_min_confidence") or 0.55))),
            default_strategy_type=raw.get("bot_default_strategy_type") or "Momentum",
            position_size_usd=max(1.0, float(raw.get("bot_position_size_usd") or 80)),
            max_open_per_wallet=max(1, int(float(raw.get("bot_max_open_per_wallet") or 25))),
            dry_run=_b(raw.get("bot_dry_run"), default=True),
            # Aggressive trading
            auto_dca_enabled=_b(raw.get("bot_auto_dca_enabled"), default=True),
            auto_dca_threshold_pct=max(0.01, min(0.20, float(raw.get("bot_auto_dca_threshold_pct") or 0.03))),
            auto_scale_in_enabled=_b(raw.get("bot_auto_scale_in_enabled"), default=True),
            diversification_target=max(1, int(float(raw.get("bot_diversification_target") or 15))),
            recovery_mode_enabled=_b(raw.get("bot_recovery_mode_enabled"), default=True),
        )


def _b(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_raw() -> dict[str, str]:
    with session_scope() as s:
        rows = s.query(AppSetting).all()
        out = {r.key: (r.value or "") for r in rows}
    # fill defaults for anything missing
    merged = dict(DEFAULTS)
    merged.update(out)
    return merged


def get(key: str) -> str:
    """Get a single setting value (with default fallback)."""
    raw = _load_raw()
    return raw.get(key, DEFAULTS.get(key, ""))


def set_many(updates: dict[str, str]) -> None:
    """Upsert a batch of key/value settings."""
    from utils.helpers import utcnow

    with session_scope() as s:
        existing = {r.key: r for r in s.query(AppSetting).all()}
        for key, value in updates.items():
            if key in existing:
                existing[key].value = str(value)
                existing[key].updated_at = utcnow()
            else:
                s.add(AppSetting(key=key, value=str(value)))


def ensure_defaults() -> None:
    """Make sure every default key exists in the table. Safe to call on every startup."""
    with session_scope() as s:
        existing = {r.key for r in s.query(AppSetting).all()}
        for key, value in DEFAULTS.items():
            if key not in existing:
                s.add(AppSetting(key=key, value=value))
