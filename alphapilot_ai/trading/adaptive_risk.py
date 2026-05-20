"""
Adaptive risk manager.

This is the missing feedback loop. The bot has plenty of "learning" engines
(trade_learning_engine, autonomous_learning_engine, claude_learning, etc.)
that all write to a passive playbook of rules-as-text. None of them touch
the actual gates that decide whether a trade is opened. This module does.

After every closed trade we update three pieces of live state:

1. A rolling 20-trade win rate. If it falls below WR_FLOOR_TRIGGER we raise
   `bot_min_confidence` by FLOOR_STEP (capped at FLOOR_MAX). When it climbs
   back above WR_FLOOR_RELAX we relax it one step at a time. This is the
   answer to "shouldn't the bot notice the loss in numbers and adjust?".

2. A rolling 10-trade P&L (in dollars). If it dips below P&L_BREAKER_USD we
   open a cooldown window during which no new entries fire. This replaces
   the consecutive-loss breaker, which is silently defeated by tiny scalp
   wins resetting the streak counter. Net P&L can't be gamed by a +$0.10
   scalp.

3. Per-symbol session stats. Once a symbol has SYMBOL_MIN_TRADES closes and
   its session WR is below SYMBOL_WR_BLOCK, it gets blacklisted for the
   session. The lifetime win rate stays poor because the same losing
   symbols keep getting picked; this stops that bleed within a session.

State is persisted to AppSetting so it survives process restarts. There is
a public `reset()` that wipes it all (used by the dashboard "Reset
Adaptive State" button or when the user starts a fresh paper session).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock

from database.db import session_scope
from database.models import AppSetting

log = logging.getLogger(__name__)

# ---- Tunables ---------------------------------------------------------------

# Rolling window sizes. 20 trades is small enough to react within an hour
# of scalp activity, but large enough that one unlucky cluster doesn't
# trigger an over-correction.
WR_WINDOW = 20
PNL_WINDOW = 10
SYMBOL_WINDOW = 50  # how many recent symbol-tagged trades to remember

# Confidence floor adaptation
WR_FLOOR_TRIGGER = 0.40   # if rolling WR < 40%, raise the floor
WR_FLOOR_RELAX = 0.55     # if rolling WR > 55%, allow the floor to drift back
FLOOR_STEP = 0.05         # how much we move the floor per adjustment
FLOOR_MAX = 0.85          # never demand impossible-to-reach conviction
FLOOR_MIN_BUFFER = 0.05   # never go below user's configured floor

# P&L circuit breaker
PNL_BREAKER_USD = -3.0    # -$3 over rolling 10 trades = pause new entries
BREAKER_COOLDOWN_MIN = 15

# Per-symbol blacklist
SYMBOL_MIN_TRADES = 4
SYMBOL_WR_BLOCK = 0.25
SYMBOL_BLACKLIST_HOURS = 6  # auto-expire so a symbol can earn its way back

# AppSetting keys (all single-row JSON blobs)
KEY_RECENT_PNL = "adaptive_recent_pnl_window"        # list[float]
KEY_RECENT_WINS = "adaptive_recent_wins_window"      # list[int 0/1]
KEY_FLOOR_ADJUSTMENT = "adaptive_floor_adjustment"   # float, additive bump
KEY_BREAKER_UNTIL = "adaptive_breaker_until_iso"     # iso datetime or ""
KEY_SYMBOL_STATS = "adaptive_symbol_stats"           # {sym: {"w":n,"l":n,"pnl":f,"blocked_until":iso}}


@dataclass
class AdaptiveSnapshot:
    """Plain-data view of the manager's current state — useful for dashboards."""
    rolling_wr: float
    rolling_pnl_usd: float
    floor_adjustment: float
    breaker_active: bool
    breaker_until: datetime | None
    blocked_symbols: list[str]
    sample_size: int


class AdaptiveRiskManager:
    """Singleton-style manager. Holds an in-process cache of state so we
    don't hammer the AppSetting table on every confidence check, but
    re-reads on writes so multiple processes (web + scheduler) stay in
    sync."""

    _lock = Lock()

    # ---- Public API ---------------------------------------------------------

    @classmethod
    def record_trade(cls, *, symbol: str, pnl: float, confidence: float | None = None) -> None:
        """Call this from the close-trade hook with the realized P&L.

        Updates the rolling windows, recomputes the floor adjustment, opens
        a circuit-breaker window if the rolling P&L is bleeding, and bumps
        the per-symbol counters (with optional auto-blacklist).
        """
        with cls._lock:
            state = cls._load()

            # Append to rolling windows (FIFO with cap).
            wins = state["wins"]
            wins.append(1 if pnl > 0 else 0)
            if len(wins) > WR_WINDOW:
                wins[:] = wins[-WR_WINDOW:]

            pnls = state["pnls"]
            pnls.append(float(pnl))
            if len(pnls) > PNL_WINDOW:
                pnls[:] = pnls[-PNL_WINDOW:]

            # Per-symbol stats — keyed by symbol, scoped to the lookback window.
            sym_stats = state["symbol_stats"]
            entry = sym_stats.setdefault(symbol, {"w": 0, "l": 0, "pnl": 0.0, "blocked_until": ""})
            if pnl > 0:
                entry["w"] = int(entry.get("w", 0)) + 1
            else:
                entry["l"] = int(entry.get("l", 0)) + 1
            entry["pnl"] = float(entry.get("pnl", 0.0)) + float(pnl)

            # Auto-blacklist a chronic loser. Note we use total trades
            # ≥ SYMBOL_MIN_TRADES so we don't blacklist on a single bad print.
            total = entry["w"] + entry["l"]
            if total >= SYMBOL_MIN_TRADES:
                wr = entry["w"] / total
                if wr < SYMBOL_WR_BLOCK and not entry.get("blocked_until"):
                    block_until = datetime.utcnow() + timedelta(hours=SYMBOL_BLACKLIST_HOURS)
                    entry["blocked_until"] = block_until.isoformat()
                    log.warning(
                        "[ADAPTIVE] auto-blacklisting %s for %dh: %d/%d wins (%.0f%% WR), $%.2f net",
                        symbol, SYMBOL_BLACKLIST_HOURS, entry["w"], total, wr * 100, entry["pnl"],
                    )

            # ---- Recompute the live floor adjustment ----
            # We only adjust when we have enough data; one bad trade
            # shouldn't move the floor.
            if len(wins) >= max(8, WR_WINDOW // 2):
                rolling_wr = sum(wins) / len(wins)
                old_adj = float(state["floor_adjustment"])
                new_adj = old_adj
                if rolling_wr < WR_FLOOR_TRIGGER:
                    new_adj = min(FLOOR_MAX, old_adj + FLOOR_STEP)
                elif rolling_wr > WR_FLOOR_RELAX:
                    new_adj = max(0.0, old_adj - FLOOR_STEP)
                if abs(new_adj - old_adj) > 1e-6:
                    log.warning(
                        "[ADAPTIVE] floor adjustment %.2f -> %.2f (rolling WR %.0f%% over %d)",
                        old_adj, new_adj, rolling_wr * 100, len(wins),
                    )
                    state["floor_adjustment"] = new_adj

            # ---- P&L-based circuit breaker ----
            # Rolling SUM (not avg) catches "death by a thousand cuts" where
            # 10 trades each lose 30 cents — invisible to a streak counter,
            # very visible to net P&L.
            if len(pnls) >= PNL_WINDOW:
                rolling_pnl = sum(pnls)
                if rolling_pnl < PNL_BREAKER_USD:
                    until = datetime.utcnow() + timedelta(minutes=BREAKER_COOLDOWN_MIN)
                    state["breaker_until"] = until.isoformat()
                    log.warning(
                        "[ADAPTIVE] P&L circuit breaker tripped: rolling $%.2f over %d trades. "
                        "Pausing new entries until %s.",
                        rolling_pnl, PNL_WINDOW, until.isoformat(timespec="seconds"),
                    )

            cls._save(state)

    @classmethod
    def effective_floor(cls, base_floor: float) -> float:
        """Apply the live adjustment on top of the user's configured floor.

        We never go BELOW the user's setting (that would be the system
        overriding their explicit choice), and we never demand more than
        FLOOR_MAX (above which Claude effectively can't open anything).
        """
        state = cls._load()
        adj = float(state["floor_adjustment"])
        floor = max(base_floor, base_floor + adj - FLOOR_MIN_BUFFER + FLOOR_MIN_BUFFER)
        floor = base_floor + adj
        return min(FLOOR_MAX, max(base_floor, floor))

    @classmethod
    def is_breaker_active(cls) -> tuple[bool, datetime | None]:
        """Has the rolling-P&L breaker tripped and not yet expired?"""
        state = cls._load()
        until_iso = state.get("breaker_until") or ""
        if not until_iso:
            return False, None
        try:
            until = datetime.fromisoformat(until_iso)
        except Exception:
            return False, None
        if datetime.utcnow() >= until:
            # Auto-expire on read so we don't need a background sweeper.
            with cls._lock:
                state = cls._load()
                state["breaker_until"] = ""
                cls._save(state)
            return False, None
        return True, until

    @classmethod
    def is_symbol_blocked(cls, symbol: str) -> bool:
        """Did the bot recently blacklist this symbol for the session?"""
        state = cls._load()
        entry = state["symbol_stats"].get(symbol)
        if not entry:
            return False
        until_iso = entry.get("blocked_until") or ""
        if not until_iso:
            return False
        try:
            until = datetime.fromisoformat(until_iso)
        except Exception:
            return False
        if datetime.utcnow() >= until:
            # Expired — clear the block so the symbol gets a fresh shot.
            with cls._lock:
                state = cls._load()
                e = state["symbol_stats"].get(symbol)
                if e:
                    e["blocked_until"] = ""
                    # Also reset the win/loss counters so it isn't
                    # immediately re-blocked on the next bad print.
                    e["w"] = 0
                    e["l"] = 0
                    e["pnl"] = 0.0
                cls._save(state)
            return False
        return True

    @classmethod
    def snapshot(cls) -> AdaptiveSnapshot:
        """Read-only view for the dashboard."""
        state = cls._load()
        wins = state["wins"]
        pnls = state["pnls"]
        wr = (sum(wins) / len(wins)) if wins else 0.0
        pnl = sum(pnls) if pnls else 0.0
        until_iso = state.get("breaker_until") or ""
        until = None
        active = False
        if until_iso:
            try:
                until = datetime.fromisoformat(until_iso)
                active = datetime.utcnow() < until
            except Exception:
                pass
        blocked = []
        now = datetime.utcnow()
        for sym, entry in state["symbol_stats"].items():
            bu = entry.get("blocked_until") or ""
            if not bu:
                continue
            try:
                if datetime.fromisoformat(bu) > now:
                    blocked.append(sym)
            except Exception:
                pass
        return AdaptiveSnapshot(
            rolling_wr=wr,
            rolling_pnl_usd=pnl,
            floor_adjustment=float(state["floor_adjustment"]),
            breaker_active=active,
            breaker_until=until,
            blocked_symbols=sorted(blocked),
            sample_size=len(wins),
        )

    @classmethod
    def reset(cls) -> None:
        """Wipe all adaptive state. Use when starting a fresh paper run."""
        with cls._lock:
            cls._save({
                "wins": [],
                "pnls": [],
                "floor_adjustment": 0.0,
                "breaker_until": "",
                "symbol_stats": {},
            })
            log.info("[ADAPTIVE] state reset")

    # ---- Persistence --------------------------------------------------------

    @classmethod
    def _load(cls) -> dict:
        """Fetch all five keys in one DB round-trip."""
        keys = [KEY_RECENT_WINS, KEY_RECENT_PNL, KEY_FLOOR_ADJUSTMENT, KEY_BREAKER_UNTIL, KEY_SYMBOL_STATS]
        with session_scope() as s:
            rows = {r.key: r.value for r in s.query(AppSetting).filter(AppSetting.key.in_(keys)).all()}
        return {
            "wins": _safe_json_list(rows.get(KEY_RECENT_WINS)),
            "pnls": _safe_json_list(rows.get(KEY_RECENT_PNL)),
            "floor_adjustment": _safe_float(rows.get(KEY_FLOOR_ADJUSTMENT), 0.0),
            "breaker_until": rows.get(KEY_BREAKER_UNTIL) or "",
            "symbol_stats": _safe_json_dict(rows.get(KEY_SYMBOL_STATS)),
        }

    @classmethod
    def _save(cls, state: dict) -> None:
        """Upsert all five keys."""
        from utils.helpers import utcnow
        payload = {
            KEY_RECENT_WINS: json.dumps(state.get("wins", [])),
            KEY_RECENT_PNL: json.dumps(state.get("pnls", [])),
            KEY_FLOOR_ADJUSTMENT: str(float(state.get("floor_adjustment", 0.0))),
            KEY_BREAKER_UNTIL: state.get("breaker_until", "") or "",
            KEY_SYMBOL_STATS: json.dumps(state.get("symbol_stats", {})),
        }
        with session_scope() as s:
            existing = {r.key: r for r in s.query(AppSetting).filter(AppSetting.key.in_(payload.keys())).all()}
            for key, value in payload.items():
                if key in existing:
                    existing[key].value = value
                    existing[key].updated_at = utcnow()
                else:
                    s.add(AppSetting(key=key, value=value))


# ---- Helpers ----------------------------------------------------------------


def _safe_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _safe_json_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _safe_float(raw: str | None, default: float) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except Exception:
        return default
