"""
Daily Mission Controller — the boss layer over the trade pipeline.

Every trade decision must pass through this controller before it reaches
execution. The controller maintains a state machine over today's P&L and
modulates the rest of the system based on which "mode" the day is in:

    SCOUT    — initial probing; tiny size, find which pairs behave today
    BUILD    — normal trading; default sizing, full strategy stack
    ATTACK   — we're in profit (>= soft zone); selectively larger size
    PROTECT  — we're near target (>= protect zone); A-grade trades only
    LOCK     — target hit; refuse new entries, let runners trail out
    RECOVERY — loss streak active; cooldowns + A+ trades only
    KILL     — daily loss limit / panic loss / manual kill; full stop

Why this exists: the rest of the pipeline (autonomous engine, strategic
router, Claude, risk gates) all reason per-trade. They don't know whether
we've already made $50 today and should stop trading marginal edges, or
whether we just lost three in a row and shouldn't be throwing capital at
the next signal. The mission controller is the OUTER LOOP that does.

Integration:
    from risk.daily_mission_controller import get_mission_controller
    mission = get_mission_controller()

    # Before placing the trade:
    decision = mission.approve_trade(
        symbol="BTC-USD",
        strategy="momentum",
        confidence=0.82,
        proposed_notional=200.0,
        expected_net_edge=0.09,         # $ edge expected net of costs
        spread_bps=3.2,                 # 0 acceptable if we don't track it
        volatility_score=0.45,          # 0..1, derived from atr_pct
        market_quality_score=0.78,      # optional composite
        router_wants_claude=True,
    )
    if not decision.approved:
        # decision.reason / decision.rejection_code explain why
        return
    final_position_usd = decision.approved_notional

    # After the trade closes:
    mission.record_trade_result(TradeResult(
        symbol="BTC-USD", strategy="momentum",
        pnl=12.50, fees=0.18,
    ))

Safety:
    The whole controller is gated by bot_config.get("mission_controller_enabled").
    When False (default), get_mission_controller() returns a NoopController that
    approves every trade. Flip the flag in Settings only when you're ready to
    enforce the Scout-mode confidence floor (~0.64) and the min-edge gate.

Persistence:
    State is serialized to an AppSetting row ("mission_state_v1") on every
    mutation. Reload happens lazily on first access. Daily rollover at UTC
    midnight wipes the per-day counters but keeps configured thresholds.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MissionMode(str, Enum):
    SCOUT = "SCOUT"
    BUILD = "BUILD"
    ATTACK = "ATTACK"
    PROTECT = "PROTECT"
    LOCK = "LOCK"
    RECOVERY = "RECOVERY"
    KILL = "KILL"


class TradeGrade(str, Enum):
    F = "F"
    D = "D"
    C = "C"
    B = "B"
    A = "A"
    A_PLUS = "A+"


class RejectionCode(str, Enum):
    NONE = "NONE"
    MODE_LOCKED = "MODE_LOCKED"
    MODE_KILL = "MODE_KILL"
    CONFIDENCE_TOO_LOW = "CONFIDENCE_TOO_LOW"
    EDGE_TOO_SMALL = "EDGE_TOO_SMALL"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    VOLATILITY_TOO_HIGH = "VOLATILITY_TOO_HIGH"
    SYMBOL_THROTTLED = "SYMBOL_THROTTLED"
    STRATEGY_THROTTLED = "STRATEGY_THROTTLED"
    SYMBOL_STRATEGY_THROTTLED = "SYMBOL_STRATEGY_THROTTLED"
    MAX_TRADES_REACHED = "MAX_TRADES_REACHED"
    MAX_HOURLY_TRADES_REACHED = "MAX_HOURLY_TRADES_REACHED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    CONSECUTIVE_LOSS_LIMIT = "CONSECUTIVE_LOSS_LIMIT"
    POSITION_SIZE_TOO_SMALL = "POSITION_SIZE_TOO_SMALL"
    POSITION_SIZE_TOO_LARGE = "POSITION_SIZE_TOO_LARGE"
    CLAUDE_NOT_ALLOWED = "CLAUDE_NOT_ALLOWED"
    MARKET_QUALITY_TOO_LOW = "MARKET_QUALITY_TOO_LOW"
    GRADE_TOO_LOW = "GRADE_TOO_LOW"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# =============================================================================
# CONFIG
# =============================================================================
# All thresholds in one place so a tuning session can edit them as data without
# hunting through code. Anything not in MissionConfig should not be a tunable.

@dataclass
class MissionConfig:
    # --- Daily target system -------------------------------------------------
    daily_net_target: float = 60.00
    soft_profit_zone: float = 30.00       # enter ATTACK
    protect_profit_zone: float = 45.00    # enter PROTECT
    lock_profit_zone: float = 60.00       # enter LOCK

    # --- Loss limits ---------------------------------------------------------
    max_daily_loss: float = -20.00        # KILL at this point (negative)
    panic_daily_loss: float = -30.00      # also KILL, harder; lockable until next day
    max_drawdown_from_peak: float = 18.00 # KILL if drawdown from peak exceeds this

    # --- Trade limits --------------------------------------------------------
    max_trades_per_day: int = 1000
    max_trades_per_hour: int = 90
    max_trades_per_symbol_per_hour: int = 35
    max_trades_per_strategy_per_hour: int = 45

    # --- Loss streak behavior ------------------------------------------------
    max_consecutive_losses: int = 4
    recovery_after_losses: int = 2
    recovery_cooldown_minutes: int = 20
    strategy_quarantine_losses: int = 3
    symbol_quarantine_losses: int = 3
    symbol_strategy_quarantine_losses: int = 2

    # --- Confidence floors per mode (0..1) -----------------------------------
    scout_confidence_floor: float = 0.64
    build_confidence_floor: float = 0.70
    attack_confidence_floor: float = 0.76
    protect_confidence_floor: float = 0.84
    recovery_confidence_floor: float = 0.86
    lock_confidence_floor: float = 1.00
    kill_confidence_floor: float = 1.00

    # --- Edge requirements per mode (dollars after costs) --------------------
    scout_min_edge: float = 0.03
    build_min_edge: float = 0.05
    attack_min_edge: float = 0.07
    protect_min_edge: float = 0.10
    recovery_min_edge: float = 0.12

    # --- Spread tolerance per mode (basis points) ----------------------------
    max_spread_bps_scout: float = 12.0
    max_spread_bps_build: float = 9.0
    max_spread_bps_attack: float = 7.0
    max_spread_bps_protect: float = 5.0
    max_spread_bps_recovery: float = 4.0

    # --- Volatility tolerance per mode (0..1) --------------------------------
    max_volatility_score_scout: float = 0.88
    max_volatility_score_build: float = 0.82
    max_volatility_score_attack: float = 0.78
    max_volatility_score_protect: float = 0.68
    max_volatility_score_recovery: float = 0.60

    # --- Position sizing multipliers per mode --------------------------------
    scout_size_multiplier: float = 0.25
    build_size_multiplier: float = 1.00
    attack_size_multiplier: float = 1.20
    protect_size_multiplier: float = 0.45
    recovery_size_multiplier: float = 0.25
    lock_size_multiplier: float = 0.00
    kill_size_multiplier: float = 0.00

    # --- Max notional per mode (dollars) -------------------------------------
    max_single_trade_notional: float = 500.00
    max_notional_scout: float = 75.00
    max_notional_build: float = 250.00
    max_notional_attack: float = 400.00
    max_notional_protect: float = 125.00
    max_notional_recovery: float = 75.00

    # --- Min notional (anything smaller is rejected as not worth the fees) ---
    min_trade_notional: float = 10.00

    # --- Claude consultation per mode ----------------------------------------
    allow_claude_in_scout: bool = True
    allow_claude_in_build: bool = True
    allow_claude_in_attack: bool = True
    allow_claude_in_protect: bool = False
    allow_claude_in_recovery: bool = False
    allow_claude_in_lock: bool = False
    allow_claude_in_kill: bool = False

    # --- Market-quality floor per mode (0..1) --------------------------------
    min_market_quality_scout: float = 0.50
    min_market_quality_build: float = 0.58
    min_market_quality_attack: float = 0.66
    min_market_quality_protect: float = 0.76
    min_market_quality_recovery: float = 0.80

    # --- Time behavior -------------------------------------------------------
    scout_min_minutes: int = 5             # stay in SCOUT for at least this long
    lock_trading_after_target: bool = True

    # --- Debug behavior ------------------------------------------------------
    verbose_debug: bool = False


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TradeDecision:
    approved: bool
    mode: MissionMode
    reason: str
    rejection_code: RejectionCode = RejectionCode.NONE

    symbol: Optional[str] = None
    strategy: Optional[str] = None

    confidence: Optional[float] = None
    confidence_floor: Optional[float] = None

    proposed_notional: Optional[float] = None
    approved_notional: Optional[float] = None
    size_multiplier: Optional[float] = None

    expected_net_edge: Optional[float] = None
    min_required_edge: Optional[float] = None

    spread_bps: Optional[float] = None
    max_allowed_spread_bps: Optional[float] = None

    volatility_score: Optional[float] = None
    max_allowed_volatility_score: Optional[float] = None

    market_quality_score: Optional[float] = None
    min_required_market_quality: Optional[float] = None

    claude_allowed: bool = False
    router_wants_claude: bool = False
    grade: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["rejection_code"] = self.rejection_code.value
        return data


@dataclass
class TradeResult:
    symbol: str
    strategy: str
    pnl: float
    fees: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    confidence: Optional[float] = None
    notional: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_net(self) -> float:
        if self.net_pnl is not None:
            return self.net_pnl
        return self.pnl - self.fees


@dataclass
class _PerKeyStats:
    """Per-symbol, per-strategy, and per-combo running stats. Shared shape so
    snapshot/persistence can serialize all three uniformly."""
    key: str
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    consecutive_losses: int = 0
    disabled_until: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    rejection_count: int = 0

    def win_rate(self) -> float:
        if self.trades_today <= 0:
            return 0.0
        return self.wins_today / self.trades_today

    def is_disabled(self, now: datetime) -> bool:
        return self.disabled_until is not None and now < self.disabled_until


# =============================================================================
# CONTROLLER
# =============================================================================

class DailyMissionController:
    """Mission state + approval gate. Thread-safe. Persisted to AppSetting."""

    # Setting key under which serialized state lives.
    _STATE_KEY = "mission_state_v1"

    def __init__(self, config: Optional[MissionConfig] = None):
        self.config = config or MissionConfig()
        self._lock = threading.RLock()

        # Run-time state (reloaded from persistence on first access).
        self.session_started_at: datetime = datetime.now(timezone.utc)
        self.current_day: date = self.session_started_at.date()

        self.mode: MissionMode = MissionMode.SCOUT
        self.previous_mode: Optional[MissionMode] = None
        self.mode_changed_at: datetime = self.session_started_at

        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.fees_paid: float = 0.0
        self.net_pnl: float = 0.0

        self.peak_net_pnl: float = 0.0
        self.lowest_net_pnl: float = 0.0
        self.drawdown_from_peak: float = 0.0

        self.total_trades_today: int = 0
        self.total_wins_today: int = 0
        self.total_losses_today: int = 0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0

        self.trade_timestamps: List[datetime] = []
        # Rejections live in-memory only (audit goes to ActivityLog too).
        self.rejections: List[TradeDecision] = []

        self.symbol_stats: Dict[str, _PerKeyStats] = {}
        self.strategy_stats: Dict[str, _PerKeyStats] = {}
        self.symbol_strategy_stats: Dict[str, _PerKeyStats] = {}

        self.manual_kill_enabled: bool = False
        self.kill_reason: Optional[str] = None

        self._loaded_from_persistence: bool = False

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """First-access lazy load. Idempotent. Failures degrade silently."""
        if self._loaded_from_persistence:
            return
        self._loaded_from_persistence = True
        try:
            from config import bot_config
            raw = bot_config.get(self._STATE_KEY)
            if raw:
                self._load_state(json.loads(raw))
        except Exception:
            logger.exception("Failed to load mission state from persistence")

    def _persist(self) -> None:
        """Save state to AppSetting. Called after every mutation. Best-effort."""
        try:
            from config import bot_config
            bot_config.set_many({self._STATE_KEY: json.dumps(self._dump_state())})
        except Exception:
            logger.exception("Failed to persist mission state")

    def _dump_state(self) -> Dict[str, Any]:
        return {
            "session_started_at": self.session_started_at.isoformat(),
            "current_day": self.current_day.isoformat(),
            "mode": self.mode.value,
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "mode_changed_at": self.mode_changed_at.isoformat(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
            "net_pnl": self.net_pnl,
            "peak_net_pnl": self.peak_net_pnl,
            "lowest_net_pnl": self.lowest_net_pnl,
            "drawdown_from_peak": self.drawdown_from_peak,
            "total_trades_today": self.total_trades_today,
            "total_wins_today": self.total_wins_today,
            "total_losses_today": self.total_losses_today,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "trade_timestamps": [t.isoformat() for t in self.trade_timestamps],
            "symbol_stats": {k: self._dump_stats(v) for k, v in self.symbol_stats.items()},
            "strategy_stats": {k: self._dump_stats(v) for k, v in self.strategy_stats.items()},
            "symbol_strategy_stats": {k: self._dump_stats(v) for k, v in self.symbol_strategy_stats.items()},
            "manual_kill_enabled": self.manual_kill_enabled,
            "kill_reason": self.kill_reason,
        }

    @staticmethod
    def _dump_stats(s: _PerKeyStats) -> Dict[str, Any]:
        return {
            "key": s.key,
            "trades_today": s.trades_today,
            "wins_today": s.wins_today,
            "losses_today": s.losses_today,
            "pnl_today": s.pnl_today,
            "consecutive_losses": s.consecutive_losses,
            "disabled_until": s.disabled_until.isoformat() if s.disabled_until else None,
            "last_trade_at": s.last_trade_at.isoformat() if s.last_trade_at else None,
            "rejection_count": s.rejection_count,
        }

    def _load_state(self, data: Dict[str, Any]) -> None:
        # Tolerate missing fields — schema may evolve.
        def _dt(v):
            return datetime.fromisoformat(v) if v else None

        self.session_started_at = _dt(data.get("session_started_at")) or datetime.now(timezone.utc)
        self.current_day = date.fromisoformat(data["current_day"]) if data.get("current_day") else self.current_day
        self.mode = MissionMode(data.get("mode", MissionMode.SCOUT.value))
        prev = data.get("previous_mode")
        self.previous_mode = MissionMode(prev) if prev else None
        self.mode_changed_at = _dt(data.get("mode_changed_at")) or self.session_started_at

        self.realized_pnl = float(data.get("realized_pnl") or 0.0)
        self.unrealized_pnl = float(data.get("unrealized_pnl") or 0.0)
        self.fees_paid = float(data.get("fees_paid") or 0.0)
        self.net_pnl = float(data.get("net_pnl") or 0.0)
        self.peak_net_pnl = float(data.get("peak_net_pnl") or 0.0)
        self.lowest_net_pnl = float(data.get("lowest_net_pnl") or 0.0)
        self.drawdown_from_peak = float(data.get("drawdown_from_peak") or 0.0)

        self.total_trades_today = int(data.get("total_trades_today") or 0)
        self.total_wins_today = int(data.get("total_wins_today") or 0)
        self.total_losses_today = int(data.get("total_losses_today") or 0)
        self.consecutive_losses = int(data.get("consecutive_losses") or 0)
        self.consecutive_wins = int(data.get("consecutive_wins") or 0)

        self.trade_timestamps = [datetime.fromisoformat(t) for t in data.get("trade_timestamps") or [] if t]

        def _load_stats(d: Dict[str, Any]) -> _PerKeyStats:
            return _PerKeyStats(
                key=d.get("key", ""),
                trades_today=int(d.get("trades_today") or 0),
                wins_today=int(d.get("wins_today") or 0),
                losses_today=int(d.get("losses_today") or 0),
                pnl_today=float(d.get("pnl_today") or 0.0),
                consecutive_losses=int(d.get("consecutive_losses") or 0),
                disabled_until=_dt(d.get("disabled_until")),
                last_trade_at=_dt(d.get("last_trade_at")),
                rejection_count=int(d.get("rejection_count") or 0),
            )

        self.symbol_stats = {k: _load_stats(v) for k, v in (data.get("symbol_stats") or {}).items()}
        self.strategy_stats = {k: _load_stats(v) for k, v in (data.get("strategy_stats") or {}).items()}
        self.symbol_strategy_stats = {k: _load_stats(v) for k, v in (data.get("symbol_strategy_stats") or {}).items()}

        self.manual_kill_enabled = bool(data.get("manual_kill_enabled"))
        self.kill_reason = data.get("kill_reason")

    # -------------------------------------------------------------------------
    # PUBLIC STATE API
    # -------------------------------------------------------------------------

    def current_mode(self) -> MissionMode:
        with self._lock:
            self._ensure_loaded()
            self._rollover_if_needed()
            self._evaluate_mode_transition()
            return self.mode

    def update_pnl(
        self,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        fees_paid: Optional[float] = None,
    ) -> MissionMode:
        with self._lock:
            self._ensure_loaded()
            self._rollover_if_needed()

            if realized_pnl is not None:
                self.realized_pnl = float(realized_pnl)
            if unrealized_pnl is not None:
                self.unrealized_pnl = float(unrealized_pnl)
            if fees_paid is not None:
                self.fees_paid = float(fees_paid)

            self._recompute_net()
            self._evaluate_mode_transition()
            self._persist()
            return self.mode

    def record_trade_result(self, result: TradeResult) -> MissionMode:
        with self._lock:
            self._ensure_loaded()
            self._rollover_if_needed()

            net = result.effective_net()

            self.trade_timestamps.append(result.timestamp)
            self.total_trades_today += 1

            sym_stat = self._stats_for(self.symbol_stats, result.symbol.upper().strip())
            strat_stat = self._stats_for(self.strategy_stats, result.strategy.strip())
            combo_key = f"{result.symbol.upper().strip()}::{result.strategy.strip()}"
            combo_stat = self._stats_for(self.symbol_strategy_stats, combo_key)

            for s in (sym_stat, strat_stat, combo_stat):
                s.trades_today += 1
                s.pnl_today += net
                s.last_trade_at = result.timestamp

            if net > 0:
                self.total_wins_today += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                for s in (sym_stat, strat_stat, combo_stat):
                    s.wins_today += 1
                    s.consecutive_losses = 0
            elif net < 0:
                self.total_losses_today += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                for s in (sym_stat, strat_stat, combo_stat):
                    s.losses_today += 1
                    s.consecutive_losses += 1
                self._apply_loss_throttles(
                    symbol_key=sym_stat.key,
                    strategy_key=strat_stat.key,
                    combo_key=combo_stat.key,
                    timestamp=result.timestamp,
                )
            # net == 0 (breakeven): record as neither win nor loss; reset losing streak.
            else:
                self.consecutive_losses = 0

            self.realized_pnl += net
            self.fees_paid += float(result.fees or 0.0)
            self._recompute_net()

            logger.info(
                "[MISSION] Trade recorded sym=%s strat=%s net=%+.4f total=%+.2f mode=%s losses_in_row=%d",
                result.symbol, result.strategy, net, self.net_pnl,
                self.mode.value, self.consecutive_losses,
            )

            self._evaluate_mode_transition()
            self._persist()
            return self.mode

    # -------------------------------------------------------------------------
    # APPROVAL GATE
    # -------------------------------------------------------------------------

    def approve_trade(
        self,
        *,
        symbol: str,
        strategy: str,
        confidence: float,
        proposed_notional: float,
        expected_net_edge: float,
        spread_bps: float = 0.0,
        volatility_score: float = 0.0,
        market_quality_score: Optional[float] = None,
        router_wants_claude: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeDecision:
        with self._lock:
            self._ensure_loaded()
            self._rollover_if_needed()
            self._evaluate_mode_transition()

            now = datetime.now(timezone.utc)
            mode = self.mode

            symbol_n = str(symbol).upper().strip()
            strategy_n = str(strategy).strip()

            confidence_floor = self.get_confidence_floor(mode)
            min_edge = self.get_min_required_edge(mode)
            max_spread = self.get_max_allowed_spread_bps(mode)
            max_vol = self.get_max_allowed_volatility_score(mode)
            min_quality = self.get_min_required_market_quality(mode)
            size_mult = self.get_position_size_multiplier(mode)
            max_notional = self.get_max_notional(mode)
            claude_allowed = self.should_call_claude(mode)

            approved_notional = min(
                proposed_notional * size_mult,
                max_notional,
                self.config.max_single_trade_notional,
            )

            base = dict(
                symbol=symbol_n, strategy=strategy_n,
                confidence=confidence, confidence_floor=confidence_floor,
                proposed_notional=proposed_notional,
                approved_notional=approved_notional,
                size_multiplier=size_mult,
                expected_net_edge=expected_net_edge,
                min_required_edge=min_edge,
                spread_bps=spread_bps, max_allowed_spread_bps=max_spread,
                volatility_score=volatility_score, max_allowed_volatility_score=max_vol,
                market_quality_score=market_quality_score,
                min_required_market_quality=min_quality,
                claude_allowed=claude_allowed, router_wants_claude=router_wants_claude,
                metadata=metadata or {},
            )

            def reject(code: RejectionCode, reason: str) -> TradeDecision:
                d = TradeDecision(approved=False, mode=mode, reason=reason, rejection_code=code, **base)
                self._record_rejection(d)
                return d

            # Hard blocks — mode-level kills, before any other check.
            if self.manual_kill_enabled:
                return reject(RejectionCode.MODE_KILL,
                              f"Manual kill switch on: {self.kill_reason or 'no reason given'}")
            if mode == MissionMode.KILL:
                return reject(RejectionCode.MODE_KILL,
                              f"Mode is KILL: {self.kill_reason or 'kill state'}")
            if mode == MissionMode.LOCK:
                return reject(RejectionCode.MODE_LOCKED,
                              f"Daily target locked at +${self.config.lock_profit_zone:.2f}")

            # Daily limits.
            if self.net_pnl <= self.config.max_daily_loss:
                self._set_mode(MissionMode.KILL, "Max daily loss")
                return reject(RejectionCode.DAILY_LOSS_LIMIT,
                              f"Max daily loss reached: net=${self.net_pnl:.2f}")
            if self.drawdown_from_peak >= self.config.max_drawdown_from_peak:
                return reject(RejectionCode.DRAWDOWN_LIMIT,
                              f"Drawdown ${self.drawdown_from_peak:.2f} exceeds ${self.config.max_drawdown_from_peak:.2f}")
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self._set_mode(MissionMode.KILL, "Consecutive loss limit")
                return reject(RejectionCode.CONSECUTIVE_LOSS_LIMIT,
                              f"{self.consecutive_losses} consecutive losses")

            # Rate limits.
            if self.total_trades_today >= self.config.max_trades_per_day:
                return reject(RejectionCode.MAX_TRADES_REACHED,
                              f"Daily trade cap {self.config.max_trades_per_day} hit")
            if self._count_trades_since(now - timedelta(hours=1)) >= self.config.max_trades_per_hour:
                return reject(RejectionCode.MAX_HOURLY_TRADES_REACHED, "Hourly trade cap hit")

            sym_stat = self._stats_for(self.symbol_stats, symbol_n)
            strat_stat = self._stats_for(self.strategy_stats, strategy_n)
            combo_stat = self._stats_for(self.symbol_strategy_stats, f"{symbol_n}::{strategy_n}")

            if sym_stat.is_disabled(now):
                return reject(RejectionCode.SYMBOL_THROTTLED,
                              f"{symbol_n} quarantined until {sym_stat.disabled_until}")
            if strat_stat.is_disabled(now):
                return reject(RejectionCode.STRATEGY_THROTTLED,
                              f"{strategy_n} quarantined until {strat_stat.disabled_until}")
            if combo_stat.is_disabled(now):
                return reject(RejectionCode.SYMBOL_STRATEGY_THROTTLED,
                              f"{symbol_n}/{strategy_n} quarantined until {combo_stat.disabled_until}")

            # Per-hour per-symbol/strategy.
            if sym_stat.trades_today and self._count_recent_symbol(symbol_n, now - timedelta(hours=1)) >= self.config.max_trades_per_symbol_per_hour:
                return reject(RejectionCode.SYMBOL_THROTTLED,
                              f"{symbol_n} hourly cap hit")
            if strat_stat.trades_today and self._count_recent_strategy(strategy_n, now - timedelta(hours=1)) >= self.config.max_trades_per_strategy_per_hour:
                return reject(RejectionCode.STRATEGY_THROTTLED,
                              f"{strategy_n} hourly cap hit")

            # Quality gates — confidence, edge, spread, volatility, market quality.
            if confidence < confidence_floor:
                return reject(RejectionCode.CONFIDENCE_TOO_LOW,
                              f"conf {confidence:.2f} < floor {confidence_floor:.2f} for {mode.value}")
            if expected_net_edge < min_edge:
                return reject(RejectionCode.EDGE_TOO_SMALL,
                              f"edge ${expected_net_edge:.3f} < min ${min_edge:.3f}")
            if spread_bps > max_spread:
                return reject(RejectionCode.SPREAD_TOO_WIDE,
                              f"spread {spread_bps:.1f}bps > max {max_spread:.1f}bps")
            if volatility_score > max_vol:
                return reject(RejectionCode.VOLATILITY_TOO_HIGH,
                              f"vol {volatility_score:.2f} > max {max_vol:.2f}")
            if market_quality_score is not None and market_quality_score < min_quality:
                return reject(RejectionCode.MARKET_QUALITY_TOO_LOW,
                              f"quality {market_quality_score:.2f} < min {min_quality:.2f}")

            # Claude budget.
            if router_wants_claude and not claude_allowed:
                return reject(RejectionCode.CLAUDE_NOT_ALLOWED,
                              f"Claude consults disabled in {mode.value}")

            # Sizing.
            if approved_notional < self.config.min_trade_notional:
                return reject(RejectionCode.POSITION_SIZE_TOO_SMALL,
                              f"approved notional ${approved_notional:.2f} < min ${self.config.min_trade_notional:.2f}")
            if approved_notional > max_notional:
                return reject(RejectionCode.POSITION_SIZE_TOO_LARGE,
                              f"approved notional ${approved_notional:.2f} > max ${max_notional:.2f}")

            # Final grade check (PROTECT/RECOVERY demand top-tier setups).
            grade = self.grade_trade(
                confidence=confidence, expected_net_edge=expected_net_edge,
                spread_bps=spread_bps, volatility_score=volatility_score,
                market_quality_score=market_quality_score,
            )
            if mode == MissionMode.PROTECT and grade not in {TradeGrade.A, TradeGrade.A_PLUS}:
                return reject(RejectionCode.GRADE_TOO_LOW,
                              f"PROTECT needs A or A+; got {grade.value}")
            if mode == MissionMode.RECOVERY and grade != TradeGrade.A_PLUS:
                return reject(RejectionCode.GRADE_TOO_LOW,
                              f"RECOVERY needs A+ only; got {grade.value}")

            decision = TradeDecision(
                approved=True, mode=mode,
                reason=f"approved in {mode.value} (grade {grade.value})",
                rejection_code=RejectionCode.NONE,
                grade=grade.value,
                **base,
            )

            logger.info(
                "[MISSION] APPROVE mode=%s sym=%s strat=%s conf=%.2f edge=$%.3f notional=$%.2f grade=%s",
                mode.value, symbol_n, strategy_n, confidence,
                expected_net_edge, approved_notional, grade.value,
            )
            return decision

    # -------------------------------------------------------------------------
    # MODE POLICIES
    # -------------------------------------------------------------------------

    def get_confidence_floor(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.scout_confidence_floor,
            MissionMode.BUILD: self.config.build_confidence_floor,
            MissionMode.ATTACK: self.config.attack_confidence_floor,
            MissionMode.PROTECT: self.config.protect_confidence_floor,
            MissionMode.RECOVERY: self.config.recovery_confidence_floor,
            MissionMode.LOCK: self.config.lock_confidence_floor,
            MissionMode.KILL: self.config.kill_confidence_floor,
        })

    def get_min_required_edge(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.scout_min_edge,
            MissionMode.BUILD: self.config.build_min_edge,
            MissionMode.ATTACK: self.config.attack_min_edge,
            MissionMode.PROTECT: self.config.protect_min_edge,
            MissionMode.RECOVERY: self.config.recovery_min_edge,
            MissionMode.LOCK: math.inf,
            MissionMode.KILL: math.inf,
        })

    def get_position_size_multiplier(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.scout_size_multiplier,
            MissionMode.BUILD: self.config.build_size_multiplier,
            MissionMode.ATTACK: self.config.attack_size_multiplier,
            MissionMode.PROTECT: self.config.protect_size_multiplier,
            MissionMode.RECOVERY: self.config.recovery_size_multiplier,
            MissionMode.LOCK: self.config.lock_size_multiplier,
            MissionMode.KILL: self.config.kill_size_multiplier,
        })

    def get_max_notional(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.max_notional_scout,
            MissionMode.BUILD: self.config.max_notional_build,
            MissionMode.ATTACK: self.config.max_notional_attack,
            MissionMode.PROTECT: self.config.max_notional_protect,
            MissionMode.RECOVERY: self.config.max_notional_recovery,
            MissionMode.LOCK: 0.0,
            MissionMode.KILL: 0.0,
        })

    def get_max_allowed_spread_bps(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.max_spread_bps_scout,
            MissionMode.BUILD: self.config.max_spread_bps_build,
            MissionMode.ATTACK: self.config.max_spread_bps_attack,
            MissionMode.PROTECT: self.config.max_spread_bps_protect,
            MissionMode.RECOVERY: self.config.max_spread_bps_recovery,
            MissionMode.LOCK: 0.0,
            MissionMode.KILL: 0.0,
        })

    def get_max_allowed_volatility_score(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.max_volatility_score_scout,
            MissionMode.BUILD: self.config.max_volatility_score_build,
            MissionMode.ATTACK: self.config.max_volatility_score_attack,
            MissionMode.PROTECT: self.config.max_volatility_score_protect,
            MissionMode.RECOVERY: self.config.max_volatility_score_recovery,
            MissionMode.LOCK: 0.0,
            MissionMode.KILL: 0.0,
        })

    def get_min_required_market_quality(self, mode: Optional[MissionMode] = None) -> float:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.min_market_quality_scout,
            MissionMode.BUILD: self.config.min_market_quality_build,
            MissionMode.ATTACK: self.config.min_market_quality_attack,
            MissionMode.PROTECT: self.config.min_market_quality_protect,
            MissionMode.RECOVERY: self.config.min_market_quality_recovery,
            MissionMode.LOCK: 1.0,
            MissionMode.KILL: 1.0,
        })

    def should_call_claude(self, mode: Optional[MissionMode] = None) -> bool:
        return self._policy_lookup(mode, {
            MissionMode.SCOUT: self.config.allow_claude_in_scout,
            MissionMode.BUILD: self.config.allow_claude_in_build,
            MissionMode.ATTACK: self.config.allow_claude_in_attack,
            MissionMode.PROTECT: self.config.allow_claude_in_protect,
            MissionMode.RECOVERY: self.config.allow_claude_in_recovery,
            MissionMode.LOCK: self.config.allow_claude_in_lock,
            MissionMode.KILL: self.config.allow_claude_in_kill,
        })

    def get_mode_policy(self, mode: Optional[MissionMode] = None) -> Dict[str, Any]:
        m = mode or self.mode
        return {
            "mode": m.value,
            "confidence_floor": self.get_confidence_floor(m),
            "min_required_edge": self.get_min_required_edge(m),
            "size_multiplier": self.get_position_size_multiplier(m),
            "max_notional": self.get_max_notional(m),
            "max_spread_bps": self.get_max_allowed_spread_bps(m),
            "max_volatility_score": self.get_max_allowed_volatility_score(m),
            "min_market_quality": self.get_min_required_market_quality(m),
            "claude_allowed": self.should_call_claude(m),
        }

    def _policy_lookup(self, mode: Optional[MissionMode], mapping: Dict[MissionMode, Any]):
        return mapping[mode or self.mode]

    # -------------------------------------------------------------------------
    # GRADING
    # -------------------------------------------------------------------------

    def grade_trade(
        self,
        confidence: float,
        expected_net_edge: float,
        spread_bps: float,
        volatility_score: float,
        market_quality_score: Optional[float] = None,
    ) -> TradeGrade:
        """Composite 0-100 trade score → letter grade. Weights are heuristic but
        intentionally weight confidence highest (45 pts) since it's the most
        information-dense input we have. Tunable but treat changes carefully —
        PROTECT and RECOVERY gates trip on grade alone."""
        score = 0.0
        score += min(max(confidence, 0.0), 1.0) * 45.0
        score += min(max(expected_net_edge / 0.20, 0.0), 1.0) * 25.0
        score += (1.0 - min(max(spread_bps / 15.0, 0.0), 1.0)) * 12.0
        score += (1.0 - min(max(volatility_score, 0.0), 1.0)) * 8.0
        if market_quality_score is not None:
            score += min(max(market_quality_score, 0.0), 1.0) * 10.0
        else:
            score += 5.0  # neutral baseline

        if score >= 90: return TradeGrade.A_PLUS
        if score >= 80: return TradeGrade.A
        if score >= 70: return TradeGrade.B
        if score >= 60: return TradeGrade.C
        if score >= 50: return TradeGrade.D
        return TradeGrade.F

    # -------------------------------------------------------------------------
    # MANUAL CONTROLS
    # -------------------------------------------------------------------------

    def enable_manual_kill(self, reason: str = "Manual kill") -> None:
        with self._lock:
            self._ensure_loaded()
            self.manual_kill_enabled = True
            self.kill_reason = reason
            self._set_mode(MissionMode.KILL, reason)
            self._persist()

    def disable_manual_kill(self) -> None:
        with self._lock:
            self._ensure_loaded()
            self.manual_kill_enabled = False
            self.kill_reason = None
            logger.warning("[MISSION] Manual kill disabled — re-evaluating mode")
            self._evaluate_mode_transition()
            self._persist()

    def force_lock(self, reason: str = "Manual lock") -> None:
        with self._lock:
            self._ensure_loaded()
            self._set_mode(MissionMode.LOCK, reason)
            self._persist()

    def disable_symbol(self, symbol: str, minutes: int, reason: str = "") -> None:
        with self._lock:
            self._ensure_loaded()
            s = self._stats_for(self.symbol_stats, symbol.upper().strip())
            s.disabled_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            logger.warning("[MISSION] Symbol %s disabled until %s (%s)", symbol, s.disabled_until, reason)
            self._persist()

    def disable_strategy(self, strategy: str, minutes: int, reason: str = "") -> None:
        with self._lock:
            self._ensure_loaded()
            s = self._stats_for(self.strategy_stats, strategy.strip())
            s.disabled_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            logger.warning("[MISSION] Strategy %s disabled until %s (%s)", strategy, s.disabled_until, reason)
            self._persist()

    # -------------------------------------------------------------------------
    # SNAPSHOTS / REPORTING
    # -------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Read-only snapshot for the UI/feed. Cheap to call from a poll loop."""
        with self._lock:
            self._ensure_loaded()
            self._rollover_if_needed()
            self._evaluate_mode_transition()

            now = datetime.now(timezone.utc)
            return {
                "timestamp": now.isoformat(),
                "mode": self.mode.value,
                "previous_mode": self.previous_mode.value if self.previous_mode else None,
                "mode_changed_at": self.mode_changed_at.isoformat(),
                "session_started_at": self.session_started_at.isoformat(),
                "manual_kill_enabled": self.manual_kill_enabled,
                "kill_reason": self.kill_reason,
                "pnl": {
                    "realized": round(self.realized_pnl, 2),
                    "unrealized": round(self.unrealized_pnl, 2),
                    "fees_paid": round(self.fees_paid, 2),
                    "net": round(self.net_pnl, 2),
                    "target": self.config.daily_net_target,
                    "remaining_to_target": round(max(self.config.daily_net_target - self.net_pnl, 0.0), 2),
                    "peak": round(self.peak_net_pnl, 2),
                    "lowest": round(self.lowest_net_pnl, 2),
                    "drawdown_from_peak": round(self.drawdown_from_peak, 2),
                },
                "trades": {
                    "total_today": self.total_trades_today,
                    "wins_today": self.total_wins_today,
                    "losses_today": self.total_losses_today,
                    "win_rate": round(self.win_rate(), 4),
                    "consecutive_losses": self.consecutive_losses,
                    "consecutive_wins": self.consecutive_wins,
                    "in_last_hour": self._count_trades_since(now - timedelta(hours=1)),
                },
                "policy": self.get_mode_policy(self.mode),
                "limits": {
                    "max_daily_loss": self.config.max_daily_loss,
                    "max_drawdown_from_peak": self.config.max_drawdown_from_peak,
                    "max_consecutive_losses": self.config.max_consecutive_losses,
                    "max_trades_per_day": self.config.max_trades_per_day,
                    "max_trades_per_hour": self.config.max_trades_per_hour,
                },
                "disabled_symbols": [
                    {"symbol": s.key, "until": s.disabled_until.isoformat()}
                    for s in self.symbol_stats.values()
                    if s.disabled_until and s.disabled_until > now
                ],
                "disabled_strategies": [
                    {"strategy": s.key, "until": s.disabled_until.isoformat()}
                    for s in self.strategy_stats.values()
                    if s.disabled_until and s.disabled_until > now
                ],
                "recent_rejections": [r.to_dict() for r in self.rejections[-20:]],
            }

    def win_rate(self) -> float:
        return (self.total_wins_today / self.total_trades_today) if self.total_trades_today else 0.0

    # -------------------------------------------------------------------------
    # INTERNAL: MODE TRANSITION
    # -------------------------------------------------------------------------

    def _evaluate_mode_transition(self) -> None:
        """Decide which mode we should be in based on current state. Idempotent.
        Each branch returns early; precedence matters — KILL/LOCK before
        profit-zone or recovery, etc."""
        now = datetime.now(timezone.utc)

        # Hardest blocks first.
        if self.manual_kill_enabled:
            self._set_mode(MissionMode.KILL, self.kill_reason or "Manual kill")
            return
        if self.net_pnl <= self.config.panic_daily_loss:
            self._set_mode(MissionMode.KILL, "Panic daily loss")
            return
        if self.net_pnl <= self.config.max_daily_loss:
            self._set_mode(MissionMode.KILL, "Max daily loss")
            return
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self._set_mode(MissionMode.KILL, "Consecutive loss limit")
            return
        if self.drawdown_from_peak >= self.config.max_drawdown_from_peak:
            self._set_mode(MissionMode.KILL, "Max drawdown from peak")
            return

        # Lock once target reached.
        if self.config.lock_trading_after_target and self.net_pnl >= self.config.lock_profit_zone:
            self._set_mode(MissionMode.LOCK, f"Daily target ${self.config.lock_profit_zone:.2f} reached")
            return

        # Recovery on a loss streak (less severe than KILL).
        if self.consecutive_losses >= self.config.recovery_after_losses:
            self._set_mode(MissionMode.RECOVERY, "Loss streak — recovery mode")
            return

        # Coming out of recovery — escalate by where P&L sits now.
        if self.mode == MissionMode.RECOVERY and self.consecutive_losses == 0:
            if self.net_pnl >= self.config.protect_profit_zone:
                self._set_mode(MissionMode.PROTECT, "Recovered into protect zone")
                return
            if self.net_pnl >= self.config.soft_profit_zone:
                self._set_mode(MissionMode.ATTACK, "Recovered into attack zone")
                return
            self._set_mode(MissionMode.BUILD, "Recovered into build mode")
            return

        # Initial scout window — give the day time to find a pair before going hot.
        minutes_running = (now - self.session_started_at).total_seconds() / 60.0
        if minutes_running < self.config.scout_min_minutes and self.total_trades_today < 5:
            self._set_mode(MissionMode.SCOUT, "Initial scout window")
            return

        # Profit-zone progression.
        if self.net_pnl >= self.config.protect_profit_zone:
            self._set_mode(MissionMode.PROTECT, "Protect zone reached")
            return
        if self.net_pnl >= self.config.soft_profit_zone:
            self._set_mode(MissionMode.ATTACK, "Soft profit zone reached")
            return

        # Default — BUILD if we have any trades, SCOUT otherwise.
        if self.total_trades_today == 0:
            self._set_mode(MissionMode.SCOUT, "No trades yet")
            return
        self._set_mode(MissionMode.BUILD, "Normal build mode")

    def _set_mode(self, new_mode: MissionMode, reason: str) -> None:
        if self.mode != new_mode:
            self.previous_mode = self.mode
            self.mode = new_mode
            self.mode_changed_at = datetime.now(timezone.utc)
            logger.warning(
                "[MISSION] mode %s -> %s | %s | net=$%.2f drawdown=$%.2f",
                self.previous_mode.value if self.previous_mode else "?",
                self.mode.value, reason, self.net_pnl, self.drawdown_from_peak,
            )
        elif self.config.verbose_debug:
            logger.debug("[MISSION] mode stays %s | %s", self.mode.value, reason)

    # -------------------------------------------------------------------------
    # INTERNAL: LOSS QUARANTINE
    # -------------------------------------------------------------------------

    def _apply_loss_throttles(self, symbol_key: str, strategy_key: str, combo_key: str, timestamp: datetime) -> None:
        sym = self._stats_for(self.symbol_stats, symbol_key)
        strat = self._stats_for(self.strategy_stats, strategy_key)
        combo = self._stats_for(self.symbol_strategy_stats, combo_key)

        recovery_x = 2 if self.mode == MissionMode.RECOVERY else 1

        if sym.consecutive_losses >= self.config.symbol_quarantine_losses:
            sym.disabled_until = timestamp + timedelta(minutes=self.config.recovery_cooldown_minutes * recovery_x)
            logger.warning("[MISSION] symbol quarantine %s losses=%d until=%s", symbol_key, sym.consecutive_losses, sym.disabled_until)
        if strat.consecutive_losses >= self.config.strategy_quarantine_losses:
            strat.disabled_until = timestamp + timedelta(minutes=self.config.recovery_cooldown_minutes * recovery_x)
            logger.warning("[MISSION] strategy quarantine %s losses=%d until=%s", strategy_key, strat.consecutive_losses, strat.disabled_until)
        if combo.consecutive_losses >= self.config.symbol_strategy_quarantine_losses:
            combo.disabled_until = timestamp + timedelta(minutes=self.config.recovery_cooldown_minutes * recovery_x)
            logger.warning("[MISSION] combo quarantine %s losses=%d until=%s", combo_key, combo.consecutive_losses, combo.disabled_until)

    def _record_rejection(self, decision: TradeDecision) -> None:
        self.rejections.append(decision)
        if len(self.rejections) > 200:
            self.rejections = self.rejections[-200:]
        if decision.symbol:
            self._stats_for(self.symbol_stats, decision.symbol).rejection_count += 1
        if decision.strategy:
            self._stats_for(self.strategy_stats, decision.strategy).rejection_count += 1
        if self.config.verbose_debug:
            logger.debug("[MISSION] reject %s %s/%s code=%s reason=%s",
                         decision.mode.value, decision.symbol, decision.strategy,
                         decision.rejection_code.value, decision.reason)

    # -------------------------------------------------------------------------
    # INTERNAL: COUNTING HELPERS
    # -------------------------------------------------------------------------

    def _count_trades_since(self, since: datetime) -> int:
        return sum(1 for ts in self.trade_timestamps if ts >= since)

    def _count_recent_symbol(self, symbol_key: str, since: datetime) -> int:
        # We don't store per-trade symbol in trade_timestamps for memory reasons.
        # Conservative approximation: use sym_stat.last_trade_at as a coarse gate.
        # Tight per-hour symbol counting requires a recent_trades ring buffer; deferred.
        sym = self._stats_for(self.symbol_stats, symbol_key)
        if sym.last_trade_at and sym.last_trade_at >= since:
            return sym.trades_today  # over-estimate when active; under when not
        return 0

    def _count_recent_strategy(self, strategy_key: str, since: datetime) -> int:
        strat = self._stats_for(self.strategy_stats, strategy_key)
        if strat.last_trade_at and strat.last_trade_at >= since:
            return strat.trades_today
        return 0

    def _stats_for(self, store: Dict[str, _PerKeyStats], key: str) -> _PerKeyStats:
        if key not in store:
            store[key] = _PerKeyStats(key=key)
        return store[key]

    def _recompute_net(self) -> None:
        self.net_pnl = self.realized_pnl + self.unrealized_pnl - self.fees_paid
        self.peak_net_pnl = max(self.peak_net_pnl, self.net_pnl)
        self.lowest_net_pnl = min(self.lowest_net_pnl, self.net_pnl)
        self.drawdown_from_peak = self.peak_net_pnl - self.net_pnl

    # -------------------------------------------------------------------------
    # DAILY ROLLOVER
    # -------------------------------------------------------------------------

    def _rollover_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.current_day:
            logger.warning("[MISSION] New UTC day — resetting daily counters (%s -> %s)", self.current_day, today)
            self.reset_for_new_day()
            self._persist()

    def reset_for_new_day(self) -> None:
        """Wipe per-day state. Config stays. Persistence is NOT erased — we
        snapshot a fresh state on top of yesterday's row."""
        self.session_started_at = datetime.now(timezone.utc)
        self.current_day = self.session_started_at.date()

        self.mode = MissionMode.SCOUT
        self.previous_mode = None
        self.mode_changed_at = self.session_started_at

        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fees_paid = 0.0
        self.net_pnl = 0.0
        self.peak_net_pnl = 0.0
        self.lowest_net_pnl = 0.0
        self.drawdown_from_peak = 0.0

        self.total_trades_today = 0
        self.total_wins_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        self.trade_timestamps.clear()
        self.rejections.clear()
        self.symbol_stats.clear()
        self.strategy_stats.clear()
        self.symbol_strategy_stats.clear()

        self.manual_kill_enabled = False
        self.kill_reason = None


# =============================================================================
# NOOP CONTROLLER — used when mission_controller_enabled is False
# =============================================================================

class _NoopMissionController:
    """Drop-in stand-in that approves everything. Used when the operator has
    not yet flipped the mission_controller_enabled flag in Settings. Keeps
    the integration call sites simple — bot_engine doesn't need to branch on
    "is the controller enabled" everywhere."""

    def __init__(self) -> None:
        self._fake_mode = MissionMode.BUILD

    def current_mode(self) -> MissionMode:
        return self._fake_mode

    def approve_trade(self, **kwargs) -> TradeDecision:
        return TradeDecision(
            approved=True,
            mode=self._fake_mode,
            reason="mission controller disabled (passthrough)",
            rejection_code=RejectionCode.NONE,
            symbol=kwargs.get("symbol"),
            strategy=kwargs.get("strategy"),
            confidence=kwargs.get("confidence"),
            proposed_notional=kwargs.get("proposed_notional"),
            approved_notional=kwargs.get("proposed_notional"),
            size_multiplier=1.0,
            expected_net_edge=kwargs.get("expected_net_edge"),
            spread_bps=kwargs.get("spread_bps"),
            volatility_score=kwargs.get("volatility_score"),
            market_quality_score=kwargs.get("market_quality_score"),
            router_wants_claude=kwargs.get("router_wants_claude", False),
            claude_allowed=True,
            grade="B",
            metadata=kwargs.get("metadata") or {},
        )

    def record_trade_result(self, *_args, **_kwargs) -> MissionMode:
        return self._fake_mode

    def update_pnl(self, *_args, **_kwargs) -> MissionMode:
        return self._fake_mode

    def snapshot(self) -> Dict[str, Any]:
        return {"mode": self._fake_mode.value, "enabled": False}

    def should_call_claude(self, *_a, **_k) -> bool:
        return True

    def get_mode_policy(self, *_a, **_k) -> Dict[str, Any]:
        return {"mode": self._fake_mode.value, "enabled": False}

    @property
    def enabled(self) -> bool:
        return False


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_controller: Optional[DailyMissionController] = None
_lock = threading.Lock()


def get_mission_controller() -> Any:
    """Return the singleton controller, or a no-op stand-in if the operator
    has not enabled it. Cheap to call repeatedly. Reads
    `mission_controller_enabled` from bot_config on every call so the flag
    can be flipped at runtime without a restart."""
    enabled = False
    try:
        from config import bot_config
        raw = bot_config.get("mission_controller_enabled")
        enabled = str(raw or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        enabled = False

    if not enabled:
        return _NoopMissionController()

    global _controller
    with _lock:
        if _controller is None:
            _controller = DailyMissionController()
        return _controller


def is_enabled() -> bool:
    return not isinstance(get_mission_controller(), _NoopMissionController)
