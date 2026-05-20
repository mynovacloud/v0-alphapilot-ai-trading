"""
Position Monitor - Auto-exit engine for open positions.

Runs every tick to check if any open positions have hit their:
  - Stop-loss price
  - Take-profit price
  - Trailing stop price
  - Max loss percentage
  - Time limit
  - Momentum reversal (smart exit)
  - Scalper profit targets

Also updates trailing stops as price moves favorably.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, List, Dict

from database.db import session_scope
from database.models import PaperTrade, ActivityLog
from utils.helpers import utcnow

# Import advanced exit manager for smarter exit decisions
try:
    from trading.advanced_exit_manager import get_exit_manager, ExitDecision
    ADVANCED_EXIT_AVAILABLE = True
except ImportError:
    ADVANCED_EXIT_AVAILABLE = False

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class ExitSignal:
    """Represents a position that should be closed."""
    trade_id: int
    symbol: str
    reason: str  # "sl" / "tp" / "trailing" / "max_loss" / "time" / "scalp_profit" / "momentum_exit"
    current_price: float
    trigger_price: float | None
    pnl_pct: float
    urgency: str = "normal"  # "high" for immediate exits, "normal" for standard


@dataclass
class PositionHealth:
    """Health assessment of a position."""
    trade_id: int
    symbol: str
    health_score: float  # 0-100, where 100 is excellent
    trend_direction: str  # "bullish", "bearish", "neutral"
    momentum: float  # -1 to 1
    time_decay: float  # 0-1, increases as position ages
    risk_level: str  # "low", "medium", "high", "critical"
    recommendation: str  # "hold", "take_profit", "cut_loss", "watch"


class PositionMonitor:
    """
    Monitors open positions and emits exit signals when thresholds are hit.

    Usage in bot_engine tick loop:
        monitor = PositionMonitor()
        exits = monitor.check_all_positions(wallet_id, price_map)
        for exit in exits:
            paper_engine.close_trade(exit.trade_id, exit.current_price, exit.reason)
    """

    def __init__(self, default_max_loss_pct: float = 0.10):
        self.default_max_loss_pct = default_max_loss_pct
        # Price history for momentum detection (symbol -> list of recent prices)
        self._price_history: dict[str, list[float]] = {}
        self._max_history = 20  # Keep last 20 price points

    def check_all_positions(
        self,
        wallet_id: int,
        price_map: dict[str, float],
    ) -> list[ExitSignal]:
        """
        Check all open positions for a wallet against current prices.

        Args:
            wallet_id: The wallet to check positions for
            price_map: Dict of symbol -> current_price

        Returns:
            List of ExitSignal objects for positions that should be closed
        """
        exits: list[ExitSignal] = []

        # Update price history for momentum detection
        for symbol, price in price_map.items():
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            self._price_history[symbol].append(price)
            # Keep only last N prices
            if len(self._price_history[symbol]) > self._max_history:
                self._price_history[symbol] = self._price_history[symbol][-self._max_history:]

        with session_scope() as s:
            open_trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.wallet_id == wallet_id)
                .filter(PaperTrade.status == "open")
                .all()
            )
            
            logging.info(f"[POSITION_MONITOR] Wallet {wallet_id}: {len(open_trades)} open trades, {len(price_map)} prices in map")

            for trade in open_trades:
                current_price = price_map.get(trade.symbol)
                if current_price is None:
                    # Try to fetch price directly if not in map
                    logging.warning(f"[POSITION_MONITOR] No price for {trade.symbol} in map, fetching...")
                    try:
                        from connectors.live_prices import get_price
                        price_result = get_price(trade.symbol)
                        if price_result.get("ok"):
                            current_price = float(price_result["price"])
                            logging.info(f"[POSITION_MONITOR] Fetched {trade.symbol} price: ${current_price}")
                        else:
                            logging.warning(f"[POSITION_MONITOR] Failed to fetch {trade.symbol}: {price_result}")
                            continue
                    except Exception as e:
                        logging.error(f"[POSITION_MONITOR] Error fetching {trade.symbol}: {e}")
                        continue

                # Calculate momentum for this symbol
                momentum = self._calculate_momentum(trade.symbol)
                
                exit_signal = self._check_single_position(s, trade, current_price, momentum)
                if exit_signal:
                    exits.append(exit_signal)
                else:
                    # No exit needed - update trailing stop if applicable
                    self._update_trailing_stop(s, trade, current_price)

            s.commit()

        return exits

    def _calculate_momentum(self, symbol: str) -> float:
        """
        Calculate momentum indicator for a symbol.
        
        Returns value from -1 (strong bearish) to +1 (strong bullish).
        """
        history = self._price_history.get(symbol, [])
        if len(history) < 3:
            return 0.0  # Not enough data
        
        # Calculate short-term momentum (last 3 prices)
        recent = history[-3:]
        if recent[0] == 0:
            return 0.0
        
        short_change = (recent[-1] - recent[0]) / recent[0]
        
        # Calculate medium-term momentum (last 10 prices if available)
        if len(history) >= 10:
            medium = history[-10:]
            medium_change = (medium[-1] - medium[0]) / medium[0] if medium[0] != 0 else 0
        else:
            medium_change = short_change
        
        # Combined momentum: 60% short-term, 40% medium-term
        # Scale to -1 to 1 range (assume 2% move is max momentum)
        raw_momentum = (short_change * 0.6 + medium_change * 0.4) / 0.02
        return max(-1.0, min(1.0, raw_momentum))

    def assess_position_health(
        self,
        trade: PaperTrade,
        current_price: float,
    ) -> PositionHealth:
        """
        Assess the health of a position for display and decision support.
        
        Returns a PositionHealth object with scores and recommendations.
        """
        from utils.helpers import time_since_minutes
        
        entry = float(trade.entry_price or 0)
        qty = float(trade.qty or 0)
        side = trade.side.upper()
        
        # Calculate P&L
        if side == "BUY":
            pnl_pct = (current_price - entry) / entry if entry > 0 else 0
        else:
            pnl_pct = (entry - current_price) / entry if entry > 0 else 0
        
        pnl_usd = pnl_pct * entry * qty
        age_minutes = time_since_minutes(trade.opened_at) if trade.opened_at else 0
        
        # Get momentum
        momentum = self._calculate_momentum(trade.symbol)
        
        # Determine trend direction based on momentum
        if momentum > 0.3:
            trend = "bullish"
        elif momentum < -0.3:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Calculate time decay (positions get riskier the longer they're held in scalping)
        time_decay = min(1.0, age_minutes / 30.0)  # Max decay at 30 minutes
        
        # Calculate health score (0-100)
        health_score = 50.0  # Start at neutral
        
        # Adjust for P&L
        if pnl_pct > 0.01:  # >1% profit
            health_score += 30
        elif pnl_pct > 0:  # Any profit
            health_score += 15
        elif pnl_pct > -0.005:  # Small loss (<0.5%)
            health_score -= 10
        elif pnl_pct > -0.01:  # Medium loss
            health_score -= 25
        else:  # Large loss
            health_score -= 40
        
        # Adjust for momentum alignment
        momentum_aligned = (side == "BUY" and momentum > 0) or (side == "SELL" and momentum < 0)
        if momentum_aligned:
            health_score += abs(momentum) * 20
        else:
            health_score -= abs(momentum) * 20
        
        # Adjust for time decay
        health_score -= time_decay * 15
        
        # Clamp to 0-100
        health_score = max(0, min(100, health_score))
        
        # Determine risk level
        if health_score >= 70:
            risk_level = "low"
        elif health_score >= 50:
            risk_level = "medium"
        elif health_score >= 30:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Determine recommendation
        if pnl_pct > 0.005 and (not momentum_aligned or age_minutes > 10):
            recommendation = "take_profit"
        elif health_score < 30 or (pnl_pct < -0.01 and not momentum_aligned):
            recommendation = "cut_loss"
        elif health_score < 50:
            recommendation = "watch"
        else:
            recommendation = "hold"
        
        return PositionHealth(
            trade_id=trade.id,
            symbol=trade.symbol,
            health_score=round(health_score, 1),
            trend_direction=trend,
            momentum=round(momentum, 2),
            time_decay=round(time_decay, 2),
            risk_level=risk_level,
            recommendation=recommendation,
        )

    def _check_single_position(
        self,
        session: "Session",
        trade: PaperTrade,
        current_price: float,
        momentum: float = 0.0,
    ) -> ExitSignal | None:
        """
        Check if a single position should be closed.

        Returns ExitSignal if position should close, None otherwise.
        """
        entry = float(trade.entry_price)
        qty = float(trade.qty)
        side = trade.side.upper()

        # Calculate P&L percentage
        if side == "BUY":
            pnl_pct = (current_price - entry) / entry if entry > 0 else 0
        else:  # SELL / SHORT
            pnl_pct = (entry - current_price) / entry if entry > 0 else 0
        
        # Calculate P&L in USD
        pnl_usd = pnl_pct * entry * qty
        
        # Calculate how long we've held this position
        from utils.helpers import time_since_minutes
        age_minutes = time_since_minutes(trade.opened_at) if trade.opened_at else 0

        # =====================================================================
        # GET WALLET SETTINGS
        # =====================================================================
        from database.models import Wallet
        wallet = session.query(Wallet).filter(Wallet.id == trade.wallet_id).first()
        
        # Default values if columns don't exist yet
        trading_style = 'scalper'  # Default to scalper for aggressive trading
        micro_target_usd = 0.25
        min_profit_pct = 0.003
        
        if wallet:
            # Read from database - use getattr for safety but log actual values
            db_style = getattr(wallet, 'trading_style', None)
            db_target = getattr(wallet, 'micro_profit_target_usd', None)
            db_min_pct = getattr(wallet, 'min_profit_pct', None)
            
            logging.info(f"[POSITION_MONITOR] Wallet DB values: style={db_style}, target={db_target}, min_pct={db_min_pct}")
            
            # Apply values with fallbacks
            trading_style = db_style if db_style else 'scalper'
            micro_target_usd = float(db_target) if db_target is not None else 0.25
            min_profit_pct = float(db_min_pct) if db_min_pct is not None else 0.003
            
            # Auto-detect scalper mode: if target is under $1, treat as scalper
            if micro_target_usd <= 1.0 and trading_style == 'hybrid':
                trading_style = 'scalper'
                logging.info(f"[POSITION_MONITOR] Auto-switching to scalper mode (target=${micro_target_usd})")
        else:
            logging.warning(f"[POSITION_MONITOR] No wallet found for trade {trade.id}, using defaults")

        # =====================================================================
        # ADAPTIVE PAYOFF SIZING — the structural fix.
        # =====================================================================
        # The historical bleed was payoff 0.75 (avg win $0.80 / avg loss $1.07).
        # Root cause: a flat $0.25 scalp target paired with a much wider stop,
        # so even a 50% win rate would lose money. We now tier the targets by
        # the entry confidence Claude already attached to the trade:
        #
        #   conf >= 0.70  → "high conviction" — let it run to Claude's full
        #                   SL/TP from the trade record (typically 5%/10%).
        #                   This keeps the upside Claude was reaching for.
        #   conf >= 0.55  → "mid conviction" — widen the scalp to $target ×
        #                   1.6 with stop at 70% of target. Payoff ≈ 2.3:1.
        #   conf <  0.55  → "low conviction" — keep it tight. Stop at 50% of
        #                   target. Payoff ≈ 2.0:1, but small.
        #
        # The crucial invariant: max_loss < target_profit, ALWAYS. That alone
        # makes the system mathematically winnable at <50% WR.
        entry_conf = float(getattr(trade, "confidence", 0.5) or 0.5)
        if entry_conf >= 0.70:
            # High-conviction: bypass scalp logic entirely, defer to Claude SL/TP.
            high_conviction = True
            scalp_max_loss_ratio = 0.50  # only used if we still hit the timeout path
            scalp_target_multiplier = 1.0
        elif entry_conf >= 0.55:
            high_conviction = False
            scalp_max_loss_ratio = 0.50  # max_loss = target × 0.50 → payoff 2:1
            scalp_target_multiplier = 1.6  # widen target so we don't clip winners
        else:
            high_conviction = False
            scalp_max_loss_ratio = 0.50  # max_loss = target × 0.50 → payoff 2:1
            scalp_target_multiplier = 1.0

        effective_target_usd = micro_target_usd * scalp_target_multiplier
        
        logging.info(f"[POSITION_MONITOR] {trade.symbol}: pnl=${pnl_usd:.2f} ({pnl_pct:.2%}), age={age_minutes:.0f}m, style={trading_style}, target=${micro_target_usd}, momentum={momentum:.2f}")

        # =====================================================================
        # SCALPER TAKE PROFIT - CHECK THIS FIRST BEFORE ANYTHING ELSE
        # This is the HIGHEST priority exit - take the money when target is hit.
        # `effective_target_usd` is the conviction-adjusted target, so a
        # high-conviction trade gets to run to a bigger TP than a low-conf one.
        # =====================================================================
        if trading_style == "scalper" and not high_conviction and pnl_usd >= effective_target_usd:
            logging.info(f"[SCALPER] TAKE PROFIT: {trade.symbol} +${pnl_usd:.2f} >= target ${effective_target_usd:.2f} (conf {entry_conf:.2f})")
            self._log_exit(session, trade, "scalp_profit", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="scalp_profit",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # =====================================================================
        # MOMENTUM-BASED EXIT: Exit early if momentum is strongly against us
        # =====================================================================
        # For BUY positions: bearish momentum = exit
        # For SELL positions: bullish momentum = exit
        momentum_against_us = (side == "BUY" and momentum < -0.5) or (side == "SELL" and momentum > 0.5)
        
        if momentum_against_us and pnl_usd < 0 and age_minutes >= 2:
            # Strong momentum reversal while we're losing - get out!
            logging.info(f"[MOMENTUM] EXIT: {trade.symbol} momentum={momentum:.2f} against {side}, pnl=${pnl_usd:.2f}")
            self._log_exit(session, trade, "momentum_reversal", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="momentum_reversal",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
                urgency="high",
            )

        # =====================================================================
        # SCALPER MODE: AGGRESSIVE loss-cutting (profit-taking handled above).
        #
        # KEY INVARIANT: max_loss MUST be smaller than target_profit.
        # Old code used 60% which was actually fine — but the old TARGET was
        # only $0.25 against an unbounded SL%, so real-world losses blew
        # through it via the time-stop and 2-min stop. We now keep the
        # ratio (50% by default, tighter for low-conf) AND make sure the
        # other exit paths use the same effective_target_usd.
        # =====================================================================
        if trading_style == "scalper" and not high_conviction:
            # Conviction-tiered max loss. With ratio=0.50 a $0.25 target
            # caps loss at $0.125 — payoff 2:1 even before considering
            # winners that overshoot.
            max_loss_usd = effective_target_usd * scalp_max_loss_ratio
            
            # IMMEDIATE CUT: If loss exceeds max_loss threshold, exit NOW
            # Don't wait for time - cut the loss immediately
            if pnl_usd <= -max_loss_usd:
                logging.info(f"[SCALPER] IMMEDIATE CUT: {trade.symbol} ${pnl_usd:.2f} <= -${max_loss_usd:.2f} threshold (conf {entry_conf:.2f})")
                self._log_exit(session, trade, "scalp_maxloss", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="scalp_maxloss",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                    urgency="high",
                )
            
            # QUICK CUT: After 2 minutes with ANY loss, exit
            # Scalping means quick in, quick out - don't let losers run
            if age_minutes >= 2 and pnl_usd < 0:
                logging.info(f"[SCALPER] QUICK CUT (2m): {trade.symbol} ${pnl_usd:.2f} after {age_minutes:.0f}m")
                self._log_exit(session, trade, "scalp_stoploss", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="scalp_stoploss",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # TIME EXIT: After 5 minutes, exit if not at least 50% to target.
            # Don't hold scalp trades hoping they'll turn around.
            if age_minutes >= 5 and pnl_usd < (effective_target_usd * 0.5):
                logging.info(f"[SCALPER] TIME EXIT (5m): {trade.symbol} ${pnl_usd:.2f} < 50% of target ${effective_target_usd:.2f}")
                self._log_exit(session, trade, "scalp_timeout", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="scalp_timeout",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )

        # =====================================================================
        # HYBRID MODE: Balance between scalping and swing trading
        # =====================================================================
        elif trading_style == "hybrid":
            # Take profits at target
            if pnl_usd >= micro_target_usd or pnl_pct >= min_profit_pct:
                logging.info(f"[HYBRID] TAKE PROFIT: {trade.symbol} +${pnl_usd:.2f}")
                self._log_exit(session, trade, "target_profit", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="target_profit",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # Cut losses after 10 minutes with >$2 loss
            if age_minutes >= 10 and pnl_usd < -2.0:
                logging.info(f"[HYBRID] CUT LOSS: {trade.symbol} ${pnl_usd:.2f} after {age_minutes:.0f}m")
                self._log_exit(session, trade, "hybrid_stoploss", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="hybrid_stoploss",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )
            
            # After 30 minutes, exit if losing
            if age_minutes >= 30 and pnl_usd < 0:
                logging.info(f"[HYBRID] TIME EXIT: {trade.symbol} ${pnl_usd:.2f} after {age_minutes:.0f}m")
                self._log_exit(session, trade, "hybrid_timeout", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="hybrid_timeout",
                    current_price=current_price,
                    trigger_price=None,
                    pnl_pct=pnl_pct,
                )

        # =====================================================================
        # STANDARD CHECKS (all styles)
        # =====================================================================
        
        # 1. Check max loss (hard cap - 10% default)
        max_loss = float(trade.max_loss_pct or self.default_max_loss_pct)
        if pnl_pct <= -max_loss:
            logging.info(f"[MAX_LOSS] {trade.symbol} hit {pnl_pct:.2%} >= {max_loss:.2%} max")
            self._log_exit(session, trade, "max_loss", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="max_loss",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # 2. Check stop-loss price
        # IMPORTANT: Add minimum hold time before SL can trigger
        # This prevents getting stopped out by normal market noise in the first few minutes
        min_hold_minutes = 15  # Don't trigger SL for first 15 minutes
        trade_age_minutes = 0
        if trade.opened_at:
            from utils.helpers import ensure_utc
            opened_utc = ensure_utc(trade.opened_at)
            trade_age_minutes = (utcnow() - opened_utc).total_seconds() / 60
        
        # Only check stop-loss after minimum hold period (unless loss exceeds max_loss_pct)
        max_loss_exceeded = pnl_pct <= -0.08  # 8% max loss always triggers
        can_trigger_sl = trade_age_minutes >= min_hold_minutes or max_loss_exceeded
        
        if trade.stop_loss_price and can_trigger_sl:
            sl_price = float(trade.stop_loss_price)
            if side == "BUY" and current_price <= sl_price:
                self._log_exit(session, trade, "sl", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="sl",
                    current_price=current_price,
                    trigger_price=sl_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price >= sl_price:
                self._log_exit(session, trade, "sl", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="sl",
                    current_price=current_price,
                    trigger_price=sl_price,
                    pnl_pct=pnl_pct,
                )

        # 3. Check trailing stop price
        if trade.trailing_stop_price:
            trailing_price = float(trade.trailing_stop_price)
            if side == "BUY" and current_price <= trailing_price:
                self._log_exit(session, trade, "trailing", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="trailing",
                    current_price=current_price,
                    trigger_price=trailing_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price >= trailing_price:
                self._log_exit(session, trade, "trailing", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="trailing",
                    current_price=current_price,
                    trigger_price=trailing_price,
                    pnl_pct=pnl_pct,
                )

        # 4. Check take-profit price (percentage-based from trade creation)
        if trade.take_profit_price:
            tp_price = float(trade.take_profit_price)
            if side == "BUY" and current_price >= tp_price:
                self._log_exit(session, trade, "tp", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="tp",
                    current_price=current_price,
                    trigger_price=tp_price,
                    pnl_pct=pnl_pct,
                )
            elif side == "SELL" and current_price <= tp_price:
                self._log_exit(session, trade, "tp", current_price, pnl_pct)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="tp",
                    current_price=current_price,
                    trigger_price=tp_price,
                    pnl_pct=pnl_pct,
                )

        # 5. Time limit for swing trades (4 hours default)
        if trading_style == "swing":
            time_limit = float(trade.time_limit_hours) if trade.time_limit_hours else 4.0
            if trade.opened_at:
                from utils.helpers import ensure_utc
                opened_utc = ensure_utc(trade.opened_at)
                deadline = opened_utc + timedelta(hours=time_limit)
                if utcnow() >= deadline and pnl_pct <= 0.005:
                    self._log_exit(session, trade, "time", current_price, pnl_pct)
                    return ExitSignal(
                        trade_id=trade.id,
                        symbol=trade.symbol,
                        reason="time",
                        current_price=current_price,
                        trigger_price=None,
                        pnl_pct=pnl_pct,
                    )

        return None

    def _update_trailing_stop(
        self,
        session: "Session",
        trade: PaperTrade,
        current_price: float,
    ) -> None:
        """
        Update the trailing stop price if price has moved favorably.
        Also handles breakeven stop activation.

        For BUY: trailing stop rises when price rises (lock in gains)
        For SELL: trailing stop falls when price falls (lock in gains)
        
        The trailing stop uses an ADAPTIVE algorithm:
        - At small profits: use the configured trailing_stop_pct
        - As profits grow: tighten the trailing stop to lock in more gains
        """
        import logging
        
        side = trade.side.upper()
        entry = float(trade.entry_price)
        
        # Calculate current profit percentage
        if side == "BUY":
            profit_pct = (current_price - entry) / entry if entry > 0 else 0
        else:
            profit_pct = (entry - current_price) / entry if entry > 0 else 0
        
        # =====================================================================
        # BREAKEVEN STOP: Once profit hits trigger, move stop to lock in small gain
        # =====================================================================
        breakeven_trigger = float(trade.breakeven_trigger_pct or 0)
        breakeven_stop = float(trade.breakeven_stop_pct or 0)
        
        if breakeven_trigger > 0 and not trade.breakeven_activated and profit_pct >= breakeven_trigger:
            # Activate breakeven stop!
            if side == "BUY":
                new_stop = entry * (1 + breakeven_stop)  # Stop above entry
            else:
                new_stop = entry * (1 - breakeven_stop)  # Stop below entry for shorts
            
            # Update stop loss to breakeven level
            current_sl = float(trade.stop_loss_price or 0)
            if side == "BUY" and new_stop > current_sl:
                trade.stop_loss_price = new_stop
                trade.breakeven_activated = True
                logging.info(f"[BREAKEVEN] {trade.symbol}: Activated! Stop moved from ${current_sl:.4f} to ${new_stop:.4f} (entry=${entry:.4f})")
            elif side == "SELL" and (current_sl == 0 or new_stop < current_sl):
                trade.stop_loss_price = new_stop
                trade.breakeven_activated = True
                logging.info(f"[BREAKEVEN] {trade.symbol}: Activated! Stop moved to ${new_stop:.4f}")
        
        # =====================================================================
        # TRAILING STOP: Ratchet stop up as price rises
        # =====================================================================
        if not trade.trailing_stop_pct:
            return

        trail_pct = float(trade.trailing_stop_pct)
        high_water = float(trade.high_water_price or entry)
        
        # ADAPTIVE TRAILING: Tighten the trailing % as profits grow
        # - At 1% profit: use base trailing %
        # - At 2% profit: tighten by 20%
        # - At 3% profit: tighten by 35%
        # - At 5%+ profit: tighten by 50%
        adaptive_multiplier = 1.0
        if profit_pct >= 0.05:
            adaptive_multiplier = 0.50  # 50% tighter trailing stop
        elif profit_pct >= 0.03:
            adaptive_multiplier = 0.65
        elif profit_pct >= 0.02:
            adaptive_multiplier = 0.80
        elif profit_pct >= 0.01:
            adaptive_multiplier = 0.90
        
        effective_trail_pct = trail_pct * adaptive_multiplier

        if side == "BUY":
            # Update high water mark if price is higher
            if current_price > high_water:
                trade.high_water_price = current_price
                high_water = current_price
                logging.debug(f"[TRAILING] {trade.symbol}: New high water ${high_water:.4f}")

            # Trailing stop is X% below high water mark
            new_trailing = high_water * (1 - effective_trail_pct)

            # Only ratchet UP the trailing stop (never lower it)
            current_trailing = float(trade.trailing_stop_price or 0)
            if new_trailing > current_trailing:
                trade.trailing_stop_price = new_trailing
                logging.info(
                    f"[TRAILING] {trade.symbol}: Stop raised ${current_trailing:.4f} -> ${new_trailing:.4f} "
                    f"(high=${high_water:.4f}, trail={effective_trail_pct:.2%}, profit={profit_pct:.2%})"
                )

        else:  # SELL / SHORT
            # Update low water mark if price is lower
            low_water = float(trade.high_water_price or entry)
            if current_price < low_water:
                trade.high_water_price = current_price
                low_water = current_price

            # Trailing stop is X% above low water mark
            new_trailing = low_water * (1 + effective_trail_pct)

            # Only ratchet DOWN the trailing stop (never raise it)
            current_trailing = float(trade.trailing_stop_price or float("inf"))
            if new_trailing < current_trailing:
                trade.trailing_stop_price = new_trailing
                logging.info(
                    f"[TRAILING] {trade.symbol} (SHORT): Stop lowered to ${new_trailing:.4f} "
                    f"(low=${low_water:.4f}, profit={profit_pct:.2%})"
                )

    def _log_exit(
        self,
        session: "Session",
        trade: PaperTrade,
        reason: str,
        price: float,
        pnl_pct: float,
    ) -> None:
        """Log the auto-exit to activity log."""
        reason_labels = {
            "sl": "Stop-Loss",
            "tp": "Take-Profit",
            "trailing": "Trailing Stop",
            "max_loss": "Max Loss Cap",
            "time": "Time Limit",
        }
        session.add(
            ActivityLog(
                category="auto_exit",
                level="info",
                message=(
                    f"Auto-exit triggered: {trade.symbol} {trade.side} "
                    f"closed by {reason_labels.get(reason, reason)} "
                    f"at ${price:.4f} (P&L: {pnl_pct:+.2%})"
                ),
                wallet_id=trade.wallet_id,
            )
        )


def initialize_trade_sl_tp(
    trade: PaperTrade,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    max_loss_pct: float | None = None,
    time_limit_hours: float | None = None,
) -> None:
    """
    Set up stop-loss, take-profit, and trailing stop for a newly opened trade.

    Converts percentage-based inputs to absolute prices based on entry price.
    """
    entry = float(trade.entry_price)
    side = trade.side.upper()

    # Store original entry for DCA tracking
    if trade.original_entry is None:
        trade.original_entry = entry

    # Calculate and set stop-loss price
    if stop_loss_pct is not None and stop_loss_pct > 0:
        if side == "BUY":
            trade.stop_loss_price = entry * (1 - stop_loss_pct)
        else:
            trade.stop_loss_price = entry * (1 + stop_loss_pct)

    # Calculate and set take-profit price
    if take_profit_pct is not None and take_profit_pct > 0:
        if side == "BUY":
            trade.take_profit_price = entry * (1 + take_profit_pct)
        else:
            trade.take_profit_price = entry * (1 - take_profit_pct)

    # Set trailing stop percentage (actual price calculated on each tick)
    if trailing_stop_pct is not None and trailing_stop_pct > 0:
        trade.trailing_stop_pct = trailing_stop_pct
        trade.high_water_price = entry
        # Initial trailing stop is at the same level as a normal SL
        if side == "BUY":
            trade.trailing_stop_price = entry * (1 - trailing_stop_pct)
        else:
            trade.trailing_stop_price = entry * (1 + trailing_stop_pct)

    # Set max loss and time limit
    if max_loss_pct is not None:
        trade.max_loss_pct = max_loss_pct
    if time_limit_hours is not None:
        trade.time_limit_hours = time_limit_hours
