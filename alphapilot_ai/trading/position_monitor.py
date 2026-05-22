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
from typing import TYPE_CHECKING, Optional, List, Dict

from database.db import session_scope
from database.models import PaperTrade, ActivityLog

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class ExitSignal:
    """Represents a position that should be closed."""
    trade_id: int
    symbol: str
    reason: str  # take_profit / max_loss / momentum_reversal / time_cap / stale / profit_lock_in / trailing
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

        # Refresh the peak-favorable price (high_water_price) BEFORE running
        # exit checks. The legacy _update_trailing_stop also maintains this
        # field, but only when (a) no exit fired AND (b) trailing_stop_pct is
        # set — neither of which the new profit lock-in floor can rely on.
        # Updating eagerly here means: peak is current as of this tick, and
        # the lock-in milestone math below sees real data on the very tick
        # a new high prints. We ratchet ONE direction (up for BUY, down for
        # SELL) and never roll back, mirroring the trailing-stop convention.
        prior_high = float(trade.high_water_price or entry)
        if side == "BUY" and current_price > prior_high:
            trade.high_water_price = current_price
            prior_high = current_price
        elif side == "SELL" and current_price < prior_high:
            trade.high_water_price = current_price
            prior_high = current_price
        
        # Calculate how long we've held this position
        from utils.helpers import time_since_minutes
        age_minutes = time_since_minutes(trade.opened_at) if trade.opened_at else 0

        # =====================================================================
        # HOLDING PROFILE — the single source of truth for this trade's exits.
        # Resolved and stamped at entry (see trading/holding_profiles.py).
        # Legacy trades opened before profiles existed carry no stamp; resolve
        # one on the fly from the current global mode + the trade's confidence.
        # =====================================================================
        from trading.holding_profiles import get_profile, resolve_profile_name
        profile_name = getattr(trade, "holding_profile", None)
        if not profile_name:
            try:
                from config.bot_config import BotConfig
                profile_name = resolve_profile_name(
                    BotConfig.load().holding_mode,
                    float(getattr(trade, "confidence", 0.5) or 0.5),
                )
            except Exception:
                profile_name = "short_swing"
        profile = get_profile(profile_name)

        logging.info(
            f"[POSITION_MONITOR] {trade.symbol}: pnl=${pnl_usd:.2f} ({pnl_pct:.2%}), "
            f"age={age_minutes:.0f}m, profile={profile.name}, momentum={momentum:.2f}"
        )

        # =====================================================================
        # 1. TAKE PROFIT — the profile's target was reached. Best outcome.
        # =====================================================================
        if pnl_pct >= profile.target_pct:
            logging.info(
                f"[EXIT/take_profit] {trade.symbol} +{pnl_pct:.2%} >= "
                f"target {profile.target_pct:.2%}"
            )
            self._log_exit(session, trade, "take_profit", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="take_profit",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # =====================================================================
        # 2. HARD MAX LOSS — the profile's stop. Fires immediately, no grace
        # period. There is deliberately no minimum-hold gate: the profile's
        # max_loss_pct IS the intended risk, so honour it the instant it hits.
        # =====================================================================
        if pnl_pct <= -profile.max_loss_pct:
            logging.info(
                f"[EXIT/max_loss] {trade.symbol} {pnl_pct:.2%} <= "
                f"-{profile.max_loss_pct:.2%}"
            )
            self._log_exit(session, trade, "max_loss", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="max_loss",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
                urgency="high",
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
        # 3. HARD TIME CAP — the universal backstop. Every trade, every
        # profile, no escape hatch. This is the rule whose ABSENCE let
        # high-conviction trades sit dead for 40-60 minutes: previously no
        # time limit applied to them at all.
        # =====================================================================
        if age_minutes >= profile.hard_cap_minutes:
            logging.info(
                f"[EXIT/time_cap] {trade.symbol} age {age_minutes:.0f}m >= "
                f"cap {profile.hard_cap_minutes:.0f}m (pnl {pnl_pct:+.2%})"
            )
            self._log_exit(session, trade, "time_cap", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="time_cap",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # =====================================================================
        # 4. STALE TIMEOUT — partway through the hold, if the trade has not
        # made meaningful progress, cut it loose rather than tying up the
        # slot. Softer than the hard cap: only fires when NOT in profit.
        # =====================================================================
        if age_minutes >= profile.stale_minutes and pnl_pct < profile.stale_min_profit_pct:
            logging.info(
                f"[EXIT/stale] {trade.symbol} age {age_minutes:.0f}m >= "
                f"{profile.stale_minutes:.0f}m, pnl {pnl_pct:+.2%} < "
                f"{profile.stale_min_profit_pct:.2%}"
            )
            self._log_exit(session, trade, "stale", current_price, pnl_pct)
            return ExitSignal(
                trade_id=trade.id,
                symbol=trade.symbol,
                reason="stale",
                current_price=current_price,
                trigger_price=None,
                pnl_pct=pnl_pct,
            )

        # =====================================================================
        # 5. PROFIT LOCK-IN FLOOR — refuse to give back a banked peak.
        # =====================================================================
        # Hard guarantee: once peak unrealized profit hits a milestone, this
        # trade can NEVER close below the floor for that milestone. Addresses
        # the "$20 profit drifts back to $2" failure mode where the
        # percentage-based trailing stop is too loose at high profit levels.
        #
        # The existing adaptive trailing tightens to 0.50× the base trail at
        # +5% profit and never tightens further. For a 3.5-4% ATR-derived
        # trail that still leaves 1.5-2% of give-back room at any profit
        # level — which is fine on a +5% peak but unacceptable on a +20% peak.
        # This lock-in floor is INDEPENDENT of trail_pct and steps up the
        # minimum-acceptable exit profit as the peak climbs.
        peak_high = float(trade.high_water_price or entry)
        peak_pct: float
        if side == "BUY":
            peak_pct = (peak_high - entry) / entry if entry > 0 else 0.0
        else:  # SELL / SHORT — peak profit is when price went DOWN
            peak_pct = (entry - peak_high) / entry if entry > 0 else 0.0

        floor_pct = _profit_lock_floor(peak_pct)
        if floor_pct >= 0.0 and peak_pct > floor_pct:
            # Compute the price below which we lock in (above for shorts).
            if side == "BUY":
                floor_price = entry * (1 + floor_pct)
                breached = current_price <= floor_price
            else:
                floor_price = entry * (1 - floor_pct)
                breached = current_price >= floor_price
            if breached:
                logging.info(
                    "[LOCK_IN] %s %s: peak=%+.2f%% gave back to %+.2f%% "
                    "(floor=%+.2f%%, entry=$%.4f, peak_px=$%.4f, exit_px=$%.4f)",
                    trade.symbol, side, peak_pct * 100, pnl_pct * 100,
                    floor_pct * 100, entry, peak_high, current_price,
                )
                self._log_exit(session, trade, "profit_lock_in", current_price, pnl_pct)
                # Audit to ActivityLog so the operator can SEE the lock-in
                # firing in the training console alongside fills.
                try:
                    session.add(ActivityLog(
                        category="auto_exit",
                        level="info",
                        wallet_id=trade.wallet_id,
                        message=(
                            # `{x:+.2f}` is the explicit-sign format spec — gives
                            # "+1.00" for positive and "-0.29" for negative. The
                            # old `+{x:.2f}` prefix-then-format produced "+-0.29"
                            # for negative numbers, which is the formatting bug
                            # you saw in the LOCK_IN console line. peak_pct is
                            # always >= 0 (it's a peak from entry) so {:.2f}
                            # with an explicit "+" prefix is fine there.
                            f"[LOCK_IN] {trade.symbol} {side}: peak +{peak_pct*100:.2f}% "
                            f"-> exit at {pnl_pct*100:+.2f}% (floor {floor_pct*100:+.2f}%)"
                        ),
                    ))
                except Exception:
                    logging.debug("Failed to write LOCK_IN ActivityLog", exc_info=True)
                return ExitSignal(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    reason="profit_lock_in",
                    current_price=current_price,
                    trigger_price=floor_price,
                    pnl_pct=pnl_pct,
                )

        # =====================================================================
        # 6. TRAILING STOP — ratcheted toward price by _update_trailing_stop
        # on every tick where no exit fired. Protects open profit against a
        # pull-back from the high-water mark.
        # =====================================================================
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
            "take_profit": "Take-Profit",
            "trailing": "Trailing Stop",
            "max_loss": "Max Loss Cap",
            "time": "Time Limit",
            "time_cap": "Hard Time Cap",
            "stale": "Stale Timeout",
            "momentum_reversal": "Momentum Reversal",
            "profit_lock_in": "Profit Lock-In",
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


# =============================================================================
# Profit lock-in floor — staircase of "minimum acceptable exit profit" tiers
# =============================================================================
# Once a trade's peak unrealized profit hits a tier, this function returns
# the floor below which the trade MUST exit. Independent of trailing %.
#
# Invariant: floor is monotonically increasing in peak_pct. Each tier locks
# in a fraction of the prior peak — small at low peaks (where chop dominates)
# and tighter at higher peaks (where giving back is unacceptable).
#
# Returns -1.0 when the peak has not yet earned a floor (i.e. trade hasn't
# made meaningful gains).
#
# Edit these tiers carefully — they're the source-of-truth for "we don't
# give back a $20 peak to $2". Each row: (peak_pct, floor_pct) where
# floor_pct is the minimum profit we accept on exit at that peak.
_PROFIT_LOCK_TIERS = (
    (0.005, 0.000),   # +0.5% peak: break-even floor (we won't go negative)
    (0.010, 0.005),   # +1.0% peak: lock +0.5%
    (0.020, 0.012),   # +2.0% peak: lock +1.2% (give back at most 0.8%)
    (0.030, 0.022),   # +3.0% peak: lock +2.2%
    (0.050, 0.040),   # +5.0% peak: lock +4.0%
    (0.080, 0.065),   # +8.0% peak: lock +6.5%
    (0.100, 0.085),   # +10% peak: lock +8.5%
    (0.150, 0.130),   # +15% peak: lock +13%
    (0.200, 0.180),   # +20% peak: lock +18%
    (0.300, 0.270),   # +30% peak: lock +27%
)


def _profit_lock_floor(peak_pct: float) -> float:
    """Return the minimum-acceptable exit profit for a given peak.

    -1.0 means no floor yet (peak hasn't crossed the lowest tier).
    """
    floor = -1.0
    for tier_peak, tier_floor in _PROFIT_LOCK_TIERS:
        if peak_pct >= tier_peak:
            floor = tier_floor
        else:
            break
    return floor


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
