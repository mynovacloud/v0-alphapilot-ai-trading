"""
Portfolio Intelligence Engine - Proactive Portfolio Management

The old bot was PASSIVE: it opened 3 positions and waited for them to recover.
This engine is ACTIVE: it continuously seeks opportunities to improve portfolio P&L.

Key behaviors:
1. ALWAYS seeking - evaluates opportunities even when slots are full
2. DCA into losers - at better prices to lower average entry
3. Offset trades - open new positions specifically to balance underwater ones
4. Scale into winners - add to positions that are working
5. Portfolio-level thinking - optimize total P&L, not individual trades
6. Recovery mode - when portfolio is red, shift to more aggressive recovery strategies

This runs at the START of every tick, BEFORE the regular entry evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TYPE_CHECKING

from database.db import session_scope
from database.models import PaperTrade, ActivityLog, Wallet, Strategy
from trading.loss_recovery import LossRecoveryEngine, get_underwater_positions
from trading.strategy_engine import evaluate_symbol
from utils.helpers import utcnow
from utils.logger import get_logger

if TYPE_CHECKING:
    from config.bot_config import BotConfig

logger = get_logger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of current portfolio health."""
    total_positions: int
    total_value_usd: float
    total_pnl_usd: float
    total_pnl_pct: float
    winning_count: int
    losing_count: int
    biggest_loser_pct: float
    biggest_winner_pct: float
    available_capital: float
    is_recovery_mode: bool  # True when portfolio P&L is significantly negative


@dataclass 
class PortfolioAction:
    """An action the portfolio intelligence engine wants to take."""
    action_type: str  # "dca", "scale_in", "offset", "trim_winner", "close_loser"
    trade_id: int | None  # Existing trade ID if modifying
    symbol: str
    side: str
    size_usd: float
    reason: str
    priority: int  # 1-10, higher = more urgent
    confidence: float


class PortfolioIntelligence:
    """
    Proactive portfolio management that thinks ahead.
    
    Instead of passively waiting for positions to recover, this engine:
    - Continuously looks for ways to improve total portfolio P&L
    - DCAs into underwater positions when prices hit better levels
    - Opens "offset trades" to balance losses (if BTC is down, maybe ETH is up)
    - Scales into winning positions (let winners run, add to strength)
    - Trims winners that are overextended (take partial profits)
    - Actively manages toward a positive portfolio
    """
    
    def __init__(
        self,
        # DCA settings
        dca_at_loss_pct: float = 0.03,  # DCA when down 3%+
        dca_size_pct: float = 0.5,  # Add 50% of original position
        max_dca_per_position: int = 3,
        
        # Scale-in settings (adding to winners)
        scale_in_at_profit_pct: float = 0.03,  # Scale in when up 3%+
        scale_in_size_pct: float = 0.3,  # Add 30% to winners
        max_scale_ins: int = 2,
        
        # Offset trading
        offset_when_portfolio_down_pct: float = 0.02,  # Start offset trades when portfolio down 2%+
        offset_trade_size_pct: float = 0.5,  # Offset trades are 50% normal size
        
        # Recovery mode thresholds
        recovery_mode_threshold_pct: float = 0.05,  # Portfolio down 5% = recovery mode
        aggressive_dca_multiplier: float = 1.5,  # In recovery mode, DCA 1.5x normal
        
        # Risk limits
        max_portfolio_concentration_pct: float = 0.4,  # No single position > 40% of portfolio
        min_diversification_count: int = 3,  # Try to have at least 3 different assets
    ):
        self.dca_at_loss_pct = dca_at_loss_pct
        self.dca_size_pct = dca_size_pct
        self.max_dca_per_position = max_dca_per_position
        self.scale_in_at_profit_pct = scale_in_at_profit_pct
        self.scale_in_size_pct = scale_in_size_pct
        self.max_scale_ins = max_scale_ins
        self.offset_when_portfolio_down_pct = offset_when_portfolio_down_pct
        self.offset_trade_size_pct = offset_trade_size_pct
        self.recovery_mode_threshold_pct = recovery_mode_threshold_pct
        self.aggressive_dca_multiplier = aggressive_dca_multiplier
        self.max_portfolio_concentration_pct = max_portfolio_concentration_pct
        self.min_diversification_count = min_diversification_count
        
        self.loss_recovery = LossRecoveryEngine()
    
    def analyze_portfolio(
        self,
        wallet_id: int,
        price_map: dict[str, float],
    ) -> PortfolioState:
        """
        Analyze current portfolio state to determine strategy.
        """
        with session_scope() as s:
            wallet = s.query(Wallet).filter(Wallet.id == wallet_id).first()
            if not wallet:
                return PortfolioState(
                    total_positions=0,
                    total_value_usd=0,
                    total_pnl_usd=0,
                    total_pnl_pct=0,
                    winning_count=0,
                    losing_count=0,
                    biggest_loser_pct=0,
                    biggest_winner_pct=0,
                    available_capital=0,
                    is_recovery_mode=False,
                )
            
            trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.wallet_id == wallet_id)
                .filter(PaperTrade.status == "open")
                .all()
            )
            
            total_value = 0.0
            total_cost = 0.0
            winning = 0
            losing = 0
            biggest_loser_pct = 0.0
            biggest_winner_pct = 0.0
            
            for trade in trades:
                current_price = price_map.get(trade.symbol)
                if current_price is None:
                    continue
                
                entry = float(trade.entry_price)
                qty = float(trade.qty)
                side = (trade.side or "BUY").upper()
                
                position_value = qty * current_price
                position_cost = qty * entry
                total_value += position_value
                total_cost += position_cost
                
                if side == "BUY":
                    pnl_pct = (current_price - entry) / entry if entry > 0 else 0
                else:
                    pnl_pct = (entry - current_price) / entry if entry > 0 else 0
                
                if pnl_pct >= 0:
                    winning += 1
                    biggest_winner_pct = max(biggest_winner_pct, pnl_pct)
                else:
                    losing += 1
                    biggest_loser_pct = min(biggest_loser_pct, pnl_pct)
            
            total_pnl_usd = total_value - total_cost
            total_pnl_pct = (total_value - total_cost) / total_cost if total_cost > 0 else 0
            
            available = float(wallet.paper_balance or 0)
            is_recovery = total_pnl_pct < -self.recovery_mode_threshold_pct
            
            return PortfolioState(
                total_positions=len(trades),
                total_value_usd=total_value,
                total_pnl_usd=total_pnl_usd,
                total_pnl_pct=total_pnl_pct,
                winning_count=winning,
                losing_count=losing,
                biggest_loser_pct=biggest_loser_pct,
                biggest_winner_pct=biggest_winner_pct,
                available_capital=available,
                is_recovery_mode=is_recovery,
            )
    
    def generate_actions(
        self,
        wallet_id: int,
        price_map: dict[str, float],
        universe: list[dict[str, Any]],
        cfg: "BotConfig",
        max_actions: int = 3,
    ) -> list[PortfolioAction]:
        """
        Generate a prioritized list of portfolio improvement actions.
        
        This is the brain - it decides what to do based on current state.
        """
        state = self.analyze_portfolio(wallet_id, price_map)
        actions: list[PortfolioAction] = []
        
        if state.total_positions == 0:
            # No positions - nothing to manage (new entries handled elsewhere)
            return actions
        
        # Log portfolio state
        mode_str = "RECOVERY MODE" if state.is_recovery_mode else "normal"
        logger.info(
            f"Portfolio [{mode_str}]: {state.total_positions} positions, "
            f"P&L: ${state.total_pnl_usd:+.2f} ({state.total_pnl_pct:+.1%}), "
            f"W/L: {state.winning_count}/{state.losing_count}"
        )
        
        # Get current positions
        with session_scope() as s:
            trades = list(
                s.query(PaperTrade)
                .filter(PaperTrade.wallet_id == wallet_id)
                .filter(PaperTrade.status == "open")
                .all()
            )
            # Materialize needed fields before session closes
            trade_data = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": (t.side or "BUY").upper(),
                    "entry_price": float(t.entry_price),
                    "qty": float(t.qty),
                    "dca_count": t.dca_count or 0,
                    "original_entry": float(t.original_entry or t.entry_price),
                }
                for t in trades
            ]
        
        # 1. DCA opportunities - positions that have dropped to better entry points
        dca_actions = self._find_dca_opportunities(
            trade_data, price_map, state, cfg
        )
        actions.extend(dca_actions)
        
        # 2. Scale-in opportunities - positions that are winning
        scale_actions = self._find_scale_in_opportunities(
            trade_data, price_map, state, cfg
        )
        actions.extend(scale_actions)
        
        # 3. Offset trades - new positions to balance losing ones
        if state.is_recovery_mode or state.total_pnl_pct < -self.offset_when_portfolio_down_pct:
            offset_actions = self._find_offset_opportunities(
                trade_data, price_map, universe, state, cfg
            )
            actions.extend(offset_actions)
        
        # Sort by priority (highest first) and return top N
        actions.sort(key=lambda a: (a.priority, a.confidence), reverse=True)
        return actions[:max_actions]
    
    def _find_dca_opportunities(
        self,
        trades: list[dict],
        price_map: dict[str, float],
        state: PortfolioState,
        cfg: "BotConfig",
    ) -> list[PortfolioAction]:
        """
        Find positions that have dropped enough to warrant averaging down.
        
        DCA is smart when:
        - The original thesis is still valid
        - Price has dropped significantly from entry
        - We haven't already DCA'd too many times
        - We have capital available
        """
        actions = []
        
        for trade in trades:
            current_price = price_map.get(trade["symbol"])
            if current_price is None:
                continue
            
            entry = trade["entry_price"]
            side = trade["side"]
            dca_count = trade["dca_count"]
            
            # Can we DCA more?
            if dca_count >= self.max_dca_per_position:
                continue
            
            # Calculate loss percentage
            if side == "BUY":
                loss_pct = (entry - current_price) / entry if entry > 0 else 0
            else:
                loss_pct = (current_price - entry) / entry if entry > 0 else 0
            
            # Only DCA if we're down enough
            threshold = self.dca_at_loss_pct
            if state.is_recovery_mode:
                threshold *= 0.7  # In recovery mode, DCA earlier
            
            if loss_pct < threshold:
                continue
            
            # Check if signal is still valid (re-evaluate)
            signal = evaluate_symbol(trade["symbol"], cfg.default_strategy_type)
            signal_side = (signal.side or "HOLD").upper()
            
            # Only DCA if signal still agrees with our direction
            if signal_side != side and signal_side != "HOLD":
                continue
            
            # Calculate DCA size
            original_value = trade["original_entry"] * trade["qty"]
            dca_size = original_value * self.dca_size_pct
            
            if state.is_recovery_mode:
                dca_size *= self.aggressive_dca_multiplier
            
            # Cap at available capital
            dca_size = min(dca_size, state.available_capital * 0.3)
            
            if dca_size < 10:  # Minimum $10
                continue
            
            # Priority based on loss amount and signal strength
            priority = min(10, int(5 + (loss_pct * 100)))
            if state.is_recovery_mode:
                priority = min(10, priority + 2)
            
            actions.append(PortfolioAction(
                action_type="dca",
                trade_id=trade["id"],
                symbol=trade["symbol"],
                side=side,
                size_usd=dca_size,
                reason=(
                    f"DCA opportunity: {trade['symbol']} down {loss_pct:.1%}, "
                    f"current ${current_price:.4f} vs entry ${entry:.4f}. "
                    f"Signal still {signal_side}. Adding ${dca_size:.0f}."
                ),
                priority=priority,
                confidence=float(signal.confidence or 0.5),
            ))
        
        return actions
    
    def _find_scale_in_opportunities(
        self,
        trades: list[dict],
        price_map: dict[str, float],
        state: PortfolioState,
        cfg: "BotConfig",
    ) -> list[PortfolioAction]:
        """
        Find winning positions worth adding to.
        
        "Let your winners run" - but also add to them when they're working.
        """
        actions = []
        
        # Don't scale in if portfolio is in recovery mode
        if state.is_recovery_mode:
            return actions
        
        for trade in trades:
            current_price = price_map.get(trade["symbol"])
            if current_price is None:
                continue
            
            entry = trade["entry_price"]
            side = trade["side"]
            
            # Calculate profit percentage
            if side == "BUY":
                profit_pct = (current_price - entry) / entry if entry > 0 else 0
            else:
                profit_pct = (entry - current_price) / entry if entry > 0 else 0
            
            # Only scale in if we're up enough
            if profit_pct < self.scale_in_at_profit_pct:
                continue
            
            # Check concentration limit
            position_value = trade["qty"] * current_price
            if position_value / state.total_value_usd > self.max_portfolio_concentration_pct:
                continue
            
            # Check if signal is still strong
            signal = evaluate_symbol(trade["symbol"], cfg.default_strategy_type)
            signal_side = (signal.side or "HOLD").upper()
            
            if signal_side != side:
                continue
            
            if float(signal.confidence or 0) < 0.55:
                continue
            
            # Calculate scale-in size
            current_value = trade["qty"] * current_price
            scale_size = current_value * self.scale_in_size_pct
            scale_size = min(scale_size, state.available_capital * 0.25)
            
            if scale_size < 10:
                continue
            
            actions.append(PortfolioAction(
                action_type="scale_in",
                trade_id=trade["id"],
                symbol=trade["symbol"],
                side=side,
                size_usd=scale_size,
                reason=(
                    f"Scale-in: {trade['symbol']} up {profit_pct:.1%}, "
                    f"signal still strong ({signal.confidence:.2f}). "
                    f"Adding ${scale_size:.0f} to winner."
                ),
                priority=4,  # Lower priority than DCA (recovery first)
                confidence=float(signal.confidence or 0.5),
            ))
        
        return actions
    
    def _find_offset_opportunities(
        self,
        trades: list[dict],
        price_map: dict[str, float],
        universe: list[dict[str, Any]],
        state: PortfolioState,
        cfg: "BotConfig",
    ) -> list[PortfolioAction]:
        """
        Find NEW positions that could offset current losses.
        
        The idea: if our BTC long is underwater, maybe ETH is showing
        a strong buy signal. Opening a new winning trade offsets the loser.
        """
        actions = []
        
        # Get symbols we already hold
        held_symbols = {t["symbol"] for t in trades}
        
        # How much to offset? Proportional to losses
        loss_to_offset = abs(state.total_pnl_usd)
        target_offset_size = min(
            loss_to_offset * 0.5,  # Try to offset half the loss
            state.available_capital * 0.4,  # Max 40% of available capital
            cfg.position_size_usd * self.offset_trade_size_pct,  # Smaller than normal
        )
        
        if target_offset_size < 20:  # Minimum $20 for offset trades
            return actions
        
        # Look for strong signals in symbols we DON'T hold
        candidates = []
        
        for product in universe[:30]:  # Check top 30 by volume
            symbol = product["product_id"]
            if symbol in held_symbols:
                continue
            
            current_price = price_map.get(symbol)
            if current_price is None:
                continue
            
            signal = evaluate_symbol(symbol, cfg.default_strategy_type)
            signal_side = (signal.side or "HOLD").upper()
            conf = float(signal.confidence or 0)
            
            if signal_side not in ("BUY", "SELL"):
                continue
            
            if conf < 0.55:  # Need decent confidence for offset trades
                continue
            
            candidates.append({
                "symbol": symbol,
                "side": signal_side,
                "confidence": conf,
                "price": current_price,
                "reasoning": signal.reasoning,
            })
        
        # Sort by confidence and take top 2
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        for cand in candidates[:2]:
            actions.append(PortfolioAction(
                action_type="offset",
                trade_id=None,  # New position
                symbol=cand["symbol"],
                side=cand["side"],
                size_usd=target_offset_size,
                reason=(
                    f"Offset trade: Portfolio down ${abs(state.total_pnl_usd):.2f}. "
                    f"{cand['symbol']} shows {cand['side']} signal ({cand['confidence']:.2f}). "
                    f"Opening ${target_offset_size:.0f} to balance losses."
                ),
                priority=7 if state.is_recovery_mode else 5,
                confidence=cand["confidence"],
            ))
        
        return actions


def execute_portfolio_action(
    action: PortfolioAction,
    wallet_id: int,
    paper_engine: Any,
    cfg: "BotConfig",
) -> dict[str, Any]:
    """
    Execute a portfolio intelligence action.
    
    Returns dict with "ok" status and details.
    """
    from connectors.live_prices import get_price
    
    result = {"ok": False, "action": action.action_type, "symbol": action.symbol}
    
    try:
        if action.action_type == "dca":
            # DCA into existing position
            current = get_price(action.symbol)
            if not current.get("ok"):
                result["error"] = "Could not get current price"
                return result
            
            current_price = float(current["price"])
            add_qty = action.size_usd / current_price
            
            recovery = LossRecoveryEngine()
            with session_scope() as s:
                trade = s.query(PaperTrade).filter(PaperTrade.id == action.trade_id).first()
                if trade and trade.status == "open":
                    success = recovery.execute_dca(trade, add_qty, current_price)
                    if success:
                        result["ok"] = True
                        result["added_qty"] = add_qty
                        result["added_usd"] = action.size_usd
                        
                        # Log the action
                        s.add(ActivityLog(
                            category="portfolio_intel",
                            level="info",
                            message=f"DCA executed: {action.reason}",
                            wallet_id=wallet_id,
                        ))
                    else:
                        result["error"] = "DCA execution failed"
                else:
                    result["error"] = "Trade not found or not open"
        
        elif action.action_type == "scale_in":
            # Scale into winning position (same as DCA mechanically)
            current = get_price(action.symbol)
            if not current.get("ok"):
                result["error"] = "Could not get current price"
                return result
            
            current_price = float(current["price"])
            add_qty = action.size_usd / current_price
            
            recovery = LossRecoveryEngine()
            with session_scope() as s:
                trade = s.query(PaperTrade).filter(PaperTrade.id == action.trade_id).first()
                if trade and trade.status == "open":
                    # For scale-in, we directly update qty (no need for avg entry recalc)
                    old_qty = float(trade.qty)
                    trade.qty = old_qty + add_qty
                    
                    s.add(ActivityLog(
                        category="portfolio_intel",
                        level="info",
                        message=f"Scale-in executed: {action.reason}",
                        wallet_id=wallet_id,
                    ))
                    s.commit()
                    
                    result["ok"] = True
                    result["added_qty"] = add_qty
                    result["added_usd"] = action.size_usd
                else:
                    result["error"] = "Trade not found or not open"
        
        elif action.action_type == "offset":
            # Open new position to offset losses
            current = get_price(action.symbol)
            if not current.get("ok"):
                result["error"] = "Could not get current price"
                return result
            
            current_price = float(current["price"])
            qty = action.size_usd / current_price
            
            # Get strategy ID
            with session_scope() as s:
                strat = (
                    s.query(Strategy)
                    .filter(Strategy.wallet_id == wallet_id)
                    .order_by(Strategy.id.asc())
                    .first()
                )
                strategy_id = strat.id if strat else None
            
            outcome = paper_engine.open_trade(
                wallet_id=wallet_id,
                symbol=action.symbol,
                side=action.side,
                qty=qty,
                entry_price=current_price,
                confidence=action.confidence,
                market_type="Crypto",
                strategy_id=strategy_id,
                notes=f"portfolio_intel/offset: {action.reason}",
            )
            
            if outcome.get("ok"):
                result["ok"] = True
                result["trade_id"] = outcome.get("trade_id")
                result["qty"] = qty
                
                with session_scope() as s:
                    s.add(ActivityLog(
                        category="portfolio_intel",
                        level="info",
                        message=f"Offset trade opened: {action.reason}",
                        wallet_id=wallet_id,
                    ))
            else:
                result["error"] = outcome.get("error", "Failed to open offset trade")
        
        else:
            result["error"] = f"Unknown action type: {action.action_type}"
    
    except Exception as e:
        result["error"] = str(e)
        logger.exception(f"Error executing portfolio action: {e}")
    
    return result
