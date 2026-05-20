"""
Paper trading engine.

Responsibilities:
- Validate trade through RiskManager
- Estimate fees + slippage
- Persist PaperTrade rows
- Open / close trades and update wallet paper balance
- Auto stop-loss and take-profit management
- Log every action via ActivityLog
"""
from __future__ import annotations

import random
from typing import Any

from database.db import session_scope
from database.models import ActivityLog, PaperTrade, Wallet
from trading.risk_manager import RiskManager
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


class PaperTradingEngine:
    def __init__(self) -> None:
        self.risk = RiskManager()

    # --- Public API -----------------------------------------------------

    def open_trade(
        self,
        wallet_id: int,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        confidence: float = 0.6,
        market_type: str = "Crypto",
        strategy_id: int | None = None,
        notes: str = "",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        enable_breakeven_stop: bool = True,
    ) -> dict[str, Any]:
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        logger.info(
            f"[OPEN_TRADE] Attempting: wallet={wallet_id}, symbol={symbol}, "
            f"side={side}, qty={qty}, price={entry_price}, conf={confidence}"
        )

        decision = self.risk.evaluate(wallet_id, qty, entry_price, confidence, strategy_id)
        if not decision.approved:
            logger.warning(f"[OPEN_TRADE] REJECTED by risk manager: {decision.reason} (code={decision.code})")
            self._log("risk", f"Trade BLOCKED: {decision.reason} (code={decision.code})", wallet_id=wallet_id, level="warn")
            return {"ok": False, "reason": decision.reason, "code": decision.code}
        
        logger.info(f"[OPEN_TRADE] APPROVED by risk manager, proceeding to open trade")

        notional = qty * entry_price
        fees = self._estimate_fees(notional)
        slippage = self._estimate_slippage(notional)
        
        # Get wallet's trading style to determine stop-loss/take-profit defaults
        with session_scope() as s:
            wallet = s.get(Wallet, wallet_id)
            trading_style = getattr(wallet, 'trading_style', 'scalper') or 'scalper'
            micro_target = getattr(wallet, 'micro_profit_target_usd', 0.25) or 0.25
        
        # Calculate stop-loss and take-profit based on trading style
        if trading_style == 'scalper':
            # SCALPER: Tight stops, quick exits
            # Target $0.25 profit = 0.05% on $500, so SL should be 0.03% (~$0.15)
            # This gives us 1.67:1 reward:risk ratio
            sl_pct = stop_loss_pct if stop_loss_pct is not None else 0.003  # 0.3% = $1.50 on $500
            tp_pct = take_profit_pct if take_profit_pct is not None else 0.005  # 0.5% = $2.50 on $500
            trail_pct = trailing_stop_pct if trailing_stop_pct is not None else 0.002  # 0.2% trailing
        elif trading_style == 'swing':
            # SWING: Wider stops, longer holds
            sl_pct = stop_loss_pct if stop_loss_pct is not None else 0.03  # 3%
            tp_pct = take_profit_pct if take_profit_pct is not None else 0.06  # 6%
            trail_pct = trailing_stop_pct if trailing_stop_pct is not None else 0.02  # 2% trailing
        else:
            # HYBRID: Balance between scalping and swing
            sl_pct = stop_loss_pct if stop_loss_pct is not None else 0.015  # 1.5%
            tp_pct = take_profit_pct if take_profit_pct is not None else 0.03  # 3%
            trail_pct = trailing_stop_pct if trailing_stop_pct is not None else 0.01  # 1% trailing
        
        if side == "BUY":
            stop_loss_price = entry_price * (1 - sl_pct)
            take_profit_price = entry_price * (1 + tp_pct)
            trailing_stop_price = entry_price * (1 - trail_pct)
        else:  # SELL
            stop_loss_price = entry_price * (1 + sl_pct)
            take_profit_price = entry_price * (1 - tp_pct)
            trailing_stop_price = entry_price * (1 + trail_pct)

        with session_scope() as s:
            wallet = s.get(Wallet, wallet_id)
            if not wallet:
                logger.error(f"[OPEN_TRADE] Wallet {wallet_id} not found")
                return {"ok": False, "reason": "Wallet not found"}

            # Reserve cash from paper balance
            wallet.paper_balance -= (notional + fees + slippage)
            wallet.last_synced = utcnow()

            trade = PaperTrade(
                wallet_id=wallet_id,
                strategy_id=strategy_id,
                symbol=symbol,
                market_type=market_type,
                side=side,
                qty=qty,
                entry_price=entry_price,
                fees=fees,
                slippage=slippage,
                confidence=confidence,
                status="open",
                notes=notes,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                # Trailing stop fields - these enable automatic profit protection
                trailing_stop_pct=trail_pct,
                trailing_stop_price=trailing_stop_price,
                high_water_price=entry_price,  # Track the best price we've seen
                # Breakeven stop: once we're up 1%, move stop to entry + 0.2%
                breakeven_trigger_pct=0.01 if enable_breakeven_stop else None,
                breakeven_stop_pct=0.002 if enable_breakeven_stop else None,
            )
            s.add(trade)
            s.flush()
            trade_id = trade.id

            s.add(
                ActivityLog(
                    category="paper_trade",
                    level="info",
                    wallet_id=wallet_id,
                    message=f"Opened {side} {qty} {symbol} @ {entry_price:.2f} (conf={confidence:.2f}, SL=${stop_loss_price:.4f}, TP=${take_profit_price:.4f})",
                )
            )
            
            logger.info(f"[OPEN_TRADE] SUCCESS: trade_id={trade_id}, symbol={symbol}, notional=${notional:.2f}, SL=${stop_loss_price:.4f}, TP=${take_profit_price:.4f}")

        return {
            "ok": True, 
            "trade_id": trade_id, 
            "fees": fees, 
            "slippage": slippage,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
        }

    def close_trade(self, trade_id: int, exit_price: float, notes: str = "", exit_reason: str = "manual") -> dict[str, Any]:
        """Close a paper trade and record the P&L."""
        logger.info(f"[CLOSE_TRADE] Attempting to close trade {trade_id} at {exit_price}, reason={exit_reason}")
        pnl = 0.0
        try:
            with session_scope() as s:
                trade = s.get(PaperTrade, trade_id)
                if not trade:
                    return {"ok": False, "reason": "Trade not found"}
                if trade.status != "open":
                    return {"ok": False, "reason": f"Trade already {trade.status}"}

                wallet = s.get(Wallet, trade.wallet_id)
                
                # Safe type conversion
                entry_price = float(trade.entry_price or 0)
                qty = float(trade.qty or 0)
                fees = float(trade.fees or 0)
                slippage = float(trade.slippage or 0)
                side = (trade.side or "BUY").upper()
                
                if entry_price <= 0 or qty <= 0:
                    return {"ok": False, "reason": f"Invalid trade data: entry={entry_price}, qty={qty}"}
                
                direction = 1 if side == "BUY" else -1
                pnl = (exit_price - entry_price) * qty * direction
                pnl -= fees + slippage
                
                # Calculate return percentage
                notional = qty * entry_price
                return_pct = (pnl / notional * 100) if notional > 0 else 0

                trade.exit_price = exit_price
                trade.realized_pnl = round(pnl, 2)
                trade.unrealized_pnl = 0.0
                trade.status = "closed"
                trade.closed_at = utcnow()
                trade.exit_reason = exit_reason

                # Return notional + pnl to paper balance
                if wallet:
                    wallet.paper_balance = float(wallet.paper_balance or 0) + notional + pnl
                    wallet.last_synced = utcnow()

                # Determine emoji/status for log
                pnl_status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAK-EVEN"
                
                s.add(
                    ActivityLog(
                        category="paper_trade",
                        level="info",
                        wallet_id=trade.wallet_id,
                        message=f"[{pnl_status}] Closed {side} {qty} {trade.symbol} @ {exit_price:.2f}, PnL=${pnl:+.2f} ({return_pct:+.2f}%), Reason={exit_reason}",
                    )
                )
        except Exception as e:
            logger.exception(f"[CLOSE_TRADE] Error closing trade {trade_id}")
            return {"ok": False, "reason": str(e)}

        # Trade is now closed. Kick the Claude reflection loop so it can
        # learn from this fill. Best-effort: reflection failures must never
        # break the close. The function is idempotent if called twice.
        try:
            from ai.claude_learning import record_trade_outcome
            record_trade_outcome(trade_id)
        except Exception:
            logger.exception("Reflection failed for trade %s", trade_id)
        
        # Also update the adaptive learning engine with the trade outcome
        # This feeds into pattern recognition and strategy optimization
        try:
            from ai.adaptive_learning_engine import learn_from_trade
            learn_from_trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=return_pct / 100,  # Convert to decimal
                duration_minutes=duration_minutes,
                exit_reason=exit_reason,
            )
            logger.debug(f"[LEARN] Adaptive learning updated for trade {trade_id}")
        except Exception:
            logger.debug("Adaptive learning update failed for trade %s", trade_id)
        
        # CRITICAL: Update the autonomous learning engine
        # This is the self-improving brain that operates WITHOUT Claude
        try:
            from ai.autonomous_learning_engine import learn_from_closed_trade
            learn_from_closed_trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=return_pct / 100,
                hold_minutes=duration_minutes,
                exit_reason=exit_reason,
            )
            logger.info(f"[AUTONOMOUS] Learned from trade {trade_id}: {symbol} {'WIN' if pnl > 0 else 'LOSS'}")
        except Exception as e:
            logger.debug(f"Autonomous learning update failed for trade {trade_id}: {e}")

        return {"ok": True, "pnl": round(pnl, 2), "exit_reason": exit_reason}
    
    def check_stop_loss_take_profit(self, prices: dict[str, float]) -> dict[str, Any]:
        """
        Check all open trades against current prices for stop-loss/take-profit triggers.
        
        Args:
            prices: Dict mapping symbol -> current_price
            
        Returns:
            Summary of closed trades
        """
        closed_trades = []
        errors = []
        
        with session_scope() as s:
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            trade_data = []
            for t in open_trades:
                trade_data.append({
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": float(t.entry_price or 0),
                    "stop_loss": float(t.stop_loss_price) if t.stop_loss_price else None,
                    "take_profit": float(t.take_profit_price) if t.take_profit_price else None,
                })
        
        for trade in trade_data:
            symbol = trade["symbol"]
            current_price = prices.get(symbol)
            
            if current_price is None:
                continue
            
            stop_loss = trade["stop_loss"]
            take_profit = trade["take_profit"]
            side = (trade["side"] or "BUY").upper()
            
            exit_reason = None
            
            # Check stop-loss
            if stop_loss:
                if side == "BUY" and current_price <= stop_loss:
                    exit_reason = "stop_loss"
                elif side == "SELL" and current_price >= stop_loss:
                    exit_reason = "stop_loss"
            
            # Check take-profit
            if take_profit and not exit_reason:
                if side == "BUY" and current_price >= take_profit:
                    exit_reason = "take_profit"
                elif side == "SELL" and current_price <= take_profit:
                    exit_reason = "take_profit"
            
            # Close if triggered
            if exit_reason:
                result = self.close_trade(trade["id"], current_price, exit_reason=exit_reason)
                if result.get("ok"):
                    closed_trades.append({
                        "trade_id": trade["id"],
                        "symbol": symbol,
                        "exit_reason": exit_reason,
                        "pnl": result.get("pnl", 0),
                    })
                    logger.info(f"[AUTO_EXIT] {symbol}: {exit_reason} triggered at ${current_price:.4f}, PnL=${result.get('pnl', 0):.2f}")
                else:
                    errors.append(f"{symbol}: {result.get('reason')}")
        
        return {
            "ok": True,
            "closed": len(closed_trades),
            "trades": closed_trades,
            "errors": errors,
        }
    
    def update_trailing_stops(self, prices: dict[str, float], trail_pct: float = 0.015) -> dict[str, Any]:
        """
        Update trailing stop-losses for profitable positions.
        
        Args:
            prices: Dict mapping symbol -> current_price
            trail_pct: Trailing stop percentage (default 1.5%)
        
        Returns:
            Summary of updated trades
        """
        updated = []
        
        with session_scope() as s:
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            
            for trade in open_trades:
                symbol = trade.symbol
                current_price = prices.get(symbol)
                
                if current_price is None:
                    continue
                
                entry_price = float(trade.entry_price or 0)
                current_stop = float(trade.stop_loss_price) if trade.stop_loss_price else None
                side = (trade.side or "BUY").upper()
                
                if side == "BUY":
                    # For long positions, trail stop up if price moves up
                    new_trail_stop = current_price * (1 - trail_pct)
                    
                    # Only update if new stop is higher than current stop
                    if current_stop is None or new_trail_stop > current_stop:
                        # Only trail if we're in profit
                        if current_price > entry_price:
                            trade.stop_loss_price = new_trail_stop
                            updated.append({
                                "trade_id": trade.id,
                                "symbol": symbol,
                                "old_stop": current_stop,
                                "new_stop": new_trail_stop,
                            })
                else:  # SELL
                    # For short positions, trail stop down if price moves down
                    new_trail_stop = current_price * (1 + trail_pct)
                    
                    if current_stop is None or new_trail_stop < current_stop:
                        if current_price < entry_price:
                            trade.stop_loss_price = new_trail_stop
                            updated.append({
                                "trade_id": trade.id,
                                "symbol": symbol,
                                "old_stop": current_stop,
                                "new_stop": new_trail_stop,
                            })
        
        if updated:
            logger.info(f"[TRAILING_STOP] Updated {len(updated)} trailing stops")
        
        return {"ok": True, "updated": len(updated), "trades": updated}
    
    def get_portfolio_summary(self, wallet_id: int | None = None) -> dict[str, Any]:
        """
        Get comprehensive portfolio summary with P&L metrics.
        
        Returns:
            Portfolio statistics including total P&L, win rate, etc.
        """
        with session_scope() as s:
            query = s.query(PaperTrade)
            if wallet_id:
                query = query.filter(PaperTrade.wallet_id == wallet_id)
            
            all_trades = query.all()
            
            open_trades = [t for t in all_trades if t.status == "open"]
            closed_trades = [t for t in all_trades if t.status == "closed"]
            
            # Open position metrics
            total_unrealized = sum(float(t.unrealized_pnl or 0) for t in open_trades)
            open_notional = sum(float(t.entry_price or 0) * float(t.qty or 0) for t in open_trades)
            
            # Closed trade metrics
            total_realized = sum(float(t.realized_pnl or 0) for t in closed_trades)
            
            wins = [t for t in closed_trades if float(t.realized_pnl or 0) > 0]
            losses = [t for t in closed_trades if float(t.realized_pnl or 0) < 0]
            
            win_count = len(wins)
            loss_count = len(losses)
            total_closed = len(closed_trades)
            
            win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
            
            # Average win/loss
            avg_win = sum(float(t.realized_pnl or 0) for t in wins) / win_count if win_count > 0 else 0
            avg_loss = sum(float(t.realized_pnl or 0) for t in losses) / loss_count if loss_count > 0 else 0
            
            # Profit factor
            gross_profit = sum(float(t.realized_pnl or 0) for t in wins)
            gross_loss = abs(sum(float(t.realized_pnl or 0) for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Max drawdown (simplified - based on trade P&L)
            cumulative_pnl = []
            running_total = 0
            for t in sorted(closed_trades, key=lambda x: x.closed_at or utcnow()):
                running_total += float(t.realized_pnl or 0)
                cumulative_pnl.append(running_total)
            
            max_drawdown = 0
            peak = 0
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                drawdown = peak - pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return {
                "open_positions": len(open_trades),
                "open_notional": round(open_notional, 2),
                "unrealized_pnl": round(total_unrealized, 2),
                "total_trades": total_closed,
                "wins": win_count,
                "losses": loss_count,
                "win_rate": round(win_rate, 2),
                "realized_pnl": round(total_realized, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "N/A",
                "max_drawdown": round(max_drawdown, 2),
                "total_pnl": round(total_realized + total_unrealized, 2),
            }

    # --- Helpers --------------------------------------------------------

    @staticmethod
    def _estimate_fees(notional: float) -> float:
        # 10 bps fee assumption
        return round(notional * 0.001, 2)

    @staticmethod
    def _estimate_slippage(notional: float) -> float:
        return round(notional * random.uniform(0.0001, 0.001), 2)

    @staticmethod
    def _log(category: str, message: str, wallet_id: int | None = None, level: str = "info") -> None:
        with session_scope() as s:
            s.add(ActivityLog(category=category, level=level, message=message, wallet_id=wallet_id))
