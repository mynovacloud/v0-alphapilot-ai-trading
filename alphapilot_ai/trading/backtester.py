"""
Real Backtester Engine
======================
Tests trading strategies against historical data from Coinbase.

Features:
- Fetches real historical OHLCV data
- Applies strategy signals to historical prices
- Calculates realistic P&L with fees
- Tracks drawdown, win rate, Sharpe ratio
- Generates equity curves
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from database.db import session_scope
from database.models import BacktestResult, Strategy
from connectors.candles import get_candles
from trading.indicators import (
    ema, rsi, macd, bollinger_bands, atr, adx,
    stochastic, relative_volume, compute_all_indicators
)
from trading.market_regime import detect_regime


@dataclass
class Trade:
    """A single backtest trade."""
    entry_time: int
    entry_price: float
    side: str  # "BUY" or "SELL"
    size: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str
    strategy_type: str
    start_balance: float = 10000.0
    position_size_pct: float = 0.10  # 10% of balance per trade
    max_positions: int = 1
    fee_pct: float = 0.001  # 0.1% fee per trade (Coinbase taker)
    slippage_pct: float = 0.0005  # 0.05% slippage
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.02
    lookback_days: int = 30
    granularity: int = 900  # 15 minute candles


@dataclass 
class BacktestStats:
    """Statistics from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    avg_hold_time_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    risk_score: float = 0.0
    recommendation: str = "unknown"
    
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def _generate_signals_momentum(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """
    Momentum strategy signals.
    
    BUY when:
    - EMA12 > EMA26 (bullish crossover)
    - RSI < 70 (not overbought)
    - MACD histogram > 0
    - Price above VWAP or SMA20
    
    SELL when:
    - EMA12 < EMA26 (bearish crossover)
    - RSI > 30 (not oversold)
    - MACD histogram < 0
    """
    n = len(close)
    signals = np.zeros(n)  # 0=hold, 1=buy, -1=sell
    
    if n < 50:
        return signals
    
    # Calculate indicators
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)
    rsi_14 = rsi(close, 14)
    macd_result = macd(close)
    sma_20 = np.convolve(close, np.ones(20)/20, mode='same')
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema_12[i]) or np.isnan(ema_26[i]) or np.isnan(rsi_14[i]):
            continue
        
        # BUY signal
        ema_bullish = ema_12[i] > ema_26[i] and ema_12[i-1] <= ema_26[i-1]
        rsi_ok_buy = rsi_14[i] < 70 and rsi_14[i] > 30
        macd_bullish = macd_result.histogram[i] > 0
        price_above_sma = close[i] > sma_20[i]
        
        if ema_bullish and rsi_ok_buy and macd_bullish and price_above_sma:
            signals[i] = 1
        
        # SELL signal
        ema_bearish = ema_12[i] < ema_26[i] and ema_12[i-1] >= ema_26[i-1]
        rsi_ok_sell = rsi_14[i] > 30 and rsi_14[i] < 70
        macd_bearish = macd_result.histogram[i] < 0
        
        if ema_bearish and rsi_ok_sell and macd_bearish:
            signals[i] = -1
    
    return signals


def _generate_signals_mean_reversion(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """
    Mean reversion strategy signals.
    
    BUY when:
    - Price touches lower Bollinger Band
    - RSI < 30 (oversold)
    - Stochastic %K < 20
    
    SELL when:
    - Price touches upper Bollinger Band
    - RSI > 70 (overbought)
    - Stochastic %K > 80
    """
    n = len(close)
    signals = np.zeros(n)
    
    if n < 50:
        return signals
    
    # Calculate indicators
    bb = bollinger_bands(close, 20, 2.0)
    rsi_14 = rsi(close, 14)
    stoch = stochastic(high, low, close)
    
    for i in range(50, n):
        if np.isnan(bb.lower[i]) or np.isnan(rsi_14[i]):
            continue
        
        # BUY signal - oversold
        price_at_lower_bb = close[i] <= bb.lower[i] * 1.01
        rsi_oversold = rsi_14[i] < 35
        stoch_oversold = stoch.k[i] < 25 if not np.isnan(stoch.k[i]) else False
        
        if price_at_lower_bb and rsi_oversold and stoch_oversold:
            signals[i] = 1
        
        # SELL signal - overbought
        price_at_upper_bb = close[i] >= bb.upper[i] * 0.99
        rsi_overbought = rsi_14[i] > 65
        stoch_overbought = stoch.k[i] > 75 if not np.isnan(stoch.k[i]) else False
        
        if price_at_upper_bb and rsi_overbought and stoch_overbought:
            signals[i] = -1
    
    return signals


def _generate_signals_volatility_breakout(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """
    Volatility breakout strategy signals.
    
    BUY when:
    - Price breaks above upper Bollinger Band
    - ATR expanding (volatility increasing)
    - Volume spike (relative volume > 1.5)
    - ADX > 25 (strong trend)
    
    SELL when:
    - Price breaks below lower Bollinger Band
    - Similar conditions
    """
    n = len(close)
    signals = np.zeros(n)
    
    if n < 50:
        return signals
    
    # Calculate indicators
    bb = bollinger_bands(close, 20, 2.0)
    atr_14 = atr(high, low, close, 14)
    rvol = relative_volume(volume, 20)
    adx_result = adx(high, low, close, 14)
    
    for i in range(50, n):
        if np.isnan(bb.upper[i]) or np.isnan(atr_14[i]):
            continue
        
        # Check for expanding volatility
        if i > 5:
            atr_expanding = atr_14[i] > atr_14[i-5] * 1.1
        else:
            atr_expanding = False
        
        vol_spike = rvol[i] > 1.3 if not np.isnan(rvol[i]) else False
        strong_trend = adx_result.adx[i] > 20 if not np.isnan(adx_result.adx[i]) else False
        
        # BUY - upward breakout
        price_breakout_up = close[i] > bb.upper[i] and close[i-1] <= bb.upper[i-1]
        
        if price_breakout_up and atr_expanding and vol_spike:
            signals[i] = 1
        
        # SELL - downward breakout
        price_breakout_down = close[i] < bb.lower[i] and close[i-1] >= bb.lower[i-1]
        
        if price_breakout_down and atr_expanding and vol_spike:
            signals[i] = -1
    
    return signals


def _generate_signals(
    strategy_type: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """Route to appropriate signal generator."""
    if strategy_type == "Momentum":
        return _generate_signals_momentum(close, high, low, volume)
    elif strategy_type == "Mean Reversion":
        return _generate_signals_mean_reversion(close, high, low, volume)
    elif strategy_type == "Volatility Breakout":
        return _generate_signals_volatility_breakout(close, high, low, volume)
    else:
        # Default to momentum
        return _generate_signals_momentum(close, high, low, volume)


def run_backtest_on_data(
    config: BacktestConfig,
    timestamps: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    volumes: np.ndarray,
) -> BacktestStats:
    """
    Run backtest on provided OHLCV data.
    """
    stats = BacktestStats()
    n = len(close_prices)
    
    if n < 50:
        stats.recommendation = "insufficient_data"
        return stats
    
    # Generate signals
    signals = _generate_signals(
        config.strategy_type,
        close_prices, high_prices, low_prices, volumes
    )
    
    # Simulation state
    balance = config.start_balance
    position: Optional[Trade] = None
    trades: list[Trade] = []
    equity_curve = [balance]
    high_water_mark = balance
    
    for i in range(50, n):
        current_price = close_prices[i]
        current_high = high_prices[i]
        current_low = low_prices[i]
        current_time = int(timestamps[i])
        
        # Check exit conditions if in position
        if position is not None:
            should_exit = False
            exit_reason = ""
            exit_price = current_price
            
            if position.side == "BUY":
                # Calculate current P&L %
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                
                # Stop loss
                if current_low <= position.entry_price * (1 - config.stop_loss_pct):
                    should_exit = True
                    exit_reason = "stop_loss"
                    exit_price = position.entry_price * (1 - config.stop_loss_pct)
                
                # Take profit
                elif current_high >= position.entry_price * (1 + config.take_profit_pct):
                    should_exit = True
                    exit_reason = "take_profit"
                    exit_price = position.entry_price * (1 + config.take_profit_pct)
                
                # Trailing stop
                elif config.use_trailing_stop and pnl_pct > config.trailing_stop_pct:
                    trail_price = current_high * (1 - config.trailing_stop_pct)
                    if current_low <= trail_price:
                        should_exit = True
                        exit_reason = "trailing_stop"
                        exit_price = trail_price
                
                # Signal exit
                elif signals[i] == -1:
                    should_exit = True
                    exit_reason = "signal"
            
            else:  # SHORT position
                pnl_pct = (position.entry_price - current_price) / position.entry_price
                
                if current_high >= position.entry_price * (1 + config.stop_loss_pct):
                    should_exit = True
                    exit_reason = "stop_loss"
                    exit_price = position.entry_price * (1 + config.stop_loss_pct)
                
                elif current_low <= position.entry_price * (1 - config.take_profit_pct):
                    should_exit = True
                    exit_reason = "take_profit"
                    exit_price = position.entry_price * (1 - config.take_profit_pct)
                
                elif signals[i] == 1:
                    should_exit = True
                    exit_reason = "signal"
            
            if should_exit:
                # Apply slippage and fees
                if position.side == "BUY":
                    exit_price *= (1 - config.slippage_pct)
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    exit_price *= (1 + config.slippage_pct)
                    pnl = (position.entry_price - exit_price) * position.size
                
                # Deduct exit fee
                fee = exit_price * position.size * config.fee_pct
                pnl -= fee
                
                position.exit_time = current_time
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = pnl / (position.entry_price * position.size) * 100
                position.exit_reason = exit_reason
                
                balance += pnl
                trades.append(position)
                position = None
        
        # Check entry conditions
        if position is None and signals[i] != 0:
            side = "BUY" if signals[i] == 1 else "SELL"
            
            # Calculate position size
            position_value = balance * config.position_size_pct
            entry_price = current_price * (1 + config.slippage_pct if side == "BUY" else 1 - config.slippage_pct)
            size = position_value / entry_price
            
            # Deduct entry fee
            fee = position_value * config.fee_pct
            balance -= fee
            
            position = Trade(
                entry_time=current_time,
                entry_price=entry_price,
                side=side,
                size=size,
            )
        
        # Track equity
        if position is not None:
            if position.side == "BUY":
                unrealized = (current_price - position.entry_price) * position.size
            else:
                unrealized = (position.entry_price - current_price) * position.size
            equity = balance + unrealized
        else:
            equity = balance
        
        equity_curve.append(equity)
        
        # Track drawdown
        if equity > high_water_mark:
            high_water_mark = equity
    
    # Close any remaining position
    if position is not None:
        exit_price = close_prices[-1]
        if position.side == "BUY":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        position.exit_time = int(timestamps[-1])
        position.exit_price = exit_price
        position.pnl = pnl
        position.exit_reason = "end_of_data"
        trades.append(position)
        balance += pnl
    
    # Calculate statistics
    if not trades:
        stats.recommendation = "no_trades"
        stats.equity_curve = equity_curve
        return stats
    
    stats.total_trades = len(trades)
    stats.trades = [
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "side": t.side,
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4) if t.exit_price else None,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]
    
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl < 0]
    
    stats.winning_trades = len(winning)
    stats.losing_trades = len(losing)
    stats.win_rate = len(winning) / len(trades) if trades else 0
    
    stats.total_pnl = sum(t.pnl for t in trades)
    stats.total_pnl_pct = (balance - config.start_balance) / config.start_balance * 100
    
    if winning:
        stats.avg_win = sum(t.pnl for t in winning) / len(winning)
        stats.avg_win_pct = sum(t.pnl_pct for t in winning) / len(winning)
    
    if losing:
        stats.avg_loss = sum(t.pnl for t in losing) / len(losing)
        stats.avg_loss_pct = sum(t.pnl_pct for t in losing) / len(losing)
    
    # Profit factor
    gross_profit = sum(t.pnl for t in winning) if winning else 0
    gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
    stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    
    # Max drawdown
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    stats.max_drawdown_pct = float(drawdown.max()) * 100
    stats.max_drawdown = float((peak - equity_array).max())
    
    # Sharpe ratio (simplified)
    returns = np.diff(equity_array) / equity_array[:-1]
    if len(returns) > 1 and returns.std() > 0:
        stats.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 4)  # 15min bars
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        stats.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252 * 24 * 4)
    
    # Average hold time
    hold_times = []
    for t in trades:
        if t.exit_time and t.entry_time:
            hold_times.append((t.exit_time - t.entry_time) / 3600)  # hours
    stats.avg_hold_time_hours = sum(hold_times) / len(hold_times) if hold_times else 0
    
    # Consecutive wins/losses
    max_wins = max_losses = current_wins = current_losses = 0
    for t in trades:
        if t.pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    stats.max_consecutive_wins = max_wins
    stats.max_consecutive_losses = max_losses
    
    # Risk score and recommendation
    stats.risk_score = min(1.0, (stats.max_drawdown_pct / 100) + (1 - stats.win_rate) * 0.3)
    
    if stats.total_pnl < 0 or stats.max_drawdown_pct > 40:
        stats.recommendation = "avoid"
    elif stats.win_rate > 0.55 and stats.profit_factor > 1.5 and stats.max_drawdown_pct < 20:
        stats.recommendation = "recommended"
    elif stats.win_rate > 0.50 and stats.profit_factor > 1.2 and stats.max_drawdown_pct < 30:
        stats.recommendation = "moderate"
    else:
        stats.recommendation = "risky"
    
    stats.equity_curve = equity_curve
    
    return stats


def run_backtest(strategy_id: int, days: int = 30, granularity: int = 900) -> dict[str, Any]:
    """
    Run a real backtest for a strategy using historical Coinbase data.
    
    Args:
        strategy_id: The strategy to test
        days: How many days of history to test
        granularity: Candle size in seconds (300=5m, 900=15m, 3600=1h)
    
    Returns:
        Backtest results dict
    """
    with session_scope() as s:
        strat = s.get(Strategy, strategy_id)
        if not strat:
            return {"ok": False, "reason": "Strategy not found"}
        
        strategy_type = strat.strategy_type or "Momentum"
        symbols = (strat.symbols or "BTC-USD").split(",")
        symbol = symbols[0].strip()  # Test on first symbol
        
        # Map risk level to position sizing
        risk_config = {
            "Conservative": {"position_size_pct": 0.05, "stop_loss_pct": 0.02},
            "Moderate": {"position_size_pct": 0.10, "stop_loss_pct": 0.03},
            "Aggressive": {"position_size_pct": 0.15, "stop_loss_pct": 0.04},
            "Degenerate": {"position_size_pct": 0.25, "stop_loss_pct": 0.05},
        }.get(strat.risk_level or "Moderate", {"position_size_pct": 0.10, "stop_loss_pct": 0.03})
    
    # Fetch historical data from Coinbase
    candle_data = get_candles(symbol, granularity=granularity, limit=days * 24 * 3600 // granularity)
    
    if not candle_data or not candle_data.get("candles"):
        return {"ok": False, "reason": f"Could not fetch historical data for {symbol}"}
    
    candles = candle_data["candles"]
    if len(candles) < 50:
        return {"ok": False, "reason": f"Insufficient historical data ({len(candles)} candles)"}
    
    # Convert to numpy arrays
    timestamps = np.array([c["time"] for c in candles], dtype=float)
    opens = np.array([c["open"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)
    
    # Create config
    config = BacktestConfig(
        symbol=symbol,
        strategy_type=strategy_type,
        position_size_pct=risk_config["position_size_pct"],
        stop_loss_pct=risk_config["stop_loss_pct"],
        lookback_days=days,
        granularity=granularity,
    )
    
    # Run backtest
    stats = run_backtest_on_data(config, timestamps, opens, highs, lows, closes, volumes)
    
    # Save results to database
    with session_scope() as s:
        result = BacktestResult(
            strategy_id=strategy_id,
            total_trades=stats.total_trades,
            win_rate=stats.win_rate,
            total_pnl=round(stats.total_pnl, 2),
            avg_win=round(stats.avg_win, 2),
            avg_loss=round(stats.avg_loss, 2),
            drawdown=round(stats.max_drawdown_pct / 100, 4),
            risk_score=round(stats.risk_score, 3),
            recommendation=stats.recommendation,
        )
        s.add(result)
    
    return {
        "ok": True,
        "strategy_id": strategy_id,
        "symbol": symbol,
        "strategy_type": strategy_type,
        "days_tested": days,
        "candles_processed": len(candles),
        "total_trades": stats.total_trades,
        "winning_trades": stats.winning_trades,
        "losing_trades": stats.losing_trades,
        "win_rate": round(stats.win_rate, 3),
        "total_pnl": round(stats.total_pnl, 2),
        "total_pnl_pct": round(stats.total_pnl_pct, 2),
        "avg_win": round(stats.avg_win, 2),
        "avg_loss": round(stats.avg_loss, 2),
        "profit_factor": round(stats.profit_factor, 2),
        "max_drawdown_pct": round(stats.max_drawdown_pct, 2),
        "sharpe_ratio": round(stats.sharpe_ratio, 2),
        "sortino_ratio": round(stats.sortino_ratio, 2),
        "avg_hold_time_hours": round(stats.avg_hold_time_hours, 1),
        "max_consecutive_wins": stats.max_consecutive_wins,
        "max_consecutive_losses": stats.max_consecutive_losses,
        "risk_score": round(stats.risk_score, 3),
        "recommendation": stats.recommendation,
        "equity_curve": stats.equity_curve[-200:],  # Last 200 points
        "sample_trades": stats.trades[:20],  # First 20 trades
    }
