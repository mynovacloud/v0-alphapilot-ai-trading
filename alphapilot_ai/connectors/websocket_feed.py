"""
Real-time WebSocket Price Feed for Coinbase.

Provides sub-second price updates instead of polling every 15-60 seconds.
This gives us a significant edge in fast-moving markets.

Features:
- Real-time ticker updates
- Order book depth (top 5 levels)
- Trade flow (buys vs sells)
- Automatic reconnection
- Price caching with microsecond timestamps
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class TickerData:
    """Real-time ticker data from websocket."""
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    volume_24h: float
    timestamp: datetime
    trade_side: str = ""  # "buy" or "sell" for last trade
    trade_size: float = 0.0


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: float
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bid/ask levels."""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)  # Highest to lowest
    asks: List[OrderBookLevel] = field(default_factory=list)  # Lowest to highest
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread as percentage."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.asks[0].price - self.bids[0].price) / self.bids[0].price
    
    @property
    def bid_depth(self) -> float:
        """Total USD value on bid side."""
        return sum(level.price * level.size for level in self.bids)
    
    @property
    def ask_depth(self) -> float:
        """Total USD value on ask side."""
        return sum(level.price * level.size for level in self.asks)
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance: positive = more bids, negative = more asks."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass
class TradeFlow:
    """Aggregated trade flow over a time window."""
    symbol: str
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    large_buys: int = 0  # Trades > $10k
    large_sells: int = 0
    window_seconds: int = 60
    
    @property
    def net_flow(self) -> float:
        """Net buy pressure: positive = buying, negative = selling."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total
    
    @property
    def large_trade_bias(self) -> float:
        """Bias from large trades: positive = big buyers, negative = big sellers."""
        total = self.large_buys + self.large_sells
        if total == 0:
            return 0.0
        return (self.large_buys - self.large_sells) / total


class WebSocketPriceFeed:
    """
    Real-time price feed using Coinbase WebSocket API.
    
    Provides:
    - Sub-second price updates
    - Order book depth analysis
    - Trade flow tracking (buy/sell pressure)
    - Automatic reconnection on disconnect
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or []
        self._prices: Dict[str, TickerData] = {}
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        self._trade_flows: Dict[str, TradeFlow] = defaultdict(TradeFlow)
        self._trade_history: Dict[str, List[dict]] = defaultdict(list)
        self._callbacks: List[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update: Dict[str, float] = {}
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        
    def subscribe(self, symbols: List[str]):
        """Add symbols to subscription list."""
        for s in symbols:
            if s not in self.symbols:
                self.symbols.append(s)
                self._trade_flows[s] = TradeFlow(symbol=s)
    
    def get_price(self, symbol: str) -> Optional[TickerData]:
        """Get latest price data for a symbol."""
        return self._prices.get(symbol)
    
    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book snapshot."""
        return self._order_books.get(symbol)
    
    def get_trade_flow(self, symbol: str) -> TradeFlow:
        """Get aggregated trade flow data."""
        return self._trade_flows.get(symbol, TradeFlow(symbol=symbol))
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices as a simple dict."""
        return {s: t.price for s, t in self._prices.items()}
    
    def is_price_fresh(self, symbol: str, max_age_seconds: float = 5.0) -> bool:
        """Check if price data is fresh enough."""
        last = self._last_update.get(symbol, 0)
        return (time.time() - last) < max_age_seconds
    
    def analyze_entry_quality(self, symbol: str, side: str) -> dict:
        """
        Analyze if this is a good entry point based on real-time data.
        
        Returns:
            dict with:
            - score: 0-100 entry quality score
            - reasons: list of factors
            - recommendation: "enter", "wait", "skip"
        """
        result = {
            "score": 50,
            "reasons": [],
            "recommendation": "enter",
        }
        
        ticker = self.get_price(symbol)
        order_book = self.get_order_book(symbol)
        trade_flow = self.get_trade_flow(symbol)
        
        if not ticker:
            result["reasons"].append("No real-time data available")
            return result
        
        # 1. Spread analysis (lower is better)
        if ticker.spread < 0.001:  # < 0.1% spread
            result["score"] += 15
            result["reasons"].append("Tight spread (<0.1%)")
        elif ticker.spread > 0.005:  # > 0.5% spread
            result["score"] -= 20
            result["reasons"].append("Wide spread (>0.5%) - poor liquidity")
            if ticker.spread > 0.01:
                result["recommendation"] = "skip"
        
        # 2. Order book imbalance
        if order_book:
            imbalance = order_book.imbalance
            if side == "BUY":
                if imbalance > 0.3:  # More bids than asks
                    result["score"] += 10
                    result["reasons"].append("Strong bid support")
                elif imbalance < -0.3:  # More asks than bids
                    result["score"] -= 15
                    result["reasons"].append("Heavy selling pressure in book")
            else:  # SELL
                if imbalance < -0.3:
                    result["score"] += 10
                    result["reasons"].append("Weak bid support - good for shorts")
                elif imbalance > 0.3:
                    result["score"] -= 15
                    result["reasons"].append("Strong bids may resist down move")
        
        # 3. Trade flow analysis
        if trade_flow.buy_volume + trade_flow.sell_volume > 0:
            net_flow = trade_flow.net_flow
            if side == "BUY":
                if net_flow > 0.2:  # Net buying
                    result["score"] += 15
                    result["reasons"].append("Active buying pressure")
                elif net_flow < -0.3:  # Heavy selling
                    result["score"] -= 20
                    result["reasons"].append("Heavy selling - wait for stabilization")
                    result["recommendation"] = "wait"
            else:  # SELL
                if net_flow < -0.2:
                    result["score"] += 15
                    result["reasons"].append("Active selling momentum")
                elif net_flow > 0.3:
                    result["score"] -= 20
                    result["reasons"].append("Buying pressure - risky short")
                    result["recommendation"] = "wait"
        
        # 4. Large trade activity
        if trade_flow.large_buys + trade_flow.large_sells > 0:
            large_bias = trade_flow.large_trade_bias
            if side == "BUY" and large_bias > 0.3:
                result["score"] += 10
                result["reasons"].append("Whales buying")
            elif side == "SELL" and large_bias < -0.3:
                result["score"] += 10
                result["reasons"].append("Whales selling")
            elif side == "BUY" and large_bias < -0.5:
                result["score"] -= 15
                result["reasons"].append("Whales dumping - caution")
        
        # Final recommendation
        if result["score"] >= 70:
            result["recommendation"] = "enter"
        elif result["score"] >= 50:
            result["recommendation"] = "enter"  # Acceptable
        elif result["score"] >= 35:
            result["recommendation"] = "wait"
        else:
            result["recommendation"] = "skip"
        
        return result
    
    def start(self):
        """Start the websocket feed in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket feed started for {len(self.symbols)} symbols")
    
    def stop(self):
        """Stop the websocket feed."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WebSocket feed stopped")
    
    def _run_async_loop(self):
        """Run the async event loop in a thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            loop.close()
    
    async def _connect_and_listen(self):
        """Connect to Coinbase WebSocket and process messages."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed - using REST API fallback")
            await self._rest_api_fallback()
            return
        
        ws_url = "wss://ws-feed.exchange.coinbase.com"
        
        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._reconnect_attempts = 0
                    
                    # Subscribe to channels
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": self.symbols,
                        "channels": ["ticker", "level2_batch", "matches"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {len(self.symbols)} symbols")
                    
                    # Process messages
                    async for message in ws:
                        if not self._running:
                            break
                        self._process_message(json.loads(message))
                        
            except Exception as e:
                self._reconnect_attempts += 1
                wait_time = min(30, 2 ** self._reconnect_attempts)
                logger.warning(f"WebSocket disconnected: {e}. Reconnecting in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def _rest_api_fallback(self):
        """Fallback to REST API polling if websockets unavailable."""
        from connectors.coinbase_connector import get_price
        
        while self._running:
            for symbol in self.symbols:
                try:
                    result = get_price(symbol)
                    if result.get("ok"):
                        self._prices[symbol] = TickerData(
                            symbol=symbol,
                            price=float(result["price"]),
                            bid=float(result.get("bid", result["price"])),
                            ask=float(result.get("ask", result["price"])),
                            spread=0.002,  # Estimate
                            volume_24h=float(result.get("volume_24h", 0)),
                            timestamp=datetime.now(timezone.utc),
                        )
                        self._last_update[symbol] = time.time()
                except Exception as e:
                    logger.debug(f"REST fallback error for {symbol}: {e}")
            
            await asyncio.sleep(1)  # Poll every second
    
    def _process_message(self, msg: dict):
        """Process incoming WebSocket message."""
        msg_type = msg.get("type")
        
        if msg_type == "ticker":
            self._handle_ticker(msg)
        elif msg_type == "l2update":
            self._handle_orderbook_update(msg)
        elif msg_type == "match":
            self._handle_match(msg)
    
    def _handle_ticker(self, msg: dict):
        """Handle ticker update."""
        symbol = msg.get("product_id")
        if not symbol:
            return
        
        try:
            price = float(msg.get("price", 0))
            bid = float(msg.get("best_bid", price))
            ask = float(msg.get("best_ask", price))
            
            self._prices[symbol] = TickerData(
                symbol=symbol,
                price=price,
                bid=bid,
                ask=ask,
                spread=(ask - bid) / bid if bid > 0 else 0,
                volume_24h=float(msg.get("volume_24h", 0)),
                timestamp=datetime.now(timezone.utc),
                trade_side=msg.get("side", ""),
                trade_size=float(msg.get("last_size", 0)),
            )
            self._last_update[symbol] = time.time()
            
            # Notify callbacks
            for cb in self._callbacks:
                try:
                    cb(symbol, self._prices[symbol])
                except Exception as e:
                    logger.debug(f"Callback error: {e}")
                    
        except (ValueError, TypeError) as e:
            logger.debug(f"Ticker parse error: {e}")
    
    def _handle_orderbook_update(self, msg: dict):
        """Handle order book update."""
        symbol = msg.get("product_id")
        if not symbol:
            return
        
        # Initialize if needed
        if symbol not in self._order_books:
            self._order_books[symbol] = OrderBookSnapshot(symbol=symbol)
        
        book = self._order_books[symbol]
        changes = msg.get("changes", [])
        
        for change in changes:
            side, price_str, size_str = change
            price = float(price_str)
            size = float(size_str)
            
            if side == "buy":
                # Update bids
                book.bids = [l for l in book.bids if l.price != price]
                if size > 0:
                    book.bids.append(OrderBookLevel(price=price, size=size))
                book.bids.sort(key=lambda x: -x.price)  # Highest first
                book.bids = book.bids[:10]  # Keep top 10
            else:
                # Update asks
                book.asks = [l for l in book.asks if l.price != price]
                if size > 0:
                    book.asks.append(OrderBookLevel(price=price, size=size))
                book.asks.sort(key=lambda x: x.price)  # Lowest first
                book.asks = book.asks[:10]
        
        book.timestamp = datetime.now(timezone.utc)
    
    def _handle_match(self, msg: dict):
        """Handle trade match (executed trade)."""
        symbol = msg.get("product_id")
        if not symbol:
            return
        
        try:
            price = float(msg.get("price", 0))
            size = float(msg.get("size", 0))
            side = msg.get("side", "")
            trade_value = price * size
            
            flow = self._trade_flows[symbol]
            flow.symbol = symbol
            
            if side == "buy":
                flow.buy_volume += trade_value
                flow.buy_count += 1
                if trade_value > 10000:  # $10k+
                    flow.large_buys += 1
            else:
                flow.sell_volume += trade_value
                flow.sell_count += 1
                if trade_value > 10000:
                    flow.large_sells += 1
            
            # Store in history (keep last 100)
            self._trade_history[symbol].append({
                "price": price,
                "size": size,
                "side": side,
                "value": trade_value,
                "time": time.time(),
            })
            self._trade_history[symbol] = self._trade_history[symbol][-100:]
            
            # Decay old trade flow data (reset every 5 minutes)
            if not hasattr(flow, '_last_reset'):
                flow._last_reset = time.time()
            if time.time() - flow._last_reset > 300:
                flow.buy_volume *= 0.5
                flow.sell_volume *= 0.5
                flow.buy_count = int(flow.buy_count * 0.5)
                flow.sell_count = int(flow.sell_count * 0.5)
                flow.large_buys = int(flow.large_buys * 0.5)
                flow.large_sells = int(flow.large_sells * 0.5)
                flow._last_reset = time.time()
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Match parse error: {e}")


# Global instance
_feed_instance: Optional[WebSocketPriceFeed] = None


def get_websocket_feed() -> WebSocketPriceFeed:
    """Get or create the global websocket feed instance."""
    global _feed_instance
    if _feed_instance is None:
        _feed_instance = WebSocketPriceFeed()
    return _feed_instance


def start_feed(symbols: List[str]):
    """Start the websocket feed with given symbols."""
    feed = get_websocket_feed()
    feed.subscribe(symbols)
    feed.start()
    return feed
