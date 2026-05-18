"""
Enhanced Activity Logging Service
=================================
Provides structured logging with filtering, search, and analytics.

Features:
- Structured activity logging with metadata
- Full-text search across log messages
- Filtering by category, level, time range
- Activity analytics and aggregations
- Log retention and cleanup
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

from sqlalchemy import desc, and_, or_, func
from sqlalchemy.orm import Query

from database.db import session_scope
from database.models import ActivityLog
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(str, Enum):
    API = "api"
    PAPER_TRADE = "paper_trade"
    LIVE_TRADE = "live_trade"
    AI = "ai"
    RISK = "risk"
    SETTINGS = "settings"
    SYSTEM = "system"
    MARKET = "market"
    WALLET = "wallet"
    STRATEGY = "strategy"
    SIGNAL = "signal"
    NOTIFICATION = "notification"


@dataclass
class ActivityEntry:
    """Structured activity log entry."""
    id: int
    category: str
    level: str
    message: str
    wallet_id: Optional[int]
    created_at: datetime
    
    # Parsed metadata from message
    symbol: Optional[str] = None
    action: Optional[str] = None
    pnl: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "level": self.level,
            "message": self.message,
            "wallet_id": self.wallet_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "symbol": self.symbol,
            "action": self.action,
            "pnl": self.pnl,
        }


@dataclass
class ActivityFilter:
    """Filter criteria for activity logs."""
    categories: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    wallet_id: Optional[int] = None
    search_query: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: list[str] = field(default_factory=list)
    limit: int = 100
    offset: int = 0


@dataclass
class ActivityStats:
    """Activity log statistics."""
    total_count: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_level: dict[str, int] = field(default_factory=dict)
    by_hour: dict[str, int] = field(default_factory=dict)
    recent_errors: int = 0
    recent_warnings: int = 0


class ActivityService:
    """Service for managing and querying activity logs."""
    
    # Known symbols for extraction
    KNOWN_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "AVAX-USD"]
    
    def __init__(self):
        pass
    
    def log(
        self,
        message: str,
        category: str = "system",
        level: str = "info",
        wallet_id: Optional[int] = None,
        **metadata: Any,
    ) -> int:
        """
        Create a new activity log entry.
        
        Args:
            message: Log message
            category: Log category (api, paper_trade, ai, etc.)
            level: Log level (debug, info, warn, error, critical)
            wallet_id: Optional associated wallet ID
            **metadata: Additional metadata to include in message
        
        Returns:
            ID of the created log entry
        """
        # Format metadata into message if provided
        if metadata:
            meta_str = " | ".join(f"{k}={v}" for k, v in metadata.items())
            message = f"{message} [{meta_str}]"
        
        with session_scope() as s:
            log_entry = ActivityLog(
                category=category,
                level=level,
                message=message,
                wallet_id=wallet_id,
                created_at=utcnow(),
            )
            s.add(log_entry)
            s.flush()
            log_id = log_entry.id
        
        # Also log to application logger
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{category}] {message}")
        
        return log_id
    
    def log_trade_event(
        self,
        event_type: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        wallet_id: Optional[int] = None,
        pnl: Optional[float] = None,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> int:
        """Log a trade-specific event with structured data."""
        if event_type == "open":
            message = f"Opened {side} {qty} {symbol} @ ${price:.4f}"
            if confidence:
                message += f" (conf={confidence:.2f})"
        elif event_type == "close":
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "N/A"
            message = f"Closed {side} {qty} {symbol} @ ${price:.4f}, PnL={pnl_str}"
            if reason:
                message += f" [{reason}]"
        elif event_type == "stop_loss":
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "N/A"
            message = f"Stop-loss triggered: {symbol} @ ${price:.4f}, PnL={pnl_str}"
        elif event_type == "take_profit":
            pnl_str = f"${pnl:+.2f}" if pnl is not None else "N/A"
            message = f"Take-profit triggered: {symbol} @ ${price:.4f}, PnL={pnl_str}"
        elif event_type == "signal":
            message = f"Signal generated: {side} {symbol} @ ${price:.4f}"
            if confidence:
                message += f" (conf={confidence:.2f})"
        else:
            message = f"{event_type}: {symbol} {side} {qty} @ ${price:.4f}"
        
        level = "info"
        if event_type in ["stop_loss", "close"] and pnl is not None and pnl < 0:
            level = "warn"
        
        return self.log(
            message=message,
            category="paper_trade",
            level=level,
            wallet_id=wallet_id,
        )
    
    def log_ai_decision(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        reasoning: str,
        wallet_id: Optional[int] = None,
    ) -> int:
        """Log an AI trading decision."""
        message = f"AI Decision: {decision} {symbol} (conf={confidence:.2f}) - {reasoning[:100]}"
        return self.log(
            message=message,
            category="ai",
            level="info",
            wallet_id=wallet_id,
        )
    
    def log_risk_event(
        self,
        event_type: str,
        details: str,
        wallet_id: Optional[int] = None,
    ) -> int:
        """Log a risk management event."""
        level = "warn" if "rejected" in event_type.lower() or "exceeded" in event_type.lower() else "info"
        message = f"Risk: {event_type} - {details}"
        return self.log(
            message=message,
            category="risk",
            level=level,
            wallet_id=wallet_id,
        )
    
    def get_activities(self, filter: ActivityFilter) -> list[ActivityEntry]:
        """
        Query activity logs with filtering.
        
        Args:
            filter: Filter criteria
        
        Returns:
            List of matching activity entries
        """
        with session_scope() as s:
            query = s.query(ActivityLog)
            
            # Apply filters
            conditions = []
            
            if filter.categories:
                conditions.append(ActivityLog.category.in_(filter.categories))
            
            if filter.levels:
                conditions.append(ActivityLog.level.in_(filter.levels))
            
            if filter.wallet_id is not None:
                conditions.append(ActivityLog.wallet_id == filter.wallet_id)
            
            if filter.start_date:
                conditions.append(ActivityLog.created_at >= filter.start_date)
            
            if filter.end_date:
                conditions.append(ActivityLog.created_at <= filter.end_date)
            
            if filter.search_query:
                # Simple LIKE search
                search_pattern = f"%{filter.search_query}%"
                conditions.append(ActivityLog.message.ilike(search_pattern))
            
            if filter.symbols:
                # Search for symbols in message
                symbol_conditions = [
                    ActivityLog.message.ilike(f"%{sym}%") 
                    for sym in filter.symbols
                ]
                conditions.append(or_(*symbol_conditions))
            
            if conditions:
                query = query.filter(and_(*conditions))
            
            # Order by most recent first
            query = query.order_by(desc(ActivityLog.created_at))
            
            # Apply pagination
            query = query.offset(filter.offset).limit(filter.limit)
            
            # Convert to ActivityEntry objects
            entries = []
            for log in query.all():
                entry = ActivityEntry(
                    id=log.id,
                    category=log.category,
                    level=log.level,
                    message=log.message,
                    wallet_id=log.wallet_id,
                    created_at=log.created_at,
                )
                
                # Extract metadata from message
                entry.symbol = self._extract_symbol(log.message)
                entry.action = self._extract_action(log.message)
                entry.pnl = self._extract_pnl(log.message)
                
                entries.append(entry)
            
            return entries
    
    def get_recent(
        self,
        limit: int = 50,
        categories: Optional[list[str]] = None,
        levels: Optional[list[str]] = None,
    ) -> list[ActivityEntry]:
        """Get most recent activity logs."""
        return self.get_activities(ActivityFilter(
            categories=categories or [],
            levels=levels or [],
            limit=limit,
        ))
    
    def get_stats(
        self,
        wallet_id: Optional[int] = None,
        hours: int = 24,
    ) -> ActivityStats:
        """Get activity log statistics."""
        stats = ActivityStats()
        
        cutoff = utcnow() - timedelta(hours=hours)
        
        with session_scope() as s:
            query = s.query(ActivityLog).filter(ActivityLog.created_at >= cutoff)
            
            if wallet_id is not None:
                query = query.filter(ActivityLog.wallet_id == wallet_id)
            
            logs = query.all()
            stats.total_count = len(logs)
            
            # Count by category
            for log in logs:
                cat = log.category or "unknown"
                stats.by_category[cat] = stats.by_category.get(cat, 0) + 1
                
                level = log.level or "info"
                stats.by_level[level] = stats.by_level.get(level, 0) + 1
                
                if log.created_at:
                    hour = log.created_at.strftime("%H:00")
                    stats.by_hour[hour] = stats.by_hour.get(hour, 0) + 1
                
                if level == "error":
                    stats.recent_errors += 1
                elif level == "warn":
                    stats.recent_warnings += 1
        
        return stats
    
    def search(
        self,
        query: str,
        limit: int = 50,
        categories: Optional[list[str]] = None,
    ) -> list[ActivityEntry]:
        """Full-text search across activity logs."""
        return self.get_activities(ActivityFilter(
            search_query=query,
            categories=categories or [],
            limit=limit,
        ))
    
    def get_trade_history(
        self,
        wallet_id: Optional[int] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[ActivityEntry]:
        """Get trade-specific activity logs."""
        filter = ActivityFilter(
            categories=["paper_trade", "live_trade"],
            wallet_id=wallet_id,
            limit=limit,
        )
        
        if symbol:
            filter.symbols = [symbol]
        
        return self.get_activities(filter)
    
    def get_ai_decisions(
        self,
        wallet_id: Optional[int] = None,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Get AI decision activity logs."""
        return self.get_activities(ActivityFilter(
            categories=["ai"],
            wallet_id=wallet_id,
            limit=limit,
        ))
    
    def get_errors_and_warnings(
        self,
        hours: int = 24,
        wallet_id: Optional[int] = None,
    ) -> list[ActivityEntry]:
        """Get recent errors and warnings."""
        return self.get_activities(ActivityFilter(
            levels=["error", "warn", "critical"],
            wallet_id=wallet_id,
            start_date=utcnow() - timedelta(hours=hours),
            limit=200,
        ))
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """Delete activity logs older than specified days."""
        cutoff = utcnow() - timedelta(days=days)
        
        with session_scope() as s:
            deleted = s.query(ActivityLog).filter(
                ActivityLog.created_at < cutoff
            ).delete()
            
            logger.info(f"Cleaned up {deleted} old activity logs")
            return deleted
    
    # Helper methods for parsing
    
    def _extract_symbol(self, message: str) -> Optional[str]:
        """Extract trading symbol from message."""
        for sym in self.KNOWN_SYMBOLS:
            if sym in message:
                return sym
        return None
    
    def _extract_action(self, message: str) -> Optional[str]:
        """Extract action type from message."""
        message_lower = message.lower()
        
        if "opened" in message_lower or "open" in message_lower:
            return "OPEN"
        elif "closed" in message_lower or "close" in message_lower:
            return "CLOSE"
        elif "stop-loss" in message_lower or "stop_loss" in message_lower:
            return "STOP_LOSS"
        elif "take-profit" in message_lower or "take_profit" in message_lower:
            return "TAKE_PROFIT"
        elif "signal" in message_lower:
            return "SIGNAL"
        elif "rejected" in message_lower:
            return "REJECTED"
        
        return None
    
    def _extract_pnl(self, message: str) -> Optional[float]:
        """Extract P&L value from message."""
        import re
        
        # Look for patterns like "PnL=$+15.00" or "PnL=-10.50"
        patterns = [
            r"PnL=\$?([+-]?\d+\.?\d*)",
            r"P&L=\$?([+-]?\d+\.?\d*)",
            r"pnl=\$?([+-]?\d+\.?\d*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None


# Singleton instance
_activity_service: Optional[ActivityService] = None


def get_activity_service() -> ActivityService:
    """Get the singleton activity service instance."""
    global _activity_service
    if _activity_service is None:
        _activity_service = ActivityService()
    return _activity_service


# Convenience functions
def log_activity(
    message: str,
    category: str = "system",
    level: str = "info",
    wallet_id: Optional[int] = None,
    **metadata: Any,
) -> int:
    """Log an activity (convenience function)."""
    return get_activity_service().log(
        message=message,
        category=category,
        level=level,
        wallet_id=wallet_id,
        **metadata,
    )


def log_trade(
    event_type: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    **kwargs: Any,
) -> int:
    """Log a trade event (convenience function)."""
    return get_activity_service().log_trade_event(
        event_type=event_type,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        **kwargs,
    )
