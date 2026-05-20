"""
System-Wide Debug Logger

Captures ALL failures, errors, warnings, and anomalies across the entire AlphaPilot system.
This is a centralized logging system that:
1. Intercepts all logger.error/exception calls
2. Tracks API failures (Coinbase, LunarCrush, etc.)
3. Monitors settings misconfigurations
4. Tracks data fetch failures
5. Records trade execution issues
6. Monitors page/route errors

All logs are persisted to the database and available via the Debug Console.
"""
from __future__ import annotations

import logging
import traceback
import functools
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Import database session
try:
    from database.db import session_scope
    from database.models import ActivityLog
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@dataclass
class DebugEntry:
    """A single debug log entry."""
    timestamp: datetime
    category: str  # api, trade, risk, settings, data, page, system
    level: str  # error, warn, info, debug
    source: str  # file/module name
    message: str
    details: dict = field(default_factory=dict)
    traceback: Optional[str] = None


class SystemDebugger:
    """
    Centralized debug logging system.
    Captures and persists all system failures for analysis.
    """
    
    # Category definitions
    CATEGORIES = {
        'api': 'API & External Services',
        'trade': 'Trade Execution',
        'risk': 'Risk Manager',
        'settings': 'Settings & Config',
        'data': 'Data Fetching',
        'page': 'Page & Route Errors',
        'system': 'System & Scheduler',
        'signal': 'Signal Generation',
        'learning': 'Learning Engine',
        'websocket': 'WebSocket & Real-time',
        'database': 'Database Operations',
    }
    
    def __init__(self):
        self._memory_buffer: list[DebugEntry] = []
        self._max_buffer = 1000
        self._stats = defaultdict(int)
        self._last_persist = time.time()
        self._persist_interval = 5  # seconds
        
    def log(
        self,
        category: str,
        level: str,
        message: str,
        source: str = "",
        details: dict = None,
        include_traceback: bool = False,
        wallet_id: int = None,
    ):
        """Log a debug entry."""
        entry = DebugEntry(
            timestamp=datetime.utcnow(),
            category=category,
            level=level,
            source=source or self._get_caller_source(),
            message=message,
            details=details or {},
            traceback=traceback.format_exc() if include_traceback else None,
        )
        
        # Update stats
        self._stats[f"{category}_{level}"] += 1
        self._stats[f"total_{level}"] += 1
        
        # Add to memory buffer
        self._memory_buffer.append(entry)
        if len(self._memory_buffer) > self._max_buffer:
            self._memory_buffer.pop(0)
        
        # Persist to database periodically
        if DB_AVAILABLE and (time.time() - self._last_persist) > self._persist_interval:
            self._persist_to_db(entry, wallet_id)
            self._last_persist = time.time()
        elif DB_AVAILABLE and level in ('error', 'warn'):
            # Always persist errors/warnings immediately
            self._persist_to_db(entry, wallet_id)
    
    def _persist_to_db(self, entry: DebugEntry, wallet_id: int = None):
        """Persist entry to database."""
        try:
            with session_scope() as s:
                # Build detailed message
                msg = f"[{entry.source}] {entry.message}"
                if entry.details:
                    details_str = " | ".join(f"{k}={v}" for k, v in entry.details.items())
                    msg += f" ({details_str})"
                if entry.traceback and entry.level == 'error':
                    # Include first 200 chars of traceback
                    msg += f" | TB: {entry.traceback[:200]}"
                
                s.add(ActivityLog(
                    category=entry.category,
                    level=entry.level if entry.level != 'warn' else 'warning',
                    message=msg[:2000],  # Limit message size
                    wallet_id=wallet_id,
                ))
        except Exception:
            pass  # Don't let logging failures cascade
    
    def _get_caller_source(self) -> str:
        """Get the source file/function of the caller."""
        import inspect
        for frame_info in inspect.stack()[3:10]:
            filename = frame_info.filename
            if 'debug_logger' not in filename and 'logging' not in filename:
                # Extract just the relevant part
                parts = filename.split('/')
                if 'alphapilot_ai' in parts:
                    idx = parts.index('alphapilot_ai')
                    return '/'.join(parts[idx+1:])
                return parts[-1] if parts else filename
        return 'unknown'
    
    def error(self, category: str, message: str, **kwargs):
        """Log an error."""
        self.log(category, 'error', message, include_traceback=True, **kwargs)
    
    def warn(self, category: str, message: str, **kwargs):
        """Log a warning."""
        self.log(category, 'warn', message, **kwargs)
    
    def info(self, category: str, message: str, **kwargs):
        """Log info."""
        self.log(category, 'info', message, **kwargs)
    
    def debug(self, category: str, message: str, **kwargs):
        """Log debug info."""
        self.log(category, 'debug', message, **kwargs)
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        return dict(self._stats)
    
    def get_recent(self, limit: int = 100, category: str = None, level: str = None) -> list[dict]:
        """Get recent log entries."""
        entries = self._memory_buffer[-limit:]
        if category:
            entries = [e for e in entries if e.category == category]
        if level:
            entries = [e for e in entries if e.level == level]
        return [
            {
                'timestamp': e.timestamp.isoformat(),
                'category': e.category,
                'level': e.level,
                'source': e.source,
                'message': e.message,
                'details': e.details,
                'traceback': e.traceback,
            }
            for e in reversed(entries)
        ]


# Global singleton
_debugger: Optional[SystemDebugger] = None


def get_debugger() -> SystemDebugger:
    """Get the global debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = SystemDebugger()
    return _debugger


# Convenience functions
def log_error(category: str, message: str, **kwargs):
    get_debugger().error(category, message, **kwargs)


def log_warn(category: str, message: str, **kwargs):
    get_debugger().warn(category, message, **kwargs)


def log_info(category: str, message: str, **kwargs):
    get_debugger().info(category, message, **kwargs)


def log_debug(category: str, message: str, **kwargs):
    get_debugger().debug(category, message, **kwargs)


# Decorator to wrap functions with debug logging
def track_errors(category: str):
    """Decorator to automatically log errors in a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                get_debugger().error(
                    category,
                    f"{func.__name__} failed: {str(e)}",
                    source=func.__module__,
                    details={'args': str(args)[:200], 'kwargs': str(kwargs)[:200]},
                )
                raise
        return wrapper
    return decorator


def track_api_call(api_name: str):
    """Decorator to track API calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                if elapsed > 5:  # Slow API call
                    get_debugger().warn(
                        'api',
                        f"{api_name} slow response: {elapsed:.2f}s",
                        source=func.__module__,
                    )
                return result
            except Exception as e:
                elapsed = time.time() - start
                get_debugger().error(
                    'api',
                    f"{api_name} failed after {elapsed:.2f}s: {str(e)}",
                    source=func.__module__,
                )
                raise
        return wrapper
    return decorator


# Settings validation helper
def validate_settings(settings: dict, expected: dict) -> list[str]:
    """
    Validate settings and log any issues.
    Returns list of issues found.
    """
    issues = []
    debugger = get_debugger()
    
    for key, config in expected.items():
        value = settings.get(key)
        
        # Check if required setting is missing
        if config.get('required') and value is None:
            msg = f"Required setting '{key}' is missing"
            issues.append(msg)
            debugger.warn('settings', msg, details={'key': key})
        
        # Check type
        if value is not None and 'type' in config:
            expected_type = config['type']
            if not isinstance(value, expected_type):
                msg = f"Setting '{key}' has wrong type: expected {expected_type.__name__}, got {type(value).__name__}"
                issues.append(msg)
                debugger.warn('settings', msg, details={'key': key, 'value': str(value)[:50]})
        
        # Check range
        if value is not None and 'min' in config and value < config['min']:
            msg = f"Setting '{key}' value {value} is below minimum {config['min']}"
            issues.append(msg)
            debugger.warn('settings', msg, details={'key': key, 'value': value})
        
        if value is not None and 'max' in config and value > config['max']:
            msg = f"Setting '{key}' value {value} is above maximum {config['max']}"
            issues.append(msg)
            debugger.warn('settings', msg, details={'key': key, 'value': value})
    
    return issues


# Hook into Python's logging system to capture all logs
class DebugLogHandler(logging.Handler):
    """Custom log handler that routes to our debug system."""
    
    LEVEL_MAP = {
        logging.ERROR: 'error',
        logging.WARNING: 'warn',
        logging.INFO: 'info',
        logging.DEBUG: 'debug',
    }
    
    CATEGORY_KEYWORDS = {
        'api': ['api', 'request', 'response', 'http', 'fetch', 'coinbase', 'lunar', 'coinglass'],
        'trade': ['trade', 'order', 'position', 'open_trade', 'close_trade', 'execute'],
        'risk': ['risk', 'reject', 'block', 'limit', 'exposure'],
        'data': ['candle', 'price', 'data', 'feed', 'universe'],
        'websocket': ['websocket', 'ws', 'socket', 'stream'],
        'database': ['db', 'database', 'query', 'session', 'commit'],
        'signal': ['signal', 'indicator', 'strategy', 'momentum', 'rsi', 'macd'],
        'learning': ['learn', 'autonomous', 'pattern', 'memory'],
        'settings': ['config', 'setting', 'parameter'],
    }
    
    def emit(self, record: logging.LogRecord):
        try:
            level = self.LEVEL_MAP.get(record.levelno, 'info')
            
            # Only capture warnings and errors by default
            if record.levelno < logging.WARNING:
                return
            
            # Determine category from message content
            category = 'system'
            msg_lower = record.getMessage().lower()
            for cat, keywords in self.CATEGORY_KEYWORDS.items():
                if any(kw in msg_lower for kw in keywords):
                    category = cat
                    break
            
            # Also check the logger name
            name_lower = record.name.lower()
            for cat, keywords in self.CATEGORY_KEYWORDS.items():
                if any(kw in name_lower for kw in keywords):
                    category = cat
                    break
            
            get_debugger().log(
                category=category,
                level=level,
                message=record.getMessage(),
                source=f"{record.name}:{record.lineno}",
                include_traceback=record.exc_info is not None,
            )
        except Exception:
            pass  # Never let the handler fail


def install_global_handler():
    """Install the debug handler on the root logger."""
    root_logger = logging.getLogger()
    
    # Check if already installed
    for handler in root_logger.handlers:
        if isinstance(handler, DebugLogHandler):
            return
    
    handler = DebugLogHandler()
    handler.setLevel(logging.WARNING)  # Only capture warnings and above
    root_logger.addHandler(handler)


# Auto-install on import
install_global_handler()
