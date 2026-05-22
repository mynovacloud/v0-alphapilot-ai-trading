"""
Server-rendered web UI for AlphaPilot AI.

Uses FastAPI + Jinja2 + HTMX. Mounted onto the main FastAPI app at "/".
Every page is a real route. Forms post to API-style routes that return
HTML fragments for HTMX to swap in.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ai.ai_engine import AIEngine
from ai.learning_memory import LearningMemory
from analytics.performance import performance_metrics
from analytics.portfolio import (
    equity_curve_df,
    get_all_trades_df,
    get_wallets,
    pnl_by_strategy,
    pnl_by_wallet,
    portfolio_summary,
)
from config import bot_config
from config.bot_config import BotConfig
from config.settings import settings
from connectors.live_prices import get_price as live_price
from connectors.live_prices import known_symbols
from connectors.registry import (
    CONNECTOR_REGISTRY,
    REAL_AUTH_PLATFORMS,
    VISIBLE_PLATFORMS,
    get_connector,
)
from database.db import reset_db, session_scope
from database.models import (
    ActivityLog,
    AppSetting,
    ApiCredentialPlaceholder,
    ClaudeDecision,
    PaperTrade,
    Position,
    Strategy,
    Wallet,
)
from trading.backtester import run_backtest
from trading.bot_engine import bot_engine
from trading.market_scanner import scan_markets
from trading.paper_trading_engine import PaperTradingEngine
from trading.reconciler import reconciler
from trading.risk_manager import RiskManager
from trading.strategy_manager import (
    create_strategy,
    delete_strategy,
    list_strategies,
)
from services.scheduler import bot_scheduler
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Number formatting filters used everywhere in templates
def _fmt_money(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "$0.00"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        v = float(value) * 100
    except Exception:
        return "0%"
    return f"{v:.{digits}f}%"


def _fmt_signed(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "$0.00"
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _fmt_dt(value: Any) -> str:
    if not value:
        return "-"
    try:
        return value.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def _fmt_timeago(value: Any) -> str:
    """Format a datetime as a human-readable 'time ago' string."""
    if not value:
        return "-"
    try:
        from utils.helpers import utcnow
        now = utcnow()
        # Handle timezone-naive datetimes
        if hasattr(value, 'tzinfo') and value.tzinfo is not None:
            value = value.replace(tzinfo=None)
        if hasattr(now, 'tzinfo') and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        
        delta = now - value
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days}d ago"
        else:
            return value.strftime("%Y-%m-%d")
    except Exception:
        return str(value) if value else "-"


templates.env.filters["money"] = _fmt_money
templates.env.filters["pct"] = _fmt_pct
templates.env.filters["signed"] = _fmt_signed
templates.env.filters["dt"] = _fmt_dt
templates.env.filters["timeago"] = _fmt_timeago

router = APIRouter()

_engine = PaperTradingEngine()
_ai = AIEngine()
_memory = LearningMemory()


def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    """Common template context."""
    return {
        "request": request,
        "app_name": settings.app_name,
        "live_trading_enabled": settings.live_trading_enabled,
        "active": "",
        **extra,
    }


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, mode: str = "all") -> HTMLResponse:
    mode = mode if mode in {"all", "paper", "live"} else "all"
    summary = portfolio_summary(mode)
    perf = performance_metrics()
    wallets = get_wallets()
    wallet_modes = {w["id"]: (w.get("trading_mode") or "paper").lower() for w in wallets}

    # Equity curve points for the SVG chart (scoped to selected mode)
    eq = equity_curve_df(mode)
    points: list[dict[str, Any]] = []
    if not eq.empty:
        for _, row in eq.iterrows():
            points.append({"date": str(row["date"]), "equity": float(row["equity"])})

    # IDs to filter on, based on mode
    if mode == "paper":
        scoped_wids = {wid for wid, m in wallet_modes.items() if m == "paper"}
    elif mode == "live":
        scoped_wids = {wid for wid, m in wallet_modes.items() if m in {"live", "live_shadow"}}
    else:
        scoped_wids = set(wallet_modes.keys())

    # Recent activity (scoped: also filter by wallet_id when scoping to a mode)
    with session_scope() as s:
        log_q = s.query(ActivityLog)
        if mode != "all":
            # ActivityLog rows from trade engine carry wallet_id; system rows
            # without wallet_id are kept so the user still sees scheduler events.
            log_q = log_q.filter(
                (ActivityLog.wallet_id == None)  # noqa: E711
                | (ActivityLog.wallet_id.in_(scoped_wids if scoped_wids else {-1}))
            )
        logs = log_q.order_by(ActivityLog.created_at.desc()).limit(8).all()
        recent_logs = [
            {
                "category": l.category,
                "level": l.level,
                "message": l.message,
                "created_at": l.created_at,
                "wallet_id": l.wallet_id,
                "trading_mode": wallet_modes.get(l.wallet_id, ""),
            }
            for l in logs
        ]

        # Recent trades, filtered by mode
        trade_q = s.query(PaperTrade)
        if mode != "all":
            trade_q = trade_q.filter(PaperTrade.wallet_id.in_(scoped_wids if scoped_wids else {-1}))
        trades = trade_q.order_by(PaperTrade.opened_at.desc()).limit(8).all()
        wallet_names = {w["id"]: w["name"] for w in wallets}
        recent_trades = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "status": t.status,
                "realized_pnl": t.realized_pnl,
                "unrealized_pnl": t.unrealized_pnl,
                "wallet": wallet_names.get(t.wallet_id, "?"),
                "wallet_id": t.wallet_id,
                "trading_mode": wallet_modes.get(t.wallet_id, "paper"),
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in trades
        ]

    # Live training session indicator so the dashboard can show "session running"
    from config.bot_config import get as cfg_get
    session_active = str(cfg_get("training_session_active") or "").lower() in {"1", "true", "yes"}

    return templates.TemplateResponse(request=request, name="dashboard.html", context=_ctx(
            request,
            active="dashboard",
            summary=summary,
            perf=perf,
            wallets=wallets,
            equity_points=points,
            recent_logs=recent_logs,
            recent_trades=recent_trades,
            current_mode=mode,
            session_active=session_active,
        ),
)


# ----------------------------------------------------------------------
# Wallets
# ----------------------------------------------------------------------

@router.get("/wallets", response_class=HTMLResponse)
def wallets_page(request: Request) -> HTMLResponse:
    wallets = get_wallets()
    # Decorate with PnL per wallet
    pnl_df = pnl_by_wallet()
    pnl_map = {row["wallet"]: float(row["pnl"]) for _, row in pnl_df.iterrows()} if not pnl_df.empty else {}

    with session_scope() as s:
        for w in wallets:
            open_count = s.query(PaperTrade).filter(
                PaperTrade.wallet_id == w["id"], PaperTrade.status == "open"
            ).count()
            closed_count = s.query(PaperTrade).filter(
                PaperTrade.wallet_id == w["id"], PaperTrade.status == "closed"
            ).count()
            w["open_trades"] = open_count
            w["closed_trades"] = closed_count
            label = f"{w['name']} ({w['platform']})"
            w["pnl"] = pnl_map.get(label, 0.0)

    return templates.TemplateResponse(request=request, name="wallets.html", context=_ctx(request, active="wallets", wallets=wallets),
)


@router.get("/wallets/new", response_class=HTMLResponse)
def add_wallet_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="add_wallet.html", context=_ctx(
            request,
            active="wallets",
            platforms=list(VISIBLE_PLATFORMS),
            real_auth_platforms=sorted(REAL_AUTH_PLATFORMS),
            risk_profiles=["Conservative", "Moderate", "Aggressive", "Degenerate"],
        ),
    )


@router.post("/wallets/new")
def add_wallet_submit(
    name: str = Form(...),
    platform: str = Form(...),
    paper_balance: float = Form(10000.0),
    risk_profile: str = Form("Moderate"),
    sandbox_mode: str = Form("on"),
    api_key: str = Form(""),
    api_secret: str = Form(""),
    api_passphrase: str = Form(""),
    account_id: str = Form(""),
) -> RedirectResponse:
    # If keys were provided AND the platform supports real auth, validate them
    # before creating the wallet, so the user gets feedback immediately.
    api_status = "no-keys"
    connection_status = "ready (paper)"
    if api_key and platform in REAL_AUTH_PLATFORMS:
        connector = get_connector(
            platform,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            account_id=account_id,
        )
        v = connector.validate_credentials()
        if v.get("valid"):
            api_status = "live (read-only)"
            connection_status = "connected (live)"
        else:
            api_status = f"invalid: {v.get('error', 'auth failed')}"
            connection_status = "auth failed"

    with session_scope() as s:
        w = Wallet(
            name=name,
            platform=platform,
            paper_balance=paper_balance,
            risk_profile=risk_profile,
            sandbox_mode=sandbox_mode == "on",
            paper_trading_only=True,
            connection_status=connection_status,
            api_status=api_status,
        )
        s.add(w)
        s.flush()
        s.add(
            ApiCredentialPlaceholder(
                wallet_id=w.id,
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
                account_id=account_id,
            )
        )
        s.add(
            ActivityLog(
                category="wallet",
                level="info",
                wallet_id=w.id,
                message=(
                    f"Wallet '{w.name}' ({w.platform}) created — "
                    f"paper balance ${paper_balance:.2f}, api_status={api_status}"
                ),
            )
        )
    return RedirectResponse(url="/wallets", status_code=303)


@router.post("/wallets/{wallet_id}/futures")
def wallet_update_futures(
    wallet_id: int,
    futures_enabled: str = Form("off"),
    max_leverage: float = Form(1.0),
    default_leverage: float = Form(1.0),
    margin_mode: str = Form("isolated"),
    liquidation_buffer_pct: float = Form(0.10),
) -> RedirectResponse:
    """
    Update perpetual-futures controls on a wallet. Bounded so users cannot
    accidentally configure dangerous values:
      - max_leverage clamped to [1, 20]
      - default_leverage clamped to [1, max_leverage]
      - liquidation_buffer_pct clamped to [0.02, 0.5]
    """
    enabled = futures_enabled in {"on", "true", "1"}
    max_lev = max(1.0, min(float(max_leverage or 1.0), 20.0))
    def_lev = max(1.0, min(float(default_leverage or 1.0), max_lev))
    buf = max(0.02, min(float(liquidation_buffer_pct or 0.10), 0.5))
    mm = "cross" if (margin_mode or "isolated").lower() == "cross" else "isolated"

    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            return RedirectResponse(url="/wallets", status_code=303)
        w.futures_enabled = enabled
        w.max_leverage = max_lev
        w.default_leverage = def_lev
        w.margin_mode = mm
        w.liquidation_buffer_pct = buf
        s.add(
            ActivityLog(
                category="wallet",
                level="info",
                wallet_id=wallet_id,
                message=(
                    f"Futures controls updated on '{w.name}': enabled={enabled}, "
                    f"max_lev={max_lev}x, default_lev={def_lev}x, mode={mm}, buf={buf:.2%}"
                ),
            )
        )
    return RedirectResponse(url=f"/wallets/{wallet_id}", status_code=303)


@router.get("/wallets/{wallet_id}", response_class=HTMLResponse)
def wallet_detail(request: Request, wallet_id: int) -> HTMLResponse:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            return HTMLResponse("Wallet not found", status_code=404)

        wallet = {
            "id": w.id,
            "name": w.name,
            "platform": w.platform,
            "paper_balance": w.paper_balance,
            "risk_profile": w.risk_profile,
            "sandbox_mode": w.sandbox_mode,
            "connection_status": w.connection_status,
            "api_status": w.api_status,
            "last_synced": w.last_synced,
            "created_at": w.created_at,
        }

        positions = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry": p.avg_entry,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in s.query(Position).filter(Position.wallet_id == wallet_id).all()
        ]

        trades = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id)
            .order_by(PaperTrade.opened_at.desc())
            .limit(50)
            .all()
        )
        trade_rows = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "realized_pnl": t.realized_pnl,
                "unrealized_pnl": t.unrealized_pnl,
                "confidence": t.confidence,
                "status": t.status,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in trades
        ]

        # Per-wallet KPIs
        closed = [t for t in trade_rows if t["status"] == "closed"]
        wins = sum(1 for t in closed if (t["realized_pnl"] or 0) > 0)
        total_pnl = sum((t["realized_pnl"] or 0) for t in closed)
        win_rate = wins / len(closed) if closed else 0.0

        strategies = [{"id": st.id, "name": st.name} for st in s.query(Strategy).all()]

    return templates.TemplateResponse(
        request=request,
        name="wallet_detail.html",
        context=_ctx(
            request,
            active="wallets",
            wallet=wallet,
            positions=positions,
            trades=trade_rows,
            strategies=strategies,
            wins=wins,
            total_pnl=total_pnl,
            win_rate=win_rate,
            closed_count=len(closed),
            open_count=len(trade_rows) - len(closed),
        ),
    )


@router.post("/wallets/{wallet_id}/delete")
def wallet_delete(wallet_id: int) -> RedirectResponse:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if w:
            s.delete(w)
            s.add(
                ActivityLog(
                    category="wallet",
                    level="warn",
                    message=f"Wallet {wallet_id} deleted.",
                )
            )
    return RedirectResponse(url="/wallets", status_code=303)


@router.post("/wallets/{wallet_id}/test", response_class=HTMLResponse)
def wallet_test(request: Request, wallet_id: int) -> HTMLResponse:
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            return HTMLResponse("<span class='badge badge-bad'>Not found</span>")
        creds = (
            s.query(ApiCredentialPlaceholder)
            .filter(ApiCredentialPlaceholder.wallet_id == wallet_id)
            .first()
        )
        connector = get_connector(
            w.platform,
            api_key=creds.api_key if creds else "",
            api_secret=creds.api_secret if creds else "",
            api_passphrase=creds.api_passphrase if creds else "",
            account_id=creds.account_id if creds else "",
            sandbox=w.sandbox_mode,
        )
        result = connector.validate_credentials()
        valid = bool(result.get("valid"))
        msg = "Connection OK" if valid else f"Failed: {result.get('error', 'unknown')}"
        if valid:
            w.connection_status = "connected (live)" if result.get("live") else "connected (mock)"
            w.api_status = "live (read-only)" if result.get("live") else "mock"
        else:
            w.connection_status = "auth failed"
            w.api_status = "invalid"
        w.last_synced = utcnow()
        s.add(
            ActivityLog(
                category="api",
                level="info" if valid else "warn",
                wallet_id=wallet_id,
                message=f"API test for {w.platform}: {msg}",
            )
        )
        cls = "badge-good" if valid else "badge-bad"
    return HTMLResponse(f"<span class='badge {cls}'>{msg}</span>")


@router.post("/wallets/{wallet_id}/sync", response_class=HTMLResponse)
def wallet_sync(request: Request, wallet_id: int) -> HTMLResponse:
    """Pull live balances from the exchange (read-only) and update the wallet."""
    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        if not w:
            return HTMLResponse("<span class='badge badge-bad'>Not found</span>")
        creds = (
            s.query(ApiCredentialPlaceholder)
            .filter(ApiCredentialPlaceholder.wallet_id == wallet_id)
            .first()
        )
        if not creds or not creds.api_key:
            return HTMLResponse(
                "<span class='badge badge-warn'>No API keys saved — paper trading only</span>"
            )
        connector = get_connector(
            w.platform,
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
            account_id=creds.account_id,
            sandbox=w.sandbox_mode,
        )
        bal = connector.fetch_balance()
        if bal.get("live"):
            w.real_balance_placeholder = float(bal.get("cash", 0.0))
            w.connection_status = "connected (live)"
            w.api_status = "live (read-only)"
            w.last_synced = utcnow()
            s.add(
                ActivityLog(
                    category="api",
                    level="info",
                    wallet_id=wallet_id,
                    message=(
                        f"Synced {w.platform}: live cash ${w.real_balance_placeholder:,.2f} "
                        f"({len(bal.get('balances', []))} non-zero balances)"
                    ),
                )
            )
            return HTMLResponse(
                f"<span class='badge badge-good'>Live cash ${w.real_balance_placeholder:,.2f}</span>"
            )
        err = bal.get("error", "unknown error")
        s.add(
            ActivityLog(
                category="api",
                level="warn",
                wallet_id=wallet_id,
                message=f"Sync failed for {w.platform}: {err}",
            )
        )
        return HTMLResponse(f"<span class='badge badge-bad'>Sync failed: {err}</span>")


@router.get("/_price")
def api_live_price(symbol: str) -> dict[str, Any]:
    """Endpoint used by JS in templates to look up a live price."""
    return live_price(symbol)


@router.post("/wallets/{wallet_id}/trade")
def wallet_open_trade(
    wallet_id: int,
    symbol: str = Form(...),
    side: str = Form("BUY"),
    qty: float = Form(...),
    entry_price: float = Form(0.0),
    confidence: float = Form(0.6),
    use_live_price: str = Form(""),
    strategy_id: int | None = Form(None),
) -> RedirectResponse:
    # If user asked for live price OR didn't provide a price, look it up.
    if use_live_price == "on" or entry_price <= 0:
        lp = live_price(symbol)
        if lp.get("ok"):
            entry_price = float(lp["price"])
        elif entry_price <= 0:
            # Fall back to refusing — better than fake price.
            with session_scope() as s:
                s.add(
                    ActivityLog(
                        category="paper_trade",
                        level="warn",
                        wallet_id=wallet_id,
                        message=f"Trade rejected: live price unavailable for {symbol}",
                    )
                )
            return RedirectResponse(url=f"/wallets/{wallet_id}", status_code=303)

    _engine.open_trade(
        wallet_id=wallet_id,
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        confidence=confidence,
        strategy_id=strategy_id if strategy_id else None,
    )
    return RedirectResponse(url=f"/wallets/{wallet_id}", status_code=303)


@router.post("/trades/{trade_id}/close")
def trade_close(
    trade_id: int,
    exit_price: float = Form(0.0),
    use_live_price: str = Form(""),
) -> RedirectResponse:
    wallet_id = None
    symbol = ""
    with session_scope() as s:
        t = s.get(PaperTrade, trade_id)
        if t:
            wallet_id = t.wallet_id
            symbol = t.symbol
    if (use_live_price == "on" or exit_price <= 0) and symbol:
        lp = live_price(symbol)
        if lp.get("ok"):
            exit_price = float(lp["price"])
    _engine.close_trade(trade_id, exit_price)
    if wallet_id:
        return RedirectResponse(url=f"/wallets/{wallet_id}", status_code=303)
    return RedirectResponse(url="/wallets", status_code=303)


# ----------------------------------------------------------------------
# Market Scanner
# ----------------------------------------------------------------------

@router.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="scanner.html", context=_ctx(request, active="scanner", opportunities=[]),
)


@router.post("/scanner/run", response_class=HTMLResponse)
def scanner_run(request: Request, n: int = Form(20)) -> HTMLResponse:
    opps = scan_markets(n=n)
    return templates.TemplateResponse(
        request=request,
        name="_scanner_table.html",
        context={"request": request, "opportunities": opps},
    )


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------

@router.get("/strategies", response_class=HTMLResponse)
def strategies_page(request: Request) -> HTMLResponse:
    strategies = list_strategies()
    pnl_df = pnl_by_strategy()
    pnl_map = {row["strategy"]: float(row["pnl"]) for _, row in pnl_df.iterrows()} if not pnl_df.empty else {}
    for s in strategies:
        s["pnl"] = pnl_map.get(s["name"], 0.0)
    return templates.TemplateResponse(request=request, name="strategies.html", context=_ctx(
            request,
            active="strategies",
            strategies=strategies,
            risk_levels=["Conservative", "Moderate", "Aggressive", "Degenerate"],
            strategy_types=[
                "Momentum",
                "Mean Reversion",
                "Volatility Breakout",
                "Probability Edge",
                "Trend Following",
                "Arbitrage",
            ],
            market_types=["Crypto", "Stocks", "Prediction Markets", "Options"],
        ),
)


@router.post("/strategies/new")
def strategies_new(
    name: str = Form(...),
    description: str = Form(""),
    market_type: str = Form("Crypto"),
    strategy_type: str = Form("Momentum"),
    max_position_size: float = Form(1000.0),
    max_daily_loss: float = Form(500.0),
    stop_loss_pct: float = Form(0.05),
    take_profit_pct: float = Form(0.10),
    min_confidence: float = Form(0.6),
    max_open_trades: int = Form(5),
    max_trades_per_day: int = Form(20),
    risk_level: str = Form("Moderate"),
) -> RedirectResponse:
    create_strategy(
        {
            "name": name,
            "description": description,
            "market_type": market_type,
            "strategy_type": strategy_type,
            "max_position_size": max_position_size,
            "max_daily_loss": max_daily_loss,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "min_confidence": min_confidence,
            "max_open_trades": max_open_trades,
            "max_trades_per_day": max_trades_per_day,
            "risk_level": risk_level,
        }
    )
    return RedirectResponse(url="/strategies", status_code=303)


@router.post("/strategies/{strategy_id}/delete")
def strategies_delete(strategy_id: int) -> RedirectResponse:
    delete_strategy(strategy_id)
    return RedirectResponse(url="/strategies", status_code=303)


@router.post("/strategies/{strategy_id}/backtest", response_class=HTMLResponse)
def strategies_backtest(request: Request, strategy_id: int, n_trades: int = Form(200)) -> HTMLResponse:
    result = run_backtest(strategy_id, n_trades=n_trades)
    return templates.TemplateResponse(
        request=request,
        name="_backtest_result.html",
        context={"request": request, "result": result},
    )


# ----------------------------------------------------------------------
# Training Lab
# ----------------------------------------------------------------------
# DEBUG CONSOLE
# ----------------------------------------------------------------------

@router.get("/debug", response_class=HTMLResponse)
def debug_console_page(request: Request) -> HTMLResponse:
    """Debug console page - shows all system errors, warnings, and execution logs."""
    return templates.TemplateResponse(request=request, name="debug_console.html", context=_ctx(request, active="debug"))


@router.get("/debug/logs")
def debug_get_logs() -> JSONResponse:
    """Get all debug logs with comprehensive stats."""
    from datetime import timedelta
    
    now = utcnow()
    day_ago = now - timedelta(hours=24)
    
    with session_scope() as s:
        # Get all activity logs
        logs = s.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(1000).all()
        
        # Count stats
        errors_24h = s.query(ActivityLog).filter(
            ActivityLog.level.in_(['error', 'exception']),
            ActivityLog.created_at >= day_ago
        ).count()
        
        warnings_24h = s.query(ActivityLog).filter(
            ActivityLog.level.in_(['warn', 'warning']),
            ActivityLog.created_at >= day_ago
        ).count()
        
        # Count blocked/rejected trades
        blocked = s.query(ActivityLog).filter(
            (ActivityLog.message.ilike('%blocked%') | 
             ActivityLog.message.ilike('%rejected%') |
             ActivityLog.message.ilike('%RISK REJECTED%')),
            ActivityLog.created_at >= day_ago
        ).count()
        
        # Count executed trades
        executed = s.query(ActivityLog).filter(
            (ActivityLog.message.ilike('%opened%') | 
             ActivityLog.message.ilike('%PAPER TRADE%')),
            ActivityLog.category == 'trade',
            ActivityLog.created_at >= day_ago
        ).count()
        
        # Count API failures
        api_failures = s.query(ActivityLog).filter(
            (ActivityLog.category == 'api') | 
            (ActivityLog.message.ilike('%api%error%')) |
            (ActivityLog.message.ilike('%fetch%fail%')) |
            (ActivityLog.message.ilike('%request%fail%')),
            ActivityLog.level.in_(['error', 'warn', 'warning']),
            ActivityLog.created_at >= day_ago
        ).count()
        
        # Count settings/config issues
        settings_issues = s.query(ActivityLog).filter(
            (ActivityLog.category == 'settings') |
            (ActivityLog.message.ilike('%config%')) |
            (ActivityLog.message.ilike('%setting%')),
            ActivityLog.level.in_(['error', 'warn', 'warning']),
            ActivityLog.created_at >= day_ago
        ).count()
        
        # Check if session is active
        session_active = str(bot_config.get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
        
        log_list = []
        for log in logs:
            # Parse source from message if available
            source = ""
            msg = log.message or ""
            if msg.startswith("[") and "]" in msg:
                source = msg[1:msg.index("]")]
                msg = msg[msg.index("]")+1:].strip()
            
            log_list.append({
                "id": log.id,
                "category": log.category or "system",
                "level": log.level or "info",
                "message": msg,
                "source": source,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "wallet_id": log.wallet_id,
                "details": {},
                "traceback": None,
            })
        
        return JSONResponse({
            "ok": True,
            "logs": log_list,
            "stats": {
                "errors_24h": errors_24h,
                "warnings_24h": warnings_24h,
                "trades_blocked": blocked,
                "trades_executed": executed,
                "api_failures": api_failures,
                "settings_issues": settings_issues,
                "session_active": session_active,
            }
        })


@router.post("/debug/logs/clear")
def debug_clear_logs() -> JSONResponse:
    """Clear all debug logs."""
    with session_scope() as s:
        s.query(ActivityLog).delete()
    return JSONResponse({"ok": True})


@router.post("/debug/log/push")
def debug_log_push(
    level: str = Form("info"),
    category: str = Form("session_error"),
    message: str = Form(...),
    source: str = Form(""),
) -> JSONResponse:
    """
    Client-side failure ingestion. The Training Center calls this whenever
    the browser encounters a session-related error (poll failure, start/stop
    failure, kill-switch event, unhandled JS exception, etc.) so the Debug
    Console — which mirrors ActivityLog — can render it in real time with a
    timestamp instead of silently swallowing the event in devtools.
    """
    lvl = (level or "info").strip().lower()
    if lvl not in ("info", "warn", "warning", "error", "success", "debug"):
        lvl = "info"
    cat = (category or "session_error").strip().lower()[:50] or "session_error"
    src = (source or "").strip()[:120]
    msg = (message or "").strip()
    if not msg:
        return JSONResponse({"ok": False, "error": "message required"}, status_code=400)
    # Prefix with [source] so /debug/logs' parser surfaces it in the source column.
    full = f"[{src}] {msg}" if src else msg
    with session_scope() as s:
        s.add(ActivityLog(category=cat, level=lvl, message=full[:2000]))
    return JSONResponse({"ok": True})


@router.get("/debug/diagnostics")
def debug_run_diagnostics() -> JSONResponse:
    """Run system diagnostics and return results.
    Wrapped in a top-level try/except so a single broken check still returns
    a structured 200 — otherwise FastAPI raises 500 and the Debug Console
    shows nothing, which is exactly the symptom we hit before."""
    import traceback as _tb
    try:
        return _run_diagnostics_inner()
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": str(e) or "Unknown diagnostics error",
            "traceback": _tb.format_exc(),
        })


def _run_diagnostics_inner() -> JSONResponse:
    from config.bot_config import BotConfig
    import os
    
    diagnostics = {}
    
    # 1. Database checks
    db_checks = []
    try:
        with session_scope() as s:
            wallet_count = s.query(Wallet).count()
            trade_count = s.query(PaperTrade).count()
            db_checks.append({"status": "ok", "message": f"Database connected ({wallet_count} wallets, {trade_count} trades)"})
    except Exception as e:
        db_checks.append({"status": "error", "message": f"Database error: {str(e)}"})
    diagnostics["Database"] = db_checks
    
    # 2. Configuration checks
    cfg_checks = []
    cfg = BotConfig.load()
    
    min_conf = cfg.min_confidence
    if min_conf > 0.7:
        cfg_checks.append({"status": "warn", "message": f"min_confidence={min_conf} is very high - few trades will execute"})
    elif min_conf < 0.2:
        cfg_checks.append({"status": "warn", "message": f"min_confidence={min_conf} is very low - many low-quality trades"})
    else:
        cfg_checks.append({"status": "ok", "message": f"min_confidence={min_conf} is reasonable"})
    
    max_open = cfg.max_open_per_wallet
    if max_open < 5:
        cfg_checks.append({"status": "warn", "message": f"max_open_per_wallet={max_open} is low - limits diversification"})
    else:
        cfg_checks.append({"status": "ok", "message": f"max_open_per_wallet={max_open}"})
    
    pos_size = cfg.position_size_usd
    cfg_checks.append({"status": "ok", "message": f"position_size_usd=${pos_size}"})
    
    session_active = str(bot_config.get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
    if session_active:
        cfg_checks.append({"status": "ok", "message": "Training session is ACTIVE"})
    else:
        cfg_checks.append({"status": "warn", "message": "Training session is NOT active"})
    
    diagnostics["Configuration"] = cfg_checks
    
    # 3. API connectivity checks
    api_checks = []
    try:
        from connectors.live_prices import get_price
        result = get_price("BTC-USD")
        if result.get("ok"):
            api_checks.append({"status": "ok", "message": f"Coinbase API connected (BTC=${result.get('price', 0):,.0f})"})
        else:
            api_checks.append({"status": "error", "message": f"Coinbase API error: {result.get('error', 'unknown')}"})
    except Exception as e:
        api_checks.append({"status": "error", "message": f"Coinbase API failed: {str(e)}"})
    
    # Check Claude API
    claude_key = os.environ.get("ANTHROPIC_API_KEY") or bot_config.get("anthropic_api_key")
    if claude_key:
        api_checks.append({"status": "ok", "message": "Claude API key configured"})
    else:
        api_checks.append({"status": "warn", "message": "Claude API key not set - AI decisions disabled"})
    
    diagnostics["API Connectivity"] = api_checks
    
    # 4. Trading system checks
    trading_checks = []
    try:
        with session_scope() as s:
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").count()
            trading_checks.append({"status": "ok", "message": f"{open_trades} open positions"})
            
            # Check for stale positions (open > 24h)
            from datetime import timedelta
            stale_cutoff = utcnow() - timedelta(hours=24)
            stale_trades = s.query(PaperTrade).filter(
                PaperTrade.status == "open",
                PaperTrade.opened_at < stale_cutoff
            ).count()
            if stale_trades > 0:
                trading_checks.append({"status": "warn", "message": f"{stale_trades} positions open > 24 hours"})
            
            # Check wallet balance
            wallet = s.query(Wallet).first()
            if wallet:
                if wallet.paper_balance < 100:
                    trading_checks.append({"status": "error", "message": f"Low paper balance: ${wallet.paper_balance:.2f}"})
                else:
                    trading_checks.append({"status": "ok", "message": f"Paper balance: ${wallet.paper_balance:.2f}"})
    except Exception as e:
        trading_checks.append({"status": "error", "message": f"Trading check failed: {str(e)}"})
    
    diagnostics["Trading System"] = trading_checks
    
    # 5. Risk manager checks
    risk_checks = []
    try:
        from trading.risk_manager import RiskManager
        rm = RiskManager()
        with session_scope() as s:
            wallet = s.query(Wallet).first()
            if wallet:
                # Daily-loss circuit breaker. The internal API is
                # `_daily_loss_tripped(session, wallet, is_paper)` returning
                # (tripped: bool, loss: float) — there is no public
                # `_check_daily_loss_breaker`, so call the real one and
                # surface the actual loss number when it has tripped.
                is_paper = bool(getattr(wallet, "is_paper", True))
                try:
                    tripped, day_loss = rm._daily_loss_tripped(s, wallet, is_paper=is_paper)
                except TypeError:
                    # Older signatures didn't take is_paper; fall back.
                    tripped, day_loss = rm._daily_loss_tripped(s, wallet)
                if tripped:
                    risk_checks.append({
                        "status": "warn",
                        "message": f"Daily loss circuit breaker ACTIVE (${day_loss:,.2f} today)",
                    })
                else:
                    risk_checks.append({"status": "ok", "message": "Circuit breaker not triggered"})

                # Cooldown after consecutive losses. The real API is
                # `_cooldown_until(session, wallet, is_paper)` and returns
                # a datetime or None — not a (bool, msg) tuple.
                try:
                    cooldown_until = rm._cooldown_until(s, wallet, is_paper=is_paper)
                except TypeError:
                    cooldown_until = rm._cooldown_until(s, wallet)
                if cooldown_until is not None:
                    risk_checks.append({
                        "status": "warn",
                        "message": f"Cooldown active until {cooldown_until.isoformat()}",
                    })
                else:
                    risk_checks.append({"status": "ok", "message": "No cooldown active"})
    except Exception as e:
        risk_checks.append({"status": "error", "message": f"Risk check failed: {str(e)}"})
    
    diagnostics["Risk Manager"] = risk_checks
    
    return JSONResponse({"ok": True, "diagnostics": diagnostics})


# ----------------------------------------------------------------------
# TRAINING CENTER
@router.get("/training", response_class=HTMLResponse)
def training_page(request: Request) -> HTMLResponse:
    from ai.claude_learning import (
        get_playbook_with_metadata,
        readiness_score,
        recent_decisions,
        recent_reflections,
    )
    from services.claude_client import is_configured as claude_is_configured
    wallets = get_wallets()
    strategies = list_strategies()
    # Symbols the bot has actually traded (open or closed paper trades).
    # Used to populate the Positions Lab grid on the page.
    with session_scope() as s:
        rows = (
            s.query(
                PaperTrade.symbol,
                PaperTrade.wallet_id,
            )
            .all()
        )
        wallet_names = {w["id"]: w["name"] for w in wallets}
        seen: dict[tuple[str, int], dict[str, Any]] = {}
        for sym, wid in rows:
            key = (sym, wid)
            if key in seen:
                continue
            seen[key] = {
                "symbol": sym,
                "wallet_id": wid,
                "wallet_name": wallet_names.get(wid, "?"),
            }
        traded_symbols = sorted(seen.values(), key=lambda r: (r["wallet_name"], r["symbol"]))

        # Portfolio P&L roll-up across every paper trade (powers the bold
        # money-strip at the top of the Training Center).
        #
        # "Session" scoping: when Settings → Paper Trading Reset has fired,
        # each wallet carries a bankroll_reset_at cursor + session starting
        # bankroll. The money strip then shows P&L SINCE that point, while
        # the underlying PaperTrade rows are untouched (the playbook,
        # reflections, autonomous-engine fingerprints all still see them).
        # Trades closed before the cursor are excluded from realized P&L
        # so the operator sees "new profits/losses against the new $10K"
        # without losing any history.
        all_trades = s.query(PaperTrade).all()
        # Pick the most recent reset across wallets as the session cutoff.
        # If no wallet has ever been reset, cutoff is None and the math
        # collapses to "all-time" (the legacy behavior).
        wallet_rows = s.query(Wallet).all()
        reset_cutoff = None
        for w in wallet_rows:
            if w.bankroll_reset_at is not None:
                if reset_cutoff is None or w.bankroll_reset_at > reset_cutoff:
                    reset_cutoff = w.bankroll_reset_at
        # Starting bankroll: prefer the session-starting value when a reset
        # has stamped one, else the live paper_balance (legacy behavior).
        if reset_cutoff is not None:
            starting = sum(
                float(w.session_starting_bankroll or w.paper_balance or 0.0)
                for w in wallet_rows
            )
        else:
            starting = sum((w.get("paper_balance") or 0.0) for w in wallets)
        realized = 0.0
        unrealized = 0.0
        invested_open = 0.0
        wins = 0
        losses = 0
        excluded_pre_reset = 0
        for t in all_trades:
            if t.status == "closed":
                # Trades closed before the cutoff belong to the OLD session.
                # Keep them in the DB (they feed learning) but don't count
                # them in this session's money strip.
                closed_at = t.closed_at
                if reset_cutoff is not None and closed_at is not None and closed_at < reset_cutoff:
                    excluded_pre_reset += 1
                    continue
                pnl = t.realized_pnl or 0.0
                realized += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
            else:
                # All currently-open trades count toward unrealized P&L
                # regardless of when they opened — they're live capital now.
                unrealized += (t.unrealized_pnl or 0.0)
                invested_open += (t.entry_price or 0.0) * (t.qty or 0.0)
        closed_count = wins + losses
        portfolio = {
            "starting": starting,
            "realized": realized,
            "unrealized": unrealized,
            "current": starting + realized + unrealized,
            "total_pl": realized + unrealized,
            "total_pl_pct": ((realized + unrealized) / starting * 100.0) if starting else 0.0,
            "invested_open": invested_open,
            "open_trades": sum(1 for t in all_trades if t.status != "closed"),
            "closed_trades": closed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / closed_count) if closed_count else 0.0,
            "session_reset_at": reset_cutoff.isoformat() if reset_cutoff else None,
            # Human-readable label for the money-strip subtitle. Server-side
            # so the template stays free of timezone-conversion JS. UTC since
            # the rest of the page uses UTC consistently in storage.
            "session_reset_label": (
                reset_cutoff.strftime("%b %d at %I:%M %p UTC").replace(" 0", " ")
                if reset_cutoff else None
            ),
            "excluded_pre_reset_trades": excluded_pre_reset,
            # Whether to apply session-scoped framing to subtitles.
            "is_session_scoped": reset_cutoff is not None,
        }
    return templates.TemplateResponse(request=request, name="training.html", context=_ctx(
            request,
            active="training",
            wallets=wallets,
            strategies=strategies,
            playbook=get_playbook_with_metadata(limit=100),
            readiness=readiness_score(),
            recent_decisions=recent_decisions(limit=20),
            recent_reflections=recent_reflections(limit=15),
            claude_configured=claude_is_configured(),
            traded_symbols=traded_symbols,
            portfolio=portfolio,
            risk_levels=["Conservative", "Moderate", "Aggressive", "Degenerate"],
            market_types=["Crypto", "Stocks", "Prediction Markets"],
        ),
)


@router.get("/training/chart-data")
def training_chart_data(
    symbol: str = Query(..., min_length=1),
    wallet_id: int | None = Query(None),
    granularity: int = Query(900, ge=60, le=86400),
) -> JSONResponse:
    """
    Return everything the Positions Lab chart needs in one payload:
      - candles:    OHLC bars from Coinbase (public endpoint, no key needed)
      - trades:     every paper trade for (symbol, wallet?) with entry+exit
      - decisions:  every Claude decision (BUY/SELL/HOLD/CLOSE) the bot made
      - stats:      symbol-level KPIs (P&L, win rate, hold time, best/worst)
    The frontend draws candles + entry/exit markers + a decision overlay.
    """
    from connectors.candles import get_candles
    from database.models import ClaudeDecision

    sym = (symbol or "").upper().strip()
    if not sym:
        return JSONResponse({"ok": False, "error": "missing symbol"}, status_code=400)

    candles = get_candles(sym, granularity=granularity, limit=300)

    with session_scope() as s:
        q = s.query(PaperTrade).filter(PaperTrade.symbol == sym)
        if wallet_id:
            q = q.filter(PaperTrade.wallet_id == wallet_id)
        trades = q.order_by(PaperTrade.opened_at.asc()).all()

        trade_rows: list[dict[str, Any]] = []
        for t in trades:
            trade_rows.append(
                {
                    "id": t.id,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "realized_pnl": t.realized_pnl or 0.0,
                    "unrealized_pnl": t.unrealized_pnl or 0.0,
                    "confidence": t.confidence or 0.0,
                    "status": t.status,
                    "opened_at_ts": int(t.opened_at.timestamp()) if t.opened_at else None,
                    "closed_at_ts": int(t.closed_at.timestamp()) if t.closed_at else None,
                    "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                    "notes": (t.notes or "")[:300],
                    "is_perp": bool(t.is_perp),
                    "leverage": t.leverage or 1.0,
                }
            )

        dq = s.query(ClaudeDecision).filter(ClaudeDecision.symbol == sym)
        if wallet_id:
            dq = dq.filter(ClaudeDecision.wallet_id == wallet_id)
        decisions = (
            dq.order_by(ClaudeDecision.created_at.desc()).limit(150).all()
        )
        decision_rows: list[dict[str, Any]] = []
        for d in decisions:
            decision_rows.append(
                {
                    "id": d.id,
                    "ts": int(d.created_at.timestamp()) if d.created_at else None,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "action": d.action,
                    "confidence": d.confidence or 0.0,
                    "size_multiplier": d.size_multiplier or 1.0,
                    "stop_loss_pct": d.stop_loss_pct or 0.0,
                    "take_profit_pct": d.take_profit_pct or 0.0,
                    "technical_side": d.technical_side,
                    "technical_confidence": d.technical_confidence or 0.0,
                    "price": d.price or 0.0,
                    "rationale": (d.rationale or "")[:500],
                    "source": d.source,
                }
            )
        decision_rows.reverse()  # oldest -> newest for charting

    closed = [t for t in trade_rows if t["status"] == "closed"]
    realized = sum(t["realized_pnl"] for t in closed)
    wins = [t for t in closed if t["realized_pnl"] > 0]
    losses = [t for t in closed if t["realized_pnl"] < 0]
    best = max((t["realized_pnl"] for t in closed), default=0.0)
    worst = min((t["realized_pnl"] for t in closed), default=0.0)
    avg_hold_min = 0.0
    holds = [
        (t["closed_at_ts"] - t["opened_at_ts"]) / 60.0
        for t in closed
        if t["opened_at_ts"] and t["closed_at_ts"]
    ]
    if holds:
        avg_hold_min = sum(holds) / len(holds)

    open_trades = [t for t in trade_rows if t["status"] == "open"]

    stats = {
        "total_trades": len(trade_rows),
        "open_trades": len(open_trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "realized_pnl": realized,
        "best_pnl": best,
        "worst_pnl": worst,
        "avg_hold_minutes": avg_hold_min,
        "avg_confidence": (
            sum(t["confidence"] for t in trade_rows) / len(trade_rows)
            if trade_rows
            else 0.0
        ),
    }

    return JSONResponse(
        {
            "ok": True,
            "symbol": sym,
            "granularity": granularity,
            "candles": candles,
            "trades": trade_rows,
            "decisions": decision_rows,
            "stats": stats,
        }
    )


# ============================================================
# LIVE TRAINING SESSION
# ============================================================
# When the user clicks "Start Live Session" on the Training Center, we:
#   1. Snapshot the current autonomous-bot config so we can restore later.
#   2. Force paper trading ON (dry_run = false), bot_enabled = true, and
#      drop the tick interval to a session-friendly cadence (default 15s).
#   3. Reload the scheduler so the new interval takes effect immediately.
#   4. Stamp `training_session_active = true` so the UI knows to poll the
#      live feed.
# Stopping reverses all of the above and restores the user's previous knobs.
# Polling is deliberately HTTP rather than websocket so it works behind any
# reverse proxy without additional configuration.

def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _mission_snapshot_for_feed() -> dict[str, Any]:
    """Lazy import + snapshot of the Daily Mission Controller.

    Returns a small dict every poll so the training UI can render the current
    mode, distance to target, and key thresholds without separate endpoints.
    When the controller is disabled (mission_controller_enabled = False) the
    snapshot returns `{"enabled": False, "mode": "BUILD"}` and the UI hides
    the panel.
    """
    try:
        from risk.daily_mission_controller import get_mission_controller, is_enabled
        snap = get_mission_controller().snapshot()
        snap["enabled"] = is_enabled()
        return snap
    except Exception:
        # Never let the controller break the polling endpoint.
        return {"enabled": False, "mode": "BUILD", "error": "snapshot_failed"}


@router.post("/training/session/start")
def training_session_start(
    tick_seconds: int = Form(15),
    min_confidence: float = Form(0.55),
    position_size_usd: float = Form(100.0),
    max_open_per_wallet: int = Form(5),
    universe_limit: int = Form(40),
    aggressive: str = Form("false"),
    trading_style: str = Form("hybrid"),
) -> JSONResponse:
    import logging
    import traceback
    
    try:
        from config.bot_config import get as cfg_get
        from config.bot_config import set_many as cfg_set
        from services.claude_client import is_configured as claude_is_configured
        from services.scheduler import bot_scheduler

        if _truthy(cfg_get("training_session_active")):
            # Session already running - return the current config so UI doesn't show "?"
            tick = int(float(cfg_get("bot_tick_seconds") or 15))
            min_conf = float(cfg_get("bot_min_confidence") or 0.55)
            pos_usd = float(cfg_get("bot_position_size_usd") or 100)
            max_open = int(float(cfg_get("bot_max_open_per_wallet") or 5))
            uni_limit = int(float(cfg_get("bot_universe_limit") or 40))
            current_style = "hybrid"
            try:
                with session_scope() as s:
                    w = s.query(Wallet).first()
                    if w and getattr(w, "trading_style", None):
                        current_style = w.trading_style
            except Exception:
                pass
            return JSONResponse({
                "ok": True, 
                "already_running": True,
                "tick_seconds": tick,
                "min_confidence": min_conf,
                "position_size_usd": pos_usd,
                "max_open_per_wallet": max_open,
                "universe_limit": uni_limit,
                "trading_style": current_style,
                "claude_configured": claude_is_configured(),
            })

        # Aggressive preset: drop confidence floor and shrink position size so the
        # bot fires often enough during a short training session for the user to
        # actually see executions stream in.
        if _truthy(aggressive):
            min_confidence = min(min_confidence, 0.35)
            max_open_per_wallet = max(max_open_per_wallet, 8)

        tick = max(2, min(120, int(tick_seconds if tick_seconds is not None else 15)))
        # Preserve 0.0 explicitly — `or 0.55` would silently bump the user's
        # "I want trades on every signal" floor of 0.0 up to 0.55, which is the
        # exact bug that kept Claude vetoing every borderline decision.
        min_conf = max(0.0, min(0.95, float(min_confidence if min_confidence is not None else 0.50)))
        pos_usd = max(5.0, min(100_000.0, float(position_size_usd if position_size_usd is not None else 80.0)))
        max_open = max(1, min(100, int(max_open_per_wallet if max_open_per_wallet is not None else 25)))
        # Universe floor: 10. Ceiling: 200. Default to 100 for good diversity.
        uni_limit = max(10, min(200, int(universe_limit if universe_limit is not None else 100)))

        # The bot's kill switch is the #1 reason "nothing happens" during a session.
        # Auto-release it when the user explicitly starts a training session — they
        # are saying "go trade", and a stale kill switch from a prior daily-loss
        # event would silently nullify everything else they configured.
        from trading.risk_manager import RiskManager
        if RiskManager.kill_switch_status():
            RiskManager.set_kill_switch(False, reason="training session start")
        
        # Enable training mode to bypass cooldown restrictions
        RiskManager.set_training_mode(True)

        # Snapshot the values we're about to overwrite so Stop can restore.
        prev = {
            "bot_enabled":         cfg_get("bot_enabled") or "",
            "bot_dry_run":         cfg_get("bot_dry_run") or "",
            "bot_tick_seconds":    cfg_get("bot_tick_seconds") or "",
            "bot_min_confidence":  cfg_get("bot_min_confidence") or "",
            "bot_position_size_usd": cfg_get("bot_position_size_usd") or "",
            "bot_max_open_per_wallet": cfg_get("bot_max_open_per_wallet") or "",
            "bot_universe_limit":  cfg_get("bot_universe_limit") or "",
        }

        cfg_set(
            {
                "training_session_active": "true",
                "training_session_started_at": utcnow().isoformat(),
                "training_session_tick_seconds": str(tick),
                "training_session_prev_bot_enabled": prev["bot_enabled"],
                "training_session_prev_dry_run": prev["bot_dry_run"],
                "training_session_prev_tick_seconds": prev["bot_tick_seconds"],
                "training_session_prev_min_confidence": prev["bot_min_confidence"],
                "training_session_prev_position_size_usd": prev["bot_position_size_usd"],
                "training_session_prev_max_open_per_wallet": prev["bot_max_open_per_wallet"],
                "training_session_prev_universe_limit": prev["bot_universe_limit"],
                # Force the autonomous loop ON in paper-live mode with the chosen knobs.
                "bot_enabled": "true",
                "bot_dry_run": "false",
                "bot_tick_seconds": str(tick),
                "bot_min_confidence": str(min_conf),
                "bot_position_size_usd": str(pos_usd),
                "bot_max_open_per_wallet": str(max_open),
                "bot_universe_limit": str(uni_limit),
            }
        )

        # Validate trading_style — fall back to "hybrid" silently rather than 400ing,
        # since this comes from a UI dropdown that can't really send anything else.
        style = (trading_style or "hybrid").strip().lower()
        if style not in ("scalper", "swing", "hybrid"):
            style = "hybrid"

        # CRITICAL: Also update the actual Wallet objects so the risk manager sees the new limits.
        # The risk manager checks wallet.max_open_positions, NOT the config setting.
        with session_scope() as s:
            wallets = s.query(Wallet).all()
            for w in wallets:
                # Save original values so we can restore on stop
                # Use getattr with a fallback since meta column may not exist yet
                try:
                    if not w.meta:
                        w.meta = {}
                    w.meta["_session_prev_max_open"] = w.max_open_positions
                    w.meta["_session_prev_max_position_usd"] = w.max_position_usd
                    w.meta["_session_prev_max_daily_trades"] = w.max_daily_trades
                    w.meta["_session_prev_trading_style"] = getattr(w, "trading_style", None)
                except Exception:
                    # meta column may not exist in older schema - skip the backup
                    pass
                # Apply session settings to wallet
                w.max_open_positions = max_open
                # The wallet hard cap must sit ABOVE the per-trade size, not
                # equal to it. The position sizer targets pos_usd and scales
                # up on high conviction; a cap == pos_usd then rejects nearly
                # every trade ("Notional $750 > cap $750", code
                # wallet_position_cap) on conviction boosts and float
                # rounding alone. 2x keeps a real safety rail with room for
                # the sizer to work.
                w.max_position_usd = pos_usd * 2.0
                # A training session exists to generate many trades to learn
                # from; the default 10-trades/day cap (code wallet_daily_count)
                # chokes it within the first minute at a 7s tick. 0 disables
                # the daily-count gate entirely for the duration of the
                # session (RiskManager skips the check when it is falsy).
                w.max_daily_trades = 0
                w.bot_paused = False  # Unpause all wallets for training
                # Force the trading style for the duration of the session so the
                # autonomous engine and exit loop both honor what the user picked
                # in the Session Settings card — not whatever was last saved on
                # the Active Positions / Trading Style preset.
                try:
                    w.trading_style = style
                except Exception:
                    pass
            logger.info(f"[SESSION_START] Updated {len(wallets)} wallets: max_open={max_open}, position_size=${pos_usd}, max_position_usd=${pos_usd * 2.0}, max_daily_trades=0 (unlimited), trading_style={style}")

        bot_scheduler.reload()  # pick up the new tick interval

        # Reset the circuit breaker so we start fresh
        try:
            from trading.bot_engine import bot_engine
            bot_engine.reset_circuit_breaker()
        except Exception as e:
            logger.warning("Could not reset circuit breaker: %s", e)

        # Fire one tick immediately so the user sees activity within seconds
        # instead of waiting for the next scheduler beat.
        try:
            from trading.bot_engine import bot_engine
            bot_engine.tick(manual=True)
        except Exception as e:
            logger.warning("Initial training tick failed: %s", e)

        with session_scope() as s:
            s.add(
                ActivityLog(
                    category="bot",
                    level="info",
                    message=(
                        f"Live training session started — tick={tick}s, "
                        f"min_conf={min_conf:.2f}, size=${pos_usd:.0f}, "
                        f"max_open={max_open}, universe={uni_limit}, style={style}, "
                        f"claude={'on' if claude_is_configured() else 'off (technical fallback)'}"
                    ),
                )
            )
        return JSONResponse(
            {
                "ok": True,
                "tick_seconds": tick,
                "min_confidence": min_conf,
                "position_size_usd": pos_usd,
                "max_open_per_wallet": max_open,
                "universe_limit": uni_limit,
                "trading_style": style,
                "claude_configured": claude_is_configured(),
            }
        )
    except Exception as e:
        logging.exception("[SESSION_START] Error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})


@router.post("/training/session/release-kill-switch")
def training_release_kill_switch() -> JSONResponse:
    """Allow the Training Center to release the kill switch in one click,
    instead of forcing the user to dig into Settings while a session is running."""
    from trading.risk_manager import RiskManager
    was_on = RiskManager.kill_switch_status()
    if was_on:
        RiskManager.set_kill_switch(False, reason="released from Training Center")
    return JSONResponse({"ok": True, "was_engaged": was_on})


@router.get("/training/session/config")
def training_session_config() -> JSONResponse:
    """Read the bot knobs the Training Center cares about so the page can
    rehydrate its form state on load."""
    from config.bot_config import get as cfg_get
    from services.claude_client import is_configured as claude_is_configured
    from trading.risk_manager import RiskManager
    from ai.claude_decision_engine import get_api_usage_stats

    api_stats = get_api_usage_stats()
    
    return JSONResponse(
        {
            "ok": True,
            "active": _truthy(cfg_get("training_session_active")),
            "tick_seconds": int(float(cfg_get("bot_tick_seconds") or 15)),
            "min_confidence": float(cfg_get("bot_min_confidence") or 0.55),
            "position_size_usd": float(cfg_get("bot_position_size_usd") or 100),
            "max_open_per_wallet": int(float(cfg_get("bot_max_open_per_wallet") or 5)),
            "universe_limit": int(float(cfg_get("bot_universe_limit") or 40)),
            "claude_configured": claude_is_configured(),
            # Surfaced so the Training Center can warn / auto-release before the
            # user wonders why their tick log is just "kill switch engaged" forever.
            "kill_switch_engaged": RiskManager.kill_switch_status(),
            # API usage stats for cost monitoring
            "api_usage": api_stats,
        }
    )


@router.get("/training/learning/stats")
def training_learning_stats() -> JSONResponse:
    """Get statistics from the autonomous learning engine."""
    try:
        from ai.autonomous_learning_engine import get_autonomous_engine
        engine = get_autonomous_engine()
        stats = engine.get_statistics()
        return JSONResponse({"ok": True, **stats})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/training/scorecard")
def training_scorecard() -> JSONResponse:
    """Trade-quality scorecard for the training UI.

    Aggregates today's session into three blocks the operator should see at
    a glance:

      1. DECISION SOURCES — across every ClaudeDecision row since the last
         bankroll reset, count what `source` produced each verdict. This
         answers "what's actually making my trade decisions today?" — is it
         the autonomous engine dominating, training_passthroughs, or Claude?

      2. TRADE CALIBRATION — across PaperTrade rows opened since the last
         bankroll reset, count which Phase B calibration tier
         (exact_pattern / knn_neighbors / raw_confidence) backed each
         approved trade. Tells the operator how many trades today were
         based on measured pattern data vs heuristic confidence.

      3. REFLECTION DEDUP — over the past N reflections, sum lessons_added
         vs lessons_reinforced. Proves whether the new dedup is matching
         paraphrases (reinforced rises) or still cloning everything
         (reinforced stays at 0).

    Cheap to compute (three small aggregations); safe to poll on the
    feed cadence.
    """
    from datetime import datetime, timezone
    from sqlalchemy import func

    try:
        with session_scope() as s:
            # --- session cutoff ----------------------------------------------
            # Use the most recent bankroll_reset_at across wallets as "today".
            # If no wallet has been reset, fall back to UTC start-of-day.
            wallets_db = s.query(Wallet).all()
            cutoff = None
            for w in wallets_db:
                if w.bankroll_reset_at is not None:
                    if cutoff is None or w.bankroll_reset_at > cutoff:
                        cutoff = w.bankroll_reset_at
            if cutoff is None:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # --- decision sources (block 1) ----------------------------------
            decision_rows = (
                s.query(ClaudeDecision.source, ClaudeDecision.action, func.count())
                .filter(ClaudeDecision.created_at >= cutoff)
                .group_by(ClaudeDecision.source, ClaudeDecision.action)
                .all()
            )
            decision_breakdown: dict[str, dict[str, int]] = {}
            total_decisions = 0
            for src, action, count in decision_rows:
                src_key = (src or "unknown").strip()
                action_key = (action or "HOLD").strip().upper()
                decision_breakdown.setdefault(src_key, {"HOLD": 0, "BUY": 0, "SELL": 0, "OTHER": 0})
                bucket = action_key if action_key in {"HOLD", "BUY", "SELL"} else "OTHER"
                decision_breakdown[src_key][bucket] += int(count or 0)
                total_decisions += int(count or 0)

            # --- trade calibration (block 2) ---------------------------------
            calib_rows = (
                s.query(PaperTrade.calibration_source, func.count(), func.avg(PaperTrade.calibration_sample_size))
                .filter(PaperTrade.opened_at >= cutoff)
                .group_by(PaperTrade.calibration_source)
                .all()
            )
            calib_breakdown = []
            total_trades = 0
            for src, count, avg_n in calib_rows:
                src_key = (src or "raw_confidence")
                cnt = int(count or 0)
                calib_breakdown.append({
                    "source": src_key,
                    "count": cnt,
                    "avg_sample_size": round(float(avg_n or 0.0), 1),
                })
                total_trades += cnt
            # Stable ordering so the UI renders in the same place every poll.
            _ORDER = {"exact_pattern": 0, "knn_neighbors": 1, "raw_confidence": 2}
            calib_breakdown.sort(key=lambda r: _ORDER.get(r["source"], 99))

            # --- reflection dedup (block 3) ----------------------------------
            # Sum lessons_added vs reinforced across the most recent N
            # reflections (since cutoff). We don't have explicit columns for
            # added/reinforced on TradeReflection — they're embedded in the
            # ActivityLog message "(new=N, reinforced=M)". We parse that.
            reflection_logs = (
                s.query(ActivityLog.message)
                .filter(ActivityLog.category == "ai")
                .filter(ActivityLog.created_at >= cutoff)
                .filter(ActivityLog.message.like("Reflection saved%"))
                .all()
            )
            import re
            lessons_new = 0
            lessons_reinforced = 0
            reflections_count = 0
            empty_reflections = 0
            for (msg,) in reflection_logs:
                reflections_count += 1
                m = re.search(r"new=(\d+),\s*reinforced=(\d+)", msg or "")
                if m:
                    lessons_new += int(m.group(1))
                    lessons_reinforced += int(m.group(2))
                # Detect the "lessons=0 ... — <cause>" pattern from Phase A.
                if "lessons=0" in (msg or ""):
                    empty_reflections += 1

            dedup_ratio = (
                lessons_reinforced / (lessons_new + lessons_reinforced)
                if (lessons_new + lessons_reinforced) > 0 else 0.0
            )

            # --- top patterns (block 4) — autonomous engine top fingerprints --
            top_patterns = []
            try:
                from ai.autonomous_learning_engine import get_autonomous_engine
                eng = get_autonomous_engine()
                eng._ensure_loaded()
                # _patterns is dict[fingerprint -> LearnedPattern]
                ranked = sorted(
                    eng._patterns.values(),
                    key=lambda p: p.total_trades,
                    reverse=True,
                )[:5]
                for p in ranked:
                    top_patterns.append({
                        "fingerprint": p.fingerprint,
                        "side": p.side,
                        "sample_size": int(p.total_trades),
                        "win_rate": round(float(p.win_rate or 0.0), 3),
                        "expectancy_pct": round(float(p.expectancy or 0.0) * 100, 2),
                    })
            except Exception:
                # Top-patterns is supplementary — don't block the scorecard
                # if the engine can't be queried right now.
                top_patterns = []

            return JSONResponse({
                "ok": True,
                "session_cutoff": cutoff.isoformat(),
                "decisions": {
                    "total": total_decisions,
                    "by_source": decision_breakdown,
                },
                "trades": {
                    "total": total_trades,
                    "by_calibration": calib_breakdown,
                },
                "reflections": {
                    "total": reflections_count,
                    "lessons_new": lessons_new,
                    "lessons_reinforced": lessons_reinforced,
                    "dedup_ratio": round(dedup_ratio, 3),
                    "empty": empty_reflections,
                },
                "top_patterns": top_patterns,
            })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/training/learning/symbol/{symbol}")
def training_learning_symbol(symbol: str) -> JSONResponse:
    """Get learned insights for a specific symbol."""
    try:
        from ai.autonomous_learning_engine import get_autonomous_engine
        engine = get_autonomous_engine()
        insights = engine.get_symbol_insights(symbol)
        if insights:
            return JSONResponse({"ok": True, **insights})
        return JSONResponse({"ok": False, "error": "No data for symbol"}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/training/session/stop")
def training_session_stop() -> JSONResponse:
    import logging
    import traceback
    
    try:
        from config.bot_config import get as cfg_get
        from config.bot_config import set_many as cfg_set
        from services.scheduler import bot_scheduler

        if not _truthy(cfg_get("training_session_active")):
            return JSONResponse({"ok": True, "already_stopped": True})

        prev_enabled = cfg_get("training_session_prev_bot_enabled") or "false"
        prev_dry = cfg_get("training_session_prev_dry_run") or "true"
        prev_tick = cfg_get("training_session_prev_tick_seconds") or "60"
        prev_min_conf = cfg_get("training_session_prev_min_confidence") or "0.65"
        prev_pos = cfg_get("training_session_prev_position_size_usd") or "100"
        prev_max_open = cfg_get("training_session_prev_max_open_per_wallet") or "5"
        prev_uni = cfg_get("training_session_prev_universe_limit") or "40"

        cfg_set(
            {
                "training_session_active": "false",
                "training_session_started_at": "",
                "training_session_prev_bot_enabled": "",
                "training_session_prev_dry_run": "",
                "training_session_prev_tick_seconds": "",
                "training_session_prev_min_confidence": "",
                "training_session_prev_position_size_usd": "",
                "training_session_prev_max_open_per_wallet": "",
                "training_session_prev_universe_limit": "",
                # Restore the user's prior config.
                "bot_enabled": prev_enabled,
                "bot_dry_run": prev_dry,
                "bot_tick_seconds": prev_tick,
                "bot_min_confidence": prev_min_conf,
                "bot_position_size_usd": prev_pos,
                "bot_max_open_per_wallet": prev_max_open,
                "bot_universe_limit": prev_uni,
            }
        )
        bot_scheduler.reload()

        # Disable training mode bypass
        from trading.risk_manager import RiskManager
        RiskManager.set_training_mode(False)

        # Restore original wallet settings
        with session_scope() as s:
            wallets = s.query(Wallet).all()
            for w in wallets:
                try:
                    if w.meta and "_session_prev_max_open" in w.meta:
                        w.max_open_positions = w.meta.get("_session_prev_max_open")
                        w.max_position_usd = w.meta.get("_session_prev_max_position_usd")
                        # Clean up the temporary keys
                        w.meta.pop("_session_prev_max_open", None)
                        w.meta.pop("_session_prev_max_position_usd", None)
                    if w.meta and "_session_prev_max_daily_trades" in w.meta:
                        w.max_daily_trades = w.meta.get("_session_prev_max_daily_trades")
                        w.meta.pop("_session_prev_max_daily_trades", None)
                    if w.meta and "_session_prev_trading_style" in w.meta:
                        prev_style = w.meta.get("_session_prev_trading_style")
                        if prev_style:
                            try:
                                w.trading_style = prev_style
                            except Exception:
                                pass
                        w.meta.pop("_session_prev_trading_style", None)
                except Exception:
                    # meta column may not exist in older schema - skip the restore
                    pass
            logger.info(f"[SESSION_STOP] Restored wallet settings for {len(wallets)} wallets")

        with session_scope() as s:
            s.add(
                ActivityLog(
                    category="bot",
                    level="info",
                    message="Live training session stopped — bot config restored.",
                )
            )
        return JSONResponse({"ok": True})
    except Exception as e:
        logging.exception("[SESSION_STOP] Error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})


@router.post("/training/session/tick")
def training_session_tick_now() -> JSONResponse:
    """Force a tick immediately so the user doesn't have to wait. Returns the result."""
    import logging
    import traceback
    
    try:
        from trading.bot_engine import bot_engine

        res = bot_engine.tick(manual=True)
        return JSONResponse(
            {
                "ok": True,
                "result": {
                    "started_at": res.started_at,
                    "ended_at": res.ended_at,
                    "universe_size": res.universe_size,
                    "wallets_evaluated": res.wallets_evaluated,
                    "decisions": res.decisions,
                    "actions": res.actions,
                    "skipped": res.skipped,
                    "errors": res.errors,
                    "notes": res.notes,
                },
            }
        )
    except Exception as e:
        logging.exception("[SESSION_TICK] Error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})


@router.get("/training/learning-stats")
def api_get_learning_stats() -> JSONResponse:
    """
    Get comprehensive statistics about the adaptive learning engine.
    Shows pattern recognition, strategy performance, and learning progress.
    """
    try:
        from ai.adaptive_learning_engine import get_adaptive_engine
        
        engine = get_adaptive_engine()
        stats = engine.get_learning_stats()
        
        return JSONResponse({
            "ok": True,
            "learning_stats": stats,
        })
    except Exception as e:
        logging.exception("[LEARNING_STATS] Error")
        return JSONResponse({"ok": False, "error": str(e)})


@router.post("/training/session/apply-settings-to-wallets")
def api_apply_session_settings_to_wallets() -> JSONResponse:
    """
    Force-apply the current session settings to all wallets.
    This fixes the issue where wallet.max_open_positions is out of sync with session settings.
    """
    import logging
    from config.bot_config import BotConfig
    
    try:
        cfg = BotConfig.load()
        max_open = cfg.max_open_per_wallet
        pos_usd = cfg.position_size_usd
        
        with session_scope() as s:
            wallets = s.query(Wallet).all()
            updated = []
            for w in wallets:
                old_max = w.max_open_positions
                old_pos = w.max_position_usd
                w.max_open_positions = max_open
                w.max_position_usd = pos_usd
                w.bot_paused = False
                updated.append({
                    "id": w.id,
                    "name": w.name,
                    "old_max_open": old_max,
                    "new_max_open": max_open,
                    "old_max_position_usd": old_pos,
                    "new_max_position_usd": pos_usd,
                })
            
            logging.info(f"[APPLY_SETTINGS] Updated {len(wallets)} wallets: max_open={max_open}, pos_usd={pos_usd}")
        
        return JSONResponse({
            "ok": True,
            "message": f"Updated {len(updated)} wallets",
            "config": {"max_open_per_wallet": max_open, "position_size_usd": pos_usd},
            "wallets": updated,
        })
    except Exception as e:
        import traceback
        logging.exception("[APPLY_SETTINGS] Error")
        return JSONResponse({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


@router.get("/training/session/feed")
def training_session_feed(
    since_decision_id: int = Query(0, ge=0),
    since_log_id: int = Query(0, ge=0),
    since_trade_id: int = Query(0, ge=0),
) -> JSONResponse:
    """
    Polled every 3s by the Training Center while a live session is running.
    
    Returns:
    - session:   active flag, started_at, tick_seconds, next_tick (from scheduler)
    - portfolio: live mark-to-market P&L across every wallet
    - decisions: new ClaudeDecision rows (BUY / SELL / HOLD / CLOSE) with rationale
    - fills:     newly opened or closed paper trades
    - logs:      bot/trade/system activity logs (the streaming console)
    - ticks:     last 5 tick summaries for the "what just happened" panel
    """
    import logging
    
    from config.bot_config import get as cfg_get
    from connectors.live_prices import get_prices_batch, get_price
    from database.models import ClaudeDecision
    from services.scheduler import bot_scheduler
    from trading.bot_engine import bot_engine
    
    sched = bot_scheduler.status()
    session_active = _truthy(cfg_get("training_session_active"))
    started_at = cfg_get("training_session_started_at") or None
    
    logging.info(f"[SESSION_FEED] session_active={session_active}, started_at={started_at}, scheduler={sched}")
    
    with session_scope() as s:
        # ---- Live portfolio mark-to-market ----
        # Query wallets directly from database to ensure we get fresh data
        wallets_db = s.query(Wallet).all()
        # Session-scoped money strip. See training_page() for the full
        # rationale: when a Paper Trading Reset has been performed, P&L is
        # computed only from trades closed AFTER the reset cutoff so the
        # operator sees "since the new $10K" instead of all-time numbers.
        # Trade history is never deleted — this is purely a display filter.
        reset_cutoff = None
        for w in wallets_db:
            if w.bankroll_reset_at is not None:
                if reset_cutoff is None or w.bankroll_reset_at > reset_cutoff:
                    reset_cutoff = w.bankroll_reset_at
        if reset_cutoff is not None:
            starting = sum(
                float(w.session_starting_bankroll or w.paper_balance or 0.0)
                for w in wallets_db
            )
        else:
            starting = sum(float(w.paper_balance or 0) for w in wallets_db)
        wallet_count = len(wallets_db)

        logging.info(
            f"[SESSION_FEED] Wallets: {wallet_count} found, starting=${starting}, "
            f"session_reset_at={reset_cutoff.isoformat() if reset_cutoff else 'never'}"
        )
        for w in wallets_db:
            logging.info(f"[SESSION_FEED]   - {w.name}: paper_balance=${w.paper_balance}")

        # If no starting balance, use a default seed amount for display
        if starting == 0:
            starting = 10000.0  # Default seed for display purposes
            logging.warning(f"[SESSION_FEED] No wallet balances found, using default seed of ${starting}")

        all_trades = s.query(PaperTrade).all()
        open_trades_list = [t for t in all_trades if t.status == "open"]
        closed_trades_list = [t for t in all_trades if t.status == "closed"]
        
        logging.info(f"[SESSION_FEED] Found {len(all_trades)} total trades: {len(open_trades_list)} open, {len(closed_trades_list)} closed")
        
        realized = 0.0
        unrealized = 0.0
        invested_open = 0.0
        wins = 0
        losses = 0
        
        # Collect all open position symbols for batch price fetch
        open_symbols = [t.symbol for t in open_trades_list]
        symbol_prices = get_prices_batch(list(set(open_symbols))) if open_symbols else {}
        
        logging.info(f"[SESSION_FEED] Fetched prices for {len(symbol_prices)} symbols")
        
        excluded_pre_reset = 0
        for t in all_trades:
            if t.status == "closed":
                # Trades closed BEFORE the bankroll reset don't count toward
                # session P&L (but stay in the DB — the autonomous engine,
                # playbook, and reflections all still see them).
                closed_at = t.closed_at
                if reset_cutoff is not None and closed_at is not None and closed_at < reset_cutoff:
                    excluded_pre_reset += 1
                    continue
                pnl = float(t.realized_pnl or 0)
                realized += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                continue
            entry = float(t.entry_price or 0)
            qty = float(t.qty or 0)
            invested_open += entry * qty
            mark = symbol_prices.get(t.symbol, 0.0)
            if mark > 0 and entry > 0:
                if (t.side or "").upper() == "BUY":
                    unrealized += (mark - entry) * qty
                else:  # SELL / SHORT
                    unrealized += (entry - mark) * qty

        closed_count = wins + losses
        open_count = len(open_trades_list)
        total_pl = realized + unrealized
        
        logging.info(f"[SESSION_FEED] Portfolio: starting=${starting}, realized=${realized:.2f}, unrealized=${unrealized:.2f}, open={open_count}, current=${starting + total_pl:.2f}")
        
        portfolio = {
            "starting": round(starting, 2),
            "realized": round(realized, 2),
            "unrealized": round(unrealized, 2),
            "current": round(starting + total_pl, 2),
            "total_pl": round(total_pl, 2),
            "total_pl_pct": round((total_pl / starting * 100.0) if starting else 0.0, 3),
            "invested_open": round(invested_open, 2),
            "open_trades": open_count,
            "closed_trades": closed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / closed_count) if closed_count else 0.0, 4),
            # When a bankroll reset has fired, these tell the UI we're
            # showing P&L "since X" instead of all-time. UI can render a
            # subtle banner like: "Showing 12 trades since reset at HH:MM.
            # Full history preserved in Recent Decisions / Playbook."
            "session_reset_at": reset_cutoff.isoformat() if reset_cutoff else None,
            "excluded_pre_reset_trades": excluded_pre_reset,
        }

        # ---- Compute the session-start cutoff ONCE, used to scope every
        # ---- live feed (logs, decisions, fills) so the console and the
        # ---- decision/fill panels never preload stale data from prior runs.
        from datetime import datetime as _dt, timezone as _tz
        session_start_naive = None
        if started_at:
            try:
                _parsed = _dt.fromisoformat(str(started_at).replace("Z", "+00:00"))
                if _parsed.tzinfo is not None:
                    _parsed = _parsed.astimezone(_tz.utc).replace(tzinfo=None)
                session_start_naive = _parsed
            except Exception:
                session_start_naive = None

        # ---- New Claude decisions since the client's last cursor ----
        decisions_q = s.query(ClaudeDecision).filter(ClaudeDecision.id > since_decision_id)
        if session_start_naive is not None:
            decisions_q = decisions_q.filter(ClaudeDecision.created_at >= session_start_naive)
        new_decisions = decisions_q.order_by(ClaudeDecision.id.asc()).limit(50).all()
        decisions_payload = [
            {
                "id": d.id,
                # created_at is naive UTC — tag it as UTC before .timestamp()
                # so the browser receives a real epoch and renders local time.
                "ts": (
                    int(d.created_at.replace(tzinfo=_tz.utc).timestamp())
                    if d.created_at else None
                ),
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "wallet_id": d.wallet_id,
                "symbol": d.symbol,
                "action": d.action,
                "confidence": float(d.confidence or 0),
                "size_multiplier": float(d.size_multiplier or 1),
                "stop_loss_pct": float(d.stop_loss_pct or 0),
                "take_profit_pct": float(d.take_profit_pct or 0),
                "technical_side": d.technical_side,
                "technical_confidence": float(d.technical_confidence or 0),
                "price": float(d.price or 0),
                "rationale": (d.rationale or "")[:400],
                "source": d.source,
            }
            for d in new_decisions
        ]

        # ---- Newly opened or closed paper trades ----
        # Use a max(opened_id, closed_id) cursor so the client gets both events.
        trades_q = s.query(PaperTrade).filter(PaperTrade.id > since_trade_id)
        if session_start_naive is not None:
            # A trade opened before the session began but still open during the
            # session is *not* part of this session's activity feed — skip it.
            trades_q = trades_q.filter(PaperTrade.opened_at >= session_start_naive)
        new_trades = trades_q.order_by(PaperTrade.id.asc()).limit(40).all()
        fills_payload = []
        for t in new_trades:
            mark = symbol_prices.get(t.symbol)
            if mark is None:
                p = get_price(t.symbol)
                mark = float(p.get("price") or 0) if p.get("ok") else 0.0
            fills_payload.append(
                {
                    "id": t.id,
                    "wallet_id": t.wallet_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": float(t.qty or 0),
                    "entry_price": float(t.entry_price or 0),
                    "exit_price": float(t.exit_price or 0) if t.exit_price else None,
                    "mark_price": mark,
                    "status": t.status,
                    "realized_pnl": float(t.realized_pnl or 0),
                    "unrealized_pnl": float(t.unrealized_pnl or 0),
                    "confidence": float(t.confidence or 0),
                    "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                    "notes": (t.notes or "")[:200],
                }
            )

        # ---- Activity logs (the live console) ----
        # Reuse the session_start_naive cutoff computed above so the console
        # never preloads pre-session log entries (e.g. old "kill switch
        # engaged" ticks from a previous day).
        logs_q = s.query(ActivityLog).filter(ActivityLog.id > since_log_id)
        if session_start_naive is not None:
            logs_q = logs_q.filter(ActivityLog.created_at >= session_start_naive)
        new_logs = logs_q.order_by(ActivityLog.id.asc()).limit(80).all()
        logs_payload = [
            {
                "id": l.id,
                # created_at is naive UTC in the DB; force-tag it as UTC so the
                # epoch we send to the browser is correct and JS can format it
                # in the user's local time zone.
                "ts": (
                    int(l.created_at.replace(tzinfo=_tz.utc).timestamp())
                    if l.created_at else None
                ),
                "category": l.category,
                "level": l.level,
                "wallet_id": l.wallet_id,
                "message": l.message,
            }
            for l in new_logs
        ]

    ticks_payload = [
        {
            "started_at": t.started_at,
            "ended_at": t.ended_at,
            "decisions": t.decisions,
            "actions": t.actions,
            "skipped": t.skipped,
            "errors": t.errors,
            "universe_size": t.universe_size,
            "notes": t.notes[:5],
        }
        for t in bot_engine.recent_ticks(limit=8)
    ]

    return JSONResponse(
        {
            "ok": True,
            "session": {
                "active": session_active,
                "started_at": started_at,
                "tick_seconds": sched.get("tick_seconds"),
                "next_tick": sched.get("next_tick"),
                "scheduler_running": sched.get("scheduler_running"),
                "dry_run": sched.get("dry_run"),
                "bot_enabled": sched.get("bot_enabled"),
            },
            "portfolio": portfolio,
            "mission": _mission_snapshot_for_feed(),
            "decisions": decisions_payload,
            "fills": fills_payload,
            "logs": logs_payload,
            "ticks": ticks_payload,
            # Cursors the client should send back next poll.
            "cursors": {
                "decision_id": (decisions_payload[-1]["id"] if decisions_payload else since_decision_id),
                "log_id": (logs_payload[-1]["id"] if logs_payload else since_log_id),
                "trade_id": (fills_payload[-1]["id"] if fills_payload else since_trade_id),
            },
        }
    )


@router.post("/training/run", response_class=HTMLResponse)
def training_run(
    request: Request,
    wallet_id: int | None = Form(None),
    strategy_id: int | None = Form(None),
    market_type: str = Form("Crypto"),
    risk_level: str = Form("Moderate"),
    num_trades: int = Form(50),
    starting_balance: float = Form(10000.0),
) -> HTMLResponse:
    result = _ai.run_training_session(
        wallet_id=wallet_id if wallet_id else None,
        strategy_id=strategy_id if strategy_id else None,
        market_type=market_type,
        risk_level=risk_level,
        num_trades=num_trades,
        starting_balance=starting_balance,
    )
    eq: list[float] = [result.starting_balance]
    for d in result.decisions:
        eq.append(d.get("balance", eq[-1]))
    return templates.TemplateResponse(
        request=request,
        name="_training_result.html",
        context={"request": request, "result": result, "equity": eq},
    )


@router.post("/training/memory/reset")
def training_memory_reset() -> RedirectResponse:
    from ai.claude_learning import reset_playbook
    reset_playbook()
    _memory.reset()
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/consolidate")
def training_memory_consolidate() -> RedirectResponse:
    from ai.claude_learning import consolidate_lessons
    consolidate_lessons()
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/compact")
def training_memory_compact() -> RedirectResponse:
    """Offline merge of duplicate playbook entries (no Claude required)."""
    from ai.claude_learning import compact_playbook_offline
    compact_playbook_offline()
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/meta-analyze")
def training_memory_meta_analyze() -> RedirectResponse:
    """
    Perform deep meta-analysis across recent trades to discover patterns,
    recurring mistakes, winning setups, and higher-order insights.
    """
    from ai.claude_learning import analyze_trade_patterns
    result = analyze_trade_patterns(lookback_trades=50)
    # Result is logged in the function, just redirect back
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/discover-edges")
def training_memory_discover_edges() -> RedirectResponse:
    """
    Run advanced learning analysis: cluster mistakes and discover
    profitable edges from trade history. Does NOT require Claude.
    """
    from ai.adaptive_learning_engine import run_advanced_learning_analysis
    result = run_advanced_learning_analysis()
    # Result is logged in the function, just redirect back
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/delete/{rule_id}")
def training_memory_delete(rule_id: int) -> RedirectResponse:
    from database.models import AILearningMemory
    with session_scope() as s:
        row = s.get(AILearningMemory, rule_id)
        if row:
            s.delete(row)
    return RedirectResponse(url="/training", status_code=303)


@router.post("/training/memory/add")
def training_memory_add(
    category: str = Form("rule"),
    content: str = Form(...),
    weight: float = Form(1.5),
) -> RedirectResponse:
    """Manually pin a rule to the playbook so Claude obeys it on every decision."""
    from database.models import AILearningMemory
    content = (content or "").strip()
    if not content:
        return RedirectResponse(url="/training", status_code=303)
    with session_scope() as s:
        s.add(AILearningMemory(
            category=(category or "rule").strip()[:60] or "rule",
            content=content[:2000],
            weight=max(0.05, min(float(weight or 1.5), 5.0)),
        ))
    return RedirectResponse(url="/training", status_code=303)


# ----------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------

@router.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request) -> HTMLResponse:
    """Comprehensive analytics page with all trades and positions."""
    import traceback
    from connectors.live_prices import get_prices_batch
    
    try:
        summary = portfolio_summary()
        perf = performance_metrics()

        eq = equity_curve_df()
        equity_points: list[dict[str, Any]] = []
        if not eq.empty:
            for _, row in eq.iterrows():
                equity_points.append({"date": str(row["date"]), "equity": float(row["equity"])})

        by_wallet = pnl_by_wallet()
        wallet_pnl = []
        if not by_wallet.empty:
            wallet_pnl = [{"label": r["wallet"], "value": float(r["pnl"])} for _, r in by_wallet.iterrows()]

        by_strat = pnl_by_strategy()
        strategy_pnl = []
        if not by_strat.empty:
            strategy_pnl = [{"label": r["strategy"], "value": float(r["pnl"])} for _, r in by_strat.iterrows()]

        # Get all trades from database
        df = get_all_trades_df()
        
        # Prepare open positions with current prices
        open_positions = []
        closed_trades = []
        symbols = set()
        
        if not df.empty:
            # Get unique symbols for batch price fetch
            open_df = df[df["status"] == "open"]
            open_symbols = open_df["symbol"].unique().tolist() if not open_df.empty else []
            price_map = get_prices_batch(open_symbols) if open_symbols else {}
            
            # Process open positions
            for _, row in open_df.iterrows():
                symbol = row["symbol"]
                symbols.add(symbol)
                entry = float(row["entry_price"] or 0)
                qty = float(row["qty"] or 0)
                side = (row["side"] or "BUY").upper()
                current = price_map.get(symbol, entry)
                
                if side == "BUY":
                    unrealized = (current - entry) * qty
                    pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                else:
                    unrealized = (entry - current) * qty
                    pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
                
                opened_at = row["opened_at"]
                duration_hours = 0
                if opened_at:
                    from utils.helpers import utcnow
                    delta = utcnow().replace(tzinfo=None) - opened_at.replace(tzinfo=None)
                    duration_hours = delta.total_seconds() / 3600
                
                open_positions.append({
                    "id": row["id"],
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "entry_price": entry,
                    "current_price": current,
                    "unrealized_pnl": round(unrealized, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "opened_at": opened_at,
                    "duration_hours": round(duration_hours, 1),
                })
            
            # Process closed trades
            closed_df = df[df["status"] == "closed"].sort_values("closed_at", ascending=False)
            for _, row in closed_df.iterrows():
                symbol = row["symbol"]
                symbols.add(symbol)
                entry = float(row["entry_price"] or 0)
                exit_price = float(row["exit_price"]) if row["exit_price"] else None
                qty = float(row["qty"] or 0)
                realized = float(row["realized_pnl"] or 0)
                
                pnl_pct = 0
                if entry > 0 and exit_price:
                    side = (row["side"] or "BUY").upper()
                    if side == "BUY":
                        pnl_pct = ((exit_price - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - exit_price) / entry) * 100
                
                opened_at = row["opened_at"]
                closed_at = row["closed_at"]
                duration_hours = 0
                if opened_at and closed_at:
                    delta = closed_at.replace(tzinfo=None) - opened_at.replace(tzinfo=None)
                    duration_hours = delta.total_seconds() / 3600
                
                closed_trades.append({
                    "id": row["id"],
                    "symbol": symbol,
                    "side": (row["side"] or "BUY").upper(),
                    "qty": qty,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "realized_pnl": round(realized, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "opened_at": opened_at,
                    "closed_at": closed_at,
                    "duration_hours": round(duration_hours, 1),
                    "exit_reason": getattr(row, "exit_reason", None) or row.get("exit_reason"),
                })

        # Win/loss histogram
        histogram: list[int] = [0] * 10
        if not df.empty:
            closed = df[df["status"] == "closed"]
            if not closed.empty:
                pnls = closed["realized_pnl"].astype(float).tolist()
                if pnls:
                    lo, hi = min(pnls), max(pnls)
                    rng = hi - lo if hi > lo else 1.0
                    for v in pnls:
                        bucket = min(9, max(0, int((v - lo) / rng * 10)))
                        histogram[bucket] += 1

        return templates.TemplateResponse(request=request, name="analytics.html", context=_ctx(
                request,
                active="analytics",
                summary=summary,
                perf=perf,
                equity_points=equity_points,
                wallet_pnl=wallet_pnl,
                strategy_pnl=strategy_pnl,
                histogram=histogram,
                open_positions=open_positions,
                closed_trades=closed_trades,
                open_count=len(open_positions),
                closed_count=len(closed_trades),
                total_trades=len(open_positions) + len(closed_trades),
                symbols=sorted(symbols),
            ),
        )
    except Exception as e:
        import logging
        logging.exception("[ANALYTICS] Error loading page")
        # Return error page instead of crashing
        return templates.TemplateResponse(request=request, name="analytics.html", context=_ctx(
            request,
            active="analytics",
            summary={"total_pnl": 0, "unrealized_pnl": 0, "daily_pnl": 0, "weekly_pnl": 0, "monthly_pnl": 0, "ytd_pnl": 0, "win_rate": 0},
            perf={"profit_factor": 0, "sharpe_placeholder": 0, "max_drawdown": 0, "avg_rr": 0, "max_consecutive_wins": 0, "max_consecutive_losses": 0, "biggest_win": 0, "biggest_loss": 0, "avg_trade_duration_hours": 0, "avg_win": 0, "avg_loss": 0, "win_rate": 0},
            equity_points=[],
            wallet_pnl=[],
            strategy_pnl=[],
            histogram=[0]*10,
            open_positions=[],
            closed_trades=[],
            open_count=0,
            closed_count=0,
            total_trades=0,
            symbols=[],
            error=str(e),
        ))


# ----------------------------------------------------------------------
# Activity
# ----------------------------------------------------------------------

@router.get("/activity", response_class=HTMLResponse)
def activity_page(
    request: Request,
    category: str = "",
    level: str = "",
    mode: str = "",
    wallet_id: int | None = None,
) -> HTMLResponse:
    wallets = get_wallets()
    wallet_lookup = {w["id"]: w for w in wallets}
    wallet_modes = {wid: (w.get("trading_mode") or "paper").lower() for wid, w in wallet_lookup.items()}

    # Resolve which wallet ids the mode filter should hit.
    mode_norm = (mode or "").lower()
    mode_wids: set[int] | None
    if mode_norm == "paper":
        mode_wids = {wid for wid, m in wallet_modes.items() if m == "paper"}
    elif mode_norm == "live":
        mode_wids = {wid for wid, m in wallet_modes.items() if m in {"live", "live_shadow"}}
    else:
        mode_wids = None

    # Pull recent paper trades to merge into the timeline so users see every
    # buy/sell/close inline with the bot's narration on a single page.
    with session_scope() as s:
        log_q = s.query(ActivityLog)
        if category:
            log_q = log_q.filter(ActivityLog.category == category)
        if level:
            log_q = log_q.filter(ActivityLog.level == level)
        if wallet_id:
            log_q = log_q.filter(ActivityLog.wallet_id == wallet_id)
        elif mode_wids is not None:
            # Keep system events with no wallet_id so scheduler/heartbeat lines stay visible.
            log_q = log_q.filter(
                (ActivityLog.wallet_id == None)  # noqa: E711
                | (ActivityLog.wallet_id.in_(mode_wids if mode_wids else {-1}))
            )
        rows = log_q.order_by(ActivityLog.created_at.desc()).limit(300).all()
        logs = [
            {
                "id": r.id,
                "category": r.category,
                "level": r.level,
                "message": r.message,
                "created_at": r.created_at,
                "wallet_id": r.wallet_id,
                "wallet_name": (wallet_lookup.get(r.wallet_id) or {}).get("name") if r.wallet_id else None,
                "trading_mode": wallet_modes.get(r.wallet_id, "") if r.wallet_id else "system",
            }
            for r in rows
        ]
        categories = sorted({r[0] for r in s.query(ActivityLog.category).distinct().all() if r[0]})
        levels = sorted({r[0] for r in s.query(ActivityLog.level).distinct().all() if r[0]})

        # Trade executions to render in the "Executions" tab.
        ex_q = s.query(PaperTrade)
        if wallet_id:
            ex_q = ex_q.filter(PaperTrade.wallet_id == wallet_id)
        elif mode_wids is not None:
            ex_q = ex_q.filter(PaperTrade.wallet_id.in_(mode_wids if mode_wids else {-1}))
        trades_recent = (
            ex_q.order_by(PaperTrade.id.desc()).limit(150).all()
        )
        executions = [
            {
                "id": t.id,
                "wallet": (wallet_lookup.get(t.wallet_id) or {}).get("name", "?"),
                "wallet_id": t.wallet_id,
                "trading_mode": wallet_modes.get(t.wallet_id, "paper"),
                "symbol": t.symbol,
                "side": t.side,
                "qty": float(t.qty or 0),
                "entry_price": float(t.entry_price or 0),
                "exit_price": float(t.exit_price or 0) if t.exit_price else None,
                "status": t.status,
                "realized_pnl": float(t.realized_pnl or 0),
                "unrealized_pnl": float(t.unrealized_pnl or 0),
                "confidence": float(t.confidence or 0),
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
                "notes": (t.notes or "")[:240],
            }
            for t in trades_recent
        ]

    return templates.TemplateResponse(request=request, name="activity.html", context=_ctx(
            request,
            active="activity",
            logs=logs,
            executions=executions,
            wallets=wallets,
            categories=categories,
            levels=levels,
            current_category=category,
            current_level=level,
            current_mode=mode_norm or "all",
            current_wallet_id=wallet_id,
        ),
)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

def _get_setting(key: str, default: str = "") -> str:
    with session_scope() as s:
        row = s.query(AppSetting).filter(AppSetting.key == key).first()
        return row.value if row else default


def _set_setting(key: str, value: str) -> None:
    with session_scope() as s:
        row = s.query(AppSetting).filter(AppSetting.key == key).first()
        if row:
            row.value = value
            row.updated_at = utcnow()
        else:
            s.add(AppSetting(key=key, value=value))


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    prefs = {
        "theme": _get_setting("theme", "dark"),
        "default_risk": _get_setting("default_risk", "Moderate"),
        "default_market": _get_setting("default_market", "Crypto"),
        "max_concurrent_trades": _get_setting("max_concurrent_trades", "5"),
        "default_position_size": _get_setting("default_position_size", "1000"),
    }
    bot_cfg = BotConfig.load()
    bot_status = bot_scheduler.status()
    recent_ticks = bot_engine.recent_ticks(limit=10)
    recent_recons = reconciler.recent(limit=10)
    kill_switch = RiskManager.kill_switch_status()
    notifier_cfg = {
        "provider": bot_config.get("notifier_provider") or "none",
        "tg_token": bot_config.get("notifier_telegram_bot_token") or "",
        "tg_chat": bot_config.get("notifier_telegram_chat_id") or "",
        "discord_url": bot_config.get("notifier_discord_webhook_url") or "",
        "min_level": bot_config.get("notifier_min_level") or "info",
        "daily": (bot_config.get("notifier_daily_summary") or "true").lower() in {"1", "true", "yes", "on"},
        "daily_hour": bot_config.get("notifier_daily_summary_hour_utc") or "23",
    }
    _claude_key = bot_config.get("anthropic_api_key") or ""
    claude_cfg = {
        "configured": bool(_claude_key),
        "key_masked": (_claude_key[:7] + "…" + _claude_key[-4:]) if len(_claude_key) > 12 else ("set" if _claude_key else ""),
        "model": bot_config.get("anthropic_model") or "claude-sonnet-4-6",
    }
    with session_scope() as s:
        paused_wallets = [
            {"id": w.id, "name": w.name}
            for w in s.query(Wallet).filter(Wallet.bot_paused.is_(True)).all()
        ]
    mission_enabled = _truthy(bot_config.get("mission_controller_enabled"))
    from trading.holding_profiles import SELECTABLE_MODES
    return templates.TemplateResponse(request=request, name="settings.html", context=_ctx(
        request,
        active="settings",
        prefs=prefs,
        bot_cfg=bot_cfg,
        holding_modes=SELECTABLE_MODES,
        bot_status=bot_status,
        recent_ticks=recent_ticks,
        recent_recons=recent_recons,
        notifier_cfg=notifier_cfg,
        claude_cfg=claude_cfg,
        kill_switch=kill_switch,
        paused_wallets=paused_wallets,
        mission_enabled=mission_enabled,
        settings=settings,
    ),
    )


@router.post("/settings/mission/save")
def settings_mission_save(
    mission_controller_enabled: str = Form("false"),
) -> RedirectResponse:
    """Toggle the Daily Mission Controller's enforce flag.

    Off (default) -> get_mission_controller() returns a no-op stand-in that
    approves every trade. On -> the real controller becomes the boss layer
    (confidence floor, edge gate, position sizing, Claude routing, throttles).
    """
    enabled = "true" if str(mission_controller_enabled).lower() in {"on", "true", "1", "yes"} else "false"
    bot_config.set_many({"mission_controller_enabled": enabled})
    with session_scope() as s:
        s.add(ActivityLog(
            category="settings",
            level="info",
            message=f"Daily Mission Controller {'ENABLED' if enabled == 'true' else 'DISABLED'} (enforce flag toggled).",
        ))
    return RedirectResponse(url="/settings#mission", status_code=303)


@router.post("/settings/save")
def settings_save(
    theme: str = Form("dark"),
    default_risk: str = Form("Moderate"),
    default_market: str = Form("Crypto"),
    max_concurrent_trades: str = Form("5"),
    default_position_size: str = Form("1000"),
) -> RedirectResponse:
    _set_setting("theme", theme)
    _set_setting("default_risk", default_risk)
    _set_setting("default_market", default_market)
    _set_setting("max_concurrent_trades", max_concurrent_trades)
    _set_setting("default_position_size", default_position_size)
    with session_scope() as s:
        s.add(
            ActivityLog(
                category="settings",
                level="info",
                message="Preferences updated.",
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/bot/save")
def settings_bot_save(
    bot_enabled: str = Form("false"),
    bot_tick_seconds: str = Form("60"),
    bot_universe: str = Form("coinbase_usd"),
    bot_universe_limit: str = Form("30"),
    bot_min_confidence: str = Form("0.65"),
    bot_default_strategy_type: str = Form("Momentum"),
    bot_position_size_usd: str = Form("100"),
    bot_max_open_per_wallet: str = Form("5"),
    bot_dry_run: str = Form("true"),
    bot_holding_mode: str = Form("mixed"),
    bot_long_only: str = Form("true"),
) -> RedirectResponse:
    """
    Persist autonomous-bot settings and reload the scheduler so the new
    interval / config takes effect immediately — no restart required.
    """
    # Checkboxes only post their value when checked. Normalize.
    enabled = "true" if str(bot_enabled).lower() in {"on", "true", "1", "yes"} else "false"
    dry = "true" if str(bot_dry_run).lower() in {"on", "true", "1", "yes"} else "false"
    long_only = "true" if str(bot_long_only).lower() in {"on", "true", "1", "yes"} else "false"

    # Validate the holding mode against the known set; fall back to default.
    from trading.holding_profiles import VALID_MODES, DEFAULT_MODE
    holding_mode = (bot_holding_mode or "").strip().lower()
    if holding_mode not in VALID_MODES:
        holding_mode = DEFAULT_MODE

    bot_config.set_many(
        {
            "bot_enabled": enabled,
            "bot_tick_seconds": str(max(5, int(float(bot_tick_seconds or 60)))),
            "bot_universe": bot_universe or "coinbase_usd",
            "bot_universe_limit": str(max(1, int(float(bot_universe_limit or 30)))),
            "bot_min_confidence": str(max(0.0, min(1.0, float(bot_min_confidence or 0.65)))),
            "bot_default_strategy_type": bot_default_strategy_type or "Momentum",
            "bot_position_size_usd": str(max(1.0, float(bot_position_size_usd or 100))),
            "bot_max_open_per_wallet": str(max(1, int(float(bot_max_open_per_wallet or 5)))),
            "bot_dry_run": dry,
            "bot_holding_mode": holding_mode,
            "bot_long_only": long_only,
        }
    )

    # Apply the new tick interval to the running scheduler immediately.
    bot_scheduler.reload()

    with session_scope() as s:
        s.add(
            ActivityLog(
                category="settings",
                level="info",
                message=(
                    f"Bot settings updated: enabled={enabled}, "
                    f"tick={bot_tick_seconds}s, universe={bot_universe}, dry_run={dry}"
                ),
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/bot/tick-now")
def settings_bot_tick_now() -> RedirectResponse:
    """Run a single tick immediately (manual override, ignores bot_enabled)."""
    bot_engine.tick(manual=True)
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/bot/reconcile-now")
def settings_bot_reconcile_now() -> RedirectResponse:
    """Run a single reconciler pass on demand."""
    reconciler.reconcile()
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/notifier/save")
def settings_notifier_save(
    notifier_provider: str = Form("none"),
    notifier_telegram_bot_token: str = Form(""),
    notifier_telegram_chat_id: str = Form(""),
    notifier_discord_webhook_url: str = Form(""),
    notifier_min_level: str = Form("info"),
    notifier_daily_summary: str = Form("false"),
    notifier_daily_summary_hour_utc: str = Form("23"),
) -> RedirectResponse:
    bot_config.set_many(
        {
            "notifier_provider": notifier_provider,
            "notifier_telegram_bot_token": notifier_telegram_bot_token.strip(),
            "notifier_telegram_chat_id": notifier_telegram_chat_id.strip(),
            "notifier_discord_webhook_url": notifier_discord_webhook_url.strip(),
            "notifier_min_level": notifier_min_level,
            "notifier_daily_summary": "true" if notifier_daily_summary in {"true", "on", "1"} else "false",
            "notifier_daily_summary_hour_utc": notifier_daily_summary_hour_utc.strip() or "23",
        }
    )
    with session_scope() as s:
        s.add(
            ActivityLog(
                category="notifier",
                level="info",
                message=f"Notifier settings updated (provider={notifier_provider}).",
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/notifier/test")
def settings_notifier_test() -> RedirectResponse:
    from services.notifier import send_test
    res = send_test()
    with session_scope() as s:
        s.add(
            ActivityLog(
                category="notifier",
                level="info" if res.get("ok") else "warn",
                message=f"Notifier test: {res}",
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/notifier/send-summary-now")
def settings_notifier_send_summary_now() -> RedirectResponse:
    from services.daily_summary import maybe_send_daily_summary
    maybe_send_daily_summary(force=True)
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/claude/save")
def settings_claude_save(
    anthropic_api_key: str = Form(""),
    anthropic_model: str = Form("claude-sonnet-4-6"),
    keep_existing: str = Form("false"),
) -> RedirectResponse:
    """
    Save Anthropic API key + model. If `keep_existing` is true and the
    submitted key is blank, we leave the existing key untouched (so the
    masked-input field doesn't accidentally wipe a configured key).
    """
    updates: dict[str, str] = {"anthropic_model": (anthropic_model or "claude-sonnet-4-6").strip()}
    submitted_key = (anthropic_api_key or "").strip()
    keep = keep_existing in {"true", "on", "1"}
    if submitted_key:
        updates["anthropic_api_key"] = submitted_key
    elif not keep:
        # User explicitly cleared the field with keep_existing=false → wipe.
        updates["anthropic_api_key"] = ""

    bot_config.set_many(updates)
    with session_scope() as s:
        s.add(
            ActivityLog(
                category="ai",
                level="info",
                message=(
                    f"Claude settings updated (model={updates['anthropic_model']}, "
                    f"key_changed={'yes' if 'anthropic_api_key' in updates else 'no'})."
                ),
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/claude/test")
def settings_claude_test() -> RedirectResponse:
    from services.claude_client import send_test
    res = send_test()
    with session_scope() as s:
        s.add(
            ActivityLog(
                category="ai",
                level="info" if res.get("ok") else "warn",
                message=f"Claude test: {res}",
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/risk/kill-switch")
def settings_kill_switch(action: str = Form("engage")) -> RedirectResponse:
    """
    Engage / release the global kill switch.
    When engaged, every paper and live order is rejected by RiskManager and
    the bot tick short-circuits before hitting the network.
    """
    if action.lower() == "engage":
        RiskManager.set_kill_switch(True, reason="manual via Settings")
    else:
        RiskManager.set_kill_switch(False, reason="manual via Settings")
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/risk/unpause-all")
def settings_unpause_all() -> RedirectResponse:
    """Clear the bot_paused flag on every wallet (after a daily-loss auto-pause)."""
    with session_scope() as s:
        wallets = s.query(Wallet).filter(Wallet.bot_paused.is_(True)).all()
        count = len(wallets)
        for w in wallets:
            w.bot_paused = False
        s.add(
            ActivityLog(
                category="risk",
                level="info",
                message=f"Manually unpaused {count} wallet(s) from Settings.",
            )
        )
    return RedirectResponse(url="/settings", status_code=303)


@router.post("/settings/reset-paper-balance")
def settings_reset_paper_balance(
    new_balance: float = Form(10000.0),
    close_positions: str = Form("true"),
    clear_history: str = Form("false"),
) -> RedirectResponse:
    """
    Reset paper trading balance for all wallets.
    
    Optionally closes all open positions first and/or clears trade history.
    This is a non-destructive way to start fresh without losing AI learning data.
    """
    import logging
    
    should_close = close_positions.lower() == "true"
    should_clear = clear_history.lower() == "true"
    
    logging.info(f"[RESET_PAPER] Resetting to ${new_balance}, close_positions={should_close}, clear_history={should_clear}")
    
    with session_scope() as s:
        wallets = s.query(Wallet).all()
        wallet_count = len(wallets)
        
        # Step 1: Close all open positions if requested
        closed_count = 0
        if should_close:
            from connectors.live_prices import get_prices_batch
            
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            if open_trades:
                # Get all prices at once
                symbols = list(set(t.symbol for t in open_trades))
                prices = get_prices_batch(symbols)
                
                for trade in open_trades:
                    price = prices.get(trade.symbol, trade.entry_price)
                    trade.status = "closed"
                    trade.exit_price = price
                    trade.closed_at = utcnow()
                    trade.exit_reason = "reset"
                    
                    # Calculate realized P&L
                    entry = float(trade.entry_price or 0)
                    qty = float(trade.qty or 0)
                    if trade.side.upper() == "BUY":
                        trade.realized_pnl = (price - entry) * qty
                    else:
                        trade.realized_pnl = (entry - price) * qty
                    
                    closed_count += 1
            
            logging.info(f"[RESET_PAPER] Closed {closed_count} open positions")
        
        # Step 2: Clear trade history if requested
        deleted_count = 0
        if should_clear:
            deleted_count = s.query(PaperTrade).delete()
            logging.info(f"[RESET_PAPER] Deleted {deleted_count} trade records")
        
        # Step 3: Reset all wallet balances AND stamp the bankroll-reset
        # cursor so the training-page money strip can scope "this session"
        # without erasing any history. paper_balance is the LIVE cash; the
        # new session_starting_bankroll preserves the post-reset starting
        # amount for the % return math, and bankroll_reset_at is the cutoff
        # the money-strip query uses to filter closed-trade realized P&L.
        reset_now = utcnow()
        for w in wallets:
            w.paper_balance = new_balance
            w.session_starting_bankroll = float(new_balance)
            w.bankroll_reset_at = reset_now

        # Log the action
        s.add(
            ActivityLog(
                category="settings",
                level="info",
                message=f"Paper balance reset to ${new_balance:,.2f} for {wallet_count} wallet(s). "
                        f"Closed {closed_count} positions. "
                        f"{'Cleared trade history.' if should_clear else 'Trade history preserved.'} "
                        f"Session P&L now counts from this point; full history retained.",
            )
        )
    
    logging.info(f"[RESET_PAPER] Complete - {wallet_count} wallets reset to ${new_balance}")
    return RedirectResponse(url="/settings?paper_reset=success", status_code=303)


@router.post("/settings/reset-data")
def settings_reset_data(confirm: str = Form("")) -> RedirectResponse:
    """
    DESTRUCTIVE: drop and recreate every table.

    Use this to wipe all wallets, trades, AI memory, and settings when you
    want to start fresh. Requires the user to type RESET in the confirm box.
    """
    if confirm.strip().upper() != "RESET":
        return RedirectResponse(url="/settings?reset=denied", status_code=303)
    reset_db()
    return RedirectResponse(url="/wallets", status_code=303)


# ============================================================================ #
# POSITION MANAGEMENT API
# ============================================================================ #


@router.get("/api/v1/wallets")
def api_wallets() -> JSONResponse:
    """Get all wallets with their current limits and status."""
    try:
        with session_scope() as s:
            wallets = s.query(Wallet).all()
            result = []
            for w in wallets:
                # Count open positions
                open_count = (
                    s.query(PaperTrade)
                    .filter(PaperTrade.wallet_id == w.id, PaperTrade.status == "open")
                    .count()
                )
                result.append({
                    "id": w.id,
                    "name": w.name,
                    "platform": w.platform,
                    "paper_balance": float(w.paper_balance or 0),
                    "max_open_positions": w.max_open_positions,
                    "max_position_usd": float(w.max_position_usd or 0),
                    "max_daily_trades": w.max_daily_trades,
                    "max_daily_loss_usd": float(w.max_daily_loss_usd or 0),
                    "bot_paused": w.bot_paused,
                    "trading_mode": w.trading_mode,
                    "trading_style": getattr(w, 'trading_style', 'hybrid'),
                    "open_positions": open_count,
                    "slots_used": f"{open_count}/{w.max_open_positions}",
                })
            return JSONResponse(result)
    except Exception as e:
        import logging
        import traceback
        logging.exception("[API_WALLETS] Error")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/v1/positions")
def api_positions() -> JSONResponse:
    """
    Get all open positions with current P&L and SL/TP proximity.
    """
    import logging
    import traceback
    
    try:
        from connectors.live_prices import get_prices_batch

        with session_scope() as s:
            trades = (
                s.query(PaperTrade)
                .filter(PaperTrade.status == "open")
                .order_by(PaperTrade.opened_at.desc())
                .all()
            )
            
            # Extract trade data while in session
            trade_data = []
            symbols = []
            for t in trades:
                trade_data.append({
                    "id": t.id,
                    "wallet_id": t.wallet_id,
                    "symbol": t.symbol,
                    "side": (t.side or "BUY").upper(),
                    "qty": float(t.qty or 0),
                    "entry_price": float(t.entry_price or 0),
                    "stop_loss_price": float(t.stop_loss_price) if t.stop_loss_price else None,
                    "take_profit_price": float(t.take_profit_price) if t.take_profit_price else None,
                    "trailing_stop_pct": float(t.trailing_stop_pct) if t.trailing_stop_pct else None,
                    "trailing_stop_price": float(t.trailing_stop_price) if t.trailing_stop_price else None,
                    "max_loss_pct": float(t.max_loss_pct or 0.10),
                    "dca_count": t.dca_count or 0,
                    "opened_at": t.opened_at,
                })
                symbols.append(t.symbol)
        
        # Batch fetch all prices at once (much faster!)
        price_map = get_prices_batch(list(set(symbols))) if symbols else {}
        
        positions = []
        for t in trade_data:
            try:
                entry = t["entry_price"]
                qty = t["qty"]
                side = t["side"]
                symbol = t["symbol"]

                # Get current price from batch results
                current = price_map.get(symbol, entry)

                # Calculate P&L
                if side == "BUY":
                    pnl = (current - entry) * qty
                    pnl_pct = (current - entry) / entry if entry > 0 else 0
                else:
                    pnl = (entry - current) * qty
                    pnl_pct = (entry - current) / entry if entry > 0 else 0

                # Calculate SL/TP proximity (0-1 where 1 = at the level)
                sl_proximity = 0.0
                tp_proximity = 0.0

                if t["stop_loss_price"] and entry > 0:
                    sl = t["stop_loss_price"]
                    if side == "BUY":
                        sl_range = entry - sl
                        current_dist = current - sl
                        sl_proximity = max(0, 1 - (current_dist / sl_range)) if sl_range > 0 else 0
                    else:
                        sl_range = sl - entry
                        current_dist = sl - current
                        sl_proximity = max(0, 1 - (current_dist / sl_range)) if sl_range > 0 else 0

                if t["take_profit_price"] and entry > 0:
                    tp = t["take_profit_price"]
                    if side == "BUY":
                        tp_range = tp - entry
                        current_dist = tp - current
                        tp_proximity = max(0, 1 - (current_dist / tp_range)) if tp_range > 0 else 0
                    else:
                        tp_range = entry - tp
                        current_dist = current - tp
                        tp_proximity = max(0, 1 - (current_dist / tp_range)) if tp_range > 0 else 0

                # Time in trade
                opened = t["opened_at"]
                time_in_trade_min = 0
                if opened:
                    from utils.helpers import time_since_minutes
                    time_in_trade_min = time_since_minutes(opened)

                positions.append({
                    "id": t["id"],
                    "wallet_id": t["wallet_id"],
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "entry_price": entry,
                    "current_price": current,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "stop_loss_price": t["stop_loss_price"],
                    "take_profit_price": t["take_profit_price"],
                    "trailing_stop_pct": t["trailing_stop_pct"],
                    "trailing_stop_price": t["trailing_stop_price"],
                    "sl_proximity": round(sl_proximity, 3),
                    "tp_proximity": round(tp_proximity, 3),
                    "max_loss_pct": t["max_loss_pct"],
                    "dca_count": t["dca_count"],
                    "time_in_trade_min": round(time_in_trade_min, 1),
                    "opened_at": opened.isoformat() if opened else None,
                })
            except Exception as e:
                logging.error(f"[POSITIONS] Error processing trade {t['id']}: {e}")
                continue

        return JSONResponse({"ok": True, "positions": positions})
    except Exception as e:
        logging.exception("[POSITIONS] Fatal error")
        return JSONResponse({"ok": False, "error": str(e), "positions": []}, status_code=200)


@router.post("/v1/positions/{trade_id}/close")
def api_close_position(trade_id: int) -> JSONResponse:
    """
    Close a single position at current market price.
    """
    from connectors.live_prices import get_price
    from trading.paper_trading_engine import PaperTradingEngine

    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if not trade:
            return JSONResponse({"ok": False, "error": "Trade not found"}, status_code=404)
        if trade.status != "open":
            return JSONResponse({"ok": False, "error": "Trade is not open"}, status_code=400)

        symbol = trade.symbol

    # Get current price
    p = get_price(symbol)
    if not p.get("ok"):
        return JSONResponse({"ok": False, "error": "Could not fetch current price"}, status_code=500)
    current_price = float(p["price"])

    engine = PaperTradingEngine()
    result = engine.close_trade(trade_id, current_price, notes="manual close via API")

    if result.get("ok"):
        # Set exit reason
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
            if trade:
                trade.exit_reason = "manual"
        return JSONResponse({"ok": True, "realized_pnl": result.get("pnl", 0)})
    else:
        return JSONResponse({"ok": False, "error": result.get("error", "Unknown error")}, status_code=500)


@router.post("/v1/positions/close-all")
def api_close_all_positions() -> JSONResponse:
    """
    Close all open positions at current market prices.
    Uses batch price fetching for speed.
    """
    import logging
    import traceback
    
    try:
        from connectors.live_prices import get_prices_batch
        from trading.paper_trading_engine import PaperTradingEngine
    except ImportError as ie:
        logging.exception("[CLOSE_ALL] Import error")
        return JSONResponse({"ok": False, "error": f"Import error: {ie}", "traceback": traceback.format_exc()})

    closed = 0
    errors = 0
    error_details = []
    total_pnl = 0.0

    try:
        engine = PaperTradingEngine()

        with session_scope() as s:
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            if not open_trades:
                return JSONResponse({"ok": True, "closed": 0, "errors": 0, "total_pnl": 0.0, "message": "No open positions"})
            trade_info = [(t.id, t.symbol) for t in open_trades]

        logging.info(f"[CLOSE_ALL] Closing {len(trade_info)} positions")

        # Batch fetch all prices at once - MUCH faster than one by one
        symbols = list(set(symbol for _, symbol in trade_info))
        prices = get_prices_batch(symbols)
        logging.info(f"[CLOSE_ALL] Fetched {len(prices)} prices for {len(symbols)} symbols")

        for trade_id, symbol in trade_info:
            try:
                price = prices.get(symbol)
                if price is None:
                    errors += 1
                    error_details.append(f"{symbol}: no price")
                    continue

                result = engine.close_trade(trade_id, float(price), notes="close-all via API")
                if result.get("ok"):
                    closed += 1
                    total_pnl += float(result.get("pnl", 0))
                else:
                    errors += 1
                    error_details.append(f"{symbol}: {result.get('reason', 'unknown')}")
            except Exception as e:
                logging.error(f"[CLOSE_ALL] Error closing {symbol}: {e}")
                errors += 1
                error_details.append(f"{symbol}: {str(e)}")

        # Batch update exit_reason for all closed trades
        try:
            with session_scope() as s:
                closed_ids = [tid for tid, _ in trade_info]
                s.query(PaperTrade).filter(
                    PaperTrade.id.in_(closed_ids),
                    PaperTrade.status == "closed"
                ).update({"exit_reason": "manual"}, synchronize_session=False)
        except Exception as db_err:
            logging.error(f"[CLOSE_ALL] Error setting exit_reason: {db_err}")

        try:
            with session_scope() as s:
                s.add(
                    ActivityLog(
                        category="positions",
                        level="info",
                        message=f"Close-all: {closed} positions closed, {errors} errors, total P&L: ${total_pnl:+.2f}",
                    )
                )
        except Exception as log_err:
            logging.error(f"[CLOSE_ALL] Error logging activity: {log_err}")
    except Exception as e:
        logging.exception("[CLOSE_ALL] Fatal error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})

    return JSONResponse({
        "ok": True,
        "closed": closed,
        "errors": errors,
        "error_details": error_details,
        "total_pnl": round(total_pnl, 2),
    })


@router.post("/v1/positions/take-profits")
def api_take_all_profits() -> JSONResponse:
    """
    Close only PROFITABLE positions - lock in gains immediately.
    OPTIMIZED: Fetches all prices in parallel for speed.
    """
    import logging
    import traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        from connectors.live_prices import get_price
        from trading.paper_trading_engine import PaperTradingEngine
    except ImportError as ie:
        logging.exception("[TAKE_PROFITS] Import error")
        return JSONResponse({"ok": False, "error": f"Import error: {ie}", "traceback": traceback.format_exc()})
    
    closed = 0
    skipped = 0
    errors = []
    total_pnl = 0.0
    
    # Get wallet's scalper target
    with session_scope() as s:
        wallet = s.query(Wallet).first()
        micro_target = 0.25  # Default
        if wallet and hasattr(wallet, 'micro_profit_target_usd'):
            micro_target = float(wallet.micro_profit_target_usd or 0.25)
        logging.info(f"[TAKE_PROFITS] Using micro-profit target: ${micro_target}")

    try:
        engine = PaperTradingEngine()
        
        with session_scope() as s:
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            if not open_trades:
                return JSONResponse({"ok": True, "closed": 0, "skipped": 0, "errors": [], "total_pnl": 0.0, "message": "No open positions"})
            trade_info = [(t.id, t.symbol, t.side or "BUY", float(t.entry_price or 0), float(t.qty or 0)) for t in open_trades]
        
        logging.info(f"[TAKE_PROFITS] Found {len(trade_info)} open trades, target=${micro_target}")
        
        # OPTIMIZATION: Fetch all prices in parallel (5 seconds max vs 5 minutes sequential)
        symbols = list(set(t[1] for t in trade_info))
        prices = {}
        
        def fetch_price(symbol):
            try:
                p = get_price(symbol)
                if p.get("ok"):
                    return symbol, float(p["price"])
            except Exception as e:
                logging.debug(f"[TAKE_PROFITS] Price fetch error for {symbol}: {e}")
            return symbol, None
        
        with ThreadPoolExecutor(max_workers=min(20, len(symbols))) as executor:
            futures = {executor.submit(fetch_price, sym): sym for sym in symbols}
            for future in as_completed(futures, timeout=10):
                try:
                    sym, price = future.result()
                    if price is not None:
                        prices[sym] = price
                except Exception as e:
                    logging.debug(f"[TAKE_PROFITS] Future error: {e}")
        
        logging.info(f"[TAKE_PROFITS] Fetched {len(prices)}/{len(symbols)} prices in parallel")

        # Now process trades with cached prices (instant)
        for trade_id, symbol, side, entry_price, qty in trade_info:
            try:
                if entry_price <= 0 or qty <= 0:
                    skipped += 1
                    continue
                
                current_price = prices.get(symbol)
                if current_price is None:
                    skipped += 1
                    continue
                
                # Calculate P&L
                if side.upper() == "BUY":
                    pnl = (current_price - entry_price) * qty
                else:
                    pnl = (entry_price - current_price) * qty
                
                # Only close if profitable AND meets scalper target
                if pnl < micro_target:
                    skipped += 1
                    continue

                result = engine.close_trade(trade_id, current_price, notes="take-profits via API")
                
                if result.get("ok"):
                    closed += 1
                    total_pnl += float(result.get("pnl", 0))
                    try:
                        with session_scope() as s:
                            trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
                            if trade:
                                trade.exit_reason = "take_profit_manual"
                    except Exception:
                        pass
                else:
                    errors.append(f"{symbol}: {result.get('reason', 'unknown')}")
                    skipped += 1
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                skipped += 1

        try:
            with session_scope() as s:
                s.add(
                    ActivityLog(
                        category="positions",
                        level="info",
                        message=f"Take-profits: {closed} winners closed (${total_pnl:+.2f}), {skipped} positions held",
                    )
                )
        except Exception:
            pass
            
    except Exception as e:
        logging.exception("[TAKE_PROFITS] Fatal error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})

    return JSONResponse({
        "ok": True,
        "closed": closed,
        "skipped": skipped,
        "errors": errors,
        "total_pnl": round(total_pnl, 2),
    })


@router.post("/v1/wallet/trading-style")
def api_update_trading_style(
    trading_style: str = Form(...),  # scalper, swing, hybrid
    micro_profit_target_usd: float = Form(0.25),
    min_profit_pct: float = Form(0.003),
    auto_reinvest: bool = Form(True),
    max_daily_trades: int = Form(100),
) -> JSONResponse:
    """
    Update wallet trading style settings for scalping vs swing trading.
    
    - scalper: Take micro-profits ($0.25 default) as soon as they hit
    - swing: Hold for larger gains (1-5%), use traditional SL/TP
    - hybrid: AI decides based on market conditions
    """
    if trading_style not in ("scalper", "swing", "hybrid"):
        return JSONResponse({"ok": False, "error": "Invalid trading_style"}, status_code=400)
    
    with session_scope() as s:
        wallet = s.query(Wallet).first()
        if not wallet:
            return JSONResponse({"ok": False, "error": "No wallet found"}, status_code=404)
        
        wallet.trading_style = trading_style
        wallet.micro_profit_target_usd = micro_profit_target_usd
        wallet.min_profit_pct = min_profit_pct
        wallet.auto_reinvest = auto_reinvest
        wallet.max_daily_trades = max_daily_trades
        
        s.add(
            ActivityLog(
                category="settings",
                level="info",
                message=f"Trading style updated: {trading_style}, micro-target: ${micro_profit_target_usd}, max trades: {max_daily_trades}",
            )
        )
    
    return JSONResponse({
        "ok": True,
        "trading_style": trading_style,
        "micro_profit_target_usd": micro_profit_target_usd,
        "min_profit_pct": min_profit_pct,
        "auto_reinvest": auto_reinvest,
        "max_daily_trades": max_daily_trades,
    })


@router.get("/v1/wallet/trading-style")
def api_get_trading_style() -> JSONResponse:
    """Get current wallet trading style settings."""
    import logging
    with session_scope() as s:
        wallet = s.query(Wallet).first()
        if not wallet:
            return JSONResponse({"ok": False, "error": "No wallet found"}, status_code=404)
        
        # Log actual database values for debugging
        logging.info(f"[TRADING_STYLE] DB values: trading_style={wallet.trading_style}, micro_target={wallet.micro_profit_target_usd}, min_pct={wallet.min_profit_pct}, auto_reinvest={wallet.auto_reinvest}")
        
        return JSONResponse({
            "ok": True,
            "trading_style": wallet.trading_style or 'hybrid',
            "micro_profit_target_usd": float(wallet.micro_profit_target_usd or 0.25),
            "min_profit_pct": float(wallet.min_profit_pct or 0.003),
            "auto_reinvest": bool(wallet.auto_reinvest) if wallet.auto_reinvest is not None else True,
            "max_daily_trades": wallet.max_daily_trades or 10,
        })


@router.post("/v1/positions/force-check")
def api_force_check_positions() -> JSONResponse:
    """
    Force the position monitor to check all positions NOW and return what it finds.
    This is a debug endpoint to test if scalper settings are working.
    """
    import logging
    import traceback
    from connectors.live_prices import get_price
    
    results = []
    exit_signals = []
    
    try:
        with session_scope() as s:
            # Get all open trades
            open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
            
            if not open_trades:
                return JSONResponse({
                    "ok": True,
                    "total_positions": 0,
                    "would_exit": 0,
                    "positions": [],
                    "exit_signals": [],
                    "message": "No open positions to check"
                })
            
            for trade in open_trades:
                try:
                    # Get wallet settings
                    wallet = s.query(Wallet).filter(Wallet.id == trade.wallet_id).first()
                    
                    trading_style = wallet.trading_style if wallet else 'hybrid'
                    micro_target = float(wallet.micro_profit_target_usd or 0.25) if wallet else 0.25
                    min_pct = float(wallet.min_profit_pct or 0.003) if wallet else 0.003
                    
                    # Get current price
                    p = get_price(trade.symbol)
                    if not p.get("ok"):
                        results.append({
                            "symbol": trade.symbol,
                            "error": f"Could not fetch price: {p.get('error', 'unknown')}"
                        })
                        continue
                    
                    current_price = float(p["price"])
                    entry = float(trade.entry_price or 0)
                    qty = float(trade.qty or 0)
                    side = (trade.side or "BUY").upper()
                    
                    if entry <= 0:
                        results.append({
                            "symbol": trade.symbol,
                            "error": "Invalid entry price"
                        })
                        continue
                    
                    # Calculate P&L
                    if side == "BUY":
                        pnl_pct = (current_price - entry) / entry
                    else:
                        pnl_pct = (entry - current_price) / entry
                    
                    pnl_usd = pnl_pct * entry * qty
                    
                    # Check if should exit
                    should_exit = False
                    exit_reason = None
                    
                    if pnl_usd > 0:
                        if trading_style == "scalper" and pnl_usd >= micro_target:
                            should_exit = True
                            exit_reason = f"SCALPER: pnl_usd ${pnl_usd:.4f} >= target ${micro_target}"
                        elif trading_style == "hybrid" and (pnl_usd >= micro_target or pnl_pct >= min_pct):
                            should_exit = True
                            exit_reason = f"HYBRID: pnl_usd ${pnl_usd:.4f} or pnl_pct {pnl_pct:.4%}"
                    
                    result = {
                        "trade_id": trade.id,
                        "symbol": trade.symbol,
                        "side": side,
                        "entry": entry,
                        "current": current_price,
                        "qty": qty,
                        "pnl_usd": round(pnl_usd, 4),
                        "pnl_pct": round(pnl_pct * 100, 4),
                        "wallet_trading_style": trading_style,
                        "wallet_micro_target": micro_target,
                        "wallet_min_pct": min_pct,
                        "should_exit": should_exit,
                        "exit_reason": exit_reason,
                    }
                    results.append(result)
                    
                    if should_exit:
                        exit_signals.append(result)
                        logging.info(f"[FORCE_CHECK] Would exit {trade.symbol}: {exit_reason}")
                except Exception as trade_err:
                    logging.error(f"[FORCE_CHECK] Error processing trade {trade.id}: {trade_err}")
                    results.append({
                        "trade_id": trade.id,
                        "symbol": trade.symbol if trade else "unknown",
                        "error": str(trade_err)
                    })
        
        return JSONResponse({
            "ok": True,
            "total_positions": len(results),
            "would_exit": len(exit_signals),
            "positions": results,
            "exit_signals": exit_signals,
        })
    except Exception as e:
        logging.exception("[FORCE_CHECK] Error")
        return JSONResponse({
            "ok": False,
            "error": str(e) or "Unknown error occurred",
            "traceback": traceback.format_exc(),
        })


@router.post("/v1/positions/{trade_id}/close-partial")
def api_close_partial(
    trade_id: int,
    fraction: float = Form(...),
) -> JSONResponse:
    """
    Partially close a position (e.g., close 25%, 50%, 75%).
    """
    from connectors.live_prices import get_price
    from trading.paper_trading_engine import PaperTradingEngine

    if fraction <= 0 or fraction >= 1:
        return JSONResponse({"ok": False, "error": "Fraction must be between 0 and 1"}, status_code=400)

    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if not trade:
            return JSONResponse({"ok": False, "error": "Trade not found"}, status_code=404)
        if trade.status != "open":
            return JSONResponse({"ok": False, "error": "Trade is not open"}, status_code=400)

        symbol = trade.symbol
        current_qty = float(trade.qty)
        entry_price = float(trade.entry_price)
        side = (trade.side or "BUY").upper()

    # Get current price
    p = get_price(symbol)
    if not p.get("ok"):
        return JSONResponse({"ok": False, "error": "Could not fetch current price"}, status_code=500)
    current_price = float(p["price"])

    # Calculate partial close
    close_qty = current_qty * fraction
    remaining_qty = current_qty - close_qty

    # Calculate P&L for the closed portion
    if side == "BUY":
        realized_pnl = (current_price - entry_price) * close_qty
    else:
        realized_pnl = (entry_price - current_price) * close_qty

    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if trade:
            # Update the position to reflect partial close
            trade.qty = remaining_qty
            # Add to realized P&L (track partial closes)
            trade.realized_pnl = float(trade.realized_pnl or 0) + realized_pnl
            # Update wallet balance
            wallet = s.query(Wallet).filter(Wallet.id == trade.wallet_id).first()
            if wallet:
                wallet.paper_balance = float(wallet.paper_balance or 0) + (close_qty * current_price) + realized_pnl

            s.add(
                ActivityLog(
                    category="positions",
                    level="info",
                    message=f"Partial close {fraction*100:.0f}% of {symbol}: realized ${realized_pnl:+.2f}",
                    wallet_id=trade.wallet_id,
                )
            )

    return JSONResponse({
        "ok": True,
        "realized_pnl": round(realized_pnl, 2),
        "remaining_qty": round(remaining_qty, 6),
        "closed_qty": round(close_qty, 6),
    })


@router.post("/v1/positions/{trade_id}/sl-tp")
def api_update_sl_tp(
    trade_id: int,
    stop_loss_price: float | None = Form(None),
    take_profit_price: float | None = Form(None),
) -> JSONResponse:
    """
    Update stop-loss and/or take-profit prices for a position.
    """
    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if not trade:
            return JSONResponse({"ok": False, "error": "Trade not found"}, status_code=404)
        if trade.status != "open":
            return JSONResponse({"ok": False, "error": "Trade is not open"}, status_code=400)

        if stop_loss_price is not None:
            trade.stop_loss_price = stop_loss_price
        if take_profit_price is not None:
            trade.take_profit_price = take_profit_price

        s.add(
            ActivityLog(
                category="positions",
                level="info",
                message=f"Updated SL/TP for {trade.symbol}: SL=${stop_loss_price}, TP=${take_profit_price}",
                wallet_id=trade.wallet_id,
            )
        )

    return JSONResponse({"ok": True})


@router.post("/v1/positions/{trade_id}/trailing")
def api_set_trailing_stop(
    trade_id: int,
    trailing_stop_pct: float = Form(...),
) -> JSONResponse:
    """
    Enable or update a trailing stop for a position.
    """
    from connectors.live_prices import get_price

    if trailing_stop_pct <= 0 or trailing_stop_pct > 0.5:
        return JSONResponse(
            {"ok": False, "error": "Trailing stop must be between 0 and 50%"},
            status_code=400,
        )

    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if not trade:
            return JSONResponse({"ok": False, "error": "Trade not found"}, status_code=404)
        if trade.status != "open":
            return JSONResponse({"ok": False, "error": "Trade is not open"}, status_code=400)

        # Get current price to initialize trailing stop
        p = get_price(trade.symbol)
        current = float(p.get("price") or trade.entry_price) if p.get("ok") else float(trade.entry_price)

        entry = float(trade.entry_price)
        side = (trade.side or "BUY").upper()

        trade.trailing_stop_pct = trailing_stop_pct

        # Initialize high water mark and trailing price
        if side == "BUY":
            trade.high_water_price = max(current, entry)
            trade.trailing_stop_price = trade.high_water_price * (1 - trailing_stop_pct)
        else:
            trade.high_water_price = min(current, entry)
            trade.trailing_stop_price = trade.high_water_price * (1 + trailing_stop_pct)

        s.add(
            ActivityLog(
                category="positions",
                level="info",
                message=(
                    f"Trailing stop set for {trade.symbol}: {trailing_stop_pct:.1%} "
                    f"(initial price: ${trade.trailing_stop_price:.4f})"
                ),
                wallet_id=trade.wallet_id,
            )
        )

    return JSONResponse({"ok": True, "trailing_stop_price": trade.trailing_stop_price})


@router.post("/v1/positions/{trade_id}/dca")
def api_dca_position(
    trade_id: int,
    add_usd: float = Form(...),
) -> JSONResponse:
    """
    Dollar-cost average into an existing position.
    """
    from connectors.live_prices import get_price
    from trading.loss_recovery import LossRecoveryEngine

    if add_usd <= 0:
        return JSONResponse({"ok": False, "error": "Amount must be positive"}, status_code=400)

    with session_scope() as s:
        trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
        if not trade:
            return JSONResponse({"ok": False, "error": "Trade not found"}, status_code=404)
        if trade.status != "open":
            return JSONResponse({"ok": False, "error": "Trade is not open"}, status_code=400)
        if (trade.dca_count or 0) >= 5:
            return JSONResponse({"ok": False, "error": "Max DCA count (5) reached"}, status_code=400)

        symbol = trade.symbol

    # Get current price
    p = get_price(symbol)
    if not p.get("ok"):
        return JSONResponse({"ok": False, "error": "Could not fetch current price"}, status_code=500)
    current_price = float(p["price"])
    add_qty = add_usd / current_price

    engine = LossRecoveryEngine()
    success = engine.execute_dca(trade, add_qty, current_price)

    if success:
        # Reload trade to get new values
        with session_scope() as s:
            trade = s.query(PaperTrade).filter(PaperTrade.id == trade_id).first()
            return JSONResponse({
                "ok": True,
                "new_entry": float(trade.entry_price) if trade else 0,
                "new_qty": float(trade.qty) if trade else 0,
                "dca_count": trade.dca_count if trade else 0,
            })
    else:
        return JSONResponse({"ok": False, "error": "DCA failed"}, status_code=500)


@router.post("/v1/positions/emergency-exit")
def api_emergency_exit() -> JSONResponse:
    """
    Emergency exit: Close all positions AND engage kill switch.
    """
    import json
    import logging
    import traceback
    
    try:
        # First engage kill switch
        RiskManager.set_kill_switch(True, reason="emergency exit via API")
        logging.warning("[EMERGENCY_EXIT] Kill switch engaged")

        # Then close all positions
        result = api_close_all_positions()
        data = result.body.decode() if hasattr(result, "body") else "{}"
        close_result = json.loads(data) if data else {}

        try:
            with session_scope() as s:
                s.add(
                    ActivityLog(
                        category="risk",
                        level="warn",
                        message="EMERGENCY EXIT: Kill switch engaged and all positions closed.",
                    )
                )
        except Exception as log_err:
            logging.error(f"[EMERGENCY_EXIT] Error logging activity: {log_err}")

        return JSONResponse({
            "ok": True,
            "kill_switch_engaged": True,
            "positions_closed": close_result.get("closed", 0),
            "total_pnl": close_result.get("total_pnl", 0),
        })
    except Exception as e:
        logging.exception("[EMERGENCY_EXIT] Fatal error")
        return JSONResponse({"ok": False, "error": str(e) or "Unknown error", "traceback": traceback.format_exc()})


@router.get("/v1/portfolio-intel")
def api_portfolio_intel() -> JSONResponse:
    """
    Get portfolio intelligence state - whether recovery mode is active,
    recent actions taken, and current portfolio health metrics.
    """
    from connectors.live_prices import get_price
    from trading.portfolio_intelligence import PortfolioIntelligence
    
    intel = PortfolioIntelligence()
    
    # Get all wallets with positions
    with session_scope() as s:
        wallet_ids = (
            s.query(PaperTrade.wallet_id)
            .filter(PaperTrade.status == "open")
            .distinct()
            .all()
        )
        wallet_ids = [w[0] for w in wallet_ids]
        
        # Get all open positions for price lookup
        positions = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
        symbols = list(set(p.symbol for p in positions))
    
    # Build price map
    price_map: dict[str, float] = {}
    for symbol in symbols:
        p = get_price(symbol)
        if p.get("ok"):
            price_map[symbol] = float(p["price"])
    
    # Analyze portfolio for each wallet
    portfolio_states = []
    for wallet_id in wallet_ids:
        state = intel.analyze_portfolio(wallet_id, price_map)
        portfolio_states.append({
            "wallet_id": wallet_id,
            "total_positions": state.total_positions,
            "total_pnl_usd": round(state.total_pnl_usd, 2),
            "total_pnl_pct": round(state.total_pnl_pct * 100, 2),
            "winning_count": state.winning_count,
            "losing_count": state.losing_count,
            "biggest_loser_pct": round(state.biggest_loser_pct * 100, 2),
            "biggest_winner_pct": round(state.biggest_winner_pct * 100, 2),
            "available_capital": round(state.available_capital, 2),
            "is_recovery_mode": state.is_recovery_mode,
        })
    
    # Get recent intel activity logs
    recent_actions = []
    with session_scope() as s:
        logs = (
            s.query(ActivityLog)
            .filter(ActivityLog.category == "portfolio_intel")
            .order_by(ActivityLog.created_at.desc())
            .limit(10)
            .all()
        )
        recent_actions = [
            {
                "message": log.message,
                "level": log.level,
                "time": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]
    
    # Aggregate
    is_any_recovery = any(ps["is_recovery_mode"] for ps in portfolio_states)
    total_pnl = sum(ps["total_pnl_usd"] for ps in portfolio_states)
    
    return JSONResponse({
        "ok": True,
        "is_recovery_mode": is_any_recovery,
        "total_pnl_usd": round(total_pnl, 2),
        "portfolios": portfolio_states,
        "recent_actions": recent_actions,
    })


@router.get("/v1/pnl-stats")
def api_pnl_stats(period: str = "today") -> JSONResponse:
    """
    Get realized P&L stats for a given time period.
    
    period: today, 3days, week, 2weeks, month, all
    """
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    
    # Calculate start date based on period
    if period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "3days":
        start_date = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start_date = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "2weeks":
        start_date = (now - timedelta(days=14)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        start_date = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:  # all
        start_date = None
    
    with session_scope() as s:
        query = s.query(PaperTrade).filter(PaperTrade.status == "closed")
        
        if start_date:
            query = query.filter(PaperTrade.closed_at >= start_date)
        
        closed_trades = query.all()
        
        # Calculate stats
        total_pnl = 0.0
        wins = 0
        losses = 0
        best_trade = 0.0
        worst_trade = 0.0
        total_volume = 0.0
        
        for trade in closed_trades:
            pnl = float(trade.realized_pnl or 0)
            total_pnl += pnl
            total_volume += float(trade.entry_price or 0) * float(trade.qty or 0)
            
            if pnl > 0:
                wins += 1
                if pnl > best_trade:
                    best_trade = pnl
            elif pnl < 0:
                losses += 1
                if pnl < worst_trade:
                    worst_trade = pnl
        
        trade_count = len(closed_trades)
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        avg_trade = (total_pnl / trade_count) if trade_count > 0 else 0
        
        # Calculate daily average if not "today"
        if start_date and period != "today":
            days = (now - start_date).days or 1
            daily_avg = total_pnl / days
        else:
            daily_avg = total_pnl
    
    return JSONResponse({
        "ok": True,
        "period": period,
        "total_pnl": round(total_pnl, 2),
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "avg_trade": round(avg_trade, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "daily_avg": round(daily_avg, 2),
        "total_volume": round(total_volume, 2),
    })


# ======================================================================
# DIAGNOSTIC ENDPOINT - Why Trades Aren't Executing
# ======================================================================

@router.get("/v1/diagnostics/trade-blockers")
def api_trade_blockers() -> JSONResponse:
    """
    Comprehensive diagnostic endpoint that shows EXACTLY why trades may not be executing.
    This checks every possible blocker in the system.
    """
    import logging
    from config.bot_config import BotConfig, get as cfg_get
    from trading.risk_manager import RiskManager
    from datetime import timedelta
    
    blockers: list[dict] = []
    warnings: list[dict] = []
    info: list[dict] = []
    
    # Load bot configuration
    cfg = BotConfig.load()
    
    # =================================================================
    # CHECK 1: Bot Enabled
    # =================================================================
    if not cfg.bot_enabled:
        blockers.append({
            "blocker": "BOT_DISABLED",
            "message": "Bot is DISABLED. Enable it in Settings > Bot Settings.",
            "setting": "bot_enabled",
            "current_value": "false",
            "fix": "Set bot_enabled to true in Settings"
        })
    else:
        info.append({"check": "bot_enabled", "status": "OK", "value": True})
    
    # =================================================================
    # CHECK 2: Dry Run Mode (MOST COMMON ISSUE!)
    # =================================================================
    if cfg.dry_run:
        blockers.append({
            "blocker": "DRY_RUN_MODE",
            "message": "DRY RUN is ON! The bot logs what it WOULD do but NEVER actually opens trades!",
            "setting": "bot_dry_run",
            "current_value": "true",
            "fix": "Disable 'Dry Run' in Settings to actually execute trades"
        })
    else:
        info.append({"check": "dry_run", "status": "OK", "value": False})
    
    # =================================================================
    # CHECK 3: Kill Switch
    # =================================================================
    if RiskManager.kill_switch_status():
        blockers.append({
            "blocker": "KILL_SWITCH_ENGAGED",
            "message": "The global KILL SWITCH is engaged. All trading is halted.",
            "setting": "kill_switch",
            "fix": "Release the kill switch in Settings or via the dashboard"
        })
    else:
        info.append({"check": "kill_switch", "status": "OK", "value": False})
    
    # =================================================================
    # CHECK 4: Wallets
    # =================================================================
    with session_scope() as s:
        wallets = s.query(Wallet).all()
        if not wallets:
            blockers.append({
                "blocker": "NO_WALLETS",
                "message": "No wallets configured. Create a wallet first.",
                "fix": "Go to Wallets page and create a paper trading wallet"
            })
        else:
            active_wallets = 0
            for w in wallets:
                wallet_issues = []
                
                if w.bot_paused:
                    wallet_issues.append("bot_paused=true")
                
                balance = float(w.paper_balance or 0)
                if balance < cfg.position_size_usd:
                    wallet_issues.append(f"balance=${balance:.2f} < position_size=${cfg.position_size_usd:.2f}")
                
                # Check max_open_positions
                open_count = s.query(PaperTrade).filter(
                    PaperTrade.wallet_id == w.id,
                    PaperTrade.status == "open"
                ).count()
                max_open = w.max_open_positions or cfg.max_open_per_wallet
                if max_open and open_count >= max_open:
                    wallet_issues.append(f"positions={open_count}/{max_open} (FULL)")
                
                # Check daily loss
                today = utcnow().date()
                day_trades = s.query(PaperTrade).filter(
                    PaperTrade.wallet_id == w.id,
                    PaperTrade.status == "closed"
                ).all()
                day_pnl = sum(
                    float(t.realized_pnl or 0) for t in day_trades
                    if t.closed_at and t.closed_at.date() == today
                )
                if w.max_daily_loss_usd and day_pnl <= -float(w.max_daily_loss_usd):
                    wallet_issues.append(f"daily_loss=${day_pnl:.2f} hit cap=${w.max_daily_loss_usd}")
                
                # Check cooldown
                recent = s.query(PaperTrade).filter(
                    PaperTrade.wallet_id == w.id,
                    PaperTrade.status == "closed"
                ).order_by(PaperTrade.closed_at.desc()).limit(3).all()
                if len(recent) >= 3 and all((t.realized_pnl or 0) < 0 for t in recent):
                    if recent[0].closed_at:
                        cooldown_until = recent[0].closed_at + timedelta(minutes=15)
                        if utcnow() < cooldown_until:
                            wallet_issues.append(f"cooldown until {cooldown_until.isoformat()}")
                
                if wallet_issues:
                    warnings.append({
                        "wallet": w.name,
                        "wallet_id": w.id,
                        "issues": wallet_issues,
                        "can_trade": len(wallet_issues) == 0
                    })
                else:
                    active_wallets += 1
                    info.append({
                        "wallet": w.name,
                        "wallet_id": w.id,
                        "balance": float(w.paper_balance or 0),
                        "open_positions": open_count,
                        "max_positions": max_open,
                        "status": "OK"
                    })
            
            if active_wallets == 0:
                blockers.append({
                    "blocker": "ALL_WALLETS_BLOCKED",
                    "message": "All wallets are either paused, over-extended, or out of balance",
                    "fix": "Check wallet settings, add paper balance, or unpause wallets"
                })
    
    # =================================================================
    # CHECK 5: Confidence Floor
    # =================================================================
    if cfg.min_confidence >= 0.70:
        warnings.append({
            "warning": "HIGH_CONFIDENCE_FLOOR",
            "message": f"Confidence floor is {cfg.min_confidence:.2f} - very few signals will pass",
            "setting": "bot_min_confidence",
            "current_value": cfg.min_confidence,
            "suggestion": "Lower to 0.55 or 0.50 to see more trades"
        })
    info.append({"check": "min_confidence", "value": cfg.min_confidence})
    
    # =================================================================
    # CHECK 6: Claude API Key
    # =================================================================
    from services.claude_client import is_configured as claude_is_configured
    if not claude_is_configured():
        warnings.append({
            "warning": "CLAUDE_NOT_CONFIGURED",
            "message": "Claude API key not set. Bot will use technical signals only (no AI enhancement).",
            "setting": "anthropic_api_key",
            "fix": "Add Anthropic API key in Settings"
        })
    else:
        info.append({"check": "claude_api", "status": "configured"})
    
    # =================================================================
    # CHECK 7: Scheduler Status
    # =================================================================
    from services.scheduler import bot_scheduler
    scheduler_status = bot_scheduler.status()
    if not scheduler_status.get("scheduler_running"):
        blockers.append({
            "blocker": "SCHEDULER_NOT_RUNNING",
            "message": "The bot scheduler is not running. Trades won't auto-execute.",
            "fix": "Start the scheduler via the Bot Control panel"
        })
    else:
        info.append({
            "check": "scheduler",
            "status": "running",
            "next_tick": scheduler_status.get("next_tick"),
            "tick_seconds": scheduler_status.get("tick_seconds")
        })
    
    # =================================================================
    # CHECK 8: Recent Activity (any trades in last 24h?)
    # =================================================================
    with session_scope() as s:
        yesterday = utcnow() - timedelta(hours=24)
        recent_opens = s.query(PaperTrade).filter(PaperTrade.opened_at >= yesterday).count()
        recent_closes = s.query(PaperTrade).filter(
            PaperTrade.closed_at >= yesterday,
            PaperTrade.status == "closed"
        ).count()
        
        if recent_opens == 0 and cfg.bot_enabled and not cfg.dry_run:
            warnings.append({
                "warning": "NO_RECENT_TRADES",
                "message": "No trades opened in the last 24 hours despite bot being enabled",
                "possible_causes": [
                    "Confidence threshold too high",
                    "All wallets paused or over-extended",
                    "No strong signals in the market"
                ]
            })
        
        info.append({
            "check": "recent_activity",
            "trades_opened_24h": recent_opens,
            "trades_closed_24h": recent_closes
        })
    
    # =================================================================
    # CHECK 9: Training Mode
    # =================================================================
    is_training = (cfg_get("training_session_active") or "").strip().lower() in {"1", "true", "yes", "on"}
    if is_training:
        info.append({
            "check": "training_mode",
            "status": "ACTIVE",
            "note": "Training session is active - bot should be trading more aggressively"
        })
    
    # =================================================================
    # Summary
    # =================================================================
    can_trade = len(blockers) == 0
    summary = {
        "can_trade": can_trade,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "status": "BLOCKED" if blockers else ("WARNINGS" if warnings else "OK")
    }
    
    # Quick fix suggestions
    quick_fixes = []
    if cfg.dry_run:
        quick_fixes.append("Disable 'Dry Run' mode in Settings")
    if not cfg.bot_enabled:
        quick_fixes.append("Enable the bot in Settings")
    if cfg.min_confidence > 0.60:
        quick_fixes.append(f"Lower min_confidence from {cfg.min_confidence} to 0.50-0.55")
    
    return JSONResponse({
        "ok": True,
        "summary": summary,
        "blockers": blockers,
        "warnings": warnings,
        "info": info,
        "quick_fixes": quick_fixes,
        "config": {
            "bot_enabled": cfg.bot_enabled,
            "dry_run": cfg.dry_run,
            "tick_seconds": cfg.tick_seconds,
            "min_confidence": cfg.min_confidence,
            "position_size_usd": cfg.position_size_usd,
            "max_open_per_wallet": cfg.max_open_per_wallet,
            "universe": cfg.universe,
            "universe_limit": cfg.universe_limit
        }
    })


# ======================================================================
# DIAGNOSTIC: Test entire trade flow
# ======================================================================

@router.post("/v1/diagnostics/test-trade-flow")
def api_test_trade_flow() -> JSONResponse:
    """
    Test the ENTIRE trade execution flow with detailed diagnostics.
    This tests every step from config loading to trade opening.
    """
    import logging
    import traceback
    
    steps = []
    errors = []
    
    def log_step(name: str, status: str, details: dict = None):
        steps.append({"step": name, "status": status, "details": details or {}})
        logging.info(f"[TEST_FLOW] {name}: {status} - {details}")
    
    try:
        # Step 1: Load config
        from config.bot_config import BotConfig
        cfg = BotConfig.load()
        log_step("1_load_config", "OK", {
            "bot_enabled": cfg.bot_enabled,
            "dry_run": cfg.dry_run,
            "position_size_usd": cfg.position_size_usd,
            "universe_limit": cfg.universe_limit,
            "min_confidence": cfg.min_confidence,
            "max_open_per_wallet": cfg.max_open_per_wallet,
        })
        
        # Step 2: Load universe
        from connectors.universe import coinbase_usd_universe
        universe = coinbase_usd_universe(limit=cfg.universe_limit)
        log_step("2_load_universe", "OK" if universe else "FAIL", {
            "count": len(universe),
            "first_10": [u["product_id"] for u in universe[:10]],
        })
        
        if not universe:
            errors.append("Universe is empty - cannot trade")
            return JSONResponse({"ok": False, "steps": steps, "errors": errors})
        
        # Step 3: Get a test price
        test_symbol = universe[0]["product_id"]
        from connectors.live_prices import get_price
        price_result = get_price(test_symbol)
        if price_result.get("ok"):
            test_price = float(price_result["price"])
            log_step("3_get_price", "OK", {"symbol": test_symbol, "price": test_price})
        else:
            log_step("3_get_price", "FAIL", {"error": price_result.get("error")})
            errors.append(f"Cannot get price for {test_symbol}")
            test_price = 100.0  # fallback
        
        # Step 4: Find wallets
        with session_scope() as s:
            wallets = s.query(Wallet).all()
            wallet_info = []
            for w in wallets:
                open_count = s.query(PaperTrade).filter(
                    PaperTrade.wallet_id == w.id,
                    PaperTrade.status == "open"
                ).count()
                wallet_info.append({
                    "id": w.id,
                    "name": w.name,
                    "paper_balance": float(w.paper_balance or 0),
                    "bot_paused": w.bot_paused,
                    "max_open_positions": w.max_open_positions,
                    "open_positions": open_count,
                })
        
        log_step("4_find_wallets", "OK" if wallet_info else "FAIL", {
            "count": len(wallet_info),
            "wallets": wallet_info,
        })
        
        if not wallet_info:
            errors.append("No wallets found - create a wallet first")
            return JSONResponse({"ok": False, "steps": steps, "errors": errors})
        
        # Step 5: Check risk manager for first active wallet
        test_wallet = None
        for w in wallet_info:
            if not w["bot_paused"]:
                test_wallet = w
                break
        
        if not test_wallet:
            log_step("5_risk_check", "FAIL", {"reason": "All wallets are paused"})
            errors.append("All wallets are paused")
            return JSONResponse({"ok": False, "steps": steps, "errors": errors})
        
        from trading.risk_manager import RiskManager
        rm = RiskManager()
        
        # Calculate position
        position_usd = cfg.position_size_usd
        qty = position_usd / test_price if test_price > 0 else 0
        
        risk_decision = rm.evaluate(
            wallet_id=test_wallet["id"],
            qty=qty,
            entry_price=test_price,
            confidence=0.70,
            strategy_id=None,
        )
        
        log_step("5_risk_check", "OK" if risk_decision.allowed else "BLOCKED", {
            "wallet": test_wallet["name"],
            "allowed": risk_decision.allowed,
            "reason": risk_decision.reason,
            "code": risk_decision.code,
            "test_qty": qty,
            "test_price": test_price,
            "test_notional": qty * test_price,
        })
        
        if not risk_decision.allowed:
            errors.append(f"Risk check blocked: {risk_decision.reason}")
        
        # Step 6: Check if dry_run is blocking
        if cfg.dry_run:
            log_step("6_dry_run_check", "WARNING", {
                "message": "DRY RUN is ON - trades will NOT actually execute",
                "fix": "Disable 'Dry Run' in Settings to actually trade"
            })
            errors.append("DRY RUN is ON - trades are logged but not executed")
        else:
            log_step("6_dry_run_check", "OK", {"dry_run": False})
        
        # Step 7: Check Claude API
        from services.claude_client import is_configured as claude_configured
        claude_ok = claude_configured()
        log_step("7_claude_api", "OK" if claude_ok else "WARNING", {
            "configured": claude_ok,
            "note": "Without Claude, bot uses technical signals only"
        })
        
        # Step 8: Simulate what WOULD happen on a tick
        simulation = {
            "would_evaluate_symbols": min(len(universe), cfg.universe_limit),
            "position_size_per_trade": cfg.position_size_usd,
            "max_concurrent_positions": cfg.max_open_per_wallet,
            "wallet_available_slots": max(0, (test_wallet.get("max_open_positions") or cfg.max_open_per_wallet) - test_wallet["open_positions"]),
            "wallet_balance": test_wallet["paper_balance"],
            "can_afford_trade": test_wallet["paper_balance"] >= cfg.position_size_usd,
        }
        log_step("8_simulation", "OK", simulation)
        
        if not simulation["can_afford_trade"]:
            errors.append(f"Wallet balance (${test_wallet['paper_balance']:.2f}) is less than position size (${cfg.position_size_usd})")
        
        if simulation["wallet_available_slots"] <= 0:
            errors.append("No available slots for new positions (max positions reached)")
        
        # Summary
        can_trade = (
            cfg.bot_enabled and
            not cfg.dry_run and
            risk_decision.allowed and
            simulation["can_afford_trade"] and
            simulation["wallet_available_slots"] > 0
        )
        
        return JSONResponse({
            "ok": True,
            "can_actually_trade": can_trade,
            "steps": steps,
            "errors": errors,
            "recommendation": (
                "System is ready to trade!" if can_trade else
                "Fix the errors above to enable trading"
            ),
        })
        
    except Exception as e:
        logging.exception("[TEST_FLOW] Error")
        return JSONResponse({
            "ok": False,
            "steps": steps,
            "errors": errors + [str(e)],
            "traceback": traceback.format_exc(),
        })
