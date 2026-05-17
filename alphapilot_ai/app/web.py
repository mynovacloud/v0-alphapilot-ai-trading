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


templates.env.filters["money"] = _fmt_money
templates.env.filters["pct"] = _fmt_pct
templates.env.filters["signed"] = _fmt_signed
templates.env.filters["dt"] = _fmt_dt

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
def dashboard(request: Request) -> HTMLResponse:
    summary = portfolio_summary()
    perf = performance_metrics()
    wallets = get_wallets()

    # Equity curve points for the SVG chart
    eq = equity_curve_df()
    points: list[dict[str, Any]] = []
    if not eq.empty:
        for _, row in eq.iterrows():
            points.append({"date": str(row["date"]), "equity": float(row["equity"])})

    # Recent activity
    with session_scope() as s:
        logs = (
            s.query(ActivityLog)
            .order_by(ActivityLog.created_at.desc())
            .limit(8)
            .all()
        )
        recent_logs = [
            {
                "category": l.category,
                "level": l.level,
                "message": l.message,
                "created_at": l.created_at,
            }
            for l in logs
        ]

        # Recent trades
        trades = (
            s.query(PaperTrade)
            .order_by(PaperTrade.opened_at.desc())
            .limit(8)
            .all()
        )
        wallet_names = {w["id"]: w["name"] for w in wallets}
        recent_trades = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "status": t.status,
                "realized_pnl": t.realized_pnl,
                "wallet": wallet_names.get(t.wallet_id, "?"),
                "opened_at": t.opened_at,
            }
            for t in trades
        ]

    return templates.TemplateResponse(request=request, name="dashboard.html", context=_ctx(
            request,
            active="dashboard",
            summary=summary,
            perf=perf,
            wallets=wallets,
            equity_points=points,
            recent_logs=recent_logs,
            recent_trades=recent_trades,
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

    # Win/loss histogram
    df = get_all_trades_df()
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
        ),
)


# ----------------------------------------------------------------------
# Activity
# ----------------------------------------------------------------------

@router.get("/activity", response_class=HTMLResponse)
def activity_page(request: Request, category: str = "", level: str = "") -> HTMLResponse:
    with session_scope() as s:
        q = s.query(ActivityLog)
        if category:
            q = q.filter(ActivityLog.category == category)
        if level:
            q = q.filter(ActivityLog.level == level)
        rows = q.order_by(ActivityLog.created_at.desc()).limit(300).all()
        logs = [
            {
                "id": r.id,
                "category": r.category,
                "level": r.level,
                "message": r.message,
                "created_at": r.created_at,
            }
            for r in rows
        ]
        categories = sorted({r[0] for r in s.query(ActivityLog.category).distinct().all() if r[0]})
        levels = sorted({r[0] for r in s.query(ActivityLog.level).distinct().all() if r[0]})

    return templates.TemplateResponse(request=request, name="activity.html", context=_ctx(
            request,
            active="activity",
            logs=logs,
            categories=categories,
            levels=levels,
            current_category=category,
            current_level=level,
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
    return templates.TemplateResponse(request=request, name="settings.html", context=_ctx(
        request,
        active="settings",
        prefs=prefs,
        bot_cfg=bot_cfg,
        bot_status=bot_status,
        recent_ticks=recent_ticks,
        recent_recons=recent_recons,
        notifier_cfg=notifier_cfg,
        claude_cfg=claude_cfg,
        kill_switch=kill_switch,
        paused_wallets=paused_wallets,
        settings=settings,
    ),
    )


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
) -> RedirectResponse:
    """
    Persist autonomous-bot settings and reload the scheduler so the new
    interval / config takes effect immediately — no restart required.
    """
    # Checkboxes only post their value when checked. Normalize.
    enabled = "true" if str(bot_enabled).lower() in {"on", "true", "1", "yes"} else "false"
    dry = "true" if str(bot_dry_run).lower() in {"on", "true", "1", "yes"} else "false"

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
