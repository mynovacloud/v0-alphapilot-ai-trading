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

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
from config.settings import settings
from connectors.live_prices import get_price as live_price
from connectors.live_prices import known_symbols
from connectors.registry import CONNECTOR_REGISTRY, REAL_AUTH_PLATFORMS, get_connector
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
from trading.market_scanner import scan_markets
from trading.paper_trading_engine import PaperTradingEngine
from trading.strategy_manager import (
    create_strategy,
    delete_strategy,
    list_strategies,
)
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
            platforms=list(CONNECTOR_REGISTRY.keys()),
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


@router.get("/api/price")
def api_live_price(symbol: str) -> dict[str, Any]:
    """Endpoint used by HTMX in templates to look up a live price."""
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
    wallets = get_wallets()
    strategies = list_strategies()
    lessons = _memory.list_lessons(limit=20)
    return templates.TemplateResponse(request=request, name="training.html", context=_ctx(
            request,
            active="training",
            wallets=wallets,
            strategies=strategies,
            lessons=lessons,
            risk_levels=["Conservative", "Moderate", "Aggressive", "Degenerate"],
            market_types=["Crypto", "Stocks", "Prediction Markets"],
        ),
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
    # Build equity curve from decisions
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
    _memory.reset()
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
    return templates.TemplateResponse(request=request, name="settings.html", context=_ctx(
            request,
            active="settings",
            prefs=prefs,
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
