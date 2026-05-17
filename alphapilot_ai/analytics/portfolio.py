"""Portfolio-level analytics computed from SQLite state."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from database.db import session_scope
from database.models import PaperTrade, Position, Strategy, Wallet
from utils.helpers import safe_div, utcnow


def _trade_to_row(t: PaperTrade) -> dict[str, Any]:
    return {
        "id": t.id,
        "wallet_id": t.wallet_id,
        "strategy_id": t.strategy_id,
        "symbol": t.symbol,
        "market_type": t.market_type,
        "side": t.side,
        "qty": t.qty,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "fees": t.fees,
        "slippage": t.slippage,
        "realized_pnl": t.realized_pnl,
        "unrealized_pnl": t.unrealized_pnl,
        "confidence": t.confidence,
        "status": t.status,
        "opened_at": t.opened_at,
        "closed_at": t.closed_at,
    }


def get_all_trades_df() -> pd.DataFrame:
    with session_scope() as s:
        rows = [_trade_to_row(t) for t in s.query(PaperTrade).all()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "id", "wallet_id", "strategy_id", "symbol", "market_type",
                "side", "qty", "entry_price", "exit_price", "fees", "slippage",
                "realized_pnl", "unrealized_pnl", "confidence", "status",
                "opened_at", "closed_at",
            ]
        )
    return pd.DataFrame(rows)


def get_wallets() -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.query(Wallet).order_by(Wallet.created_at.asc()).all()
        return [
            {
                "id": w.id,
                "name": w.name,
                "platform": w.platform,
                "paper_balance": w.paper_balance,
                "real_balance_placeholder": w.real_balance_placeholder,
                "risk_profile": w.risk_profile,
                "sandbox_mode": w.sandbox_mode,
                "paper_trading_only": w.paper_trading_only,
                # Authoritative mode flag used across the UI ("paper" | "live" | "live_shadow").
                "trading_mode": (w.trading_mode or "paper"),
                "connection_status": w.connection_status,
                "api_status": w.api_status,
                "last_synced": w.last_synced,
                "created_at": w.created_at,
            }
            for w in rows
        ]


def _wallets_in_mode(mode: str) -> set[int]:
    """Return wallet ids that match a mode filter.

    `mode` is one of: "all", "paper", "live".
    "live" includes both `live` and `live_shadow` wallets so users see real
    money activity grouped together. "paper" only includes pure paper wallets.
    """
    mode = (mode or "all").lower()
    ids: set[int] = set()
    for w in get_wallets():
        wm = (w.get("trading_mode") or "paper").lower()
        if mode == "all":
            ids.add(w["id"])
        elif mode == "paper" and wm == "paper":
            ids.add(w["id"])
        elif mode == "live" and wm in {"live", "live_shadow"}:
            ids.add(w["id"])
    return ids


def portfolio_summary(mode: str = "all") -> dict[str, Any]:
    """Aggregate portfolio metrics, optionally scoped to one trading mode.

    `mode` is "all" (default), "paper", or "live". The returned dict always
    includes a `by_mode` breakdown so the UI can show paper vs. live side-by-side.
    """
    wallets_all = get_wallets()
    wallet_mode_map = {w["id"]: (w.get("trading_mode") or "paper").lower() for w in wallets_all}
    selected_ids = _wallets_in_mode(mode)
    wallets = [w for w in wallets_all if w["id"] in selected_ids]

    trades_df = get_all_trades_df()
    if not trades_df.empty:
        trades_df = trades_df[trades_df["wallet_id"].isin(selected_ids)]
    closed = trades_df[trades_df["status"] == "closed"] if not trades_df.empty else trades_df
    open_ = trades_df[trades_df["status"] == "open"] if not trades_df.empty else trades_df

    total_paper_value = sum(w["paper_balance"] for w in wallets)
    total_pnl = float(closed["realized_pnl"].sum()) if not closed.empty else 0.0
    unrealized = float(open_["unrealized_pnl"].sum()) if not open_.empty else 0.0

    now = utcnow()
    def pnl_since(days: int) -> float:
        if closed.empty:
            return 0.0
        cutoff = now - timedelta(days=days)
        # Make tz-naive comparison safe
        recent = closed[closed["closed_at"].apply(lambda d: d is not None and d.replace(tzinfo=None) >= cutoff.replace(tzinfo=None))]
        return float(recent["realized_pnl"].sum())

    daily = pnl_since(1)
    weekly = pnl_since(7)
    monthly = pnl_since(30)
    ytd_cutoff = datetime(now.year, 1, 1)
    ytd = (
        float(
            closed[
                closed["closed_at"].apply(
                    lambda d: d is not None and d.replace(tzinfo=None) >= ytd_cutoff
                )
            ]["realized_pnl"].sum()
        )
        if not closed.empty
        else 0.0
    )

    wins = int((closed["realized_pnl"] > 0).sum()) if not closed.empty else 0
    losses = int((closed["realized_pnl"] <= 0).sum()) if not closed.empty else 0
    total_trades = int(len(closed))
    win_rate = safe_div(wins, total_trades, 0.0)

    # Best/worst wallet
    best_wallet = worst_wallet = None
    if not closed.empty and wallets:
        agg = closed.groupby("wallet_id")["realized_pnl"].sum()
        if not agg.empty:
            best_id = int(agg.idxmax())
            worst_id = int(agg.idxmin())
            id_to_name = {w["id"]: f"{w['name']} ({w['platform']})" for w in wallets}
            best_wallet = {"name": id_to_name.get(best_id, "?"), "pnl": float(agg.max())}
            worst_wallet = {"name": id_to_name.get(worst_id, "?"), "pnl": float(agg.min())}

    # Best/worst strategy
    best_strat = worst_strat = None
    with session_scope() as s:
        strat_map = {st.id: st.name for st in s.query(Strategy).all()}
    if not closed.empty and strat_map:
        agg = closed.groupby("strategy_id")["realized_pnl"].sum()
        agg = agg[agg.index.notnull()]
        if not agg.empty:
            best_id = int(agg.idxmax())
            worst_id = int(agg.idxmin())
            best_strat = {"name": strat_map.get(best_id, "?"), "pnl": float(agg.max())}
            worst_strat = {"name": strat_map.get(worst_id, "?"), "pnl": float(agg.min())}

    # AI confidence (avg of recent closed)
    avg_conf = float(closed["confidence"].mean()) if not closed.empty else 0.0

    # Drawdown on equity curve (cumulative PnL)
    drawdown = 0.0
    if not closed.empty:
        sorted_closed = closed.sort_values("closed_at")
        equity = sorted_closed["realized_pnl"].cumsum().values
        peak = np.maximum.accumulate(equity)
        denom = np.maximum(np.abs(peak), 1)
        drawdown = float(((peak - equity) / denom).max())

    # Breakdown of paper vs live (regardless of selected `mode`) so the
    # Dashboard can show both columns side-by-side without a second query.
    by_mode: dict[str, dict[str, float]] = {
        "paper": {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "open": 0, "closed": 0,
                  "wins": 0, "losses": 0, "balance": 0.0, "wallets": 0},
        "live":  {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "open": 0, "closed": 0,
                  "wins": 0, "losses": 0, "balance": 0.0, "wallets": 0},
    }
    full_df = get_all_trades_df()
    for w in wallets_all:
        bucket = "live" if (w.get("trading_mode") or "paper").lower() in {"live", "live_shadow"} else "paper"
        by_mode[bucket]["wallets"] += 1
        # For "live" wallets the canonical bankroll lives in real_balance_placeholder.
        if bucket == "live":
            by_mode[bucket]["balance"] += float(w.get("real_balance_placeholder") or 0)
        else:
            by_mode[bucket]["balance"] += float(w.get("paper_balance") or 0)
    if not full_df.empty:
        full_df = full_df.assign(
            _mode=full_df["wallet_id"].map(
                lambda wid: "live" if wallet_mode_map.get(wid, "paper") in {"live", "live_shadow"} else "paper"
            )
        )
        for bucket in ("paper", "live"):
            sub = full_df[full_df["_mode"] == bucket]
            sub_closed = sub[sub["status"] == "closed"]
            sub_open = sub[sub["status"] == "open"]
            by_mode[bucket]["realized_pnl"] = round(float(sub_closed["realized_pnl"].sum()) if not sub_closed.empty else 0.0, 2)
            by_mode[bucket]["unrealized_pnl"] = round(float(sub_open["unrealized_pnl"].sum()) if not sub_open.empty else 0.0, 2)
            by_mode[bucket]["open"] = int(len(sub_open))
            by_mode[bucket]["closed"] = int(len(sub_closed))
            by_mode[bucket]["wins"] = int((sub_closed["realized_pnl"] > 0).sum()) if not sub_closed.empty else 0
            by_mode[bucket]["losses"] = int((sub_closed["realized_pnl"] <= 0).sum()) if not sub_closed.empty else 0
            by_mode[bucket]["balance"] = round(by_mode[bucket]["balance"], 2)

    return {
        "mode": mode,
        "by_mode": by_mode,
        "total_paper_value": round(total_paper_value, 2),
        "total_portfolio_value": round(total_paper_value + unrealized, 2),
        "total_pnl": round(total_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "daily_pnl": round(daily, 2),
        "weekly_pnl": round(weekly, 2),
        "monthly_pnl": round(monthly, 2),
        "ytd_pnl": round(ytd, 2),
        "win_rate": round(win_rate, 3),
        "loss_rate": round(1 - win_rate, 3) if total_trades else 0.0,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "open_positions": int(len(open_)),
        "closed_positions": int(len(closed)),
        "active_wallets": len(wallets),
        "best_wallet": best_wallet,
        "worst_wallet": worst_wallet,
        "best_strategy": best_strat,
        "worst_strategy": worst_strat,
        "ai_confidence": round(avg_conf, 3),
        "drawdown": round(drawdown, 4),
        "risk_exposure": round(min(1.0, sum(p.get("unrealized_pnl", 0) for p in []) / max(1, total_paper_value)), 4),
    }


def equity_curve_df(mode: str = "all") -> pd.DataFrame:
    """Return a date-indexed cumulative-PnL series across closed paper trades, optionally scoped by mode."""
    df = get_all_trades_df()
    if df.empty:
        return pd.DataFrame(columns=["date", "equity"])
    selected = _wallets_in_mode(mode)
    df = df[df["wallet_id"].isin(selected)]
    closed = df[(df["status"] == "closed") & df["closed_at"].notna()].copy()
    if closed.empty:
        return pd.DataFrame(columns=["date", "equity"])
    closed["date"] = closed["closed_at"].apply(lambda d: d.replace(tzinfo=None).date())
    daily = closed.groupby("date")["realized_pnl"].sum().sort_index()
    equity = daily.cumsum()
    return pd.DataFrame({"date": equity.index, "equity": equity.values})


def pnl_by_wallet() -> pd.DataFrame:
    df = get_all_trades_df()
    if df.empty:
        return pd.DataFrame(columns=["wallet", "pnl"])
    closed = df[df["status"] == "closed"]
    wallets = {w["id"]: f"{w['name']} ({w['platform']})" for w in get_wallets()}
    if closed.empty:
        return pd.DataFrame(columns=["wallet", "pnl"])
    agg = closed.groupby("wallet_id")["realized_pnl"].sum().reset_index()
    agg["wallet"] = agg["wallet_id"].map(wallets).fillna("?")
    return agg[["wallet", "realized_pnl"]].rename(columns={"realized_pnl": "pnl"})


def pnl_by_strategy() -> pd.DataFrame:
    df = get_all_trades_df()
    if df.empty:
        return pd.DataFrame(columns=["strategy", "pnl"])
    closed = df[df["status"] == "closed"]
    with session_scope() as s:
        strat_map = {st.id: st.name for st in s.query(Strategy).all()}
    if closed.empty:
        return pd.DataFrame(columns=["strategy", "pnl"])
    agg = closed.groupby("strategy_id")["realized_pnl"].sum().reset_index()
    agg["strategy"] = agg["strategy_id"].map(strat_map).fillna("Unassigned")
    return agg[["strategy", "realized_pnl"]].rename(columns={"realized_pnl": "pnl"})
