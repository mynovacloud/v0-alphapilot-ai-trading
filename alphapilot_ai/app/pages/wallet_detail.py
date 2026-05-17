"""Wallet Detail page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.charts import equity_curve_chart
from database.db import session_scope
from database.models import ActivityLog, PaperTrade, Position, Wallet
from utils.helpers import fmt_money, fmt_pct, safe_div


def render() -> None:
    st.title("Wallet Detail")

    with session_scope() as s:
        wallets = s.query(Wallet).order_by(Wallet.created_at.asc()).all()
        wallet_options = {f"{w.name} ({w.platform})": w.id for w in wallets}

    if not wallet_options:
        st.info("No wallets yet. Add one from **Add Wallet**.")
        return

    default_label = next(iter(wallet_options))
    selected_id = st.session_state.get("_selected_wallet")
    default_idx = 0
    if selected_id is not None:
        for i, (lbl, wid) in enumerate(wallet_options.items()):
            if wid == selected_id:
                default_idx = i
                default_label = lbl
                break

    selected_label = st.selectbox("Wallet", list(wallet_options.keys()), index=default_idx)
    wallet_id = wallet_options[selected_label]

    with session_scope() as s:
        w = s.get(Wallet, wallet_id)
        trades = (
            s.query(PaperTrade)
            .filter(PaperTrade.wallet_id == wallet_id)
            .order_by(PaperTrade.opened_at.desc())
            .all()
        )
        positions = s.query(Position).filter(Position.wallet_id == wallet_id).all()
        logs = (
            s.query(ActivityLog)
            .filter(ActivityLog.wallet_id == wallet_id)
            .order_by(ActivityLog.created_at.desc())
            .limit(50)
            .all()
        )
        # Detach to dicts so we can use after session closes
        trade_rows = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "market_type": t.market_type,
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
        position_rows = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry": p.avg_entry,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]
        log_rows = [
            {
                "time": l.created_at.strftime("%Y-%m-%d %H:%M"),
                "category": l.category,
                "level": l.level,
                "message": l.message,
            }
            for l in logs
        ]
        wallet_info = {
            "name": w.name,
            "platform": w.platform,
            "paper_balance": w.paper_balance,
            "risk_profile": w.risk_profile,
            "connection_status": w.connection_status,
            "api_status": w.api_status,
            "last_synced": w.last_synced,
        }

    # Header card
    st.markdown(
        f"""
        <div class="card">
            <h3 style="margin:0;">{wallet_info['name']}</h3>
            <span class="badge blue">{wallet_info['platform']}</span>
            <span class="badge green">{wallet_info['connection_status']}</span>
            <span class="badge amber">LIVE LOCKED</span>
            <span class="badge gray">Risk: {wallet_info['risk_profile']}</span>
            <div class="small" style="margin-top:8px;">Last synced {wallet_info['last_synced'].strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    closed = [t for t in trade_rows if t["status"] == "closed"]
    wins = sum(1 for t in closed if t["realized_pnl"] > 0)
    pnl = sum(t["realized_pnl"] for t in closed)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper Balance", fmt_money(wallet_info["paper_balance"]))
    c2.metric("Total P&L", fmt_money(pnl))
    c3.metric("Win Rate", fmt_pct(safe_div(wins, len(closed), 0.0)))
    c4.metric("Open Trades", sum(1 for t in trade_rows if t["status"] == "open"))

    st.divider()

    # Equity curve from trades
    if closed:
        df_closed = pd.DataFrame(closed).sort_values("closed_at")
        df_closed["date"] = df_closed["closed_at"].apply(
            lambda d: d.replace(tzinfo=None).date() if d else None
        )
        eq = df_closed.groupby("date")["realized_pnl"].sum().cumsum().reset_index()
        eq.columns = ["date", "equity"]
        st.plotly_chart(equity_curve_chart(eq), width="stretch")

    tabs = st.tabs(["Open Trades", "Closed Trades", "Positions", "Activity Logs"])

    open_df = pd.DataFrame([t for t in trade_rows if t["status"] == "open"])
    closed_df = pd.DataFrame([t for t in trade_rows if t["status"] == "closed"])
    positions_df = pd.DataFrame(position_rows)
    logs_df = pd.DataFrame(log_rows)

    with tabs[0]:
        if open_df.empty:
            st.info("No open trades.")
        else:
            st.dataframe(open_df, width="stretch", hide_index=True)
    with tabs[1]:
        if closed_df.empty:
            st.info("No closed trades.")
        else:
            st.dataframe(closed_df, width="stretch", hide_index=True)
    with tabs[2]:
        if positions_df.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions_df, width="stretch", hide_index=True)
    with tabs[3]:
        if logs_df.empty:
            st.info("No activity yet.")
        else:
            st.dataframe(logs_df, width="stretch", hide_index=True)
