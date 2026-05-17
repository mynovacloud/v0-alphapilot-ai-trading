"""Dashboard page — top-level command center."""
from __future__ import annotations

import streamlit as st

from analytics.charts import (
    confidence_over_time,
    drawdown_chart,
    equity_curve_chart,
    pnl_bar,
    trade_volume_chart,
    win_loss_donut,
)
from analytics.performance import performance_metrics
from analytics.portfolio import (
    equity_curve_df,
    get_all_trades_df,
    pnl_by_strategy,
    pnl_by_wallet,
    portfolio_summary,
)
from database.db import session_scope
from database.models import ActivityLog
from utils.helpers import fmt_money, fmt_pct


def _delta_color(v: float) -> str:
    return "normal"


def render() -> None:
    st.title("Dashboard")
    st.caption("AlphaPilot AI command center — paper trading only.")

    summary = portfolio_summary()
    perf = performance_metrics()

    # Top KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", fmt_money(summary["total_portfolio_value"]))
    c2.metric("Paper Cash", fmt_money(summary["total_paper_value"]))
    c3.metric("Total P&L", fmt_money(summary["total_pnl"]),
              delta=fmt_money(summary["unrealized_pnl"]) + " unrealized")
    c4.metric("Win Rate", fmt_pct(summary["win_rate"]))
    c5.metric("AI Confidence", fmt_pct(summary["ai_confidence"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily P&L", fmt_money(summary["daily_pnl"]))
    c2.metric("Weekly P&L", fmt_money(summary["weekly_pnl"]))
    c3.metric("Monthly P&L", fmt_money(summary["monthly_pnl"]))
    c4.metric("YTD P&L", fmt_money(summary["ytd_pnl"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", summary["total_trades"])
    c2.metric("Open Positions", summary["open_positions"])
    c3.metric("Closed Positions", summary["closed_positions"])
    c4.metric("Active Wallets", summary["active_wallets"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drawdown", fmt_pct(summary["drawdown"]))
    c2.metric("Profit Factor", perf["profit_factor"])
    c3.metric("Sharpe (placeholder)", perf["sharpe_placeholder"])
    c4.metric("Avg R:R", perf["avg_rr"])

    st.divider()

    # Best / worst
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Best & Worst Wallets")
        if summary["best_wallet"]:
            st.markdown(
                f'<div class="card"><span class="badge green">BEST</span> {summary["best_wallet"]["name"]} — '
                f'<b>{fmt_money(summary["best_wallet"]["pnl"])}</b></div>',
                unsafe_allow_html=True,
            )
        if summary["worst_wallet"]:
            st.markdown(
                f'<div class="card"><span class="badge red">WORST</span> {summary["worst_wallet"]["name"]} — '
                f'<b>{fmt_money(summary["worst_wallet"]["pnl"])}</b></div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("#### Best & Worst Strategies")
        if summary["best_strategy"]:
            st.markdown(
                f'<div class="card"><span class="badge green">BEST</span> {summary["best_strategy"]["name"]} — '
                f'<b>{fmt_money(summary["best_strategy"]["pnl"])}</b></div>',
                unsafe_allow_html=True,
            )
        if summary["worst_strategy"]:
            st.markdown(
                f'<div class="card"><span class="badge red">WORST</span> {summary["worst_strategy"]["name"]} — '
                f'<b>{fmt_money(summary["worst_strategy"]["pnl"])}</b></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Charts
    eq_df = equity_curve_df()
    trades_df = get_all_trades_df()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(equity_curve_chart(eq_df), width="stretch")
    with c2:
        st.plotly_chart(win_loss_donut(summary["wins"], summary["losses"]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(pnl_bar(pnl_by_wallet(), "wallet", "P&L by Wallet"), width="stretch")
    with c2:
        st.plotly_chart(pnl_bar(pnl_by_strategy(), "strategy", "P&L by Strategy"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(confidence_over_time(trades_df), width="stretch")
    with c2:
        st.plotly_chart(drawdown_chart(eq_df), width="stretch")

    st.plotly_chart(trade_volume_chart(trades_df), width="stretch")

    st.divider()

    # Recent activity
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Recent Paper Trades")
        if not trades_df.empty:
            recent = trades_df.sort_values("opened_at", ascending=False).head(10)[
                ["symbol", "side", "qty", "entry_price", "realized_pnl", "status", "confidence"]
            ]
            st.dataframe(recent, width="stretch", hide_index=True)
        else:
            st.info("No paper trades yet.")
    with c2:
        st.markdown("#### Recent AI Decisions & Warnings")
        with session_scope() as s:
            logs = (
                s.query(ActivityLog)
                .filter(ActivityLog.category.in_(["ai", "risk", "paper_trade", "system"]))
                .order_by(ActivityLog.created_at.desc())
                .limit(15)
                .all()
            )
            for r in logs:
                color = {"warn": "amber", "error": "red", "info": "blue"}.get(r.level, "gray")
                st.markdown(
                    f'<div class="card"><span class="badge {color}">{r.category.upper()}</span>'
                    f'<span class="small"> {r.created_at.strftime("%Y-%m-%d %H:%M")}</span><br/>{r.message}</div>',
                    unsafe_allow_html=True,
                )
