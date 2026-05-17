"""Analytics page."""
from __future__ import annotations

import pandas as pd
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
from utils.helpers import fmt_money, fmt_pct


def render() -> None:
    st.title("Analytics")
    st.caption("Deep portfolio, strategy, and AI performance analytics.")

    summary = portfolio_summary()
    perf = performance_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit Factor", perf["profit_factor"])
    c2.metric("Sharpe (placeholder)", perf["sharpe_placeholder"])
    c3.metric("Max Drawdown", fmt_pct(perf["max_drawdown"]))
    c4.metric("Avg R:R", perf["avg_rr"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Biggest Win", fmt_money(perf["biggest_win"]))
    c2.metric("Biggest Loss", fmt_money(perf["biggest_loss"]))
    c3.metric("Max Consec. Wins", perf["max_consecutive_wins"])
    c4.metric("Max Consec. Losses", perf["max_consecutive_losses"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Trade Duration (h)", perf["avg_trade_duration_hours"])
    c2.metric("Win Rate", fmt_pct(summary["win_rate"]))
    c3.metric("Total P&L", fmt_money(summary["total_pnl"]))
    c4.metric("Active Wallets", summary["active_wallets"])

    st.divider()

    eq = equity_curve_df()
    trades = get_all_trades_df()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(equity_curve_chart(eq), width="stretch")
    with c2:
        st.plotly_chart(win_loss_donut(summary["wins"], summary["losses"]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(pnl_bar(pnl_by_wallet(), "wallet", "P&L by Wallet"), width="stretch")
    with c2:
        st.plotly_chart(pnl_bar(pnl_by_strategy(), "strategy", "P&L by Strategy"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(drawdown_chart(eq), width="stretch")
    with c2:
        st.plotly_chart(confidence_over_time(trades), width="stretch")

    st.plotly_chart(trade_volume_chart(trades), width="stretch")

    st.divider()
    st.markdown("### Market Type Comparison")
    if not trades.empty:
        closed = trades[trades["status"] == "closed"]
        if not closed.empty:
            agg = (
                closed.groupby("market_type")
                .agg(
                    trades=("id", "count"),
                    pnl=("realized_pnl", "sum"),
                    win_rate=("realized_pnl", lambda x: (x > 0).mean()),
                )
                .reset_index()
            )
            agg["win_rate"] = agg["win_rate"].apply(lambda v: f"{v:.0%}")
            agg["pnl"] = agg["pnl"].round(2)
            st.dataframe(agg, width="stretch", hide_index=True)
        else:
            st.info("No closed trades to compare.")
