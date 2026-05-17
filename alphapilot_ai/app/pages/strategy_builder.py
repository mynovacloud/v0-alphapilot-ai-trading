"""Strategy Builder page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.charts import _apply_theme
from database.db import session_scope
from database.models import Strategy, Wallet
from trading.backtester import run_backtest
from trading.strategy_manager import create_strategy, delete_strategy, list_strategies
from utils.constants import MARKET_TYPES, RISK_LEVELS, STRATEGY_TYPES


def _equity_curve_chart(equity: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(equity))),
            y=equity,
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.10)",
        )
    )
    return _apply_theme(fig, "Backtest Equity Curve")


def render() -> None:
    st.title("Strategy Builder")
    st.caption("Create, save, and backtest paper-trading strategies.")

    with session_scope() as s:
        wallet_choices = {"—": None}
        for w in s.query(Wallet).all():
            wallet_choices[f"{w.name} ({w.platform})"] = w.id

    st.markdown("### New Strategy")
    with st.form("new_strategy"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name", value="My Strategy")
            stype = st.selectbox("Type", STRATEGY_TYPES, index=0)
            market = st.selectbox("Market", MARKET_TYPES, index=0)
            risk = st.selectbox("Risk", RISK_LEVELS, index=1)
            wallet_label = st.selectbox("Wallet (optional)", list(wallet_choices.keys()))
        with c2:
            max_position = st.number_input("Max position size ($)", value=1000.0, step=100.0)
            max_daily_loss = st.number_input("Max daily loss ($)", value=500.0, step=100.0)
            stop_loss = st.number_input("Stop loss %", value=0.05, step=0.01, format="%.2f")
            take_profit = st.number_input("Take profit %", value=0.10, step=0.01, format="%.2f")
        with c3:
            min_conf = st.number_input("Min AI confidence", value=0.6, step=0.05, format="%.2f")
            max_per_day = st.number_input("Max trades / day", value=20, step=1)
            max_open = st.number_input("Max open trades", value=5, step=1)
            paper_only = st.toggle("Paper trading only", value=True)
            allow_ai = st.toggle("Allow AI adjustments", value=True)
        description = st.text_area("Description", value="")

        save = st.form_submit_button("💾 Save Strategy")

    if save:
        sid = create_strategy(
            {
                "name": name,
                "description": description,
                "wallet_id": wallet_choices.get(wallet_label),
                "market_type": market,
                "strategy_type": stype,
                "max_position_size": max_position,
                "max_daily_loss": max_daily_loss,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
                "min_confidence": min_conf,
                "max_trades_per_day": int(max_per_day),
                "max_open_trades": int(max_open),
                "risk_level": risk,
                "paper_trading_only": paper_only,
                "allow_ai_adjustments": allow_ai,
            }
        )
        st.success(f"Strategy saved (id={sid}).")

    st.divider()
    st.markdown("### Saved Strategies")
    strategies = list_strategies()
    if not strategies:
        st.info("No strategies yet.")
        return

    df = pd.DataFrame(strategies)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("### Backtest a Strategy")
    name_to_id = {s["name"]: s["id"] for s in strategies}
    sel = st.selectbox("Select strategy", list(name_to_id.keys()))
    n = st.slider("Simulated trades", 50, 1000, 200, step=50)
    if st.button("▶️ Run Backtest"):
        with st.spinner("Backtesting..."):
            result = run_backtest(name_to_id[sel], n_trades=int(n))
        if not result.get("ok"):
            st.error(result.get("reason", "Failed"))
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", result["total_trades"])
        c2.metric("Win Rate", f"{result['win_rate']:.0%}")
        c3.metric("Total P&L", f"${result['total_pnl']:.2f}")
        c4.metric("Drawdown", f"{result['drawdown']:.1%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Win", f"${result['avg_win']:.2f}")
        c2.metric("Avg Loss", f"${result['avg_loss']:.2f}")
        rec = result["recommendation"]
        color = {"safe": "green", "moderate": "blue", "risky": "amber", "avoid": "red"}.get(rec, "gray")
        c3.markdown(
            f'<div style="margin-top:8px;"><span class="badge {color}">{rec.upper()}</span></div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(_equity_curve_chart(result["equity_curve"]), width="stretch")

    st.divider()
    st.markdown("### Delete a Strategy")
    del_name = st.selectbox("Strategy to delete", ["—"] + list(name_to_id.keys()), key="del_strat")
    if del_name != "—" and st.button("🗑️ Delete"):
        ok = delete_strategy(name_to_id[del_name])
        if ok:
            st.warning(f"Strategy '{del_name}' deleted.")
            st.rerun()
        else:
            st.error("Could not delete.")
