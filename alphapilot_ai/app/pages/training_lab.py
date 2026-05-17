"""AI Training Lab page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.ai_engine import AIEngine
from ai.learning_memory import LearningMemory
from analytics.charts import _apply_theme  # noqa
from database.db import session_scope
from database.models import AITrainingSession, Strategy, Wallet
from utils.constants import MARKET_TYPES, RISK_LEVELS
from utils.helpers import fmt_money, fmt_pct


_engine = AIEngine()
_memory = LearningMemory()


def _equity_chart(decisions: list[dict], starting: float) -> go.Figure:
    eq = [starting]
    bal = starting
    for d in decisions:
        bal += d.get("pnl", 0.0)
        eq.append(bal)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(eq))),
            y=eq,
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.12)",
        )
    )
    return _apply_theme(fig, "Training Equity Curve")


def render() -> None:
    st.title("AI Training Lab")
    st.caption(
        "Train the AI engine on mock market data with paper cash. "
        "All trades here are simulated."
    )

    with session_scope() as s:
        wallets = {f"{w.name} ({w.platform})": w.id for w in s.query(Wallet).all()}
        strategies = {st_.name: st_.id for st_ in s.query(Strategy).all()}

    with st.form("training_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            wallet_label = st.selectbox(
                "Wallet (optional)", ["—"] + list(wallets.keys())
            )
            strat_label = st.selectbox(
                "Strategy (optional)", ["—"] + list(strategies.keys())
            )
            market_type = st.selectbox("Market", MARKET_TYPES, index=0)
        with c2:
            risk_level = st.selectbox("Risk level", RISK_LEVELS, index=1)
            num_trades = st.slider("Number of simulated trades", 10, 500, 100, step=10)
            speed = st.selectbox("Training speed", ["fast", "normal", "slow"], index=0)
        with c3:
            starting_balance = st.number_input(
                "Starting paper balance ($)", min_value=100.0, value=10_000.0, step=500.0
            )
            st.caption("Live trading is locked by default — this is paper-only.")
            run = st.form_submit_button("▶️ Start Training Session")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 Reset AI Memory"):
            n = _memory.reset()
            st.warning(f"Cleared {n} stored lessons.")
    with c2:
        st.caption("Stop training: training runs synchronously today; close the browser to abort.")

    if run:
        wallet_id = wallets.get(wallet_label) if wallet_label != "—" else None
        strategy_id = strategies.get(strat_label) if strat_label != "—" else None
        with st.spinner("Running AI training session..."):
            result = _engine.run_training_session(
                wallet_id=wallet_id,
                strategy_id=strategy_id,
                market_type=market_type,
                risk_level=risk_level,
                num_trades=int(num_trades),
                starting_balance=float(starting_balance),
            )
        st.success("Training complete.")
        st.session_state["_last_training"] = result.__dict__

    last = st.session_state.get("_last_training")
    if last:
        st.divider()
        st.markdown("### Last Training Session")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Starting", fmt_money(last["starting_balance"]))
        c2.metric("Ending", fmt_money(last["ending_balance"]),
                  delta=fmt_money(last["pnl"]))
        c3.metric("Win Rate", fmt_pct(last["win_rate"]))
        c4.metric("Avg Confidence", fmt_pct(last["avg_confidence"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", last["trades"])
        c2.metric("Wins", last["wins"])
        c3.metric("Max Drawdown", fmt_pct(last["max_drawdown"]))

        st.plotly_chart(
            _equity_chart(last["decisions"], last["starting_balance"]),
            width="stretch",
        )

        st.markdown("#### AI Decision Log")
        if last["decisions"]:
            df = pd.DataFrame(last["decisions"])
            cols = [c for c in ["i", "platform", "symbol", "side", "confidence", "reasoning", "pnl", "balance"] if c in df.columns]
            st.dataframe(df[cols], width="stretch", hide_index=True)

        st.markdown("#### Lessons Learned This Session")
        lessons = last["lessons"]
        if not lessons:
            st.info("No new lessons logged.")
        for l in lessons:
            st.markdown(
                f'<div class="card"><span class="badge amber">LESSON</span> {l}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Persistent AI Learning Memory")
    mem = _memory.list_lessons(limit=50)
    if not mem:
        st.info("No lessons stored yet — run a training session to populate memory.")
    else:
        for m in mem:
            st.markdown(
                f'<div class="card"><span class="badge blue">{m["category"].upper()}</span>'
                f' <span class="small">weight {m["weight"]:.2f} · {m["created_at"].strftime("%Y-%m-%d %H:%M")}</span><br/>{m["content"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Recent Training Sessions")
    with session_scope() as s:
        sess = (
            s.query(AITrainingSession)
            .order_by(AITrainingSession.started_at.desc())
            .limit(10)
            .all()
        )
        rows = [
            {
                "id": r.id,
                "market": r.market_type,
                "risk": r.risk_level,
                "trades": r.trades_simulated,
                "wins": r.wins,
                "losses": r.losses,
                "start": round(r.starting_balance, 2),
                "end": round(r.ending_balance, 2),
                "pnl": round(r.ending_balance - r.starting_balance, 2),
                "drawdown": round(r.max_drawdown, 4),
                "avg_conf": round(r.avg_confidence, 3),
                "status": r.status,
                "started_at": r.started_at,
            }
            for r in sess
        ]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No training sessions yet.")
