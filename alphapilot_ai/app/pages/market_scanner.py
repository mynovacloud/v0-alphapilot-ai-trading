"""Market Scanner page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trading.market_scanner import scan_markets


def render() -> None:
    st.title("Market Scanner")
    st.caption("Mock scanner across crypto, equities, and prediction markets.")

    c1, c2 = st.columns([1, 4])
    with c1:
        n = st.number_input("# Opportunities", min_value=5, max_value=100, value=25, step=5)
        if st.button("🔍 Scan Markets"):
            st.session_state["_scan_results"] = scan_markets(n=int(n))

    rows = st.session_state.get("_scan_results")
    if not rows:
        st.info("Click **Scan Markets** to generate fresh mock opportunities.")
        return

    df = pd.DataFrame(rows)
    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        platforms = st.multiselect("Platform", sorted(df["platform"].unique()))
    with c2:
        markets = st.multiselect("Market type", sorted(df["market_type"].unique()))
    with c3:
        actions = st.multiselect("Suggested action", sorted(df["suggested_action"].unique()))

    filt = df.copy()
    if platforms:
        filt = filt[filt["platform"].isin(platforms)]
    if markets:
        filt = filt[filt["market_type"].isin(markets)]
    if actions:
        filt = filt[filt["suggested_action"].isin(actions)]

    # Format columns nicely
    show = filt.copy()
    show["edge_pct"] = (show["edge_pct"] * 100).round(2)
    show["ai_probability"] = show["ai_probability"].round(3)
    show["market_probability"] = show["market_probability"].round(3)
    show["confidence"] = show["confidence"].round(2)
    show = show[
        [
            "platform", "symbol", "market_type", "current_price", "fair_value",
            "ai_probability", "market_probability", "edge_pct", "confidence",
            "liquidity", "volatility", "risk_rating", "suggested_action", "reasoning",
        ]
    ]
    st.dataframe(show, width="stretch", hide_index=True)

    # Highlight strong opportunities
    strong = filt[filt["suggested_action"] == "Strong Opportunity"]
    if not strong.empty:
        st.markdown("### Strong Opportunities")
        for _, r in strong.iterrows():
            st.markdown(
                f"""
                <div class="card">
                    <span class="badge green">STRONG</span>
                    <span class="badge blue">{r['platform']}</span>
                    <span class="badge gray">{r['market_type']}</span>
                    <h4 style="margin:6px 0;">{r['symbol']}</h4>
                    <div class="small">{r['reasoning']}</div>
                    <div style="display:flex;gap:18px;margin-top:8px;">
                        <div><div class="small">Edge</div><b>{r['edge_pct']*100:+.2f}%</b></div>
                        <div><div class="small">Confidence</div><b>{r['confidence']:.2f}</b></div>
                        <div><div class="small">Liquidity</div><b>{r['liquidity']:.2f}</b></div>
                        <div><div class="small">Risk</div><b>{r['risk_rating']}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
