"""Wallets list page."""
from __future__ import annotations

import streamlit as st

from database.db import session_scope
from database.models import PaperTrade, Wallet
from utils.helpers import fmt_money, fmt_pct, safe_div, utcnow


def _wallet_stats(wallet_id: int) -> dict:
    with session_scope() as s:
        trades = s.query(PaperTrade).filter(PaperTrade.wallet_id == wallet_id).all()
    closed = [t for t in trades if t.status == "closed"]
    open_ = [t for t in trades if t.status == "open"]
    wins = sum(1 for t in closed if t.realized_pnl > 0)
    pnl = sum(t.realized_pnl for t in closed)
    return {
        "open": len(open_),
        "closed": len(closed),
        "pnl": pnl,
        "win_rate": safe_div(wins, len(closed), 0.0),
    }


def render() -> None:
    st.title("Wallets")
    st.caption("Each wallet represents a connected trading platform / broker account.")

    if st.button("➕ Add Wallet"):
        st.session_state["_nav_target"] = "Add Wallet"
        st.info("Use the sidebar to switch to **Add Wallet**.")

    with session_scope() as s:
        wallets = s.query(Wallet).order_by(Wallet.created_at.asc()).all()

    if not wallets:
        st.info("No wallets yet. Click **Add Wallet** in the sidebar to create one.")
        return

    cols_per_row = 3
    rows = [wallets[i : i + cols_per_row] for i in range(0, len(wallets), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, w in zip(cols, row):
            stats = _wallet_stats(w.id)
            with col:
                pnl_color = "green" if stats["pnl"] >= 0 else "red"
                st.markdown(
                    f"""
                    <div class="card">
                        <h4 style="margin:0 0 4px 0;">{w.name}</h4>
                        <span class="badge blue">{w.platform}</span>
                        <span class="badge green">{w.connection_status}</span>
                        <span class="badge amber">LIVE LOCKED</span>
                        <hr style="border-color:#1f2937;margin:10px 0;"/>
                        <div class="small">Paper Balance</div>
                        <div style="font-size:22px;font-weight:700;">{fmt_money(w.paper_balance)}</div>
                        <div class="small" style="margin-top:8px;">Total P&L</div>
                        <div style="font-size:18px;font-weight:600;color:{'#22c55e' if stats['pnl']>=0 else '#ef4444'};">
                            {fmt_money(stats['pnl'])}
                        </div>
                        <div style="display:flex;gap:14px;margin-top:10px;">
                            <div><div class="small">Open</div><b>{stats['open']}</b></div>
                            <div><div class="small">Closed</div><b>{stats['closed']}</b></div>
                            <div><div class="small">Win rate</div><b>{fmt_pct(stats['win_rate'])}</b></div>
                            <div><div class="small">Risk</div><b>{w.risk_profile}</b></div>
                        </div>
                        <div class="small" style="margin-top:8px;">
                            Last synced {w.last_synced.strftime('%Y-%m-%d %H:%M')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                bcol1, bcol2, bcol3 = st.columns(3)
                if bcol1.button("View", key=f"v{w.id}"):
                    st.session_state["_selected_wallet"] = w.id
                    st.info(f"Selected wallet #{w.id}. Switch to **Wallet Detail** in the sidebar.")
                if bcol2.button("Sync", key=f"s{w.id}"):
                    with session_scope() as s2:
                        ww = s2.get(Wallet, w.id)
                        if ww:
                            ww.last_synced = utcnow()
                    st.success("Mock sync complete.")
                if bcol3.button("Remove", key=f"r{w.id}"):
                    with session_scope() as s2:
                        ww = s2.get(Wallet, w.id)
                        if ww:
                            s2.delete(ww)
                    st.warning(f"Wallet '{w.name}' removed.")
                    st.rerun()
