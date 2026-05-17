"""
AlphaPilot AI — Streamlit dashboard entry point.

Run via the launcher (`python main.py`) or directly:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable when Streamlit launches this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.pages import (
    activity_logs,
    add_wallet,
    analytics,
    dashboard,
    market_scanner,
    settings_page,
    strategy_builder,
    training_lab,
    wallet_detail,
    wallets,
)
from database.db import init_db
from database.seed import seed_if_empty

# ---------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AlphaPilot AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Make sure DB exists even if user runs streamlit directly
init_db()
seed_if_empty()


# ---------------------------------------------------------------------
# Global dark fintech CSS
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f17;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1623;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] * { color: #e5e7eb !important; }

    h1, h2, h3, h4 { color: #f8fafc !important; letter-spacing: -0.01em; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700; }

    /* DataFrame container */
    div[data-testid="stDataFrame"] {
        background: #0f1623;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #1e293b, #0f172a);
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
    }

    /* Custom badges via markdown */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge.green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
    .badge.red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
    .badge.blue { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); }
    .badge.amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
    .badge.gray { background: #1f2937; color: #94a3b8; border: 1px solid #334155; }

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .small { color: #94a3b8; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
PAGES = {
    "Dashboard": dashboard.render,
    "Wallets": wallets.render,
    "Add Wallet": add_wallet.render,
    "Wallet Detail": wallet_detail.render,
    "Market Scanner": market_scanner.render,
    "AI Training Lab": training_lab.render,
    "Strategy Builder": strategy_builder.render,
    "Analytics": analytics.render,
    "Activity Logs": activity_logs.render,
    "Settings": settings_page.render,
}

with st.sidebar:
    st.markdown("## AlphaPilot AI")
    st.caption("AI Trading Intelligence — Paper Trading")
    st.markdown(
        '<span class="badge amber">LIVE TRADING LOCKED</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    page_name = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("This is not financial advice. Mock data only.")


# ---------------------------------------------------------------------
# Render selected page
# ---------------------------------------------------------------------
PAGES[page_name]()
