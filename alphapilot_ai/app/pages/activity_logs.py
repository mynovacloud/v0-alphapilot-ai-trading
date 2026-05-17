"""Activity Logs page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from database.db import session_scope
from database.models import ActivityLog


def render() -> None:
    st.title("Activity Logs")
    st.caption("Every API call, AI decision, paper trade, risk event, and warning.")

    with session_scope() as s:
        rows = (
            s.query(ActivityLog)
            .order_by(ActivityLog.created_at.desc())
            .limit(1000)
            .all()
        )
        data = [
            {
                "time": l.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "category": l.category,
                "level": l.level,
                "wallet_id": l.wallet_id,
                "message": l.message,
            }
            for l in rows
        ]

    if not data:
        st.info("No activity logged yet.")
        return

    df = pd.DataFrame(data)

    c1, c2 = st.columns(2)
    with c1:
        cats = st.multiselect("Category", sorted(df["category"].dropna().unique()))
    with c2:
        levels = st.multiselect("Level", sorted(df["level"].dropna().unique()))

    filt = df.copy()
    if cats:
        filt = filt[filt["category"].isin(cats)]
    if levels:
        filt = filt[filt["level"].isin(levels)]

    st.dataframe(filt, width="stretch", hide_index=True)
    st.caption(f"{len(filt)} of {len(df)} log entries shown.")
