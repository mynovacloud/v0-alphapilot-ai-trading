"""Settings page — account, trading, wallet, AI, security, database."""
from __future__ import annotations

import streamlit as st

from ai.learning_memory import LearningMemory
from config.settings import settings as app_settings
from database.db import reset_db, session_scope
from database.models import ActivityLog, AppSetting, PaperTrade
from database.seed import seed_if_empty


def _get_setting(key: str, default: str = "") -> str:
    with session_scope() as s:
        row = s.query(AppSetting).filter(AppSetting.key == key).first()
        return row.value if row else default


def _set_setting(key: str, value: str) -> None:
    with session_scope() as s:
        row = s.query(AppSetting).filter(AppSetting.key == key).first()
        if row:
            row.value = value
        else:
            s.add(AppSetting(key=key, value=value))


def render() -> None:
    st.title("Settings")
    st.caption("Account, trading rules, AI behavior, security, and database controls.")

    tabs = st.tabs(["Account", "Trading", "AI", "Security", "Database"])

    # --- Account ---
    with tabs[0]:
        user_name = st.text_input("Display name", value=_get_setting("user_name", "Trader"))
        theme = st.selectbox(
            "Theme", ["Dark", "Light"], index=0 if _get_setting("theme", "Dark") == "Dark" else 1
        )
        landing = st.selectbox(
            "Default landing page",
            ["Dashboard", "Wallets", "Market Scanner", "AI Training Lab"],
            index=0,
        )
        notif = st.toggle("Enable notifications", value=_get_setting("notif", "true") == "true")
        if st.button("Save Account Settings"):
            _set_setting("user_name", user_name)
            _set_setting("theme", theme)
            _set_setting("landing", landing)
            _set_setting("notif", "true" if notif else "false")
            st.success("Account settings saved.")

    # --- Trading ---
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            default_balance = st.number_input(
                "Default paper balance ($)",
                value=float(_get_setting("default_paper_balance", "10000") or 10000),
                step=500.0,
            )
            max_daily_loss = st.number_input(
                "Max daily loss ($)",
                value=float(_get_setting("max_daily_loss", "500") or 500),
                step=100.0,
            )
            max_position = st.number_input(
                "Max position size ($)",
                value=float(_get_setting("max_position", "1000") or 1000),
                step=100.0,
            )
        with c2:
            max_open = st.number_input(
                "Max open trades",
                value=int(_get_setting("max_open", "5") or 5),
                step=1,
            )
            require_approval = st.toggle("Require manual approval before live trades", value=True)
            paper_default = st.toggle("Paper trading default enabled", value=True)
        if st.button("Save Trading Settings"):
            _set_setting("default_paper_balance", str(default_balance))
            _set_setting("max_daily_loss", str(max_daily_loss))
            _set_setting("max_position", str(max_position))
            _set_setting("max_open", str(int(max_open)))
            st.success("Trading settings saved.")

        st.markdown("---")
        st.error("⛔ Live trading is **LOCKED** by default at the framework level.")
        st.caption(
            "To enable live trading, set LIVE_TRADING_ENABLED=true in your `.env`, "
            "implement real `place_live_trade` per connector, then review every risk control."
        )
        if st.button("🚨 Emergency Stop (close all open paper trades)"):
            n = 0
            with session_scope() as s:
                for t in s.query(PaperTrade).filter(PaperTrade.status == "open").all():
                    t.status = "cancelled"
                    n += 1
                s.add(
                    ActivityLog(
                        category="risk",
                        level="warn",
                        message=f"Emergency stop triggered — {n} open paper trades cancelled.",
                    )
                )
            st.warning(f"Emergency stop: cancelled {n} open paper trades.")

    # --- AI ---
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            ai_training = st.toggle("AI training mode enabled", value=True)
            learn_aggr = st.slider("Learning aggressiveness", 0.0, 1.0, 0.5, 0.05)
            risk_tol = st.slider("Risk tolerance", 0.0, 1.0, 0.4, 0.05)
        with c2:
            memory_retention = st.slider("Memory retention (lessons)", 10, 5000, 500, step=10)
            learn_losses = st.toggle("Allow AI to learn from losses", value=True)
            learn_missed = st.toggle("Allow AI to learn from missed trades", value=True)
            adjust_weights = st.toggle("Allow AI to adjust strategy weights", value=True)

        c1, c2, c3 = st.columns(3)
        if c1.button("Reset AI Memory"):
            n = LearningMemory().reset()
            st.warning(f"Cleared {n} lessons.")
        if c2.button("Export AI Memory"):
            data = LearningMemory().export()
            st.download_button(
                "Download lessons.json",
                data=str(data),
                file_name="alphapilot_lessons.json",
                mime="application/json",
            )
        c3.caption("Import is a future placeholder.")

    # --- Security ---
    with tabs[3]:
        st.toggle("Lock live trading", value=not app_settings.live_trading_enabled, disabled=True)
        st.toggle("Require confirmation before any trade", value=True)
        st.text("API key encryption: placeholder (planned)")
        st.text("Session timeout: placeholder (planned)")
        st.toggle("Local-only mode (no outbound calls)", value=True)
        retention = st.slider("Activity log retention (days)", 7, 365, 90)
        if st.button("Save Security Settings"):
            _set_setting("log_retention_days", str(retention))
            st.success("Saved.")

    # --- Database ---
    with tabs[4]:
        st.markdown(f"**Database URL:** `{app_settings.database_url}`")
        with session_scope() as s:
            from database.models import (
                ActivityLog as AL,
                PaperTrade as PT,
                Wallet as W,
                Strategy as ST,
                AILearningMemory as AM,
            )
            counts = {
                "Wallets": s.query(W).count(),
                "Paper trades": s.query(PT).count(),
                "Strategies": s.query(ST).count(),
                "Activity logs": s.query(AL).count(),
                "AI memory": s.query(AM).count(),
            }
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (k, v) in zip([c1, c2, c3, c4, c5], counts.items()):
            col.metric(k, v)

        st.markdown("---")
        if st.button("🔄 Re-seed mock data (only if empty)"):
            seed_if_empty()
            st.success("Seed run.")
        if st.button("🗑️ Reset Paper Trades"):
            with session_scope() as s:
                n = s.query(PaperTrade).delete()
                s.add(ActivityLog(category="system", level="warn", message=f"Cleared {n} paper trades."))
            st.warning(f"Deleted {n} paper trades.")
        if st.button("🧨 Reset ENTIRE Database (drops all tables)"):
            reset_db()
            seed_if_empty()
            st.error("Database reset and re-seeded.")
