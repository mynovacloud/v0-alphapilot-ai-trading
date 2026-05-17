"""Add Wallet page."""
from __future__ import annotations

import streamlit as st

from connectors.registry import get_connector
from database.db import session_scope
from database.models import ActivityLog, ApiCredentialPlaceholder, Wallet
from utils.constants import RISK_LEVELS, SUPPORTED_PLATFORMS, DEFAULT_PAPER_BALANCE


def render() -> None:
    st.title("Add Wallet")
    st.caption(
        "Configure a new trading-platform wallet. API fields are placeholders — "
        "real connections are mocked in this build."
    )

    with st.form("add_wallet_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            platform = st.selectbox("Platform", SUPPORTED_PLATFORMS, index=0)
            name = st.text_input("Wallet name", value=f"My {platform} Wallet")
            risk_profile = st.selectbox("Risk profile", RISK_LEVELS, index=1)
            paper_balance = st.number_input(
                "Starting paper balance ($)",
                min_value=0.0,
                value=DEFAULT_PAPER_BALANCE,
                step=500.0,
            )
        with c2:
            api_key = st.text_input("API key (optional, mocked)", type="password")
            api_secret = st.text_input("API secret (optional, mocked)", type="password")
            api_passphrase = st.text_input(
                "API passphrase (optional, mocked)", type="password"
            )
            account_id = st.text_input("Account ID (optional)")

        c3, c4, c5 = st.columns(3)
        with c3:
            sandbox = st.toggle("Sandbox mode", value=True)
        with c4:
            paper_only = st.toggle("Paper trading only", value=True)
        with c5:
            test_connection = st.form_submit_button("Test Connection (mock)")

        save = st.form_submit_button("Save Wallet")

    if test_connection:
        connector = get_connector(
            platform,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            account_id=account_id,
            sandbox=sandbox,
        )
        result = connector.connect()
        if result.get("ok"):
            st.success(f"Mock connection to {platform} successful (sandbox={sandbox}).")
        else:
            st.error("Mock connection failed.")

    if save:
        with session_scope() as s:
            w = Wallet(
                name=name,
                platform=platform,
                paper_balance=paper_balance,
                risk_profile=risk_profile,
                sandbox_mode=sandbox,
                paper_trading_only=paper_only,
                connection_status="connected (mock)",
                api_status="mock",
            )
            s.add(w)
            s.flush()
            s.add(
                ApiCredentialPlaceholder(
                    wallet_id=w.id,
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                    account_id=account_id,
                )
            )
            s.add(
                ActivityLog(
                    category="wallet",
                    level="info",
                    wallet_id=w.id,
                    message=f"Wallet '{name}' ({platform}) created with paper balance ${paper_balance:.2f}",
                )
            )
        st.success(f"Wallet '{name}' saved. Open the **Wallets** page to view it.")
        st.info(
            "Reminder: real credential storage will be encrypted in a future version. "
            "Today this is mock-only paper trading."
        )
