"""Plotly chart factories — all return go.Figure objects ready for Streamlit."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# AlphaPilot dark fintech palette
DARK_BG = "#0b0f17"
GRID = "#1f2937"
GREEN = "#22c55e"
RED = "#ef4444"
ACCENT = "#38bdf8"
TEXT = "#e5e7eb"


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _apply_theme(fig, "Portfolio Growth")
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            name="Cumulative P&L",
        )
    )
    return _apply_theme(fig, "Portfolio Growth (Cumulative P&L)")


def pnl_bar(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _apply_theme(fig, title)
    colors = [GREEN if v >= 0 else RED for v in df["pnl"]]
    fig.add_trace(go.Bar(x=df[label_col], y=df["pnl"], marker_color=colors))
    return _apply_theme(fig, title)


def win_loss_donut(wins: int, losses: int) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Wins", "Losses"],
                values=[wins, losses],
                hole=0.6,
                marker=dict(colors=[GREEN, RED]),
                textinfo="label+percent",
            )
        ]
    )
    return _apply_theme(fig, "Win / Loss Ratio")


def confidence_over_time(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "confidence" not in df.columns:
        return _apply_theme(fig, "AI Confidence Over Time")
    sub = df.dropna(subset=["closed_at"]).sort_values("closed_at")
    if sub.empty:
        return _apply_theme(fig, "AI Confidence Over Time")
    fig.add_trace(
        go.Scatter(
            x=sub["closed_at"],
            y=sub["confidence"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=5, color=ACCENT),
        )
    )
    return _apply_theme(fig, "AI Confidence Over Time")


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """df expected to have 'date' and 'equity' columns."""
    fig = go.Figure()
    if df.empty:
        return _apply_theme(fig, "Drawdown Over Time")
    eq = df["equity"].values
    peak = []
    cur = float("-inf")
    for v in eq:
        cur = max(cur, v)
        peak.append(cur)
    import numpy as np

    peak_arr = np.array(peak)
    dd = (peak_arr - eq) / np.maximum(np.abs(peak_arr), 1)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=-dd * 100,
            mode="lines",
            line=dict(color=RED, width=2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.18)",
            name="Drawdown %",
        )
    )
    fig.update_yaxes(title="Drawdown (%)")
    return _apply_theme(fig, "Drawdown Over Time")


def trade_volume_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "closed_at" not in df.columns:
        return _apply_theme(fig, "Trade Volume Over Time")
    sub = df.dropna(subset=["closed_at"]).copy()
    if sub.empty:
        return _apply_theme(fig, "Trade Volume Over Time")
    sub["date"] = sub["closed_at"].apply(lambda d: d.replace(tzinfo=None).date())
    counts = sub.groupby("date").size().reset_index(name="trades")
    fig.add_trace(go.Bar(x=counts["date"], y=counts["trades"], marker_color=ACCENT))
    return _apply_theme(fig, "Trade Volume Over Time")
