"""
Daily P&L summary — sent once per day to the configured notifier channel.

Pulls realized P&L, win rate, open exposure, and top winners/losers from
the closed paper trades over the last 24h. Designed to be run by the
scheduler at the configured UTC hour. Idempotency is handled by stamping
the date of the last sent summary in AppSetting (`notifier_last_summary_date`).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from config import bot_config
from database.db import session_scope
from database.models import AppSetting, PaperTrade, Wallet
from services.notifier import notify
from utils.helpers import utcnow
from utils.logger import get_logger

logger = get_logger(__name__)

LAST_SENT_KEY = "notifier_last_summary_date"


def _last_sent_date() -> str:
    return bot_config.get(LAST_SENT_KEY) or ""


def _mark_sent(today: str) -> None:
    with session_scope() as s:
        row = s.query(AppSetting).filter(AppSetting.key == LAST_SENT_KEY).first()
        if row:
            row.value = today
            row.updated_at = utcnow()
        else:
            s.add(AppSetting(key=LAST_SENT_KEY, value=today))


def build_summary() -> dict[str, Any]:
    """
    Build the day's P&L summary dict. Always returns — empty when no data.
    """
    end = utcnow()
    start = end - timedelta(hours=24)
    with session_scope() as s:
        closed = (
            s.query(PaperTrade)
            .filter(PaperTrade.status == "closed")
            .filter(PaperTrade.closed_at >= start)
            .all()
        )
        open_trades = s.query(PaperTrade).filter(PaperTrade.status == "open").all()
        wallets = {w.id: w.name for w in s.query(Wallet).all()}

        rows = [
            {
                "id": t.id,
                "wallet": wallets.get(t.wallet_id, "?"),
                "symbol": t.symbol,
                "side": t.side,
                "pnl": float(t.realized_pnl or 0.0),
                "closed_at": t.closed_at,
            }
            for t in closed
        ]
        open_exposure_usd = sum(
            float((t.entry_price or 0) * (t.qty or 0)) for t in open_trades
        )

    total_pnl = sum(r["pnl"] for r in rows)
    wins = [r for r in rows if r["pnl"] > 0]
    losses = [r for r in rows if r["pnl"] < 0]
    n = len(rows)
    win_rate = (len(wins) / n) if n else 0.0

    by_wallet: dict[str, float] = defaultdict(float)
    for r in rows:
        by_wallet[r["wallet"]] += r["pnl"]

    top_win = max(rows, key=lambda r: r["pnl"], default=None)
    top_loss = min(rows, key=lambda r: r["pnl"], default=None)

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "trades_closed": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "open_positions": len(open_trades),
        "open_exposure_usd": open_exposure_usd,
        "by_wallet": dict(by_wallet),
        "top_win": top_win,
        "top_loss": top_loss,
    }


def format_summary(summary: dict[str, Any]) -> str:
    sign = "+" if summary["total_pnl"] >= 0 else "-"
    lines = [
        f"AlphaPilot Daily Summary",
        f"24h P&L: {sign}${abs(summary['total_pnl']):,.2f}  "
        f"({summary['wins']}W / {summary['losses']}L, "
        f"win rate {summary['win_rate']*100:.1f}%, {summary['trades_closed']} trades)",
        f"Open positions: {summary['open_positions']}  "
        f"(${summary['open_exposure_usd']:,.2f} exposure)",
    ]
    if summary["by_wallet"]:
        lines.append("By wallet:")
        for w, pnl in sorted(summary["by_wallet"].items(), key=lambda kv: -kv[1]):
            s = "+" if pnl >= 0 else "-"
            lines.append(f"  • {w}: {s}${abs(pnl):,.2f}")
    if summary.get("top_win"):
        tw = summary["top_win"]
        lines.append(f"Top win: {tw['symbol']} +${tw['pnl']:,.2f} ({tw['wallet']})")
    if summary.get("top_loss") and summary["top_loss"]["pnl"] < 0:
        tl = summary["top_loss"]
        lines.append(f"Top loss: {tl['symbol']} -${abs(tl['pnl']):,.2f} ({tl['wallet']})")
    return "\n".join(lines)


def maybe_send_daily_summary(*, force: bool = False) -> dict[str, Any]:
    """
    Idempotent — only sends once per UTC date. Call from the scheduler every
    hour or so; it cheaply no-ops if already sent today or it's not yet the
    configured hour.
    """
    enabled = (bot_config.get("notifier_daily_summary") or "true").lower() in {"1", "true", "on", "yes"}
    if not enabled and not force:
        return {"ok": True, "skipped": "disabled"}

    today_iso = utcnow().date().isoformat()
    if not force and _last_sent_date() == today_iso:
        return {"ok": True, "skipped": "already_sent_today"}

    if not force:
        try:
            target_hour = int(bot_config.get("notifier_daily_summary_hour_utc") or "23")
        except ValueError:
            target_hour = 23
        if utcnow().hour < target_hour:
            return {"ok": True, "skipped": "before_target_hour"}

    summary = build_summary()
    text = format_summary(summary)
    res = notify(text, level="info", category="daily")
    if res.get("ok") and not res.get("skipped"):
        _mark_sent(today_iso)
    return {"ok": True, "sent": True, "result": res, "summary": summary}
