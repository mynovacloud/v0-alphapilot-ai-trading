"""
Signal-edge harness — does the live technical signal actually predict?

This runs the REAL strategy_engine signal functions (the exact code the
bot trades on) over recent historical candles for the liquid universe,
then measures forward returns. It answers the one question that has to
be answered before tuning anything else:

    When the bot says BUY, does price rise more than when it says SELL?

If the answer is no, then gating / holding / universe work is just
making a coin-flip lose more slowly — the signal itself needs to change.

Honest about its limits: it samples only the recent candles Coinbase's
public API returns (~300 per symbol), so it is a snapshot, not a deep
multi-month backtest. But ~300 candles across ~55 liquid symbols is
thousands of signal evaluations — more than enough to tell an edge from
noise.

Usage (run on the machine that can reach Coinbase — i.e. where the app
runs):
    python signal_edge.py
    python signal_edge.py --strategy "Mean Reversion" --granularity 300
    python signal_edge.py --cost-bps 30 --limit 300
"""
from __future__ import annotations

import argparse
import datetime
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# Repo root is this file's directory; make intra-package imports resolve
# whether run as `python signal_edge.py` or `python -m signal_edge`.
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from connectors.universe import _LIQUID_UNIVERSE
from trading.strategy_engine import _STRATEGY_REGISTRY

_COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{pid}/candles"

WARMUP_BARS = 40            # bars the indicators need before a signal is meaningful
DEFAULT_HORIZONS = (5, 15, 30, 60)   # forward-return horizons, in bars


@dataclass
class _Sample:
    side: str                       # BUY / SELL / HOLD
    confidence: float
    fwd: dict[int, float]           # horizon -> forward return (fraction)


@dataclass
class _SideStat:
    n: int = 0
    returns: list[float] = field(default_factory=list)   # signed returns (bps)

    def add(self, signed_bps: float) -> None:
        self.n += 1
        self.returns.append(signed_bps)

    @property
    def mean(self) -> float:
        return statistics.mean(self.returns) if self.returns else 0.0

    @property
    def hit_rate(self) -> float:
        if not self.returns:
            return 0.0
        return sum(1 for r in self.returns if r > 0) / len(self.returns)

    @property
    def t_stat(self) -> float:
        """Crude significance: mean / standard error. |t| >= 2 ~ real."""
        if len(self.returns) < 2:
            return 0.0
        sd = statistics.pstdev(self.returns)
        if sd <= 0:
            return 0.0
        return self.mean / (sd / (len(self.returns) ** 0.5))


def measure(
    candles_by_symbol: dict[str, list[dict]],
    strategy_name: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    warmup: int = WARMUP_BARS,
) -> list[_Sample]:
    """Walk every symbol bar-by-bar, evaluate the real signal, record
    forward returns. Pure (no network) so it is unit-testable."""
    signal_fn = _STRATEGY_REGISTRY.get(strategy_name)
    if signal_fn is None:
        raise ValueError(
            f"unknown strategy {strategy_name!r}; "
            f"choose from {sorted(_STRATEGY_REGISTRY)}"
        )

    max_h = max(horizons)
    samples: list[_Sample] = []

    for candles in candles_by_symbol.values():
        n = len(candles)
        if n < warmup + max_h + 1:
            continue
        for i in range(warmup, n - max_h):
            try:
                sig = signal_fn(candles[: i + 1])
            except Exception:
                continue
            entry = float(candles[i]["close"] or 0.0)
            if entry <= 0:
                continue
            fwd: dict[int, float] = {}
            for h in horizons:
                exit_px = float(candles[i + h]["close"] or 0.0)
                if exit_px > 0:
                    fwd[h] = (exit_px - entry) / entry
            samples.append(_Sample(
                side=(sig.side or "HOLD").upper(),
                confidence=float(sig.confidence or 0.0),
                fwd=fwd,
            ))
    return samples


def _signed_bps(side: str, raw_return: float) -> float:
    """Return in the direction of the trade, in basis points.
    BUY profits when price rises, SELL when it falls."""
    directional = raw_return if side == "BUY" else -raw_return
    return directional * 10_000.0


def summarize(
    samples: list[_Sample],
    horizons: tuple[int, ...],
    cost_bps: float,
) -> str:
    """Build the human-readable verdict report."""
    out: list[str] = []
    actionable = [s for s in samples if s.side in ("BUY", "SELL")]
    n_buy = sum(1 for s in samples if s.side == "BUY")
    n_sell = sum(1 for s in samples if s.side == "SELL")
    n_hold = sum(1 for s in samples if s.side == "HOLD")

    out.append("=" * 72)
    out.append("SIGNAL-EDGE REPORT")
    out.append("=" * 72)
    out.append(
        f"Signal evaluations: {len(samples):,}  "
        f"(BUY {n_buy:,} · SELL {n_sell:,} · HOLD {n_hold:,})"
    )
    out.append(f"Round-trip cost assumption: {cost_bps:.0f} bps")
    if not actionable:
        out.append("")
        out.append("No actionable (BUY/SELL) signals were produced. Nothing to measure.")
        return "\n".join(out)

    # Per-horizon directional edge.
    out.append("")
    out.append("DIRECTIONAL EDGE BY HORIZON  (signed return = return in trade's favor)")
    out.append("-" * 72)
    out.append(f"{'horizon':>8} {'BUY mean':>10} {'SELL mean':>10} "
               f"{'edge gross':>11} {'edge net':>10} {'hit%':>7} {'t-stat':>8}")

    best = None  # (net_edge, horizon)
    for h in horizons:
        buy = _SideStat()
        sell = _SideStat()
        both = _SideStat()
        for s in actionable:
            if h not in s.fwd:
                continue
            sb = _signed_bps(s.side, s.fwd[h])
            both.add(sb)
            (buy if s.side == "BUY" else sell).add(sb)
        if both.n == 0:
            continue
        gross = both.mean
        net = gross - cost_bps
        if best is None or net > best[0]:
            best = (net, h)
        out.append(
            f"{h:>6}b  {buy.mean:>+9.1f} {sell.mean:>+9.1f} "
            f"{gross:>+10.1f} {net:>+9.1f} {both.hit_rate*100:>6.1f}% {both.t_stat:>+8.2f}"
        )

    # HOLD control — should drift near zero; a large value means the
    # universe itself trended over the sample window (bias to note).
    hold_samples = [s for s in samples if s.side == "HOLD"]
    if hold_samples:
        h0 = horizons[len(horizons) // 2]
        drift = [s.fwd[h0] * 10_000.0 for s in hold_samples if h0 in s.fwd]
        if drift:
            out.append("")
            out.append(f"HOLD control @ {h0}b: mean raw move {statistics.mean(drift):+.1f} bps "
                       f"(n={len(drift):,}) — market drift baseline")

    # Confidence buckets at the best horizon — does a higher confidence
    # score actually correspond to more edge?
    if best is not None:
        _, h = best
        out.append("")
        out.append(f"CONFIDENCE CALIBRATION  @ {h}-bar horizon")
        out.append("-" * 72)
        out.append(f"{'conf bucket':>14} {'n':>8} {'net edge bps':>14} {'hit%':>8}")
        buckets = [(0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
        for lo, hi in buckets:
            st = _SideStat()
            for s in actionable:
                if h in s.fwd and lo <= s.confidence < hi:
                    st.add(_signed_bps(s.side, s.fwd[h]))
            if st.n:
                out.append(f"  [{lo:.2f},{hi:.2f}) {st.n:>8,} "
                           f"{st.mean - cost_bps:>+13.1f} {st.hit_rate*100:>7.1f}%")

    # Verdict.
    out.append("")
    out.append("=" * 72)
    out.append("VERDICT")
    out.append("=" * 72)
    if best is None:
        out.append("Inconclusive — not enough samples.")
        return "\n".join(out)
    net, h = best
    gross_at_best = net + cost_bps
    # Recompute t-stat at the best horizon for the verdict.
    bs = _SideStat()
    for s in actionable:
        if h in s.fwd:
            bs.add(_signed_bps(s.side, s.fwd[h]))
    t = bs.t_stat
    if net > 0 and t >= 2.0:
        out.append(f"EDGE FOUND: +{net:.1f} bps/trade net of cost at the {h}-bar horizon "
                   f"(gross {gross_at_best:+.1f} bps, t={t:+.2f}).")
        out.append("Statistically real and survives costs. Point the bot at THIS horizon")
        out.append("and tighten everything else around it.")
    elif gross_at_best > 0 and t >= 2.0:
        out.append(f"WEAK EDGE: the signal has direction (+{gross_at_best:.1f} bps gross, "
                   f"t={t:+.2f}) but the {cost_bps:.0f} bps round-trip cost eats it "
                   f"(net {net:+.1f} bps).")
        out.append("The signal predicts, but not enough to pay the fees. Options: cheaper")
        out.append("execution, longer holds, or a higher-conviction subset only.")
    else:
        out.append(f"NO EDGE: best horizon ({h}b) nets {net:+.1f} bps with t={t:+.2f}.")
        out.append("The signal does not predict direction beyond noise. No amount of")
        out.append("gating, holding-profile, or universe work changes this — those")
        out.append("reduce losses, they do not create edge. The signal itself must")
        out.append("change: a different timeframe, a different feature set, or a")
        out.append("different market. Tuning the current loop further is wasted effort.")
    return "\n".join(out)


def alpha_report(
    samples: list[_Sample],
    horizons: tuple[int, ...],
    cost_bps: float,
) -> str:
    """Does acting on BUY signals beat just holding the coins?

    The honest test for a long-only bot is not "do BUY signals make
    money" (in an up-drifting market almost anything long does) — it is
    "do BUY signals make MORE money than being in the market at a random
    moment". That gap is timing alpha.

      buy&hold  = mean forward return over EVERY bar (the unconditional
                  experience of someone simply holding)
      BUY       = mean forward return on bars the signal said BUY
      alpha     = BUY - buy&hold
      net alpha = (BUY - cost) - buy&hold   — the signal pays a fee on
                  every trade; the holder does not.

    If net alpha is not clearly positive, the signal's timing earns
    nothing and the bot is just fee drag on top of buy-and-hold.
    """
    out: list[str] = []
    out.append("=" * 72)
    out.append("ALPHA REPORT — does the BUY signal beat buy-and-hold?")
    out.append("=" * 72)
    buys = [s for s in samples if s.side == "BUY"]
    out.append(f"BUY signals: {len(buys):,} of {len(samples):,} bars evaluated")
    if not buys or not samples:
        out.append("Not enough data to measure timing alpha.")
        return "\n".join(out)
    out.append(f"Round-trip cost: {cost_bps:.0f} bps "
               f"(buy-and-hold pays this once; the signal pays it every trade)")
    out.append("")
    out.append(f"{'horizon':>8} {'buy&hold':>10} {'BUY signal':>11} "
               f"{'alpha':>9} {'net alpha':>10} {'t-stat':>8}")
    out.append("-" * 72)

    best = None  # (net_alpha, horizon, t)
    for h in horizons:
        all_r = [s.fwd[h] * 10_000.0 for s in samples if h in s.fwd]
        buy_r = [s.fwd[h] * 10_000.0 for s in buys if h in s.fwd]
        if not all_r or not buy_r:
            continue
        hold_mean = statistics.mean(all_r)
        buy_mean = statistics.mean(buy_r)
        alpha = buy_mean - hold_mean
        net_alpha = (buy_mean - cost_bps) - hold_mean
        if len(buy_r) >= 2 and statistics.pstdev(buy_r) > 0:
            t = (buy_mean - hold_mean) / (statistics.pstdev(buy_r) / (len(buy_r) ** 0.5))
        else:
            t = 0.0
        if best is None or net_alpha > best[0]:
            best = (net_alpha, h, t)
        out.append(f"{h:>6}b  {hold_mean:>+9.1f} {buy_mean:>+10.1f} "
                   f"{alpha:>+8.1f} {net_alpha:>+9.1f} {t:>+8.2f}")

    if best is not None:
        _, h, _ = best
        out.append("")
        out.append(f"BUY-SIGNAL ALPHA BY CONFIDENCE  @ {h}-bar horizon")
        out.append("-" * 72)
        out.append(f"{'conf bucket':>14} {'n':>8} {'net alpha bps':>15}")
        hold_mean_h = statistics.mean([s.fwd[h] * 10_000.0 for s in samples if h in s.fwd])
        for lo, hi in ((0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)):
            bucket = [s.fwd[h] * 10_000.0 for s in buys
                      if h in s.fwd and lo <= s.confidence < hi]
            if bucket:
                na = (statistics.mean(bucket) - cost_bps) - hold_mean_h
                out.append(f"  [{lo:.2f},{hi:.2f}) {len(bucket):>8,} {na:>+14.1f}")

    out.append("")
    out.append("=" * 72)
    out.append("VERDICT")
    out.append("=" * 72)
    if best is None:
        out.append("Inconclusive — not enough samples.")
        return "\n".join(out)
    net_alpha, h, t = best
    if net_alpha > 0 and t >= 2.0:
        out.append(f"TIMING ALPHA CONFIRMED: BUY signals beat buy-and-hold by "
                   f"{net_alpha:+.1f} bps/trade net of fees at the {h}-bar horizon "
                   f"(t={t:+.2f}).")
        out.append("The signal's entry timing is genuinely worth something. THIS is")
        out.append("the horizon and edge to build the bot around.")
    elif t >= 2.0 and net_alpha <= 0:
        out.append(f"MARGINAL: BUY signals do pick better-than-average moments "
                   f"(t={t:+.2f}) but the {cost_bps:.0f} bps round-trip fee eats the "
                   f"advantage (net {net_alpha:+.1f} bps).")
        out.append("The timing has some skill but not enough to pay for itself.")
        out.append("Cheaper execution or longer holds might rescue it; as-is it loses.")
    else:
        out.append(f"NO TIMING ALPHA: best horizon ({h}b) nets {net_alpha:+.1f} bps "
                   f"vs buy-and-hold with t={t:+.2f}.")
        out.append("The BUY signal does not pick better moments than simply holding.")
        out.append("An autonomous bot on this signal cannot beat — and after fees")
        out.append("will trail — just buying the basket and sitting. The signal")
        out.append("itself has to change, or the honest move is buy-and-hold.")
    return "\n".join(out)


def _fetch_extended(product_id: str, granularity: int, target_bars: int) -> list[dict]:
    """Paginate Coinbase's candle endpoint backward in time.

    The public endpoint returns at most 300 candles per request, so a
    longer history needs several windowed requests walking into the
    past. Returns candles oldest -> newest, deduped by timestamp.
    """
    by_time: dict[int, dict] = {}
    end = time.time()
    span = 300 * granularity
    max_requests = (target_bars // 300) + 2
    for _ in range(max_requests):
        if len(by_time) >= target_bars:
            break
        start = end - span
        try:
            with httpx.Client(timeout=20.0) as c:
                r = c.get(
                    _COINBASE_CANDLES.format(pid=product_id),
                    params={
                        "granularity": granularity,
                        "start": datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
                        "end": datetime.datetime.fromtimestamp(end, datetime.timezone.utc).isoformat(),
                    },
                )
                r.raise_for_status()
                rows = r.json()
        except Exception:
            break
        if not rows:
            break
        for row in rows:  # [time, low, high, open, close, volume]
            t = int(row[0])
            by_time[t] = {
                "time": t, "low": float(row[1]), "high": float(row[2]),
                "open": float(row[3]), "close": float(row[4]), "volume": float(row[5]),
            }
        end = start
        time.sleep(0.12)   # ~8 req/s — under the public-endpoint limit
    return [by_time[t] for t in sorted(by_time)]


def _fetch(symbols, granularity: int, target_bars: int) -> dict[str, list[dict]]:
    """Pull deep candle history for each symbol; tolerate per-symbol failures."""
    series: dict[str, list[dict]] = {}
    need = WARMUP_BARS + max(DEFAULT_HORIZONS) + 1
    ok = fail = 0
    for sym in symbols:
        try:
            candles = _fetch_extended(sym, granularity, target_bars)
        except Exception:
            candles = []
        if len(candles) >= need:
            series[sym] = candles
            ok += 1
        else:
            fail += 1
    total = sum(len(c) for c in series.values())
    print(f"  fetched {total:,} candles across {ok} symbols ({fail} skipped — no/insufficient data)")
    return series


# A single-runner lock so a button-spammer can't kick off two parallel
# Coinbase-pounding audits at once. The lock is process-local — fine for
# the single-process FastAPI server; if the app is ever scaled out we'd
# need an external coordinator.
_AUDIT_LOCK = threading.Lock()


def run_audit_lite(
    strategy: str = "Momentum",
    granularity: int = 60,
    bars: int = 600,
    cost_bps: float = 30.0,
) -> dict[str, Any]:
    """Synchronous, web-callable signal-edge audit.

    Lighter than the CLI default (600 bars per symbol = ~2 paginated
    requests, ~30-50 second wall time over 57 symbols) so it returns
    inside a sane HTTP wait. The CLI version (`python signal_edge.py`)
    remains the canonical deeper run.

    Raises RuntimeError if another audit is already in flight; the
    caller (the route handler) maps that to a 409 response so the UI
    can show a wait message instead of starting a second Coinbase
    pounding alongside the first.
    """
    if not _AUDIT_LOCK.acquire(blocking=False):
        raise RuntimeError("A signal audit is already running")
    try:
        t0 = time.time()
        series = _fetch(_LIQUID_UNIVERSE, granularity, bars)
        samples = measure(series, strategy, DEFAULT_HORIZONS)
        report = "\n".join((
            alpha_report(samples, DEFAULT_HORIZONS, cost_bps),
            "",
            summarize(samples, DEFAULT_HORIZONS, cost_bps),
        ))
        return {
            "report_text": report,
            "samples_evaluated": len(samples),
            "symbols_fetched": len(series),
            "duration_seconds": time.time() - t0,
            "strategy": strategy,
            "granularity": granularity,
            "bars_per_symbol": bars,
            "cost_bps": cost_bps,
        }
    finally:
        _AUDIT_LOCK.release()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure whether the technical signal predicts.")
    p.add_argument("--strategy", default="Momentum",
                   help=f"one of {sorted(_STRATEGY_REGISTRY)}")
    p.add_argument("--granularity", type=int, default=60,
                   help="candle size in seconds (60 = what the live bot trades)")
    p.add_argument("--bars", type=int, default=2400,
                   help="target candles per symbol (paginated, ~300 per request)")
    p.add_argument("--cost-bps", type=float, default=30.0,
                   help="round-trip fee+slippage in bps (paper engine models ~30)")
    args = p.parse_args(argv)

    horizons = DEFAULT_HORIZONS
    print(f"Signal-edge harness — strategy={args.strategy!r}, granularity={args.granularity}s, "
          f"target {args.bars} bars/symbol, horizons={horizons}")
    print(f"Fetching deep candle history for {len(_LIQUID_UNIVERSE)} liquid symbols "
          f"(paginated — takes a few minutes)...")
    series = _fetch(_LIQUID_UNIVERSE, args.granularity, args.bars)
    if not series:
        print("ERROR: no candle data fetched. Is this machine able to reach Coinbase?")
        return 1
    spans = [c[-1]["time"] - c[0]["time"] for c in series.values() if len(c) > 1]
    if spans:
        hrs = statistics.median(spans) / 3600.0
        print(f"  median history per symbol: {hrs:.1f}h ({hrs / 24:.1f} days)")
    print("Evaluating the real signal at every bar...")
    samples = measure(series, args.strategy, horizons)
    print("")
    print(alpha_report(samples, horizons, args.cost_bps))
    print("")
    print(summarize(samples, horizons, args.cost_bps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
