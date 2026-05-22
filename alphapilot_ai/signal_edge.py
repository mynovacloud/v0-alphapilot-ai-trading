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
import statistics
import sys
import time
from dataclasses import dataclass, field

# Repo root is this file's directory; make intra-package imports resolve
# whether run as `python signal_edge.py` or `python -m signal_edge`.
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from connectors.candles import get_candles
from connectors.universe import _LIQUID_UNIVERSE
from trading.strategy_engine import _STRATEGY_REGISTRY

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


def _fetch(symbols, granularity: int, limit: int) -> dict[str, list[dict]]:
    """Pull candles for each symbol; tolerate per-symbol failures."""
    series: dict[str, list[dict]] = {}
    ok = fail = 0
    for sym in symbols:
        try:
            candles = get_candles(sym, granularity=granularity, limit=limit)
        except Exception:
            candles = []
        if len(candles) >= WARMUP_BARS + max(DEFAULT_HORIZONS) + 1:
            series[sym] = candles
            ok += 1
        else:
            fail += 1
        time.sleep(0.15)   # be polite to the public endpoint (429 guard)
    print(f"  fetched candles for {ok} symbols ({fail} skipped — no/insufficient data)")
    return series


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure whether the technical signal predicts.")
    p.add_argument("--strategy", default="Momentum",
                   help=f"one of {sorted(_STRATEGY_REGISTRY)}")
    p.add_argument("--granularity", type=int, default=60,
                   help="candle size in seconds (60 = what the live bot trades)")
    p.add_argument("--limit", type=int, default=300, help="candles per symbol (Coinbase caps ~300)")
    p.add_argument("--cost-bps", type=float, default=30.0,
                   help="round-trip fee+slippage in bps (paper engine models ~30)")
    args = p.parse_args(argv)

    horizons = DEFAULT_HORIZONS
    print(f"Signal-edge harness — strategy={args.strategy!r}, granularity={args.granularity}s, "
          f"horizons={horizons} bars")
    print(f"Fetching candles for {len(_LIQUID_UNIVERSE)} liquid symbols...")
    series = _fetch(_LIQUID_UNIVERSE, args.granularity, args.limit)
    if not series:
        print("ERROR: no candle data fetched. Is this machine able to reach Coinbase?")
        return 1
    print("Evaluating the real signal at every bar...")
    samples = measure(series, args.strategy, horizons)
    print("")
    print(summarize(samples, horizons, args.cost_bps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
