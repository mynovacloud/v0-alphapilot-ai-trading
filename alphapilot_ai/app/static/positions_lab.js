/* AlphaPilot — Positions Lab
 *
 * Vanilla canvas chart that renders:
 *   - OHLC candles (Coinbase public data)
 *   - Entry markers   (green up arrow long / red down arrow short)
 *   - Exit markers    (triangle, colored by P&L sign)
 *   - Decision dots   (every Claude BUY/SELL/HOLD/CLOSE the bot considered)
 *   - Hover tooltip   showing whatever overlay you're nearest to
 *
 * No external libraries — keeps the bot deployable as a single FastAPI
 * process with no JS build step.
 */
(function () {
  "use strict";

  const sel = (q, root) => (root || document).querySelector(q);
  const all = (q, root) => Array.from((root || document).querySelectorAll(q));

  const symbolList = sel("#lab-symbol-list");
  const granInput = sel("#lab-granularity");
  const emptyEl = sel("#lab-empty");
  const contentEl = sel("#lab-content");
  const canvas = sel("#lab-chart");
  const tooltip = sel("#lab-tooltip");
  const ledgerBody = sel("#lab-ledger-body");
  const ledgerSym = sel("#lab-ledger-symbol");

  if (!symbolList || !canvas) return;

  const state = {
    symbol: null,
    walletId: null,
    granularity: 900,
    data: null,
    layout: null, // {x, y, w, h, candleW, priceMin, priceMax, t0, t1, points}
  };

  const COLORS = {
    grid: "rgba(255,255,255,0.06)",
    axis: "rgba(255,255,255,0.45)",
    up: "rgba(74,222,128,0.95)",
    down: "rgba(248,113,113,0.95)",
    wick: "rgba(255,255,255,0.4)",
    long: "#4ade80",
    short: "#f87171",
    win: "#4ade80",
    loss: "#f87171",
    flat: "#94a3b8",
    decisionBuy: "#4ade80",
    decisionSell: "#f87171",
    decisionHold: "#64748b",
    decisionClose: "#fbbf24",
  };

  function fmtMoney(n) {
    const v = Number(n) || 0;
    const sign = v < 0 ? "-" : "";
    const abs = Math.abs(v);
    return sign + "$" + abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function fmtPrice(n) {
    const v = Number(n) || 0;
    if (v >= 1000) return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (v >= 1) return v.toFixed(2);
    return v.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }
  function fmtPct(n, digits) {
    return ((Number(n) || 0) * 100).toFixed(digits || 1) + "%";
  }
  function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  }

  // -----------------------------------------------------------
  // Symbol list selection
  // -----------------------------------------------------------
  function selectSymbol(btn) {
    all(".lab-symbol", symbolList).forEach((b) => {
      b.style.borderColor = "var(--border)";
      b.style.background = "var(--bg-elev)";
    });
    btn.style.borderColor = "var(--accent, #4aa3ff)";
    btn.style.background = "rgba(74,163,255,0.10)";
    state.symbol = btn.dataset.symbol;
    state.walletId = btn.dataset.walletId || null;
    loadData();
  }

  symbolList.addEventListener("click", (e) => {
    const btn = e.target.closest(".lab-symbol");
    if (btn) selectSymbol(btn);
  });

  granInput.addEventListener("change", () => {
    state.granularity = parseInt(granInput.value, 10) || 900;
    if (state.symbol) loadData();
  });

  // -----------------------------------------------------------
  // Data loading
  // -----------------------------------------------------------
  async function loadData() {
    if (!state.symbol) return;
    emptyEl.style.display = "none";
    contentEl.style.display = "";
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    const params = new URLSearchParams({
      symbol: state.symbol,
      granularity: String(state.granularity),
    });
    if (state.walletId) params.set("wallet_id", state.walletId);
    try {
      const r = await fetch("/training/chart-data?" + params.toString());
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "load failed");
      state.data = data;
      renderStats();
      renderLedger();
      renderChart();
    } catch (err) {
      console.warn("[positions-lab] load failed", err);
    }
  }

  function renderStats() {
    const s = state.data.stats;
    sel("#lab-kpi-symbol").textContent = state.data.symbol;
    sel("#lab-kpi-pnl").textContent = fmtMoney(s.realized_pnl);
    sel("#lab-kpi-pnl").style.color =
      s.realized_pnl > 0 ? "var(--good)" : s.realized_pnl < 0 ? "var(--bad)" : "var(--text)";
    sel("#lab-kpi-winrate").textContent = fmtPct(s.win_rate, 0);
    sel("#lab-kpi-trades").textContent = s.total_trades;
    sel("#lab-kpi-open").textContent = s.open_trades;
    sel("#lab-kpi-hold").textContent =
      s.avg_hold_minutes > 0 ? Math.round(s.avg_hold_minutes) + "m" : "—";
  }

  function renderLedger() {
    ledgerSym.textContent = state.data.symbol;
    const trades = state.data.trades.slice().reverse(); // newest first
    if (!trades.length) {
      ledgerBody.innerHTML =
        '<tr><td colspan="11" class="empty">No trades for this symbol yet.</td></tr>';
      return;
    }
    const rows = trades.map((t) => {
      const pnl = t.realized_pnl || t.unrealized_pnl || 0;
      const pnlColor = pnl > 0 ? "var(--good)" : pnl < 0 ? "var(--bad)" : "var(--text-muted)";
      const sideClass = t.side === "BUY" ? "badge-good" : "badge-bad";
      const statusClass = t.status === "open" ? "badge-info" : "badge";
      const opened = t.opened_at_ts ? fmtTime(t.opened_at_ts) : "—";
      const closed = t.closed_at_ts ? fmtTime(t.closed_at_ts) : "—";
      const notes = (t.notes || "").replace(/</g, "&lt;");
      return (
        '<tr>' +
        '<td class="mono text-dim">#' + t.id + '</td>' +
        '<td><span class="badge ' + sideClass + '">' + t.side + '</span></td>' +
        '<td class="mono">' + t.qty + '</td>' +
        '<td class="mono">' + fmtPrice(t.entry_price) + '</td>' +
        '<td class="mono">' + (t.exit_price ? fmtPrice(t.exit_price) : "—") + '</td>' +
        '<td class="mono" style="color:' + pnlColor + '">' + fmtMoney(pnl) + '</td>' +
        '<td class="mono">' + (t.confidence || 0).toFixed(2) + '</td>' +
        '<td><span class="badge ' + statusClass + '">' + t.status + '</span></td>' +
        '<td class="mono text-dim" style="font-size:0.72rem;">' + opened + '</td>' +
        '<td class="mono text-dim" style="font-size:0.72rem;">' + closed + '</td>' +
        '<td title="' + notes + '" style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + notes + '</td>' +
        '</tr>'
      );
    });
    ledgerBody.innerHTML = rows.join("");
  }

  // -----------------------------------------------------------
  // Chart rendering
  // -----------------------------------------------------------
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: rect.width, h: rect.height };
  }

  function renderChart() {
    if (!state.data) return;
    const { w, h } = resizeCanvas();
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    const candles = state.data.candles || [];
    const trades = state.data.trades || [];
    const decisions = state.data.decisions || [];

    if (!candles.length) {
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.font = "13px -apple-system, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No candle data available for this symbol.", w / 2, h / 2);
      return;
    }

    // ---- price range (auto-fit, including markers) ------------
    let priceMin = Infinity, priceMax = -Infinity;
    for (const c of candles) {
      if (c.low < priceMin) priceMin = c.low;
      if (c.high > priceMax) priceMax = c.high;
    }
    for (const t of trades) {
      if (t.entry_price && t.entry_price < priceMin) priceMin = t.entry_price;
      if (t.entry_price && t.entry_price > priceMax) priceMax = t.entry_price;
      if (t.exit_price && t.exit_price < priceMin) priceMin = t.exit_price;
      if (t.exit_price && t.exit_price > priceMax) priceMax = t.exit_price;
    }
    const pad = (priceMax - priceMin) * 0.08 || priceMax * 0.02 || 1;
    priceMin -= pad;
    priceMax += pad;

    // ---- time range ------------------------------------------
    const t0 = candles[0].time;
    const t1 = candles[candles.length - 1].time + state.granularity; // include trailing slot

    // ---- layout ---------------------------------------------
    const padL = 10, padR = 64, padT = 14, padB = 28;
    const plotX = padL, plotY = padT;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    function xFor(ts) {
      return plotX + ((ts - t0) / (t1 - t0)) * plotW;
    }
    function yFor(price) {
      return plotY + plotH - ((price - priceMin) / (priceMax - priceMin)) * plotH;
    }

    // ---- grid + price axis -----------------------------------
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.font = "10px -apple-system, system-ui, sans-serif";
    ctx.fillStyle = COLORS.axis;
    ctx.textAlign = "left";
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const p = priceMin + ((priceMax - priceMin) * i) / ticks;
      const y = yFor(p);
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.stroke();
      ctx.fillText(fmtPrice(p), plotX + plotW + 4, y + 3);
    }
    // time axis ticks (4 labels)
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const ts = t0 + ((t1 - t0) * i) / 4;
      const x = xFor(ts);
      ctx.fillText(fmtTime(ts), x, h - 8);
    }

    // ---- candles ---------------------------------------------
    const candleW = Math.max(2, (plotW / Math.max(candles.length, 1)) * 0.7);
    for (const c of candles) {
      const x = xFor(c.time + state.granularity / 2);
      const yO = yFor(c.open), yC = yFor(c.close);
      const yH = yFor(c.high), yL = yFor(c.low);
      const up = c.close >= c.open;
      ctx.strokeStyle = COLORS.wick;
      ctx.beginPath();
      ctx.moveTo(x, yH);
      ctx.lineTo(x, yL);
      ctx.stroke();
      ctx.fillStyle = up ? COLORS.up : COLORS.down;
      const top = Math.min(yO, yC);
      const bodyH = Math.max(1, Math.abs(yC - yO));
      ctx.fillRect(x - candleW / 2, top, candleW, bodyH);
    }

    // ---- decision dots --------------------------------------
    const overlayPoints = []; // {x, y, type, payload}
    for (const d of decisions) {
      if (!d.ts || d.ts < t0 || d.ts > t1) continue;
      if (!d.price) continue;
      const x = xFor(d.ts);
      const y = yFor(d.price);
      let color = COLORS.decisionHold;
      if (d.action === "BUY") color = COLORS.decisionBuy;
      else if (d.action === "SELL") color = COLORS.decisionSell;
      else if (d.action === "CLOSE") color = COLORS.decisionClose;
      // HOLDs are rendered very faint so they don't overwhelm
      const isHold = d.action === "HOLD";
      ctx.globalAlpha = isHold ? 0.32 : 0.85;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, isHold ? 2.5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      overlayPoints.push({ x, y, r: isHold ? 4 : 6, type: "decision", payload: d });
    }

    // ---- entries (arrows) -----------------------------------
    for (const t of trades) {
      if (!t.opened_at_ts || !t.entry_price) continue;
      if (t.opened_at_ts < t0 || t.opened_at_ts > t1) continue;
      const x = xFor(t.opened_at_ts);
      const y = yFor(t.entry_price);
      const isLong = t.side === "BUY";
      const color = isLong ? COLORS.long : COLORS.short;
      // Arrow body
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (isLong) {
        // up-arrow below the entry price
        ctx.moveTo(x, y - 2);
        ctx.lineTo(x - 6, y + 10);
        ctx.lineTo(x + 6, y + 10);
      } else {
        // down-arrow above the entry price
        ctx.moveTo(x, y + 2);
        ctx.lineTo(x - 6, y - 10);
        ctx.lineTo(x + 6, y - 10);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // Entry-price dashed line spanning the trade duration
      const xEnd = t.closed_at_ts && t.closed_at_ts <= t1 ? xFor(t.closed_at_ts) : xFor(t1);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = color + "aa";
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(xEnd, y);
      ctx.stroke();
      ctx.setLineDash([]);
      overlayPoints.push({ x, y, r: 10, type: "entry", payload: t });
    }

    // ---- exits (triangles colored by P&L) -------------------
    for (const t of trades) {
      if (!t.closed_at_ts || !t.exit_price) continue;
      if (t.closed_at_ts < t0 || t.closed_at_ts > t1) continue;
      const x = xFor(t.closed_at_ts);
      const y = yFor(t.exit_price);
      const pnl = t.realized_pnl || 0;
      const color = pnl > 0 ? COLORS.win : pnl < 0 ? COLORS.loss : COLORS.flat;
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.beginPath();
      ctx.moveTo(x - 6, y - 6);
      ctx.lineTo(x + 6, y - 6);
      ctx.lineTo(x + 6, y + 6);
      ctx.lineTo(x - 6, y + 6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // little × through the box
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.beginPath();
      ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4);
      ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4);
      ctx.stroke();
      overlayPoints.push({ x, y, r: 10, type: "exit", payload: t });
    }

    state.layout = { points: overlayPoints, w, h };
  }

  // -----------------------------------------------------------
  // Hover tooltip
  // -----------------------------------------------------------
  canvas.addEventListener("mousemove", (e) => {
    if (!state.layout) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null, bestDist = 14;
    for (const p of state.layout.points) {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestDist) {
        best = p;
        bestDist = d;
      }
    }
    if (!best) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.innerHTML = renderTooltipHtml(best);
    tooltip.style.display = "block";
    // Position tooltip with right-side flip if near edge
    const tw = 320;
    let tx = best.x + 12;
    if (tx + tw > rect.width) tx = best.x - tw - 12;
    let ty = best.y + 12;
    if (ty + 120 > rect.height) ty = best.y - 130;
    tooltip.style.left = tx + "px";
    tooltip.style.top = ty + "px";
  });
  canvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });

  function renderTooltipHtml(p) {
    if (p.type === "entry") {
      const t = p.payload;
      const sideTag = t.side === "BUY"
        ? '<span class="badge badge-good">LONG</span>'
        : '<span class="badge badge-bad">SHORT</span>';
      return (
        '<div style="font-weight:600;margin-bottom:4px;">' + sideTag + ' Entry · trade #' + t.id + '</div>' +
        '<div>Price: <span class="mono">' + fmtPrice(t.entry_price) + '</span></div>' +
        '<div>Qty: <span class="mono">' + t.qty + '</span> · Conf: <span class="mono">' + (t.confidence || 0).toFixed(2) + '</span></div>' +
        (t.is_perp ? '<div>Leverage: <span class="mono">' + (t.leverage || 1) + 'x</span></div>' : '') +
        '<div style="color:var(--text-muted);font-size:0.72rem;margin-top:3px;">' + fmtTime(t.opened_at_ts) + '</div>' +
        (t.notes ? '<div style="margin-top:4px;color:var(--text-muted);font-size:0.74rem;line-height:1.35;">' + escapeHtml(t.notes) + '</div>' : '')
      );
    }
    if (p.type === "exit") {
      const t = p.payload;
      const pnl = t.realized_pnl || 0;
      const pnlColor = pnl > 0 ? "var(--good)" : pnl < 0 ? "var(--bad)" : "var(--text-muted)";
      const heldMin = t.opened_at_ts && t.closed_at_ts
        ? Math.round((t.closed_at_ts - t.opened_at_ts) / 60)
        : null;
      return (
        '<div style="font-weight:600;margin-bottom:4px;">Exit · trade #' + t.id + '</div>' +
        '<div>Entry → Exit: <span class="mono">' + fmtPrice(t.entry_price) + ' → ' + fmtPrice(t.exit_price) + '</span></div>' +
        '<div>P&amp;L: <span class="mono" style="color:' + pnlColor + '">' + fmtMoney(pnl) + '</span></div>' +
        (heldMin !== null ? '<div>Held: <span class="mono">' + heldMin + 'm</span></div>' : '') +
        '<div style="color:var(--text-muted);font-size:0.72rem;margin-top:3px;">' + fmtTime(t.closed_at_ts) + '</div>'
      );
    }
    if (p.type === "decision") {
      const d = p.payload;
      const tag = d.action === "BUY" ? '<span class="badge badge-good">BUY</span>'
        : d.action === "SELL" ? '<span class="badge badge-bad">SELL</span>'
        : d.action === "CLOSE" ? '<span class="badge badge-warn">CLOSE</span>'
        : '<span class="badge">HOLD</span>';
      return (
        '<div style="font-weight:600;margin-bottom:4px;">' + tag + ' · ' + d.source + '</div>' +
        '<div>Price: <span class="mono">' + fmtPrice(d.price) + '</span> · Conf: <span class="mono">' + (d.confidence || 0).toFixed(2) + '</span></div>' +
        '<div>Tech: <span class="mono">' + d.technical_side + ' (' + (d.technical_confidence || 0).toFixed(2) + ')</span></div>' +
        '<div>SL/TP: <span class="mono">' + fmtPct(d.stop_loss_pct, 1) + ' / ' + fmtPct(d.take_profit_pct, 1) + '</span></div>' +
        '<div style="color:var(--text-muted);font-size:0.72rem;margin-top:3px;">' + fmtTime(d.ts) + '</div>' +
        (d.rationale ? '<div style="margin-top:4px;color:var(--text-muted);font-size:0.74rem;line-height:1.4;">' + escapeHtml(d.rationale) + '</div>' : '')
      );
    }
    return "";
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Re-render on resize
  let resizeTimer;
  window.addEventListener("resize", () => {
    if (!state.data) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderChart, 100);
  });

  // Auto-select first symbol
  const first = sel(".lab-symbol", symbolList);
  if (first) selectSymbol(first);
})();
