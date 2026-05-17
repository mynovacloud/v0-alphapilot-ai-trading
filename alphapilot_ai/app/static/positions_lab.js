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
    // Pan/zoom state
    zoomLevel: 1.0,
    panOffset: 0, // time offset in seconds (positive = looking at older data)
    isDragging: false,
    dragStartX: 0,
    dragStartPan: 0,
    // Filter state
    showEntries: true,
    showExits: true,
    showDecisions: true,
    showHolds: false,
  };

  // Filter checkboxes
  const filterEntries = sel("#filter-entries");
  const filterExits = sel("#filter-exits");
  const filterDecisions = sel("#filter-decisions");
  const filterHolds = sel("#filter-holds");
  
  // Zoom/pan buttons
  const zoomInBtn = sel("#chart-zoom-in");
  const zoomOutBtn = sel("#chart-zoom-out");
  const zoomResetBtn = sel("#chart-zoom-reset");
  const zoomLevelEl = sel("#chart-zoom-level");
  const panLeftBtn = sel("#chart-pan-left");
  const panRightBtn = sel("#chart-pan-right");
  const goLatestBtn = sel("#chart-go-latest");

  // Initialize filter listeners
  if (filterEntries) filterEntries.addEventListener("change", () => { state.showEntries = filterEntries.checked; renderChart(); });
  if (filterExits) filterExits.addEventListener("change", () => { state.showExits = filterExits.checked; renderChart(); });
  if (filterDecisions) filterDecisions.addEventListener("change", () => { state.showDecisions = filterDecisions.checked; renderChart(); });
  if (filterHolds) filterHolds.addEventListener("change", () => { state.showHolds = filterHolds.checked; renderChart(); });

  // Zoom controls
  if (zoomInBtn) zoomInBtn.addEventListener("click", () => { zoom(1.25); });
  if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => { zoom(0.8); });
  if (zoomResetBtn) zoomResetBtn.addEventListener("click", () => { state.zoomLevel = 1.0; state.panOffset = 0; updateZoomLabel(); renderChart(); });
  
  // Pan controls
  if (panLeftBtn) panLeftBtn.addEventListener("click", () => { pan(-0.25); });
  if (panRightBtn) panRightBtn.addEventListener("click", () => { pan(0.25); });
  if (goLatestBtn) goLatestBtn.addEventListener("click", () => { state.panOffset = 0; renderChart(); });

  function zoom(factor) {
    state.zoomLevel = Math.max(0.25, Math.min(10, state.zoomLevel * factor));
    updateZoomLabel();
    renderChart();
  }

  function pan(fraction) {
    if (!state.data || !state.data.candles.length) return;
    const candles = state.data.candles;
    const timeSpan = (candles[candles.length - 1].time - candles[0].time) / state.zoomLevel;
    state.panOffset += timeSpan * fraction;
    state.panOffset = Math.max(0, state.panOffset); // Can't pan into the future
    renderChart();
  }

  function updateZoomLabel() {
    if (zoomLevelEl) zoomLevelEl.textContent = Math.round(state.zoomLevel * 100) + "%";
  }

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
    const trades = state.data.trades || [];

    // Aggregate notional dollar amounts at entry and exit, per symbol.
    // "Bought $" sums (qty * entry_price) for every BUY (long) trade and
    // every SHORT exit's covering price; we treat entry_price * qty as the
    // capital "bought in" regardless of side for clarity to the user.
    let boughtTotal = 0, soldTotal = 0;
    let buyCount = 0, sellCount = 0;
    let openNotional = 0, openUnreal = 0, openCount = 0;
    for (const t of trades) {
      const entryNotional = (t.entry_price || 0) * (t.qty || 0);
      boughtTotal += entryNotional;
      buyCount += 1;
      if (t.exit_price) {
        soldTotal += (t.exit_price || 0) * (t.qty || 0);
        sellCount += 1;
      }
      if (t.status === "open") {
        openNotional += entryNotional;
        openUnreal += t.unrealized_pnl || 0;
        openCount += 1;
      }
    }

    sel("#lab-kpi-symbol").textContent = state.data.symbol;
    sel("#lab-kpi-pnl").textContent = fmtMoney(s.realized_pnl);
    sel("#lab-kpi-pnl").style.color =
      s.realized_pnl > 0 ? "var(--good)" : s.realized_pnl < 0 ? "var(--bad)" : "var(--text)";
    const pnlSub = sel("#lab-kpi-pnl-sub");
    if (pnlSub) pnlSub.textContent =
      "best " + fmtMoney(s.best_pnl) + " / worst " + fmtMoney(s.worst_pnl);

    sel("#lab-kpi-bought").textContent = fmtMoney(boughtTotal);
    const boughtSub = sel("#lab-kpi-bought-sub");
    if (boughtSub) boughtSub.textContent = buyCount + " entr" + (buyCount === 1 ? "y" : "ies");

    sel("#lab-kpi-sold").textContent = fmtMoney(soldTotal);
    const soldSub = sel("#lab-kpi-sold-sub");
    if (soldSub) soldSub.textContent = sellCount + " exit" + (sellCount === 1 ? "" : "s");

    const openVal = sel("#lab-kpi-openval");
    if (openVal) {
      openVal.textContent = fmtMoney(openNotional);
      openVal.style.color =
        openUnreal > 0 ? "var(--good)" : openUnreal < 0 ? "var(--bad)" : "var(--text)";
    }
    const openSub = sel("#lab-kpi-openval-sub");
    if (openSub)
      openSub.textContent =
        openCount + " position" + (openCount === 1 ? "" : "s") +
        " · unreal " + fmtMoney(openUnreal);

    sel("#lab-kpi-winrate").textContent = fmtPct(s.win_rate, 0);
    sel("#lab-kpi-trades").textContent = s.total_trades;
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
      const boughtNotional = (t.entry_price || 0) * (t.qty || 0);
      const soldNotional = t.exit_price ? (t.exit_price * t.qty) : 0;
      const boughtCell =
        '<div class="mono" style="font-weight:700;font-size:0.95rem;color:var(--good);">' +
        fmtMoney(boughtNotional) + '</div>' +
        '<div class="text-dim mono" style="font-size:0.7rem;">@ ' + fmtPrice(t.entry_price) + '</div>';
      const soldCell = t.exit_price
        ? ('<div class="mono" style="font-weight:700;font-size:0.95rem;color:var(--accent,#4aa3ff);">' +
           fmtMoney(soldNotional) + '</div>' +
           '<div class="text-dim mono" style="font-size:0.7rem;">@ ' + fmtPrice(t.exit_price) + '</div>')
        : '<span class="text-dim">— still open —</span>';
      return (
        '<tr>' +
        '<td class="mono text-dim">#' + t.id + '</td>' +
        '<td><span class="badge ' + sideClass + '">' + t.side + '</span></td>' +
        '<td class="mono">' + t.qty + '</td>' +
        '<td style="text-align:right;">' + boughtCell + '</td>' +
        '<td style="text-align:right;">' + soldCell + '</td>' +
        '<td class="mono" style="text-align:right;font-weight:700;color:' + pnlColor + '">' + fmtMoney(pnl) + '</td>' +
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

    // ---- time range with zoom/pan ------------------------------------------
    const fullT0 = candles[0].time;
    const fullT1 = candles[candles.length - 1].time + state.granularity;
    const fullSpan = fullT1 - fullT0;
    
    // Apply zoom: narrower time window
    const visibleSpan = fullSpan / state.zoomLevel;
    
    // Apply pan: shift the visible window (panOffset is how far back we're looking)
    // t1 is the right edge (most recent), t0 is the left edge
    const t1 = fullT1 - state.panOffset;
    const t0 = t1 - visibleSpan;
    
    // Filter candles to visible range
    const visibleCandles = candles.filter(c => c.time + state.granularity >= t0 && c.time <= t1);
    
    // Recalculate price range for visible candles only
    priceMin = Infinity; priceMax = -Infinity;
    for (const c of visibleCandles) {
      if (c.low < priceMin) priceMin = c.low;
      if (c.high > priceMax) priceMax = c.high;
    }
    if (priceMin === Infinity) { priceMin = 0; priceMax = 1; }
    for (const t of trades) {
      if (t.opened_at_ts >= t0 && t.opened_at_ts <= t1) {
        if (t.entry_price && t.entry_price < priceMin) priceMin = t.entry_price;
        if (t.entry_price && t.entry_price > priceMax) priceMax = t.entry_price;
      }
      if (t.closed_at_ts >= t0 && t.closed_at_ts <= t1) {
        if (t.exit_price && t.exit_price < priceMin) priceMin = t.exit_price;
        if (t.exit_price && t.exit_price > priceMax) priceMax = t.exit_price;
      }
    }
    const pricePad = (priceMax - priceMin) * 0.08 || priceMax * 0.02 || 1;
    priceMin -= pricePad;
    priceMax += pricePad;

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

    // ---- candles (use visibleCandles) ---------------------------------------------
    const candleW = Math.max(2, (plotW / Math.max(visibleCandles.length, 1)) * 0.7 * Math.min(state.zoomLevel, 2));
    for (const c of visibleCandles) {
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

    // ---- decision dots (with filter) --------------------------------------
    const overlayPoints = []; // {x, y, type, payload}
    if (state.showDecisions || state.showHolds) {
      for (const d of decisions) {
        if (!d.ts || d.ts < t0 || d.ts > t1) continue;
        if (!d.price) continue;
        const isHold = d.action === "HOLD";
        // Apply filters
        if (isHold && !state.showHolds) continue;
        if (!isHold && !state.showDecisions) continue;
        
        const x = xFor(d.ts);
        const y = yFor(d.price);
        let color = COLORS.decisionHold;
        if (d.action === "BUY") color = COLORS.decisionBuy;
        else if (d.action === "SELL") color = COLORS.decisionSell;
        else if (d.action === "CLOSE") color = COLORS.decisionClose;
        // HOLDs are rendered very faint so they don't overwhelm
        ctx.globalAlpha = isHold ? 0.32 : 0.85;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, isHold ? 2.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        overlayPoints.push({ x, y, r: isHold ? 4 : 6, type: "decision", payload: d });
      }
    }

    // ---- entries (arrows) with filter -----------------------------------
    if (state.showEntries) {
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
    }

    // ---- exits (triangles colored by P&L) with filter -------------------
    if (state.showExits) {
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
    }

    state.layout = { points: overlayPoints, w, h };
  }

  // -----------------------------------------------------------
  // Drag to pan
  // -----------------------------------------------------------
  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return; // left click only
    state.isDragging = true;
    state.dragStartX = e.clientX;
    state.dragStartPan = state.panOffset;
    canvas.style.cursor = "grabbing";
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!state.isDragging) return;
    if (!state.data || !state.data.candles.length) return;
    
    const dx = e.clientX - state.dragStartX;
    const candles = state.data.candles;
    const fullSpan = candles[candles.length - 1].time - candles[0].time;
    const visibleSpan = fullSpan / state.zoomLevel;
    const rect = canvas.getBoundingClientRect();
    
    // Convert pixel drag to time offset
    const timeDelta = (dx / rect.width) * visibleSpan;
    state.panOffset = Math.max(0, state.dragStartPan + timeDelta);
    renderChart();
  });

  canvas.addEventListener("mouseup", () => {
    state.isDragging = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mouseleave", () => {
    if (state.isDragging) {
      state.isDragging = false;
      canvas.style.cursor = "grab";
    }
  });

  // Scroll to zoom
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    state.zoomLevel = Math.max(0.25, Math.min(10, state.zoomLevel * factor));
    updateZoomLabel();
    renderChart();
  }, { passive: false });

  // Double-click to reset
  canvas.addEventListener("dblclick", () => {
    state.zoomLevel = 1.0;
    state.panOffset = 0;
    updateZoomLabel();
    renderChart();
  });

  // -----------------------------------------------------------
  // Hover tooltip
  // -----------------------------------------------------------
  canvas.addEventListener("mousemove", (e) => {
    if (state.isDragging) return; // Don't show tooltip while dragging
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
