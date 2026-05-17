/* AlphaPilot — Live Trading Session controller.
 *
 * Drives the Training Center "Live Trading Session" card:
 *   - Start / Stop / Tick-Now buttons hit POST /training/session/...
 *   - Polls GET /training/session/feed every POLL_MS while a session is active
 *     (and once on page load to rehydrate state if the bot was already running).
 *   - Streams Claude decisions, paper fills, and activity logs into three
 *     scrolling feeds, and live-marks the equity strip every poll.
 *
 * Pure vanilla JS. No deps.
 */
(function () {
  "use strict";

  const POLL_MS = 1500;
  const MAX_FEED_ITEMS = 120;

  // --- DOM ---
  const root = document.getElementById("live-session");
  if (!root) return;

  const startBtn = document.getElementById("live-start-btn");
  const stopBtn = document.getElementById("live-stop-btn");
  const tickBtn = document.getElementById("live-tick-btn");
  const tickSel = document.getElementById("live-tick-seconds");
  const tickValEl = document.getElementById("live-tick-val");
  const confSel = document.getElementById("live-min-confidence");
  const confValEl = document.getElementById("live-conf-val");
  const sizeSel = document.getElementById("live-position-size");
  const sizeValEl = document.getElementById("live-size-val");
  const maxOpenSel = document.getElementById("live-max-open");
  const maxOpenValEl = document.getElementById("live-maxopen-val");
  const universeLimitEl = document.getElementById("live-universe-limit");
  const aggressiveEl = document.getElementById("live-aggressive");
  const settingsSummaryEl = document.getElementById("live-settings-summary");
  const settingsDetails = document.getElementById("live-settings");
  const pulse = document.getElementById("live-pulse");
  const statusBadge = document.getElementById("live-status-badge");

  const equityEl = document.getElementById("live-equity");
  const equitySub = document.getElementById("live-equity-sub");
  const totalPlEl = document.getElementById("live-total-pl");
  const totalPlPct = document.getElementById("live-total-pl-pct");
  const realizedEl = document.getElementById("live-realized");
  const realizedSub = document.getElementById("live-realized-sub");
  const unrealEl = document.getElementById("live-unrealized");
  const unrealSub = document.getElementById("live-unrealized-sub");
  const winrateEl = document.getElementById("live-winrate");
  const winrateSub = document.getElementById("live-winrate-sub");

  const decisionFeed = document.getElementById("live-decision-feed");
  const fillFeed = document.getElementById("live-fill-feed");
  const consoleEl = document.getElementById("live-console");
  const tickInfo = document.getElementById("live-tick-info");
  const decCountEl = document.getElementById("live-decision-count");
  const fillCountEl = document.getElementById("live-fill-count");

  // --- State ---
  const state = {
    polling: false,
    timer: null,
    cursors: { decision_id: 0, log_id: 0, trade_id: 0 },
    decisionCount: 0,
    fillCount: 0,
    sessionActive: false,
  };

  // --- Helpers ---
  function fmtMoney(v) {
    const sign = v < 0 ? "−" : "";
    return sign + "$" + Math.abs(v).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  function signedMoney(v) {
    const sign = v >= 0 ? "+" : "−";
    return sign + "$" + Math.abs(v).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  function fmtPrice(v) {
    if (!v) return "—";
    if (v >= 100) return v.toFixed(2);
    if (v >= 1) return v.toFixed(4);
    return v.toFixed(6);
  }
  function colorFor(v) {
    if (v > 0.005) return "var(--good)";
    if (v < -0.005) return "var(--bad)";
    return "var(--text)";
  }
  function clockFor(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  function trimFeed(el) {
    while (el.children.length > MAX_FEED_ITEMS) {
      el.removeChild(el.lastChild);
    }
  }
  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // --- Status / pulse ---
  function setStatus(active, info) {
    state.sessionActive = !!active;
    if (active) {
      pulse.style.background = "var(--good)";
      pulse.style.boxShadow = "0 0 8px var(--good)";
      statusBadge.className = "badge badge-good";
      statusBadge.textContent = "LIVE — paper";
      startBtn.style.display = "none";
      stopBtn.style.display = "";
    } else {
      pulse.style.background = "var(--text-muted)";
      pulse.style.boxShadow = "none";
      statusBadge.className = "badge";
      statusBadge.textContent = "Idle";
      startBtn.style.display = "";
      stopBtn.style.display = "none";
    }
    if (info && info.tick_seconds) {
      tickInfo.textContent = "tick " + info.tick_seconds + "s · next " +
        (info.next_tick ? new Date(info.next_tick).toLocaleTimeString() : "—");
    }
  }

  // --- Equity strip ---
  function renderPortfolio(p) {
    if (!p) return;
    equityEl.textContent = fmtMoney(p.current);
    equitySub.textContent = signedMoney(p.total_pl) + " (" +
      (p.total_pl_pct >= 0 ? "+" : "") + p.total_pl_pct.toFixed(2) + "%) vs. " +
      fmtMoney(p.starting) + " seeded";
    totalPlEl.textContent = signedMoney(p.total_pl);
    totalPlEl.style.color = colorFor(p.total_pl);
    totalPlPct.textContent = (p.total_pl_pct >= 0 ? "+" : "") + p.total_pl_pct.toFixed(2) + "%";
    totalPlPct.style.color = colorFor(p.total_pl);
    realizedEl.textContent = signedMoney(p.realized);
    realizedEl.style.color = colorFor(p.realized);
    realizedSub.textContent = p.closed_trades + " closed · " + p.wins + "W / " + p.losses + "L";
    unrealEl.textContent = signedMoney(p.unrealized);
    unrealEl.style.color = colorFor(p.unrealized);
    unrealSub.textContent = p.open_trades + " open · marked live";
    winrateEl.textContent = (p.win_rate * 100).toFixed(1) + "%";
    winrateSub.textContent = p.wins + "W / " + p.losses + "L";
  }

  // --- Decision feed ---
  function actionStyle(a) {
    if (a === "BUY") return { color: "var(--good)", icon: "▲" };
    if (a === "SELL") return { color: "var(--bad)", icon: "▼" };
    if (a === "CLOSE") return { color: "var(--accent,#4aa3ff)", icon: "■" };
    return { color: "var(--text-muted)", icon: "◇" }; // HOLD / other
  }
  function renderDecision(d) {
    const st = actionStyle(d.action);
    const conf = (d.confidence || 0).toFixed(2);
    const node = document.createElement("div");
    node.style.cssText =
      "padding:0.4rem 0.5rem;background:var(--bg);border:1px solid var(--border);" +
      "border-left:3px solid " + st.color + ";border-radius:var(--radius-sm);";
    node.innerHTML =
      '<div class="flex items-center justify-between" style="gap:0.4rem;">' +
        '<span style="color:' + st.color + ';font-weight:700;">' +
          st.icon + " " + escapeHtml(d.action) + "</span>" +
        '<span class="mono" style="font-weight:700;">' + escapeHtml(d.symbol) + "</span>" +
        '<span class="text-dim" style="font-size:0.7rem;">' + clockFor(d.ts) + "</span>" +
      "</div>" +
      '<div class="text-dim" style="font-size:0.7rem;margin-top:0.2rem;">' +
        "@ " + fmtPrice(d.price) + " · conf " + conf +
        " · sl " + (d.stop_loss_pct * 100).toFixed(1) + "%" +
        " · tp " + (d.take_profit_pct * 100).toFixed(1) + "%" +
        " · src " + escapeHtml(d.source || "?") +
      "</div>" +
      (d.rationale
        ? '<div style="font-size:0.72rem;margin-top:0.25rem;color:var(--text);opacity:0.85;">' +
            escapeHtml(d.rationale) + "</div>"
        : "");
    decisionFeed.insertBefore(node, decisionFeed.firstChild);
    trimFeed(decisionFeed);
  }

  // --- Fill feed ---
  function renderFill(t) {
    const isOpen = t.status === "open";
    const sideColor = t.side === "BUY" ? "var(--good)" : "var(--bad)";
    const headColor = isOpen ? sideColor : (t.realized_pnl >= 0 ? "var(--good)" : "var(--bad)");
    const headLabel = isOpen
      ? "OPEN " + t.side
      : "CLOSE " + (t.realized_pnl >= 0 ? "+ WIN" : "− LOSS");
    const notional = (t.entry_price || 0) * (t.qty || 0);
    const exitNotional = t.exit_price ? t.exit_price * t.qty : 0;
    const pnl = isOpen ? t.unrealized_pnl : t.realized_pnl;
    const node = document.createElement("div");
    node.style.cssText =
      "padding:0.4rem 0.5rem;background:var(--bg);border:1px solid var(--border);" +
      "border-left:3px solid " + headColor + ";border-radius:var(--radius-sm);";
    node.innerHTML =
      '<div class="flex items-center justify-between" style="gap:0.4rem;">' +
        '<span style="color:' + headColor + ';font-weight:700;">' + headLabel + "</span>" +
        '<span class="mono" style="font-weight:700;">' + escapeHtml(t.symbol) + "</span>" +
        '<span class="text-dim" style="font-size:0.7rem;">#' + t.id + "</span>" +
      "</div>" +
      '<div style="font-size:0.74rem;margin-top:0.2rem;">' +
        "qty " + t.qty + " · in " + fmtMoney(notional) +
        (t.exit_price ? " → out " + fmtMoney(exitNotional) : "") +
      "</div>" +
      '<div class="mono" style="font-size:0.78rem;margin-top:0.2rem;color:' +
        colorFor(pnl) + ';font-weight:700;">' +
        (isOpen ? "unreal " : "P&L ") + signedMoney(pnl) +
      "</div>";
    fillFeed.insertBefore(node, fillFeed.firstChild);
    trimFeed(fillFeed);
  }

  // --- Console ---
  function renderLog(l) {
    const colorMap = {
      info: "#aaa", warn: "#f0c674", error: "#ff6b6b", success: "#26c281",
    };
    const c = colorMap[l.level] || "#aaa";
    const node = document.createElement("div");
    node.style.cssText = "color:" + c + ";line-height:1.35;";
    const t = l.ts ? clockFor(l.ts) : "";
    node.innerHTML =
      '<span style="color:#666;">[' + t + "][" + escapeHtml(l.category || "log") + "]</span> " +
      escapeHtml(l.message || "");
    consoleEl.insertBefore(node, consoleEl.firstChild);
    while (consoleEl.children.length > 200) consoleEl.removeChild(consoleEl.lastChild);
  }

  // --- Polling ---
  async function pollOnce() {
    try {
      const url =
        "/training/session/feed?since_decision_id=" + state.cursors.decision_id +
        "&since_log_id=" + state.cursors.log_id +
        "&since_trade_id=" + state.cursors.trade_id;
      const res = await fetch(url, { headers: { Accept: "application/json" } });
      if (!res.ok) return;
      const data = await res.json();
      if (!data.ok) return;

      setStatus(!!data.session.active, data.session);
      renderPortfolio(data.portfolio);

      (data.decisions || []).forEach(renderDecision);
      (data.fills || []).forEach(renderFill);
      (data.logs || []).forEach(renderLog);

      if (data.decisions && data.decisions.length) {
        state.decisionCount += data.decisions.length;
        decCountEl.textContent = state.decisionCount;
      }
      if (data.fills && data.fills.length) {
        state.fillCount += data.fills.length;
        fillCountEl.textContent = state.fillCount;
      }
      if (data.cursors) {
        state.cursors = data.cursors;
      }
      
      // Poll portfolio intelligence status
      pollPortfolioIntel();
    } catch (e) {
      console.error("[v0] live feed poll failed", e);
    }
  }
  
  // Portfolio Intelligence polling
  async function pollPortfolioIntel() {
    try {
      const res = await fetch("/api/portfolio-intel");
      if (!res.ok) return;
      const data = await res.json();
      if (!data.ok) return;
      
      // Update the intel card if it exists
      const card = document.getElementById("portfolio-intel-card");
      if (!card) return;
      
      // Show card if there are positions
      if (data.portfolios && data.portfolios.length > 0) {
        card.style.display = "block";
        
        // Set recovery mode styling
        if (data.is_recovery_mode) {
          card.classList.add("recovery-mode");
          const badge = document.getElementById("intel-mode-badge");
          if (badge) badge.textContent = "RECOVERY MODE";
        } else {
          card.classList.remove("recovery-mode");
          const badge = document.getElementById("intel-mode-badge");
          if (badge) badge.textContent = "ACTIVE";
        }
        
        // Update action list from recent activity logs
        const list = document.getElementById("intel-actions-list");
        if (list && data.recent_actions && data.recent_actions.length > 0) {
          list.innerHTML = data.recent_actions.slice(0, 5).map(a => {
            // Parse action type from message
            let type = "unknown";
            if (a.message.toLowerCase().includes("dca")) type = "dca";
            else if (a.message.toLowerCase().includes("scale-in")) type = "scale_in";
            else if (a.message.toLowerCase().includes("offset")) type = "offset";
            
            // Extract symbol if present
            const symbolMatch = a.message.match(/([A-Z]{2,6}-USD)/);
            const symbol = symbolMatch ? symbolMatch[1] : "—";
            
            return `
              <div class="intel-action-item">
                <div class="flex items-center gap-2">
                  <span class="intel-action-type ${type}">${type.replace("_", " ")}</span>
                  <span class="mono">${symbol}</span>
                </div>
                <span class="text-dim">${a.message.substring(0, 50)}...</span>
              </div>
            `;
          }).join("");
        } else if (list) {
          list.innerHTML = '<div class="text-dim">No recent portfolio intelligence actions</div>';
        }
        
        // Update counts (approximate from recent actions)
        const dcaCount = (data.recent_actions || []).filter(a => a.message.toLowerCase().includes("dca")).length;
        const offsetCount = (data.recent_actions || []).filter(a => a.message.toLowerCase().includes("offset")).length;
        const el1 = document.getElementById("intel-action-count");
        const el2 = document.getElementById("intel-dca-count");
        const el3 = document.getElementById("intel-offset-count");
        if (el1) el1.textContent = data.recent_actions ? data.recent_actions.length : 0;
        if (el2) el2.textContent = dcaCount;
        if (el3) el3.textContent = offsetCount;
      } else {
        card.style.display = "none";
      }
    } catch (e) {
      console.error("[v0] portfolio intel poll failed", e);
    }
  }
  function startPolling() {
    if (state.polling) return;
    state.polling = true;
    pollOnce();
    state.timer = setInterval(pollOnce, POLL_MS);
  }
  function stopPolling() {
    state.polling = false;
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  // --- Settings panel wiring ---
  function fmtMoneyShort(v) {
    return "$" + Math.round(v).toLocaleString();
  }
  function refreshSettingsSummary() {
    if (!settingsSummaryEl) return;
    settingsSummaryEl.textContent =
      "tick " + tickSel.value + "s · floor " + parseFloat(confSel.value).toFixed(2) +
      " · " + fmtMoneyShort(parseFloat(sizeSel.value)) + "/trade · " +
      maxOpenSel.value + " open" +
      (aggressiveEl && aggressiveEl.checked ? " · aggressive" : "");
  }
  function bindSlider(input, valueEl, fmt) {
    if (!input || !valueEl) return;
    const upd = () => { valueEl.textContent = fmt(input.value); refreshSettingsSummary(); };
    input.addEventListener("input", upd);
    upd();
  }
  bindSlider(tickSel, tickValEl, (v) => v + "s");
  bindSlider(confSel, confValEl, (v) => parseFloat(v).toFixed(2));
  bindSlider(sizeSel, sizeValEl, (v) => fmtMoneyShort(parseFloat(v)));
  bindSlider(maxOpenSel, maxOpenValEl, (v) => v);
  if (aggressiveEl) {
    aggressiveEl.addEventListener("change", () => {
      if (aggressiveEl.checked) {
        // Snap sliders to aggressive defaults but keep them user-editable.
        confSel.value = "0.30";
        confValEl.textContent = "0.30";
        if (parseInt(maxOpenSel.value, 10) < 8) {
          maxOpenSel.value = "8";
          maxOpenValEl.textContent = "8";
        }
        if (parseInt(tickSel.value, 10) > 5) {
          tickSel.value = "5";
          tickValEl.textContent = "5s";
        }
      }
      refreshSettingsSummary();
    });
  }

  // --- Buttons ---
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    try {
      const fd = new FormData();
      fd.append("tick_seconds", tickSel.value);
      fd.append("min_confidence", confSel.value);
      fd.append("position_size_usd", sizeSel.value);
      fd.append("max_open_per_wallet", maxOpenSel.value);
      fd.append("universe_limit", universeLimitEl ? universeLimitEl.value : "40");
      fd.append("aggressive", aggressiveEl && aggressiveEl.checked ? "true" : "false");
      const res = await fetch("/training/session/start", { method: "POST", body: fd });
      const data = await res.json();
      if (!data.ok) {
        alert("Could not start live session.");
        return;
      }
      if (root.dataset.claudeConfigured === "false") {
        renderLog({
          ts: Math.floor(Date.now() / 1000),
          level: "warn",
          category: "session",
          message: "Claude is not configured — bot will trade on technical signals only.",
        });
      }
      renderLog({
        ts: Math.floor(Date.now() / 1000),
        level: "success",
        category: "session",
        message:
          "Session started — tick " + data.tick_seconds + "s · floor " +
          (data.min_confidence != null ? data.min_confidence.toFixed(2) : "?") +
          " · $" + Math.round(data.position_size_usd || 0) + "/trade · " +
          (data.max_open_per_wallet || "?") + " open · universe " + (data.universe_limit || "?"),
      });
      // Collapse the settings card once running so the live feed has more room.
      if (settingsDetails) settingsDetails.open = false;
      setStatus(true, { tick_seconds: data.tick_seconds });
      startPolling();
    } finally {
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    try {
      const res = await fetch("/training/session/stop", { method: "POST" });
      const data = await res.json();
      if (!data.ok) return;
      setStatus(false);
      // Keep polling once more so we render any final logs.
      pollOnce();
      stopPolling();
    } finally {
      stopBtn.disabled = false;
    }
  });

  tickBtn.addEventListener("click", async () => {
    tickBtn.disabled = true;
    try {
      await fetch("/training/session/tick", { method: "POST" });
      pollOnce();
    } finally {
      tickBtn.disabled = false;
    }
  });

  // --- Boot ---
  // Show / hide the kill-switch banner based on backend state.
  const killBanner = document.getElementById("live-kill-banner");
  const killReleaseBtn = document.getElementById("live-kill-release");
  function setKillBanner(engaged) {
    if (!killBanner) return;
    killBanner.style.display = engaged ? "" : "none";
  }
  if (killReleaseBtn) {
    killReleaseBtn.addEventListener("click", async () => {
      killReleaseBtn.disabled = true;
      try {
        const res = await fetch("/training/session/release-kill-switch", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
          setKillBanner(false);
          renderLog({
            ts: Math.floor(Date.now() / 1000),
            level: "success",
            category: "session",
            message: "Kill switch released — bot will tick again on the next interval.",
          });
        }
      } finally {
        killReleaseBtn.disabled = false;
      }
    });
  }

  // Pull current bot knobs so the sliders reflect the real backend state
  // (e.g. after a reload during an active session).
  async function hydrateSettings() {
    try {
      const res = await fetch("/training/session/config", { headers: { Accept: "application/json" } });
      if (!res.ok) return;
      const data = await res.json();
      if (!data.ok) return;
      if (tickSel)    { tickSel.value = String(data.tick_seconds); tickValEl.textContent = data.tick_seconds + "s"; }
      if (confSel)    { confSel.value = String(data.min_confidence); confValEl.textContent = Number(data.min_confidence).toFixed(2); }
      if (sizeSel)    { sizeSel.value = String(data.position_size_usd); sizeValEl.textContent = fmtMoneyShort(data.position_size_usd); }
      if (maxOpenSel) { maxOpenSel.value = String(data.max_open_per_wallet); maxOpenValEl.textContent = String(data.max_open_per_wallet); }
      if (universeLimitEl) universeLimitEl.value = String(Math.max(10, parseInt(data.universe_limit, 10) || 40));
      setKillBanner(!!data.kill_switch_engaged);
      refreshSettingsSummary();
    } catch (e) {
      console.error("[v0] settings hydrate failed", e);
    }
  }

  // Always do an initial poll so the page rehydrates portfolio numbers and
  // detects an already-running session (e.g. if the user reloaded mid-run).
  // Also re-check kill switch state every 10s so a daily-loss event that
  // re-engages it mid-session is surfaced immediately.
  hydrateSettings().then(pollOnce).then(() => {
    if (state.sessionActive) startPolling();
  });
  setInterval(() => {
    fetch("/training/session/config", { headers: { Accept: "application/json" } })
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d && d.ok) setKillBanner(!!d.kill_switch_engaged); })
      .catch(() => {});
  }, 10000);
})();
