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

  const POLL_MS = 3000;  // Poll every 3 seconds
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
  const hideHoldsToggle = document.getElementById("live-decision-hide-holds");

  // --- State ---
  const state = {
    polling: false,
    timer: null,
    cursors: { decision_id: 0, log_id: 0, trade_id: 0 },
    decisionCount: 0,
    actionableCount: 0,
    fillCount: 0,
    sessionActive: false,
  };

  // Toggle visibility of HOLD entries without re-fetching. Each rendered
  // decision card carries data-action="HOLD"|"BUY"|... so we can show/hide
  // them with a single class flip on the parent feed.
  function applyDecisionFilter() {
    if (!decisionFeed) return;
    const hide = !!(hideHoldsToggle && hideHoldsToggle.checked);
    decisionFeed.classList.toggle("hide-holds", hide);
  }
  if (hideHoldsToggle) {
    hideHoldsToggle.addEventListener("change", applyDecisionFilter);
  }

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
  // Full ISO timestamp in UTC for tooltip — useful when correlating with
  // server logs which are timestamped in UTC.
  function utcTooltipFor(ts) {
    if (!ts) return "";
    return new Date(ts * 1000).toISOString().replace("T", " ").replace("Z", " UTC");
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

  // --- Debug Console bridge ---
  // The Debug Console page polls GET /debug/logs which mirrors ActivityLog.
  // Anything we POST here lands in that stream with a real DB timestamp, so
  // the user can leave this page and still see the same failures on /debug.
  function pushDebug(level, message, extra) {
    try {
      const fd = new FormData();
      fd.append("level", level || "info");
      fd.append("category", (extra && extra.category) || "session_error");
      fd.append("message", String(message || ""));
      fd.append("source", (extra && extra.source) || "training_ui");
      // Use keepalive so a push initiated right before navigation still
      // flushes — this is the whole point: the user wants failures to
      // survive a page change.
      fetch("/debug/log/push", { method: "POST", body: fd, keepalive: true })
        .catch(() => {});
    } catch (_) { /* never let logging break the app */ }
  }
  // Mirror unhandled JS errors and rejections to the Debug Console.
  window.addEventListener("error", (e) => {
    pushDebug("error", "JS error: " + (e && e.message ? e.message : "unknown") +
      (e && e.filename ? " @ " + e.filename + ":" + (e.lineno || "?") : ""),
      { source: "training_ui" });
  });
  window.addEventListener("unhandledrejection", (e) => {
    const reason = e && e.reason ? (e.reason.message || String(e.reason)) : "unknown";
    pushDebug("error", "Unhandled rejection: " + reason, { source: "training_ui" });
  });
  
  // --- Scroll preservation ---
  // Prevents auto-scroll-to-top when DOM is updated
  function preserveScroll(fn) {
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;
    fn();
    // Only restore if we actually scrolled (prevents fighting with user)
    if (Math.abs(window.scrollY - scrollY) > 5) {
      window.scrollTo(scrollX, scrollY);
    }
  }

  // --- Status / pulse ---
  function setStatus(active, info) {
    state.sessionActive = !!active;
    const bgHint = document.getElementById("live-bg-hint");
    if (active) {
      pulse.style.background = "var(--good)";
      pulse.style.boxShadow = "0 0 8px var(--good)";
      statusBadge.className = "badge badge-good";
      statusBadge.textContent = "LIVE — paper";
      startBtn.style.display = "none";
      stopBtn.style.display = "";
      if (bgHint) bgHint.style.display = "";
    } else {
      pulse.style.background = "var(--text-muted)";
      pulse.style.boxShadow = "none";
      statusBadge.className = "badge";
      statusBadge.textContent = "Idle";
      startBtn.style.display = "";
      stopBtn.style.display = "none";
      if (bgHint) bgHint.style.display = "none";
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
  // The `source` column on ClaudeDecision tracks which code path produced
  // the row. Surface it as a tiny colored badge so you can see at a glance
  // whether Claude actually weighed in or it was a passthrough/fallback.
  //   claude              → real Anthropic API call, parsed cleanly
  //   technical_strong    → tech conf high enough to skip Claude (saves $)
  //   training_passthrough→ same, but during a training session
  //   tech_hold           → technical signal said HOLD, Claude not called
  //   cache               → reused a recent identical Claude decision
  //   budget_fallback     → daily Claude budget exhausted
  //   technical / fallback→ Claude failed (network, JSON parse), used tech
  function sourceBadgeStyle(src) {
    const s = String(src || "").toLowerCase();
    if (s === "claude") return { label: "claude", bg: "rgba(74,163,255,0.18)", fg: "#7cb9ff" };
    if (s === "technical_strong") return { label: "tech-strong", bg: "rgba(38,194,129,0.18)", fg: "#5fdca0" };
    if (s === "training_passthrough") return { label: "passthrough", bg: "rgba(38,194,129,0.18)", fg: "#5fdca0" };
    if (s === "tech_hold") return { label: "tech-hold", bg: "rgba(160,160,160,0.18)", fg: "var(--text-muted)" };
    if (s === "cache") return { label: "cache", bg: "rgba(212,175,55,0.18)", fg: "#d4af37" };
    if (s === "budget_fallback") return { label: "budget", bg: "rgba(255,107,107,0.18)", fg: "#ff8b8b" };
    if (s === "fallback" || s === "technical") return { label: "fallback", bg: "rgba(255,107,107,0.18)", fg: "#ff8b8b" };
    return { label: s || "?", bg: "rgba(160,160,160,0.18)", fg: "var(--text-muted)" };
  }
  function renderDecision(d) {
    const action = (d.action || "HOLD").toUpperCase();
    const isActionable = !!d.actionable && action !== "HOLD";
    const st = actionStyle(action);
    const conf = (d.confidence || 0).toFixed(2);
    const techConf = (d.technical_confidence || 0).toFixed(2);
    const techSide = (d.technical_side || "?").toUpperCase();
    const src = sourceBadgeStyle(d.source);
    const localTime = clockFor(d.ts);
    const utcTime = utcTooltipFor(d.ts);

    // Show the tech→Claude delta only when something actually changed —
    // either the side flipped (tech said BUY, Claude said HOLD) or the
    // confidence shifted by more than 0.05. Otherwise it's just noise.
    const sideFlipped = techSide !== action && techSide !== "?" && action !== "?";
    const confDelta = Math.abs((d.confidence || 0) - (d.technical_confidence || 0));
    const showDelta = sideFlipped || confDelta > 0.05;

    // Only show key_factors / risk_flags when present and there's something
    // actionable to read — for HOLD/0.00 these are usually empty anyway.
    const factors = Array.isArray(d.key_factors) ? d.key_factors.slice(0, 4) : [];
    const flags = Array.isArray(d.risk_flags) ? d.risk_flags.slice(0, 3) : [];

    const node = document.createElement("div");
    node.dataset.action = action;
    node.dataset.actionable = isActionable ? "1" : "0";
    node.style.cssText =
      "padding:0.4rem 0.5rem;background:var(--bg);border:1px solid var(--border);" +
      "border-left:3px solid " + st.color + ";border-radius:var(--radius-sm);" +
      // Subtle dimming for non-actionable rows so the actionable ones pop.
      (isActionable ? "" : "opacity:0.62;");

    let html =
      '<div class="flex items-center justify-between" style="gap:0.4rem;">' +
        '<span style="color:' + st.color + ';font-weight:700;">' +
          st.icon + " " + escapeHtml(action) + "</span>" +
        '<span class="mono" style="font-weight:700;">' + escapeHtml(d.symbol || "?") + "</span>" +
        '<span class="text-dim mono" style="font-size:0.7rem;" title="' + escapeHtml(utcTime) + '">' +
          escapeHtml(localTime) + "</span>" +
      "</div>" +
      '<div class="flex items-center" style="gap:0.4rem;flex-wrap:wrap;font-size:0.7rem;margin-top:0.2rem;">' +
        '<span style="background:' + src.bg + ';color:' + src.fg +
          ';padding:0.05rem 0.3rem;border-radius:3px;font-weight:600;letter-spacing:0.04em;">' +
          escapeHtml(src.label) + "</span>" +
        '<span class="text-dim">@ ' + fmtPrice(d.price) + "</span>" +
        '<span class="text-dim">conf <span style="color:var(--text);font-weight:600;">' +
          conf + "</span></span>" +
        '<span class="text-dim">sl ' + (d.stop_loss_pct * 100).toFixed(1) + "%</span>" +
        '<span class="text-dim">tp ' + (d.take_profit_pct * 100).toFixed(1) + "%</span>" +
      "</div>";

    // Tech vs Claude delta — only when meaningful.
    if (showDelta) {
      const flipColor = sideFlipped ? "var(--accent,#4aa3ff)" : "var(--text-muted)";
      html +=
        '<div class="text-dim" style="font-size:0.68rem;margin-top:0.2rem;">' +
          'tech ' + escapeHtml(techSide) + " " + techConf +
          ' <span style="color:' + flipColor + ';">→</span> ' +
          escapeHtml(action) + " " + conf +
          (sideFlipped ? ' <span style="color:' + flipColor + ';">(side flip)</span>' : "") +
        "</div>";
    }

    if (d.rationale) {
      html +=
        '<div style="font-size:0.72rem;margin-top:0.25rem;color:var(--text);opacity:0.9;line-height:1.45;">' +
          escapeHtml(d.rationale) + "</div>";
    }

    if (factors.length) {
      html +=
        '<div class="flex" style="gap:0.25rem;flex-wrap:wrap;margin-top:0.25rem;">' +
          factors.map(function (f) {
            return '<span class="mono text-dim" style="font-size:0.64rem;background:var(--bg-elev);' +
              'padding:0.05rem 0.3rem;border-radius:3px;border:1px solid var(--border);">' +
              escapeHtml(f) + "</span>";
          }).join("") +
        "</div>";
    }

    if (flags.length) {
      html +=
        '<div class="flex" style="gap:0.25rem;flex-wrap:wrap;margin-top:0.2rem;">' +
          flags.map(function (f) {
            return '<span class="mono" style="font-size:0.64rem;background:rgba(255,107,107,0.12);' +
              'color:#ff8b8b;padding:0.05rem 0.3rem;border-radius:3px;">⚠ ' +
              escapeHtml(f) + "</span>";
          }).join("") +
        "</div>";
    }

    node.innerHTML = html;
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
      if (!res.ok) {
        pushDebug("error", "Session feed HTTP " + res.status + " — " + res.statusText,
          { source: "training_ui", category: "session_feed" });
        return;
      }
      const data = await res.json();
      if (!data.ok) {
        pushDebug("warn", "Session feed responded ok=false" + (data.error ? " — " + data.error : ""),
          { source: "training_ui", category: "session_feed" });
        return;
      }
      
      // Preserve scroll position during all DOM updates
      preserveScroll(() => {
        setStatus(!!data.session.active, data.session);
        renderPortfolio(data.portfolio);

        (data.decisions || []).forEach(renderDecision);
        (data.fills || []).forEach(renderFill);
        (data.logs || []).forEach(renderLog);

        if (data.decisions && data.decisions.length) {
          state.decisionCount += data.decisions.length;
          // Track BUY/SELL/CLOSE separately so the user can see the
          // signal-to-noise ratio at a glance ("3 actionable out of 412").
          for (const d of data.decisions) {
            if (d && d.actionable) state.actionableCount += 1;
          }
          decCountEl.textContent = state.actionableCount + " / " + state.decisionCount;
        }
        if (data.fills && data.fills.length) {
          state.fillCount += data.fills.length;
          fillCountEl.textContent = state.fillCount;
        }
        if (data.cursors) {
          state.cursors = data.cursors;
        }
      });
      
      // Poll portfolio intelligence status (outside preserveScroll since it's async)
      pollPortfolioIntel();
    } catch (e) {
      console.error("[v0] live feed poll failed", e);
      pushDebug("error", "Session feed exception: " + (e && e.message ? e.message : String(e)),
        { source: "training_ui", category: "session_feed" });
    }
  }
  
  // Portfolio Intelligence polling
  async function pollPortfolioIntel() {
    try {
      const res = await fetch("/v1/portfolio-intel");
      if (!res.ok) return;
      const data = await res.json();
      if (!data.ok) return;
      
      // Preserve scroll during DOM updates
      preserveScroll(() => {
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
      });
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
      const styleSel = document.getElementById("live-trading-style");
      fd.append("trading_style", styleSel ? styleSel.value : "hybrid");
      let res, data;
      try {
        res = await fetch("/training/session/start", { method: "POST", body: fd });
        data = await res.json();
      } catch (e) {
        pushDebug("error", "Start session network exception: " + (e && e.message ? e.message : String(e)),
          { source: "training_ui", category: "session_start" });
        alert("Could not start live session — network error. See Debug Console for details.");
        return;
      }
      if (!data.ok) {
        alert("Could not start live session.");
        pushDebug("error", "Start session failed: " + (data.error || "unknown"),
          { source: "training_ui", category: "session_start" });
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
      
      // Safely handle potentially missing response fields
      const tickSec = data.tick_seconds != null ? data.tick_seconds : "?";
      const minConf = data.min_confidence != null ? data.min_confidence.toFixed(2) : "?";
      const posSize = data.position_size_usd != null ? Math.round(data.position_size_usd) : "?";
      const maxOpenVal = data.max_open_per_wallet != null ? data.max_open_per_wallet : "?";
      const uniLimit = data.universe_limit != null ? data.universe_limit : "?";
      const styleVal = data.trading_style || (styleSel ? styleSel.value : "hybrid");
      // Echo the server-confirmed style back into the dropdown so a reconnecting
      // user sees the *actual* effective setting rather than a stale UI default.
      if (styleSel && data.trading_style) styleSel.value = data.trading_style;
      
      renderLog({
        ts: Math.floor(Date.now() / 1000),
        level: "success",
        category: "session",
        message:
          "Session started — tick " + tickSec + "s · floor " + minConf +
          " · $" + posSize + "/trade · " + maxOpenVal + " open · universe " + uniLimit +
          " · style " + styleVal,
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
    // The session lives in the *backend* scheduler — it keeps ticking even
    // when the browser is closed or the user navigates to another page.
    // The Stop button is the ONLY way to actually halt it. Confirm explicitly
    // so users don't kill it just because they're switching tabs.
    if (!confirm(
      "Stop the live training session?\n\n" +
      "(You don't need to stop it just to leave this page — the bot keeps " +
      "running in the background. This button fully halts trading.)"
    )) return;
    stopBtn.disabled = true;
    try {
      const res = await fetch("/training/session/stop", { method: "POST" });
      const data = await res.json();
      if (!data.ok) {
        pushDebug("error", "Stop session failed: " + (data.error || "unknown"),
          { source: "training_ui", category: "session_stop" });
        return;
      }
      setStatus(false);
      // Keep polling once more so we render any final logs.
      pollOnce();
      stopPolling();
    } catch (e) {
      pushDebug("error", "Stop session exception: " + (e && e.message ? e.message : String(e)),
        { source: "training_ui", category: "session_stop" });
    } finally {
      stopBtn.disabled = false;
    }
  });

  tickBtn.addEventListener("click", async () => {
    tickBtn.disabled = true;
    try {
      const res = await fetch("/training/session/tick", { method: "POST" });
      if (!res.ok) {
        pushDebug("warn", "Manual tick HTTP " + res.status,
          { source: "training_ui", category: "session_tick" });
      }
      pollOnce();
    } catch (e) {
      pushDebug("error", "Manual tick exception: " + (e && e.message ? e.message : String(e)),
        { source: "training_ui", category: "session_tick" });
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
      if (!res.ok) {
        pushDebug("warn", "Hydrate config HTTP " + res.status,
          { source: "training_ui", category: "session_hydrate" });
        return;
      }
      const data = await res.json();
      if (!data.ok) return;
      if (tickSel)    { tickSel.value = String(data.tick_seconds); tickValEl.textContent = data.tick_seconds + "s"; }
      if (confSel)    { confSel.value = String(data.min_confidence); confValEl.textContent = Number(data.min_confidence).toFixed(2); }
      if (sizeSel)    { sizeSel.value = String(data.position_size_usd); sizeValEl.textContent = fmtMoneyShort(data.position_size_usd); }
      if (maxOpenSel) { maxOpenSel.value = String(data.max_open_per_wallet); maxOpenValEl.textContent = String(data.max_open_per_wallet); }
      if (universeLimitEl) universeLimitEl.value = String(Math.max(10, parseInt(data.universe_limit, 10) || 100));
      setKillBanner(!!data.kill_switch_engaged);
      refreshSettingsSummary();
    } catch (e) {
      console.error("[v0] settings hydrate failed", e);
      pushDebug("error", "Hydrate config exception: " + (e && e.message ? e.message : String(e)),
        { source: "training_ui", category: "session_hydrate" });
    }
  }

  // Always do an initial poll so the page rehydrates portfolio numbers and
  // detects an already-running session (e.g. if the user reloaded mid-run).
  // Also re-check kill switch state every 10s so a daily-loss event that
  // re-engages it mid-session is surfaced immediately.
  // Apply the "actionable only" filter immediately on load so the default
  // checked state is honored before the first tick arrives.
  applyDecisionFilter();
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
