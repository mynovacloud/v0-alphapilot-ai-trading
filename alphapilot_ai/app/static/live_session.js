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
  // --- Mission Mode (Daily Mission Controller) ---
  // Driven by /training/session/feed.mission. When mission.enabled is false
  // the panel stays hidden — the operator hasn't flipped the controller on
  // in Settings yet. Once enabled, we render the current mode badge, the
  // five tiles (Net Today / Distance to Lock / Trades · WR / From Peak /
  // Throttles), and the policy strip showing the active thresholds for
  // this mode. All ids are addressed by getElementById once and cached
  // implicitly by browser DOM lookup — cheap on every poll.
  // --- Trade Quality Scorecard (Phase B+ visibility) ---
  // Polls /training/scorecard separately from the feed (it's a much
  // cheaper aggregation query and only the operator's eye uses it, so
  // it doesn't need the per-2s cadence the feed uses). Renders four
  // blocks: decision-source breakdown, calibration-tier breakdown,
  // reflection dedup ratio, top accumulated patterns.
  async function fetchScorecard() {
    try {
      const res = await fetch("/training/scorecard", { cache: "no-store" });
      const data = await res.json();
      if (data && data.ok) renderScorecard(data);
    } catch (err) {
      // Non-fatal: scorecard is observability, not pipeline.
      if (window.console) console.warn("scorecard fetch failed", err);
    }
  }

  function renderScorecard(data) {
    const panel = document.getElementById("scorecard-panel");
    if (!panel) return;
    panel.style.display = "";

    // Session-cutoff label
    const sessEl = document.getElementById("scorecard-session");
    if (sessEl && data.session_cutoff) {
      try {
        const t = new Date(data.session_cutoff);
        const hh = t.getHours().toString().padStart(2, "0");
        const mm = t.getMinutes().toString().padStart(2, "0");
        sessEl.textContent = "since " + hh + ":" + mm;
      } catch (e) { sessEl.textContent = ""; }
    }

    // --- Block 1: decision sources ---------------------------------------
    const decTotalEl = document.getElementById("sc-decisions-total");
    const decBarsEl = document.getElementById("sc-decisions-bars");
    if (decTotalEl) decTotalEl.textContent = "(" + (data.decisions.total || 0) + ")";
    if (decBarsEl) {
      decBarsEl.innerHTML = "";
      const total = data.decisions.total || 1;
      // Flatten the by_source dict to rows {label, count}.
      const rows = [];
      for (const [src, actions] of Object.entries(data.decisions.by_source || {})) {
        const count = (actions.HOLD || 0) + (actions.BUY || 0) + (actions.SELL || 0) + (actions.OTHER || 0);
        rows.push({ label: src, count, klass: src });
      }
      // Sort by count desc so the dominant sources appear first.
      rows.sort((a, b) => b.count - a.count);
      if (rows.length === 0) {
        decBarsEl.innerHTML = '<div class="text-dim" style="font-size:0.78rem;">no decisions yet</div>';
      } else {
        rows.forEach(r => {
          const pct = total > 0 ? (r.count / total * 100) : 0;
          const row = document.createElement("div");
          row.className = "sc-row";
          row.innerHTML =
            '<div class="sc-label">' + r.label + '</div>' +
            '<div class="sc-track"><div class="sc-fill ' + r.klass + '" style="width:' + pct.toFixed(0) + '%;"></div></div>' +
            '<div class="sc-count">' + r.count + '</div>';
          decBarsEl.appendChild(row);
        });
      }
    }

    // --- Block 2: trade calibration --------------------------------------
    const tradTotalEl = document.getElementById("sc-trades-total");
    const calibBarsEl = document.getElementById("sc-calib-bars");
    if (tradTotalEl) tradTotalEl.textContent = "(" + (data.trades.total || 0) + ")";
    if (calibBarsEl) {
      calibBarsEl.innerHTML = "";
      const total = data.trades.total || 1;
      const rows = data.trades.by_calibration || [];
      if (rows.length === 0) {
        calibBarsEl.innerHTML = '<div class="text-dim" style="font-size:0.78rem;">no trades yet</div>';
      } else {
        rows.forEach(r => {
          const pct = total > 0 ? (r.count / total * 100) : 0;
          const row = document.createElement("div");
          row.className = "sc-row";
          const labelText = r.source + (r.avg_sample_size ? " <span class=\"text-dim mono\" style=\"font-weight:400;\">n=" + r.avg_sample_size + "</span>" : "");
          row.innerHTML =
            '<div class="sc-label">' + labelText + '</div>' +
            '<div class="sc-track"><div class="sc-fill ' + r.source + '" style="width:' + pct.toFixed(0) + '%;"></div></div>' +
            '<div class="sc-count">' + r.count + '</div>';
          calibBarsEl.appendChild(row);
        });
      }
    }

    // --- Block 3: reflection dedup ---------------------------------------
    const ratioEl = document.getElementById("sc-dedup-ratio");
    const newEl = document.getElementById("sc-dedup-new");
    const reinfEl = document.getElementById("sc-dedup-reinforced");
    const emptyEl = document.getElementById("sc-dedup-empty");
    const refl = data.reflections || {};
    if (ratioEl) {
      const ratio = (refl.dedup_ratio || 0) * 100;
      ratioEl.textContent = ratio.toFixed(0) + "%";
      // Colour: green if dedup is doing work (>20%), warn if low, gray if no data.
      const total = (refl.lessons_new || 0) + (refl.lessons_reinforced || 0);
      if (total === 0) ratioEl.style.color = "var(--text-muted)";
      else if (ratio >= 20) ratioEl.style.color = "var(--good)";
      else if (ratio >= 5) ratioEl.style.color = "var(--warn)";
      else ratioEl.style.color = "var(--bad)";
    }
    if (newEl) newEl.textContent = refl.lessons_new || 0;
    if (reinfEl) reinfEl.textContent = refl.lessons_reinforced || 0;
    if (emptyEl) {
      const e = refl.empty || 0;
      emptyEl.textContent = e > 0 ? "· " + e + " empty" : "";
      emptyEl.style.color = e > 0 ? "var(--warn)" : "var(--text-muted)";
    }

    // --- Block 4: top patterns -------------------------------------------
    const topEl = document.getElementById("sc-top-patterns");
    if (topEl) {
      topEl.innerHTML = "";
      const patterns = data.top_patterns || [];
      if (patterns.length === 0) {
        topEl.innerHTML = '<div class="text-dim">no patterns accumulated yet</div>';
      } else {
        patterns.forEach(p => {
          const row = document.createElement("div");
          row.style.cssText = "display:grid; grid-template-columns: 120px 50px 70px 60px 1fr; gap:0.5rem; align-items:center;";
          const wrColor =
            p.win_rate >= 0.6 ? "var(--good)" :
            p.win_rate >= 0.4 ? "var(--warn)" : "var(--bad)";
          const evColor =
            p.expectancy_pct > 0 ? "var(--good)" :
            p.expectancy_pct < 0 ? "var(--bad)" : "var(--text-muted)";
          row.innerHTML =
            '<div class="mono text-dim" style="font-size:0.7rem;">' + p.fingerprint + '</div>' +
            '<div class="mono">' + p.side + '</div>' +
            '<div class="mono">' + p.sample_size + 'tr</div>' +
            '<div class="mono" style="color:' + wrColor + '">' + (p.win_rate * 100).toFixed(0) + '%</div>' +
            '<div class="mono" style="color:' + evColor + '">ev ' + (p.expectancy_pct >= 0 ? '+' : '') + p.expectancy_pct.toFixed(2) + '%</div>';
          topEl.appendChild(row);
        });
      }
    }
  }

  function renderMission(m) {
    const panel = document.getElementById("mission-panel");
    if (!panel) return;
    if (!m || !m.enabled) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";

    const mode = m.mode || "BUILD";
    const badge = document.getElementById("mission-mode-badge");
    if (badge) {
      badge.textContent = mode;
      badge.setAttribute("data-mode", mode);
    }
    const reason = document.getElementById("mission-mode-reason");
    if (reason) {
      // mode_changed_at is ISO UTC — format as "since HH:MM"
      let since = "";
      if (m.mode_changed_at) {
        try {
          const t = new Date(m.mode_changed_at);
          const hh = t.getHours().toString().padStart(2, "0");
          const mm = t.getMinutes().toString().padStart(2, "0");
          since = " · since " + hh + ":" + mm;
        } catch (e) { /* ignore */ }
      }
      // Short policy hint that explains what this mode does.
      const hints = {
        SCOUT:    "small probes, find which pairs behave today",
        BUILD:    "normal trading, full strategy stack",
        ATTACK:   "in profit; selectively larger size",
        PROTECT:  "near target; A-grade trades only",
        LOCK:     "target reached — refusing new entries",
        RECOVERY: "loss streak; A+ trades only, doubled cooldowns",
        KILL:     "trading halted: daily loss / panic / manual kill",
      };
      reason.textContent = (hints[mode] || "") + since;
    }

    const killInd = document.getElementById("mission-kill-indicator");
    if (killInd) killInd.style.display = (mode === "KILL" || m.manual_kill_enabled) ? "" : "none";

    const pnl = m.pnl || {};
    const trades = m.trades || {};
    const policy = m.policy || {};

    // Net P&L today + target reference
    const netEl = document.getElementById("mission-net-pnl");
    if (netEl) {
      netEl.textContent = signedMoney(pnl.net || 0);
      netEl.style.color = colorFor(pnl.net || 0);
    }
    const tgtSub = document.getElementById("mission-target-sub");
    if (tgtSub) tgtSub.textContent = "target " + fmtMoney(pnl.target || 0);

    // Distance to lock
    const distEl = document.getElementById("mission-distance");
    if (distEl) {
      if (mode === "LOCK") {
        distEl.textContent = "✓ LOCKED";
        distEl.style.color = "var(--good)";
      } else {
        const remaining = pnl.remaining_to_target;
        distEl.textContent = remaining !== undefined ? fmtMoney(remaining) : "—";
        distEl.style.color = "var(--text)";
      }
    }

    // Trades / WR
    const trEl = document.getElementById("mission-trades");
    if (trEl) {
      const wr = ((trades.win_rate || 0) * 100).toFixed(0);
      trEl.textContent = (trades.total_today || 0) + " · " + wr + "%";
    }
    const streakEl = document.getElementById("mission-streak");
    if (streakEl) {
      const wins = trades.consecutive_wins || 0;
      const losses = trades.consecutive_losses || 0;
      if (losses > 0) {
        streakEl.textContent = losses + " loss streak";
        streakEl.style.color = "var(--bad)";
      } else if (wins > 0) {
        streakEl.textContent = wins + " win streak";
        streakEl.style.color = "var(--good)";
      } else {
        streakEl.textContent = "no active streak";
        streakEl.style.color = "var(--text-muted)";
      }
    }

    // Drawdown from peak
    const ddEl = document.getElementById("mission-drawdown");
    if (ddEl) {
      const dd = pnl.drawdown_from_peak || 0;
      ddEl.textContent = "−" + fmtMoney(dd);
      ddEl.style.color = dd > 0.01 ? "var(--warn)" : "var(--text)";
    }
    const peakEl = document.getElementById("mission-peak");
    if (peakEl) peakEl.textContent = "peak " + signedMoney(pnl.peak || 0);

    // Throttles
    const ds = (m.disabled_symbols || []).length;
    const dst = (m.disabled_strategies || []).length;
    const throttlesEl = document.getElementById("mission-throttles");
    if (throttlesEl) {
      const total = ds + dst;
      throttlesEl.textContent = total === 0 ? "—" : ("" + total);
      throttlesEl.style.color = total > 0 ? "var(--warn)" : "var(--text)";
    }
    const throttlesSub = document.getElementById("mission-throttles-sub");
    if (throttlesSub) {
      if (ds === 0 && dst === 0) {
        throttlesSub.textContent = "nothing quarantined";
      } else {
        throttlesSub.textContent = ds + " symbols · " + dst + " strategies";
      }
    }

    // Policy strip
    const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setText("mp-conf",     (policy.confidence_floor || 0).toFixed(2));
    setText("mp-edge",     "$" + (policy.min_required_edge || 0).toFixed(2));
    setText("mp-size",     (policy.size_multiplier || 0).toFixed(2) + "×");
    setText("mp-notional", "$" + (policy.max_notional || 0).toFixed(0));
    setText("mp-claude",   policy.claude_allowed ? "allowed" : "blocked");
  }

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
        renderMission(data.mission);
      });
      // Scorecard fetches from a separate endpoint (cheaper than bundling
      // it into the feed JSON every tick) and renders outside the
      // preserveScroll wrapper since its bars are below the fold.
      fetchScorecard().catch(() => {});
      preserveScroll(() => {

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
