/* GeoRegime Evolution Visualization */
(function () {
  "use strict";

  const state = {
    data: null,
    current: 1,
    autoplay: null,
  };

  // ---------- Utilities ----------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const fmt = (n, digits = 4) => (n == null || isNaN(n) ? "—" : Number(n).toFixed(digits));

  function deltaSpan(curr, prev, opts = {}) {
    if (curr == null || prev == null) return { text: "", cls: "flat" };
    const lowerBetter = !!opts.lowerBetter;
    const diff = curr - prev;
    const eps = 1e-9;
    let cls = "flat";
    if (Math.abs(diff) > eps) {
      const better = lowerBetter ? diff < 0 : diff > 0;
      cls = better ? "up" : "down";
    }
    const sign = diff > 0 ? "+" : "";
    const digits = opts.digits != null ? opts.digits : 4;
    return { text: `${sign}${diff.toFixed(digits)} vs prev`, cls };
  }

  // ---------- Diff: Myers' LCS-based line diff (compact) ----------
  function lineDiff(a, b) {
    const n = a.length, m = b.length;
    // Build LCS lengths (DP). Cap to avoid blowing up; our files are <1k lines.
    const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
        else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const ops = [];
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) { ops.push({ op: "ctx", line: a[i], i: i + 1, j: j + 1 }); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push({ op: "del", line: a[i], i: i + 1, j: 0 }); i++; }
      else { ops.push({ op: "add", line: b[j], i: 0, j: j + 1 }); j++; }
    }
    while (i < n) { ops.push({ op: "del", line: a[i], i: i + 1, j: 0 }); i++; }
    while (j < m) { ops.push({ op: "add", line: b[j], i: 0, j: j + 1 }); j++; }
    return ops;
  }

  /** Render a unified diff with hunks (3 lines context). */
  function renderDiff(host, oldText, newText) {
    host.innerHTML = "";
    if (!oldText) {
      // First round, no prior — show newText as all-add but visually muted
      const lines = (newText || "").split("\n");
      const frag = document.createDocumentFragment();
      const hint = document.createElement("div");
      hint.className = "diff-line hunk";
      hint.innerHTML = `<div class="ln"></div><div class="code">@@ initial baseline @@</div>`;
      frag.appendChild(hint);
      lines.forEach((ln, idx) => {
        const row = document.createElement("div");
        row.className = "diff-line ctx";
        row.innerHTML = `<div class="ln">${idx + 1}</div><div class="code"></div>`;
        row.querySelector(".code").textContent = ln;
        frag.appendChild(row);
      });
      host.appendChild(frag);
      return { added: 0, removed: 0 };
    }

    const a = oldText.split("\n");
    const b = newText.split("\n");
    const ops = lineDiff(a, b);

    const ctxN = 3;
    // Decide which ops to show: any add/del + ctxN ops on either side.
    const keep = new Array(ops.length).fill(false);
    for (let k = 0; k < ops.length; k++) {
      if (ops[k].op !== "ctx") {
        for (let t = Math.max(0, k - ctxN); t <= Math.min(ops.length - 1, k + ctxN); t++) {
          keep[t] = true;
        }
      }
    }

    const frag = document.createDocumentFragment();
    let added = 0, removed = 0;
    let lastShown = -2;
    for (let k = 0; k < ops.length; k++) {
      if (!keep[k]) continue;
      if (k !== lastShown + 1) {
        const op = ops[k];
        const hunk = document.createElement("div");
        hunk.className = "diff-line hunk";
        hunk.innerHTML = `<div class="ln"></div><div class="code"></div>`;
        hunk.querySelector(".code").textContent = `@@ -${op.i || "?"} +${op.j || "?"} @@`;
        frag.appendChild(hunk);
      }
      const o = ops[k];
      const row = document.createElement("div");
      row.className = `diff-line ${o.op}`;
      const lnLabel = o.op === "del" ? `${o.i}` : (o.op === "add" ? `${o.j}` : `${o.i}`);
      row.innerHTML = `<div class="ln">${lnLabel}</div><div class="code"></div>`;
      row.querySelector(".code").textContent = o.line;
      frag.appendChild(row);
      if (o.op === "add") added++;
      if (o.op === "del") removed++;
      lastShown = k;
    }
    host.appendChild(frag);
    return { added, removed };
  }

  // ---------- Rendering ----------
  function buildTimeline() {
    const tl = $("#timeline");
    tl.innerHTML = "";
    const rounds = state.data.rounds;
    const scores = rounds.map((r) => r.info.metrics.combined_score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const span = max - min || 1;
    rounds.forEach((rd, idx) => {
      const score = rd.info.metrics.combined_score;
      const prev = idx > 0 ? rounds[idx - 1].info.metrics.combined_score : null;
      const tag = prev == null ? "start" : (score > prev + 1e-9 ? "▲" : (score < prev - 1e-9 ? "▼" : "—"));
      const tagCls = prev == null || tag === "—" ? "flat" : (tag === "▲" ? "up" : "flat");
      const cell = document.createElement("button");
      cell.className = "tl-cell";
      cell.dataset.round = String(rd.round);
      const pct = ((score - min) / span) * 100;
      cell.innerHTML = `
        <div class="r-num">ROUND ${rd.round}</div>
        <div class="r-score">${score.toFixed(4)}</div>
        <div class="r-bar"><i style="width:${pct.toFixed(1)}%"></i></div>
        <div class="r-tag ${tagCls}">${tag}</div>
      `;
      cell.addEventListener("click", () => selectRound(rd.round));
      tl.appendChild(cell);
    });
    setActiveCell(state.current);
  }

  function setActiveCell(round) {
    $$(".tl-cell").forEach((el) => el.classList.toggle("active", Number(el.dataset.round) === round));
  }

  const sparkCharts = {};

  function buildSparkline(metricKey, color, opts = {}) {
    const lowerBetter = !!opts.lowerBetter;
    const digits = opts.digits != null ? opts.digits : 4;
    const canvas = document.getElementById(`spark-${metricKey}`);
    const labels = state.data.rounds.map((r) => `R${r.round}`);
    const data = state.data.rounds.map((r) => r.info.metrics[metricKey]);
    // Auto-zoom y-axis with 8% padding so even small deltas are visible
    const min = Math.min(...data);
    const max = Math.max(...data);
    const span = max - min || Math.abs(max) * 0.05 || 1;
    const yMin = min - span * 0.15;
    const yMax = max + span * 0.15;
    // Range hint
    const rangeEl = document.getElementById(`spark-${metricKey}-range`);
    if (rangeEl) {
      const arrow = lowerBetter ? "↓ better" : "↑ better";
      rangeEl.textContent = `${data[0].toFixed(digits)} → ${data[data.length - 1].toFixed(digits)}  · ${arrow}`;
    }
    const pointBg = data.map((_, i) => (i === state.current - 1 ? color : color + "55"));
    const pointRadii = data.map((_, i) => (i === state.current - 1 ? 5 : 2.5));
    const chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: metricKey,
          data,
          borderColor: color,
          backgroundColor: color + "22",
          tension: 0.3,
          fill: true,
          pointBackgroundColor: pointBg,
          pointBorderColor: pointBg,
          pointRadius: pointRadii,
          pointHoverRadius: 5,
          borderWidth: 1.6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f1530",
            borderColor: "#243064",
            borderWidth: 1,
            titleColor: "#fff",
            bodyColor: "#cbd5ff",
            displayColors: false,
            callbacks: {
              label: (ctx) => `${metricKey}: ${ctx.parsed.y.toFixed(digits)}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: "#98a2c7", font: { size: 10 }, autoSkip: false, maxRotation: 0 },
            grid: { display: false, drawBorder: false },
          },
          y: {
            min: yMin,
            max: yMax,
            ticks: {
              color: "#98a2c7",
              font: { size: 10 },
              maxTicksLimit: 3,
              callback: (v) => v.toFixed(digits),
            },
            grid: { color: "rgba(255,255,255,0.05)", drawBorder: false },
          },
        },
        onClick: (_, elements) => {
          if (elements && elements.length) {
            selectRound(elements[0].index + 1);
          }
        },
      },
    });
    sparkCharts[metricKey] = { chart, color };
  }

  function buildSparklines() {
    buildSparkline("randi", "#6ea8ff", { digits: 4, lowerBetter: false });
    buildSparkline("nmi", "#9b8cff", { digits: 4, lowerBetter: false });
    buildSparkline("ssr", "#f87171", { digits: 2, lowerBetter: true });
    buildSparkline("mae", "#facc15", { digits: 4, lowerBetter: true });
  }

  function highlightSparkPoint(round) {
    Object.entries(sparkCharts).forEach(([key, { chart, color }]) => {
      const n = state.data.rounds.length;
      const radii = Array.from({ length: n }, (_, i) => (i === round - 1 ? 5 : 2.5));
      const bgs = Array.from({ length: n }, (_, i) => (i === round - 1 ? color : color + "55"));
      const ds = chart.data.datasets[0];
      ds.pointRadius = radii;
      ds.pointBackgroundColor = bgs;
      ds.pointBorderColor = bgs;
      chart.update("none");
    });
  }

  function setMetric(id, value, deltaInfo) {
    const el = $("#" + id);
    el.textContent = value;
    if (deltaInfo) {
      const d = $("#" + id.replace("m-", "d-"));
      d.textContent = deltaInfo.text;
      d.className = "metric-delta " + deltaInfo.cls;
    }
  }

  function renderMetrics(round) {
    const idx = round - 1;
    const r = state.data.rounds[idx];
    const m = r.info.metrics;
    const prev = idx > 0 ? state.data.rounds[idx - 1].info.metrics : null;
    setMetric("m-randi", fmt(m.randi, 4), prev && deltaSpan(m.randi, prev.randi, { digits: 4 }));
    setMetric("m-nmi", fmt(m.nmi, 4), prev && deltaSpan(m.nmi, prev.nmi, { digits: 4 }));
    setMetric("m-ssr", fmt(m.ssr, 2), prev && deltaSpan(m.ssr, prev.ssr, { digits: 2, lowerBetter: true }));
    setMetric("m-mae", fmt(m.mae, 4), prev && deltaSpan(m.mae, prev.mae, { digits: 4, lowerBetter: true }));
    setMetric("m-time", fmt(m.used_time, 2), prev && deltaSpan(m.used_time, prev.used_time, { digits: 2, lowerBetter: true }));
  }

  function renderKnowledge(round) {
    const r = state.data.rounds[round - 1];
    const log = r.log || {};
    const kn = log.knowledge_needed;
    $("#round-badge").textContent = `Round ${round}`;
    if (!kn) {
      $("#kn-missing").textContent = "No knowledge-retrieval entry was logged for this round.";
      $("#kn-need-theory").textContent = "—";
      $("#kn-need-theory").className = "kn-tag no";
      $("#kn-keyword").textContent = "—";
      $("#kn-queries").innerHTML = "";
      $("#kn-retrieved").innerHTML = `<p class="muted">No retrieved knowledge logged for this round.</p>`;
      $("#kn-retrieved-len").textContent = "";
      return;
    }
    $("#kn-missing").textContent = kn.missing_or_problematic_knowledge || "—";
    const need = !!kn.need_new_geographical_theory;
    const tag = $("#kn-need-theory");
    tag.textContent = need ? "Yes — fetching new geo theory" : "No";
    tag.className = "kn-tag " + (need ? "yes" : "no");
    const ng = kn.new_geo_knowledge_to_fetch || {};
    $("#kn-keyword").innerHTML = ng.keyword
      ? `<strong>${escapeHtml(ng.keyword)}</strong> <span class="muted small">— ${escapeHtml(ng.category || "")}</span>`
      : "—";
    const ul = $("#kn-queries");
    ul.innerHTML = "";
    (kn.search_queries || []).forEach((q) => {
      const li = document.createElement("li");
      li.innerHTML = `<span></span><span>${escapeHtml(q)}</span>`;
      ul.appendChild(li);
    });

    const retrieved = log.knowledge_retrieved || "";
    $("#kn-retrieved-len").textContent = retrieved
      ? `${(retrieved.length / 1000).toFixed(1)}k chars`
      : "";
    if (retrieved) {
      try {
        $("#kn-retrieved").innerHTML = marked.parse(retrieved);
      } catch (e) {
        $("#kn-retrieved").textContent = retrieved;
      }
    } else {
      $("#kn-retrieved").innerHTML = `<p class="muted">No retrieved knowledge logged for this round.</p>`;
    }
  }

  function renderPrompt(round) {
    const r = state.data.rounds[round - 1];
    const cur = r.system_message || "";
    $("#prompt-text").textContent = cur;
    if (round > 1) {
      const prev = state.data.rounds[round - 2].system_message || "";
      renderDiff($("#prompt-diff-view"), prev, cur);
    } else {
      renderDiff($("#prompt-diff-view"), "", cur);
    }
  }

  function renderCode(round) {
    const r = state.data.rounds[round - 1];
    const code = r.code || "";
    const codeEl = $("#code-text");
    codeEl.textContent = code;
    try { hljs.highlightElement(codeEl); } catch (_) {}
    let stats = { added: 0, removed: 0 };
    if (round > 1) {
      const prev = state.data.rounds[round - 2].code || "";
      stats = renderDiff($("#code-diff-view"), prev, code);
    } else {
      renderDiff($("#code-diff-view"), "", code);
      stats = { added: code.split("\n").length, removed: 0 };
    }
    const totalLines = code.split("\n").length;
    $("#code-stats").innerHTML = `
      <span class="pill">${totalLines} lines</span>
      <span class="pill add">+${stats.added}</span>
      <span class="pill del">−${stats.removed}</span>
    `;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------- Tabs ----------
  function bindTabs(rootSel) {
    const root = $(rootSel);
    root.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab");
      if (!btn) return;
      const target = btn.dataset.tab;
      const card = root.closest(".card") || root.parentElement;
      card.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
      card.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === target));
    });
  }

  // ---------- Round selection ----------
  function selectRound(round) {
    state.current = round;
    setActiveCell(round);
    renderMetrics(round);
    renderKnowledge(round);
    renderPrompt(round);
    renderCode(round);
    if (Object.keys(sparkCharts).length) highlightSparkPoint(round);
  }

  function nextRound() {
    const next = state.current >= state.data.rounds.length ? 1 : state.current + 1;
    selectRound(next);
  }
  function prevRound() {
    const prev = state.current <= 1 ? state.data.rounds.length : state.current - 1;
    selectRound(prev);
  }

  function toggleAutoplay() {
    const btn = $("#play-btn");
    if (state.autoplay) {
      clearInterval(state.autoplay);
      state.autoplay = null;
      btn.textContent = "▶ Auto Play";
    } else {
      btn.textContent = "❚❚ Pause";
      state.autoplay = setInterval(nextRound, 3500);
    }
  }

  // ---------- Init ----------
  async function init() {
    try {
      const res = await fetch("data.json", { cache: "no-store" });
      state.data = await res.json();
    } catch (e) {
      document.body.innerHTML = `<pre style="color:#f87171;padding:24px;">Failed to load data.json: ${e}\n` +
        `If you opened this page directly with file://, please serve the web/ folder over HTTP, e.g.:\n` +
        `    python -m http.server 8000\n` +
        `Then browse to http://localhost:8000/</pre>`;
      return;
    }
    document.getElementById("title").textContent = state.data.title;
    document.getElementById("subtitle").textContent = state.data.subtitle;

    // Configure marked
    if (window.marked) {
      marked.setOptions({ breaks: true, mangle: false, headerIds: false });
    }
    if (window.hljs) {
      hljs.registerLanguage("python", window.hljsDefineLanguage ? window.hljsDefineLanguage : hljs.getLanguage("python"));
    }

    buildTimeline();
    buildSparklines();

    // Buttons
    $("#prev-btn").addEventListener("click", prevRound);
    $("#next-btn").addEventListener("click", nextRound);
    $("#play-btn").addEventListener("click", toggleAutoplay);

    // Tabs
    bindTabs("#prompt-tabs");
    bindTabs("#code-tabs");

    // Keyboard: ←/→ to navigate
    window.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowLeft") { prevRound(); }
      else if (e.key === "ArrowRight") { nextRound(); }
      else if (e.key === " ") { e.preventDefault(); toggleAutoplay(); }
    });

    selectRound(1);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
