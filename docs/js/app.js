/* Page logic. Every number comes from assets/results.json (written by src/export_web.py);
 * the live classifier uses js/model.js. Nothing is sent to a server. */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const f3 = (x) => Number(x).toFixed(3);
  const ci = (c) => `${f3(c[0])} to ${f3(c[1])}`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pretty = (s) => s.replace(/_/g, " ").replace(/\?$/, "");

  let R = null;              // results.json
  const redraw = [];         // re-run when the colour theme changes
  const charts = {};

  // Translucent version of a #rrggbb token, as rgba() so canvases and Chart.js can parse it.
  function alpha(color, a) {
    const h = color.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a.toFixed(3)})`;
  }

  // Charts are updated in place. Destroying and recreating one on every slider tick crashed
  // Chart.js's shared animation loop and blanked every chart on the page.
  function chart(id, config) {
    const c = charts[id];
    if (c && c.config.type === config.type) {
      c.data = config.data;
      c.options = config.options;
      c.update("none");
      return c;
    }
    if (c) c.destroy();
    charts[id] = new Chart(document.getElementById(id), config);
    return charts[id];
  }

  function theme() {
    const d = Chart.defaults;
    d.color = css("--muted");
    d.borderColor = css("--line");
    d.font.family = css("--sans");
    d.font.size = 12;
    d.animation = false;
    d.maintainAspectRatio = false;
    d.elements.line.borderWidth = 2;
    d.elements.point.hitRadius = 8;
    d.plugins.legend.position = "bottom";
    d.plugins.legend.align = "start";
    d.plugins.legend.labels.usePointStyle = true;
    d.plugins.legend.labels.pointStyle = "line";
    d.plugins.legend.labels.boxWidth = 22;
    d.plugins.legend.labels.padding = 16;
    d.plugins.legend.labels.filter = (item, data) => data.datasets[item.datasetIndex].inLegend !== false;
    d.plugins.legend.onClick = null;
    const t = d.plugins.tooltip;
    t.backgroundColor = css("--text"); t.titleColor = css("--bg"); t.bodyColor = css("--bg");
    t.padding = 10; t.cornerRadius = 8; t.displayColors = false; t.titleFont = { weight: "600" };
  }
  // Axis with a title, faint grid and no border line.
  const ax = (title, extra = {}) => ({
    title: { display: !!title, text: title, color: css("--muted"), font: { size: 12 } },
    grid: { color: css("--line"), drawTicks: false }, border: { display: false },
    ...extra, ticks: { padding: 8, maxTicksLimit: 7, ...(extra.ticks || {}) },
  });
  const pctTick = { callback: (x) => pct(x, 0) };

  function seg(id, onChange) {
    const el = document.getElementById(id);
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.v);
    });
    return {
      get: () => el.querySelector("button.on").dataset.v,
      set: (v) => { el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.v === v)); onChange(v); },
    };
  }
  function stats(el, items) {
    el.innerHTML = items.map(([v, l, cls]) => `<div><b class="${cls || ""}">${v}</b><span>${l}</span></div>`).join("");
  }

  // ---------- Figure 1: live classifier ----------

  function initDemo() {
    const input = $("#demo-input"), out = $("#demo-out"), ex = $("#demo-examples");
    let k = 5, truth = null, model = null;
    const thr = () => R.slider.frozen_thresholds.find((t) => t.k === k).threshold;
    const groupOf = (intent) => R.h1.groups.find((g) => g.includes(intent));

    for (const kind of ["clear", "ambiguous", "misclassified"]) {
      for (const e of R.examples[kind].slice(0, 2)) {
        const b = document.createElement("button");
        b.className = `chip ${kind}`;
        b.innerHTML = `<i></i>${esc(e.text)}`;
        b.title = `${kind} · true label: ${e.label}`;
        b.onclick = () => { input.value = e.text; truth = e; run(); };
        ex.appendChild(b);
      }
    }
    IntentModel.load("assets/").then((m) => { model = m; run(); })
      .catch((e) => { out.innerHTML = `<p class="loading">Could not load the model: ${esc(e)}</p>`; });

    function run() {
      if (!model) return;
      const text = input.value;
      if (!normalizeText(text)) { out.innerHTML = `<p class="loading">Type a message above.</p>`; return; }
      const r = model.predict(text);
      const top = model.classes[r.top], p = r.proba[r.top], t = thr(), answer = p >= t;
      const ranks = r.order.slice(0, 3).map((i) => `<div class="rank"><span class="name" title="${esc(model.classes[i])}">${esc(model.classes[i])}</span>
        <span class="p">${pct(r.proba[i])}</span><span class="barline"><i style="width:${(100 * r.proba[i]).toFixed(1)}%"></i></span></div>`).join("");
      const maxAbs = Math.max(1e-9, ...r.wordScore.map(Math.abs));
      const hl = r.words.map((w, i) => {
        const s = r.wordScore[i] / maxAbs;
        const bg = s >= 0 ? alpha(css("--good"), 0.42 * s) : alpha(css("--bad"), 0.42 * -s);
        return `<span style="background:${bg}" title="contribution ${r.wordScore[i].toFixed(3)}">${esc(w)}</span>`;
      }).join(" ");
      const feats = r.contrib.filter((c) => c.value > 0).slice(0, 8).map((c) =>
        `<span class="feat" title="${c.label.kind} n-gram, +${c.value.toFixed(3)}">${esc(c.label.kind === "char" ? c.label.text.replace(/ /g, "␣") : c.label.text)}</span>`).join("");
      const g = groupOf(top);
      const others = g ? g.filter((x) => x !== top) : [];
      const truthNote = truth && normalizeText(text) === normalizeText(truth.text)
        ? `<p class="groupnote">True label in the test set: <code>${esc(truth.label)}</code>. ${truth.label === top ? "The model is right." : "<b>The model is wrong here.</b>"}</p>` : "";
      out.innerHTML = `<div class="demo-grid">
        <div>
          <p class="verdict ${answer ? "answer" : "handoff"}">${answer ? "Answer automatically" : "Hand off to a human"}</p>
          <p class="small muted" style="margin:0">Top probability ${pct(p)} is ${answer ? "at or above" : "below"} the threshold.</p>
          <div class="meter"><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div>
            <div class="tick" style="left:${(100 * t).toFixed(1)}%"><span>threshold ${pct(t, 0)}</span></div></div>
          <div class="ctl" style="margin-top:14px">A wrong answer costs
            <div class="seg" id="demo-k">${[2, 5, 10].map((v) => `<button data-v="${v}" class="${v === k ? "on" : ""}">${v}×</button>`).join("")}</div>a handoff</div>
          ${truthNote}
          <p class="sub">Top three intents</p>${ranks}
        </div>
        <div>
          <p class="sub">Why <code>${esc(top)}</code></p>
          <p class="hl">${hl}</p>
          <p class="sub">Strongest features</p>
          <div class="feats">${feats || '<span class="muted small">None of these words are in the vocabulary.</span>'}</div>
          <p class="groupnote">Word and character n-grams that raise this intent most. ␣ marks a word edge.</p>
          ${g ? `<p class="groupnote"><b style="color:var(--amber)">Confusable intent.</b> <code>${esc(top)}</code> shares a group with ${others.length} other intent${others.length > 1 ? "s" : ""}${others.length <= 6 ? ": " + others.map((x) => `<code>${esc(x)}</code>`).join(" ") : ""}. See <a href="#h1">H1</a>.</p>` : ""}
        </div></div>`;
      $("#demo-k").addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (b) { k = +b.dataset.v; run(); }
      });
    }
    let timer;
    input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 100); });
    redraw.push(run);
  }

  // ---------- Figure 2: scores with intervals ----------

  function renderScores() {
    const lo = 0.5, hi = 1.0, pos = (x) => (100 * (Math.min(hi, Math.max(lo, x)) - lo)) / (hi - lo);
    const rows = R.scores.filter((r) => r.id !== "B0_majority").sort((a, b) => b.test.macro_f1 - a.test.macro_f1);
    const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const grid = ticks.map((t) => `<i class="gridl" style="left:${pos(t)}%"></i>`).join("");
    const iv = (r) => `<div class="iv">${grid}
      <i class="ci cv" style="left:${pos(r.cv.ci[0])}%;width:${pos(r.cv.ci[1]) - pos(r.cv.ci[0])}%"></i><i class="dot cv" style="left:${pos(r.cv.macro_f1)}%"></i>
      <i class="ci te" style="left:${pos(r.test.ci[0])}%;width:${pos(r.test.ci[1]) - pos(r.test.ci[0])}%"></i><i class="dot te" style="left:${pos(r.test.macro_f1)}%"></i></div>`;
    const b0 = R.scores.find((r) => r.id === "B0_majority");
    $("#scores").innerHTML =
      `<span class="h">Model</span><span class="h ivh">Macro-F1</span><span class="h v">CV</span><span class="h v">Test</span>` +
      rows.map((r) => `<span class="rule"></span>
        <span class="name ${r.id === "M1_L2" ? "main" : ""}">${esc(r.name)}${r.id === "M1_L2" ? "<small>main model</small>" : r.id.startsWith("B") ? "<small>baseline</small>" : ""}</span>
        ${iv(r)}<span class="v">${f3(r.cv.macro_f1)}</span><span class="v ${r.id === "M1_L2" ? "main" : ""}">${f3(r.test.macro_f1)}</span>`).join("") +
      `<span class="rule"></span><span class="axpad"></span><div class="axis">${ticks.map((t) => `<span style="left:${pos(t)}%">${t.toFixed(1)}</span>`).join("")}</div><span></span><span></span>
       <span class="rule"></span><span class="name">${esc(b0.name)}<small>baseline, off the scale</small></span><span class="iv"></span><span class="v">${f3(b0.cv.macro_f1)}</span><span class="v">${f3(b0.test.macro_f1)}</span>`;
  }

  // ---------- Figure 3: H2 handoff ----------

  function prepare(conf, correct) {
    const idx = conf.map((_, i) => i).sort((a, b) => conf[b] - conf[a]);
    const c = idx.map((i) => conf[i]);
    const cumW = [0];
    for (const i of idx) cumW.push(cumW[cumW.length - 1] + (1 - correct[i]));
    return { conf: c, cumW, n: c.length, err: cumW[c.length] / c.length };
  }
  // Number of messages with confidence >= t, for confidences sorted descending.
  function answeredCount(conf, t) {
    let lo = 0, hi = conf.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (conf[m] >= t) lo = m + 1; else hi = m; }
    return lo;
  }
  function at(d, t, k) {
    const m = answeredCount(d.conf, t), wrong = d.cumW[m], cov = m / d.n;
    return { m, cov, wrong, acc: m ? 1 - wrong / m : NaN, cost: (k * wrong + (d.n - m)) / d.n,
             randomCost: k * d.err * cov + (1 - cov), answerAll: k * d.err };
  }
  // Same rule as src/analysis.py cost_threshold: only cut between distinct confidence values.
  function bestThreshold(d, k) {
    let best = { cost: 1, t: 1.0 };
    for (let m = 1; m <= d.n; m++) {
      if (m < d.n && d.conf[m - 1] === d.conf[m]) continue;
      const cost = (k * d.cumW[m] + (d.n - m)) / d.n;
      if (cost < best.cost) best = { cost, t: d.conf[m - 1] };
    }
    return best.t;
  }
  // Split 100 dots across shares (largest-remainder rounding).
  function allot(shares, n = 100) {
    const raw = shares.map((s) => s * n), base = raw.map(Math.floor);
    let left = n - base.reduce((a, b) => a + b, 0);
    raw.map((x, i) => [x - base[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left-- > 0) base[i]++; });
    return base;
  }
  function waffle(title, sub, shares, classes) {
    const counts = allot(shares);
    const dots = counts.map((c, i) => `<i class="${classes[i]}"></i>`.repeat(c)).join("");
    return `<div class="waffle"><h4>${title}</h4><p>${sub}</p><div class="dots">${dots}</div></div>`;
  }

  function initH2() {
    const D = { test: prepare(R.slider.test.conf, R.slider.test.correct), cv: prepare(R.slider.cv.conf, R.slider.cv.correct) };
    const U = { seen: R.slider.unseen.seen.slice().sort((a, b) => b - a), unseen: R.slider.unseen.unseen.slice().sort((a, b) => b - a) };
    const tEl = $("#h2-t"), kEl = $("#h2-k"), best = $("#h2-best");
    const mode = seg("h2-data", () => draw());
    best.onclick = () => { tEl.value = bestThreshold(D.cv, +kEl.value).toFixed(3); draw(); };
    tEl.oninput = draw; kEl.oninput = draw;
    document.querySelectorAll('[data-jump="unseen"]').forEach((a) => a.addEventListener("click", () => mode.set("unseen")));
    const key = (items) => `<div class="keyrow" style="grid-column:1/-1;margin-top:0">${items.map(([cls, style, label]) =>
      `<span class="key"><i class="${cls}" style="${style}"></i>${label}</span>`).join("")}<span>Each dot is 1% of messages.</span></div>`;

    function draw() {
      const t = +tEl.value, k = +kEl.value, v = mode.get();
      $("#h2-t-val").textContent = t.toFixed(3);
      $("#h2-k-val").textContent = `${k}× a handoff`;
      const frozen = R.slider.frozen_thresholds.find((x) => x.k === k);
      best.textContent = frozen ? `Use the threshold frozen from cross-validation` : `Use the best cross-validation threshold`;
      if (v === "unseen") return drawUnseen(t, k);
      const d = D[v], s = at(d, t, k), tot = d.cumW[d.n];
      stats($("#h2-stats"), [
        [pct(s.cov), "answered automatically"],
        [isNaN(s.acc) ? "–" : pct(s.acc), `accuracy on those answers (${pct(1 - d.err)} if it answered everything)`],
        [s.cost.toFixed(3), `cost per message (answering everything: ${s.answerAll.toFixed(3)}, handing off everything: 1)`, "accent"],
        [s.randomCost.toFixed(3), "cost if the same number were handed off at random", "warm"],
      ]);
      const cls = ["d-right", "d-wrong", "d-caught", "d-waste"];
      const per100 = (x) => (100 * x).toFixed(1);
      $("#h2-waffles").innerHTML =
        waffle("Hand off the least confident", `${per100(s.wrong / d.n)} wrong answers reach customers per 100 messages.`,
          [(s.m - s.wrong) / d.n, s.wrong / d.n, (tot - s.wrong) / d.n, (d.n - s.m - (tot - s.wrong)) / d.n], cls) +
        waffle("Hand off the same number at random", `${per100(d.err * s.cov)} wrong answers reach customers per 100 messages.`,
          [s.cov * (1 - d.err), s.cov * d.err, (1 - s.cov) * d.err, (1 - s.cov) * (1 - d.err)], cls) +
        key([["", "background:var(--accent)", "Answered, right"], ["", "background:var(--bad)", "Answered, wrong"],
             ["ring", "color:var(--good)", "Handed off, would have been wrong"], ["ring", "color:var(--ghost)", "Handed off, would have been right"]]);

      $("#h2-title-a").textContent = "Error on answered messages";
      const pts = [];
      for (let m = Math.max(1, Math.round(0.02 * d.n)); m <= d.n; m += Math.max(1, Math.floor(d.n / 300))) pts.push({ x: m / d.n, y: d.cumW[m] / m });
      pts.push({ x: 1, y: d.err });
      chart("h2-risk", { type: "scatter",
        data: { datasets: [
          { label: "Confidence rule", data: pts, showLine: true, pointRadius: 0, borderColor: css("--accent") },
          { label: "Random handoff", data: [{ x: 0.02, y: d.err }, { x: 1, y: d.err }], showLine: true, pointRadius: 0, borderColor: css("--warm"), borderDash: [5, 4] },
          { label: "Current threshold", inLegend: false, order: -1, data: s.m ? [{ x: s.cov, y: 1 - s.acc }] : [], pointRadius: 6, pointBackgroundColor: css("--text"), pointBorderColor: css("--panel"), pointBorderWidth: 2 }] },
        options: { scales: { x: ax("Share of messages answered", { min: 0, max: 1, ticks: pctTick }), y: ax("", { min: 0, ticks: pctTick }) },
          plugins: { tooltip: { callbacks: { title: (c) => c[0].dataset.label, label: (c) => `${pct(c.raw.x)} answered, ${pct(c.raw.y)} wrong` } } } } });

      $("#h2-title-b").textContent = `Cost per message at ${k}×`;
      const conf = [], rnd = [];
      for (let x = 0; x <= 1.0001; x += 0.01) { const a = at(d, x, k); conf.push({ x, y: a.cost }); rnd.push({ x, y: a.randomCost }); }
      chart("h2-cost", { type: "scatter",
        data: { datasets: [
          { label: "Confidence rule", data: conf, showLine: true, pointRadius: 0, borderColor: css("--accent") },
          { label: "Random handoff", data: rnd, showLine: true, pointRadius: 0, borderColor: css("--warm"), borderDash: [5, 4] },
          ...(frozen ? [{ label: "Frozen threshold", data: [{ x: frozen.threshold, y: at(d, frozen.threshold, k).cost }], pointRadius: 6, pointStyle: "rectRot", pointBackgroundColor: css("--good"), pointBorderColor: css("--panel"), pointBorderWidth: 2, inLegend: false }] : []),
          { label: "Current threshold", inLegend: false, order: -1, data: [{ x: t, y: s.cost }], pointRadius: 6, pointBackgroundColor: css("--text"), pointBorderColor: css("--panel"), pointBorderWidth: 2 }] },
        options: { scales: { x: ax("Threshold", { min: 0, max: 1 }), y: ax("", { min: 0 }) },
          plugins: { tooltip: { callbacks: { title: (c) => c[0].dataset.label, label: (c) => `threshold ${c.raw.x.toFixed(2)}, cost ${c.raw.y.toFixed(3)}` } } } } });

      $("#h2-note").innerHTML = `<b>Figure 3.</b> ` + (v === "test"
        ? `Test set, ${d.n.toLocaleString()} messages scored once by the frozen model. The black dot is your threshold. The green diamond is the threshold frozen from cross-validation for this cost, so it is an honest out-of-sample choice.`
        : `Cross-validation, out-of-fold predictions for all ${d.n.toLocaleString()} training messages. This is where the thresholds were chosen. The green diamond marks the frozen one.`)
        + ` Area under the risk–coverage curve: ${f3(R.h2[v].aurc)} against ${f3(R.h2[v].aurc_random)} for random handoff.`;
    }

    function drawUnseen(t, k) {
      const off = (arr) => 1 - answeredCount(arr, t) / arr.length;
      const ou = off(U.unseen), os = off(U.seen);
      const ref = R.unseen.cv.at_cost_thresholds[String(k)];
      stats($("#h2-stats"), [
        [pct(ou), "of unseen-intent messages handed off", "accent"],
        [pct(os), "of seen-intent messages handed off, the price", "warm"],
        [f3(R.unseen.cv.auroc_seen_vs_unseen), `AUROC from confidence alone (${f3(R.unseen.test.auroc_seen_vs_unseen)} on the test set)`],
        [ref ? ref.threshold.toFixed(3) : "–", ref ? `threshold for a ${k}× cost, chosen on seen intents only` : "reference thresholds exist for 2×, 5× and 10×"],
      ]);
      $("#h2-waffles").innerHTML =
        waffle("Messages from the 10 unseen intents", "Every automatic answer here is wrong, because the right intent does not exist.", [ou, 1 - ou], ["d-caught", "d-miss"]) +
        waffle("Messages from the 67 seen intents", "Handing these off is the price of catching the unseen ones.", [1 - os, os], ["d-right", "d-waste"]) +
        key([["ring", "color:var(--good)", "Unseen, handed off"], ["", "background:var(--bad)", "Unseen, answered"],
             ["", "background:var(--accent)", "Seen, answered"], ["ring", "color:var(--ghost)", "Seen, handed off"]]);

      $("#h2-title-a").textContent = "Confidence, seen and unseen";
      const bins = 25, hist = (arr) => {
        const h = new Array(bins).fill(0);
        for (const x of arr) h[Math.min(bins - 1, Math.floor(x * bins))]++;
        return h.map((c, i) => ({ x: (i + 0.5) / bins, y: c / arr.length }));
      };
      const hs = hist(U.seen), hu = hist(U.unseen);
      const ymax = Math.max(...hs.map((p) => p.y), ...hu.map((p) => p.y)) * 1.05;
      chart("h2-risk", { type: "bar",
        data: { datasets: [
          { type: "line", label: "Threshold", inLegend: false, data: [{ x: t, y: 0 }, { x: t, y: ymax }], borderColor: css("--text"), borderWidth: 1.5, pointRadius: 0 },
          { label: "Seen intents", data: hs, backgroundColor: alpha(css("--accent"), 0.6), barPercentage: 1, categoryPercentage: 1, grouped: false, pointStyle: "rect" },
          { label: "Unseen intents", data: hu, backgroundColor: alpha(css("--warm"), 0.6), barPercentage: 1, categoryPercentage: 1, grouped: false, pointStyle: "rect" }] },
        options: { scales: { x: ax("Top probability", { type: "linear", min: 0, max: 1 }), y: ax("", { max: ymax, ticks: pctTick }) },
          plugins: { tooltip: { enabled: false } } } });

      $("#h2-title-b").textContent = "Share handed off";
      const cs = [], cu = [];
      for (let x = 0; x <= 1.0001; x += 0.01) { cs.push({ x, y: off2(U.seen, x) }); cu.push({ x, y: off2(U.unseen, x) }); }
      chart("h2-cost", { type: "scatter",
        data: { datasets: [
          { label: "Unseen intents", data: cu, showLine: true, pointRadius: 0, borderColor: css("--warm") },
          { label: "Seen intents", data: cs, showLine: true, pointRadius: 0, borderColor: css("--accent") },
          { label: "Current threshold", inLegend: false, order: -1, data: [{ x: t, y: ou }, { x: t, y: os }], pointRadius: 6, pointBackgroundColor: css("--text"), pointBorderColor: css("--panel"), pointBorderWidth: 2 }] },
        options: { scales: { x: ax("Threshold", { min: 0, max: 1 }), y: ax("", { min: 0, max: 1, ticks: pctTick }) },
          plugins: { tooltip: { callbacks: { title: (c) => c[0].dataset.label, label: (c) => `threshold ${c.raw.x.toFixed(2)}: ${pct(c.raw.y)} handed off` } } } } });
      const f5 = R.unseen.test.at_frozen_thresholds["5"];
      $("#h2-note").innerHTML = `<b>Figure 3.</b> Unseen-intent mode. A model trained on 67 intents, with out-of-fold confidences for ${U.seen.length.toLocaleString()} seen and ${U.unseen.length.toLocaleString()} unseen training messages. On the test set, the threshold for a 5× cost hands off ${pct(f5.unseen_handed_off)} of unseen messages and ${pct(f5.seen_handed_off)} of seen ones.`;
    }
    const off2 = (arr, x) => 1 - answeredCount(arr, x) / arr.length;
    redraw.push(draw);
    draw();
  }

  function renderE1() {
    const cv = R.e1.cv, te = R.e1.test;
    $("#e1-table").innerHTML = `<thead><tr><th></th><th class="n">Max probability</th><th class="n">E1</th></tr></thead><tbody>
      <tr><td>Error-detection AUROC, cross-validation</td><td class="n">${f3(cv.auroc_error_detection.max_prob)}</td><td class="n"><b>${f3(cv.auroc_error_detection.e1)}</b></td></tr>
      <tr><td>Error-detection AUROC, test set</td><td class="n">${f3(te.auroc.max_prob)}</td><td class="n"><b>${f3(te.auroc.e1)}</b></td></tr>
      <tr><td>Gain in AUROC, 95% interval</td><td class="n" colspan="2">+${f3(cv.e1_minus_max_prob.auroc_diff)} (${ci(cv.e1_minus_max_prob.auroc_diff_ci95)})</td></tr></tbody>`;
  }

  // ---------- Figure 4: H3 reliability ----------

  function initH3() {
    const ids = Object.keys(R.h3.cv);
    let sel = "M1_L2";
    const mode = seg("h3-data", () => draw());
    $("#h3-picker").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) { sel = b.dataset.id; draw(); }
    });
    function draw() {
      const src = R.h3[mode.get()], m = src[sel];
      const maxN = Math.max(...m.bins.map((b) => b.n));
      const ghosts = ids.filter((id) => id !== sel).map((id) => ({ label: src[id].name, inLegend: false, data: src[id].bins.map((b) => ({ x: b.conf, y: b.acc })),
        showLine: true, pointRadius: 0, pointHitRadius: 0, borderColor: css("--ghost"), borderWidth: 1 }));
      chart("h3-chart", { type: "scatter",
        data: { datasets: [
          { label: m.name, data: m.bins.map((b) => ({ x: b.conf, y: b.acc, n: b.n })), showLine: true, borderColor: css("--accent"), backgroundColor: css("--accent"),
            pointRadius: m.bins.map((b) => 2.5 + 8 * Math.sqrt(b.n / maxN)), pointBorderColor: css("--panel"), pointBorderWidth: 1.5 },
          { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, pointHitRadius: 0, borderColor: css("--muted"), borderDash: [4, 4], borderWidth: 1.2 },
          ...ghosts] },
        options: { scales: { x: ax("Confidence", { min: 0, max: 1, ticks: pctTick }), y: ax("Accuracy", { min: 0, max: 1, ticks: pctTick }) },
          plugins: { tooltip: { filter: (c) => c.datasetIndex === 0, callbacks: { title: () => m.name, label: (c) => `says ${pct(c.raw.x, 0)}, right ${pct(c.raw.y, 0)} of the time (${c.raw.n} messages)` } } } } });
      const gap = m.accuracy - m.mean_confidence;
      stats($("#h3-stats"), [
        [f3(m.ece), `calibration error (ECE), 95% interval ${ci(m.ece_ci)}`, "accent"],
        [`${pct(m.mean_confidence, 0)} → ${pct(m.accuracy, 0)}`, `says, then is right: ${gap >= 0 ? "underconfident" : "overconfident"} by ${Math.abs(100 * gap).toFixed(0)} points`],
      ]);
      const order = ids.slice().sort((a, b) => src[b].macro_f1 - src[a].macro_f1);
      const bestEce = ids.reduce((a, b) => (src[a].ece <= src[b].ece ? a : b));
      $("#h3-picker").innerHTML = `<div class="prow h"><span>Model</span><span>Macro-F1</span><span>ECE</span></div>` +
        order.map((id, i) => `<button class="prow ${id === sel ? "on" : ""}" data-id="${id}" aria-pressed="${id === sel}">
          <span>${esc(src[id].name)}${i === 0 ? " <small>· most accurate</small>" : ""}${id === bestEce ? " <small>· best calibrated</small>" : ""}</span>
          <span>${f3(src[id].macro_f1)}</span><span>${f3(src[id].ece)}</span></button>`).join("");
    }
    redraw.push(draw);
    draw();
  }

  // ---------- Figure 5: H4 ----------

  function initH4() {
    const N_FEATURES = 37106;   // size of the full word + char TF-IDF vocabulary (docs/assets/model.json)
    const mode = seg("h4-data", () => draw());
    const dEl = $("#h4-d");
    dEl.oninput = draw;
    function draw() {
      const v = mode.get(), curve = R.h4[v], full = R.h4[v + "_full"], i = +dEl.value, c = curve[i];
      $("#h4-d-val").textContent = c.d;
      const ev = R.h4.cv[i].explained_variance;
      stats($("#h4-stats"), [
        [pct(c.share), `of the full model's macro-F1 (${f3(c.macro_f1)} of ${f3(full)})`, "accent"],
        [pct(ev, 0), "of the TF-IDF variance explained", "warm"],
        [`${Math.round(N_FEATURES / c.d).toLocaleString()}×`, `fewer features: ${c.d} instead of about 37,000`],
        [c.share >= 0.9 && c.d <= 300 ? "Passes" : c.d > 300 ? "Above 300" : "Below 90%", "against the pre-registered bar"],
      ]);
      const ds = [
        { label: "Macro-F1 kept", data: curve.map((p) => ({ x: p.d, y: p.share })), showLine: true, borderColor: css("--accent"), backgroundColor: css("--accent"),
          pointRadius: curve.map((_, j) => (j === i ? 7 : 3.5)), pointBorderColor: css("--panel"), pointBorderWidth: 1.5 },
        { label: "90% bar", data: [{ x: 20, y: 0.9 }, { x: 1200, y: 0.9 }], showLine: true, pointRadius: 0, pointHitRadius: 0, borderColor: css("--muted"), borderDash: [4, 4], borderWidth: 1.2 }];
      if (v === "cv") ds.push({ label: "Variance explained", data: R.h4.cv.map((p) => ({ x: p.d, y: p.explained_variance })), showLine: true, borderColor: css("--warm"), backgroundColor: css("--warm"),
        borderDash: [2, 3], pointRadius: curve.map((_, j) => (j === i ? 6 : 2.5)), pointBorderColor: css("--panel"), pointBorderWidth: 1.5 });
      chart("h4-chart", { type: "scatter", data: { datasets: ds },
        options: { scales: { x: ax("SVD dimensions (log scale)", { type: "logarithmic", min: 20, max: 1200, ticks: { maxTicksLimit: 20, callback: (x) => ([25, 50, 100, 200, 300, 500, 1000].includes(x) ? x : "") } }),
                             y: ax("", { min: 0, max: 1, ticks: pctTick }) },
          plugins: { tooltip: { filter: (c) => c.dataset.label !== "90% bar", callbacks: { title: (c) => `${c[0].raw.x} dimensions`, label: (c) => `${c.dataset.label}: ${pct(c.raw.y)}` } } } } });
    }
    redraw.push(draw);
    draw();
  }

  // ---------- Figure 6: H1 ----------

  function initH1() {
    const names = R.h1.confusion.order, K = names.length;
    const mode = seg("h1-data", () => draw());
    const mergeEl = $("#h1-merge"), edgeEl = $("#h1-edge");
    mergeEl.onchange = draw; edgeEl.oninput = draw;
    const canvas = $("#h1-matrix"), tip = $("#h1-tip");
    let view = null, hover = null;

    // Connected components of "confused at least t times in total", from cross-validation errors.
    // Same rule as src/h1.py confusable_groups.
    function groupsAt(t) {
      const M = R.h1.confusion.cv, parent = names.map((_, i) => i);
      const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) if (M[i][j] + M[j][i] >= t) parent[find(i)] = find(j);
      const comp = new Map();
      for (let i = 0; i < K; i++) { const r = find(i); if (!comp.has(r)) comp.set(r, []); comp.get(r).push(i); }
      return [...comp.values()].filter((g) => g.length >= 2)
        .map((g) => g.sort((a, b) => names[a].localeCompare(names[b])))
        .sort((a, b) => b.length - a.length || names[a[0]].localeCompare(names[b[0]]));
    }
    function metrics(M) {
      const n = M.length, col = new Array(n).fill(0);
      let total = 0, diag = 0, f1 = 0;
      M.forEach((row) => row.forEach((v, j) => { col[j] += v; total += v; }));
      for (let i = 0; i < n; i++) {
        const tp = M[i][i], rs = M[i].reduce((a, b) => a + b, 0);
        diag += tp;
        const p = col[i] ? tp / col[i] : 0, r = rs ? tp / rs : 0;
        f1 += p + r ? (2 * p * r) / (p + r) : 0;
      }
      return { acc: diag / total, f1: f1 / n, errors: total - diag, n };
    }

    function draw() {
      const t = +edgeEl.value, v = mode.get(), M = R.h1.confusion[v], merged = mergeEl.checked;
      $("#h1-edge-val").textContent = `${t} times${t === 5 ? " (pre-registered)" : ""}`;
      const groups = groupsAt(t);
      const gOf = new Array(K).fill(-1);
      groups.forEach((g, gi) => g.forEach((i) => (gOf[i] = gi)));
      const rest = names.map((_, i) => i).filter((i) => gOf[i] < 0).sort((a, b) => names[a].localeCompare(names[b]));
      const perm = groups.flat().concat(rest);

      // (a) share of errors inside groups, and what chance alone would give (src/h1.py chance_share)
      let errors = 0, inside = 0, chance = 0;
      const perGroup = groups.map(() => 0);
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        if (i === j || !M[i][j]) continue;
        errors += M[i][j];
        if (gOf[i] >= 0) {
          chance += (M[i][j] * (groups[gOf[i]].length - 1)) / (K - 1);
          if (gOf[i] === gOf[j]) { inside += M[i][j]; perGroup[gOf[i]] += M[i][j]; }
        }
      }
      const base = metrics(M);
      // merged matrix: one label per group
      const label = perm.map((i) => (gOf[i] >= 0 ? gOf[i] : groups.length + rest.indexOf(i)));
      const nL = groups.length + rest.length;
      const MM = Array.from({ length: nL }, () => new Array(nL).fill(0));
      perm.forEach((i, a) => perm.forEach((j, b) => { MM[label[a]][label[b]] += M[i][j]; }));
      const after = metrics(MM);

      const share = errors ? inside / errors : 0, ch = errors ? chance / errors : 0;
      stats($("#h1-stats"), [
        [pct(share), `of ${errors} errors fall inside a group`, "accent"],
        [pct(ch), `expected if errors were random${ch > 0 ? `, so ${(share / ch).toFixed(1)}× chance` : ""}`, "warm"],
        [`${groups.length}`, groups.length ? `groups, the largest with ${groups[0].length} intents` : "groups"],
        merged ? [`${pct(base.acc)} → ${pct(after.acc)}`, `accuracy after merging 77 labels into ${after.n} (macro-F1 ${f3(base.f1)} → ${f3(after.f1)}, errors ${base.errors} → ${after.errors})`]
               : [pct(base.acc), `accuracy on 77 intents (macro-F1 ${f3(base.f1)}, top-3 accuracy ${pct(v === "cv" ? R.h1.top3_cv : R.h1.test.top3_accuracy)})`],
      ]);

      if (merged) {
        view = { M: MM, n: nL, spans: [], name: (a) => (a < groups.length ? `group ${a + 1} (${groups[a].length} intents)` : names[rest[a - groups.length]]) };
      } else {
        const P = perm.map((i) => perm.map((j) => M[i][j]));
        const spans = [];
        let pos = 0;
        for (const g of groups) { spans.push([pos, pos + g.length]); pos += g.length; }
        view = { M: P, n: K, spans, name: (a) => names[perm[a]] };
      }
      paint();
      $("#h1-groups").innerHTML = groups.length ? groups.map((g, gi) => `<div class="group ${merged ? "merged" : ""}">
        <div class="head"><b>Group ${gi + 1}</b> · ${g.length} intents · ${perGroup[gi]} errors inside${merged ? ", now counted as right" : ""}</div>
        ${g.map((i) => `<span class="pill">${esc(names[i])}</span>`).join("")}</div>`).join("") : `<p class="muted small">No pair of intents is confused this often.</p>`;
    }

    function paint() {
      const { M, n, spans } = view;
      const size = canvas.parentElement.getBoundingClientRect().width, dpr = window.devicePixelRatio || 1;
      canvas.style.height = size + "px";
      canvas.width = Math.round(size * dpr); canvas.height = Math.round(size * dpr);
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      const cell = size / n, ink = css("--accent");
      ctx.fillStyle = css("--bg"); ctx.fillRect(0, 0, size, size);
      let max = 1;
      M.forEach((row, i) => row.forEach((x, j) => { if (i !== j) max = Math.max(max, x); }));
      M.forEach((row, i) => row.forEach((x, j) => {
        if (i === j || !x) return;
        ctx.fillStyle = alpha(ink, 0.14 + 0.86 * Math.log1p(x) / Math.log1p(max));
        ctx.fillRect(j * cell, i * cell, Math.max(cell - 0.5, 1), Math.max(cell - 0.5, 1));
      }));
      ctx.strokeStyle = css("--warm"); ctx.lineWidth = 1.3;
      for (const [a, b] of spans) ctx.strokeRect(a * cell + 0.5, a * cell + 0.5, (b - a) * cell - 1, (b - a) * cell - 1);
      if (hover) { ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5; ctx.strokeRect(hover.j * cell, hover.i * cell, cell, cell); }
    }
    canvas.addEventListener("mousemove", (e) => {
      if (!view) return;
      const rect = canvas.getBoundingClientRect(), cell = rect.width / view.n;
      const j = Math.floor((e.clientX - rect.left) / cell), i = Math.floor((e.clientY - rect.top) / cell);
      if (i < 0 || j < 0 || i >= view.n || j >= view.n) return;
      hover = { i, j };
      const x = view.M[i][j];
      tip.innerHTML = i === j ? `<b>${esc(view.name(i))}</b><br>${x} right` : `<b>${x}</b> message${x === 1 ? "" : "s"}<br>true: ${esc(view.name(i))}<br>predicted: ${esc(view.name(j))}`;
      tip.style.left = Math.min(rect.width - 200, Math.max(0, e.clientX - rect.left + 14)) + "px";
      tip.style.top = e.clientY - rect.top + 16 + "px";
      tip.classList.add("show");
      paint();
    });
    canvas.addEventListener("mouseleave", () => { hover = null; tip.classList.remove("show"); paint(); });
    window.addEventListener("resize", () => view && paint());
    redraw.push(draw);
    draw();
  }

  function renderH1Evidence() {
    const s = R.h1.hand_labels.shares;
    const items = [["ambiguous_label", "Ambiguous label", "--accent"], ["vague_message", "Vague message", "--warm"],
                   ["possible_mislabel", "Possible mislabel", "--muted"], ["vocabulary_gap", "Vocabulary gap", "--ghost"]];
    $("#h1-labels").innerHTML = `<div class="stack" role="img" aria-label="Causes of 50 hand-labelled errors">${items.map(([k, , c]) => `<i style="width:${100 * s[k]}%;background:var(${c})"></i>`).join("")}</div>
      <div class="stack-key">${items.map(([k, l, c]) => `<span><i style="background:var(${c})"></i><b>${pct(s[k], 0)}</b>${l}</span>`).join("")}</div>
      <p class="small muted" style="margin-top:14px">Two thirds are intents whose definitions overlap. Only 6% are words the model had never seen, which is what a weak model would look like.</p>`;
    const cm = R.h1.cross_model.top10, keyOf = (p) => `${p[0]}|${p[1]}`;
    const knn = new Set(cm.KNN.map(keyOf)), rf = new Set(cm.RF.map(keyOf));
    const yes = `<span class="mark">✓</span>`, no = `<span class="muted">–</span>`;
    $("#h1-pairs").innerHTML = `<thead><tr><th>Pair confused by logistic regression</th><th class="n">Times</th><th class="n">KNN</th><th class="n">Forest</th></tr></thead><tbody>` +
      cm.LR.map((p) => `<tr><td>${esc(pretty(p[0]))} <span class="muted">↔</span> ${esc(pretty(p[1]))}</td><td class="n">${p[2]}</td>
        <td class="n">${knn.has(keyOf(p)) ? yes : no}</td><td class="n">${rf.has(keyOf(p)) ? yes : no}</td></tr>`).join("") +
      `<tr><td><b>Shared with logistic regression</b></td><td></td><td class="n"><b>${R.h1.cross_model.shared_with_LR.KNN} of 10</b></td><td class="n"><b>${R.h1.cross_model.shared_with_LR.RF} of 10</b></td></tr></tbody>`;
  }

  // ---------- Figure 7: unseen intents ----------

  function renderUnseen() {
    const u = R.unseen.cv, k5 = u.at_cost_thresholds["5"];
    stats($("#unseen-stats"), [
      [pct(k5.unseen_handed_off), "of unseen-intent messages handed off at a 5× cost", "accent"],
      [pct(k5.seen_handed_off), "of seen-intent messages handed off at the same threshold", "warm"],
      [`${u.mean_confidence.unseen.toFixed(2)} vs ${u.mean_confidence.seen.toFixed(2)}`, "mean confidence, unseen against seen"],
      [f3(u.auroc_seen_vs_unseen), `AUROC (${f3(R.unseen.test.auroc_seen_vs_unseen)} on the test set)`],
    ]);
    const rows = Object.entries(u.by_unseen_intent).sort((a, b) => b[1].handed_off_at_k5 - a[1].handed_off_at_k5);
    $("#unseen-bars").innerHTML = rows.map(([name, v]) => {
      const [mis] = Object.keys(v.most_common_predictions);
      return `<span class="name" title="${esc(name)}">${esc(name)}<small>mistaken for ${esc(pretty(mis))}</small></span>
        <span class="t"><i style="width:${(100 * v.handed_off_at_k5).toFixed(1)}%"></i></span><span class="p">${pct(v.handed_off_at_k5, 0)}</span>`;
    }).join("");
  }

  // ---------- Figure 8: learning curve ----------

  function initCurve() {
    const cv = R.learning_curve.cv, te = R.learning_curve.test, el = $("#lc-f");
    const b1 = R.scores.find((s) => s.id === "B1_keyword_k20");
    el.oninput = draw;
    function draw() {
      const i = +el.value, c = cv[i], t = te[i];
      $("#lc-f-val").textContent = pct(c.fraction, 0);
      stats($("#lc-stats"), [
        [f3(t.mean), `test macro-F1, trained on ${Math.round(c.fraction * R.dataset.n_train).toLocaleString()} messages`, "accent"],
        [`${f3(c.mean)} ± ${f3(c.sd)}`, `cross-validated macro-F1, about ${Math.round(c.n).toLocaleString()} messages per fold`, "warm"],
        [f3(b1.cv.macro_f1), "keyword rule trained on everything"],
      ]);
      const big = (arr) => arr.map((_, j) => (j === i ? 7 : 3.5));
      chart("lc-chart", { type: "scatter",
        data: { datasets: [
          { label: "Test set", data: te.map((p, j) => ({ x: cv[j].fraction * R.dataset.n_train, y: p.mean })), showLine: true, borderColor: css("--accent"), backgroundColor: css("--accent"), pointRadius: big(te), pointBorderColor: css("--panel"), pointBorderWidth: 1.5 },
          { label: "Cross-validation", data: cv.map((p) => ({ x: p.n, y: p.mean })), showLine: true, borderColor: css("--warm"), backgroundColor: css("--warm"), pointRadius: big(cv), pointBorderColor: css("--panel"), pointBorderWidth: 1.5 },
          { label: "Keyword rule, all data", data: [{ x: 300, y: b1.cv.macro_f1 }, { x: 12000, y: b1.cv.macro_f1 }], showLine: true, pointRadius: 0, pointHitRadius: 0, borderColor: css("--muted"), borderDash: [4, 4], borderWidth: 1.2 }] },
        options: { scales: { x: ax("Training messages (log scale)", { type: "logarithmic", min: 300, max: 12000, ticks: { maxTicksLimit: 20, callback: (x) => ([500, 1000, 2000, 5000, 10000].includes(x) ? x.toLocaleString() : "") } }),
                             y: ax("Macro-F1", { min: 0.55, max: 1 }) },
          plugins: { tooltip: { filter: (c) => c.datasetIndex < 2, callbacks: { title: (c) => c[0].dataset.label, label: (c) => `${Math.round(c.raw.x).toLocaleString()} messages: ${f3(c.raw.y)}` } } } } });
    }
    redraw.push(draw);
    draw();
  }

  // ---------- chrome ----------

  function initNav() {
    const links = [...document.querySelectorAll("#nav a")];
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) {
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
        const act = links.find((a) => a.classList.contains("active"));
        if (act) act.parentElement.scrollTo({ left: act.offsetLeft - 60, behavior: "smooth" });
      }
    }, { rootMargin: "-35% 0px -60% 0px" });
    document.querySelectorAll("main section[id]").forEach((s) => obs.observe(s));
  }
  function initTheme() {
    const root = document.documentElement;
    const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const apply = () => { if (R) { theme(); redraw.forEach((f) => f()); } };
    $("#theme").onclick = () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* private mode: the choice just won't persist */ }
      apply();
    };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", apply);
  }

  async function main() {
    initNav(); initTheme();
    try {
      R = await fetch("assets/results.json").then((r) => r.json());
    } catch (e) {
      $("#demo-out").innerHTML = `<p class="loading">Could not load the results (${esc(e)}). Open this page through a web server, not as a file.</p>`;
      return;
    }
    theme();
    initDemo();
    renderScores(); renderE1(); renderH1Evidence(); renderUnseen();
    initH2(); initH3(); initH4(); initH1(); initCurve();
    window.__ready = true;
  }
  main();
})();
