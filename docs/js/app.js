/* Page logic: live classifier, H2 slider, H3 picker, H4 chart, H1 merge view, learning curve.
 * All numbers come from assets/results.json (written by src/export_web.py); nothing is computed on a server. */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const f3 = (x) => Number(x).toFixed(3);
  const ci = (c) => `[${f3(c[0])}, ${f3(c[1])}]`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pretty = (intent) => intent.replace(/_/g, " ").replace(/\?$/, "");

  let R = null;            // results.json
  const renderers = [];    // re-run when the colour scheme changes
  const charts = {};

  // Update an existing chart in place instead of destroying it. Destroying and recreating a chart
  // on every slider tick, while the previous one was still animating, crashed Chart.js's shared
  // animation loop ("this._fn is not a function") and blanked every chart on the page.
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
    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--grid");
    Chart.defaults.font.family = css("--sans");
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.animation = false;      // sliders redraw on every tick; animation only adds lag
    Chart.defaults.maintainAspectRatio = false;
  }
  const S = (i) => css(`--series-${i}`);
  // Translucent version of a #rrggbb token, as rgba() so every canvas and Chart.js can parse it.
  const alpha = (color, a) => {
    const h = color.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a.toFixed(3)})`;
  };

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
    el.innerHTML = items.map(([v, l]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }

  // ---------- 1. Live classifier ----------

  function groupOf(intent) {
    return R.h1.groups.find((g) => g.includes(intent));
  }

  function initDemo() {
    const input = $("#demo-input"), out = $("#demo-out"), kSel = $("#demo-k");
    const thr = () => R.slider.frozen_thresholds.find((t) => String(t.k) === kSel.value).threshold;
    const ex = $("#demo-examples");
    let truth = null;
    for (const kind of ["clear", "ambiguous", "misclassified"]) {
      for (const e of R.examples[kind].slice(0, 2)) {
        const b = document.createElement("button");
        b.className = `chip ${kind}`;
        b.innerHTML = `<span class="tag">${kind}</span>${esc(e.text)}`;
        b.title = `True label: ${e.label}`;
        b.onclick = () => { input.value = e.text; truth = e; run(); };
        ex.appendChild(b);
      }
    }
    let model = null;
    IntentModel.load("assets/").then((m) => { model = m; run(); })
      .catch((e) => { out.innerHTML = `<p class="muted">Could not load the model: ${esc(e)}</p>`; });

    function run() {
      $("#demo-threshold").textContent = `Hand off when the top probability is below ${thr().toFixed(3)} (chosen on cross-validation for this cost).`;
      if (!model) return;
      const text = input.value;
      if (!normalizeText(text)) { out.innerHTML = `<p class="muted">Type a message above.</p>`; return; }
      const r = model.predict(text);
      const top = model.classes[r.top], p = r.proba[r.top];
      const answer = p >= thr();
      const bars = r.order.slice(0, 3).map((i) => `<div class="bar"><span class="name" title="${esc(model.classes[i])}">${esc(model.classes[i])}</span>
          <span class="track"><span class="fill" style="width:${(100 * r.proba[i]).toFixed(1)}%"></span></span><span class="pct">${pct(r.proba[i])}</span></div>`).join("");
      const maxAbs = Math.max(1e-9, ...r.wordScore.map(Math.abs));
      const hl = r.words.map((w, i) => {
        const s = r.wordScore[i] / maxAbs;
        const bg = s >= 0 ? alpha(css("--good"), 0.55 * s) : alpha(css("--bad"), 0.55 * -s);
        return `<span style="background:${bg}" title="contribution ${r.wordScore[i].toFixed(3)}">${esc(w)}</span>`;
      }).join(" ");
      const feats = r.contrib.filter((c) => c.value > 0).slice(0, 8).map((c) =>
        `<span class="feat" title="${c.label.kind} feature, +${c.value.toFixed(3)}">${c.label.kind === "char" ? esc(c.label.text.replace(/ /g, "␣")) : esc(c.label.text)}</span>`).join("");
      const g = groupOf(top);
      const truthNote = truth && normalizeText(text) === normalizeText(truth.text)
        ? `<p class="small">True label (from the test set): <code>${esc(truth.label)}</code> ${truth.label === top ? "✓ the model is right" : "✗ the model is wrong here"}</p>` : "";
      out.innerHTML = `
        <div class="decision ${answer ? "answer" : "handoff"}">${answer ? "Answer automatically" : "Hand off to a human"}
          <span class="why">top probability ${pct(p)} is ${answer ? "at or above" : "below"} the threshold ${pct(thr())}</span></div>
        ${truthNote}
        <h3>Top 3 intents</h3><div class="bars">${bars}</div>
        <h3>Which words pushed it towards <code>${esc(top)}</code></h3>
        <div class="highlight">${hl}</div>
        <p class="small muted">Green words raise this intent's score relative to the average intent, red words lower it. Strongest features
          (␣ marks a word edge in a character n-gram):</p>
        <div class="feats">${feats || '<span class="muted small">none of the words are in the vocabulary</span>'}</div>
        ${g ? `<p class="groupnote">⚠ <code>${esc(top)}</code> is in a confusable group (H1) with ${g.length - 1} other intent${g.length > 2 ? "s" : ""}${g.length <= 8 ? ": " + g.filter((x) => x !== top).map((x) => `<code>${esc(x)}</code>`).join(", ") : ""}. These are the intents it most often mixes up.</p>` : ""}`;
    }
    let timer;
    input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 120); });
    kSel.addEventListener("change", run);
    run();
  }

  // ---------- 6. Scores ----------

  function renderScores() {
    const rows = R.scores;
    $("#scores-table").innerHTML = `<thead><tr><th>Model</th><th class="n">CV macro-F1 [95% CI]</th><th class="n">Test macro-F1 [95% CI]</th><th class="n">Test accuracy</th></tr></thead><tbody>` +
      rows.map((r) => `<tr class="${r.id === "M1_L2" ? "hl" : ""}"><td>${r.id.startsWith("B") ? "Baseline: " : ""}${esc(r.name)}</td>
        <td class="n">${f3(r.cv.macro_f1)} ${ci(r.cv.ci)}</td><td class="n">${f3(r.test.macro_f1)} ${ci(r.test.ci)}</td><td class="n">${f3(r.test.accuracy)}</td></tr>`).join("") + "</tbody>";
    const shown = rows.filter((r) => r.id !== "B0_majority").sort((a, b) => b.test.macro_f1 - a.test.macro_f1);
    chart("scores-chart", {
      type: "bar",
      data: { labels: shown.map((r) => r.name), datasets: [
        { label: "Cross-validation", data: shown.map((r) => r.cv.macro_f1), backgroundColor: S(2) },
        { label: "Test set", data: shown.map((r) => r.test.macro_f1), backgroundColor: S(1) }] },
      options: { indexAxis: "y", scales: { x: { min: 0.5, max: 1, title: { display: true, text: "Macro-F1" } } },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f3(c.raw)}` } } } },
    });
  }

  // ---------- 7. H2 slider ----------

  function prepare(conf, correct) {
    const idx = conf.map((_, i) => i).sort((a, b) => conf[b] - conf[a]);
    const c = idx.map((i) => conf[i]), w = idx.map((i) => 1 - correct[i]);
    const cumW = [0];
    for (const x of w) cumW.push(cumW[cumW.length - 1] + x);
    return { conf: c, cumW, n: c.length, err: cumW[c.length] / c.length };
  }
  // Number of messages with conf >= t (conf sorted descending).
  function answeredCount(d, t) {
    let lo = 0, hi = d.n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (d.conf[m] >= t) lo = m + 1; else hi = m; }
    return lo;
  }
  function at(d, t, k) {
    const m = answeredCount(d, t), wrong = d.cumW[m];
    return { m, coverage: m / d.n, acc: m ? 1 - wrong / m : NaN, cost: (k * wrong + (d.n - m)) / d.n,
             randomCost: (k * d.err * m + (d.n - m)) / d.n, answerAll: k * d.err };
  }
  // Same rule as src/analysis.py cost_threshold: cut only between distinct confidence values.
  function bestThreshold(d, k) {
    let best = { cost: 1, t: 1.0 };   // answer nothing
    for (let m = 1; m <= d.n; m++) {
      if (m < d.n && d.conf[m - 1] === d.conf[m]) continue;
      const cost = (k * d.cumW[m] + (d.n - m)) / d.n;
      if (cost < best.cost) best = { cost, t: d.conf[m - 1] };
    }
    return best.t;
  }

  function initH2() {
    const D = { test: prepare(R.slider.test.conf, R.slider.test.correct), cv: prepare(R.slider.cv.conf, R.slider.cv.correct) };
    const U = { seen: R.slider.unseen.seen.slice().sort((a, b) => b - a), unseen: R.slider.unseen.unseen.slice().sort((a, b) => b - a) };
    const tEl = $("#h2-t"), kEl = $("#h2-k");
    const mode = seg("h2-data", () => draw());
    $("#h2-best").onclick = () => { tEl.value = bestThreshold(D.cv, +kEl.value).toFixed(3); draw(); };
    tEl.oninput = () => draw(); kEl.oninput = () => draw();
    document.querySelectorAll('[data-jump="unseen"]').forEach((a) => a.addEventListener("click", () => mode.set("unseen")));

    function draw() {
      const t = +tEl.value, k = +kEl.value, v = mode.get();
      $("#h2-t-val").textContent = t.toFixed(3);
      $("#h2-k-val").textContent = `${k} : 1`;
      const frozen = R.slider.frozen_thresholds.find((x) => x.k === k);
      $("#h2-best").textContent = frozen ? `Use the frozen CV threshold for k = ${k}` : `Use the best CV threshold for k = ${k}`;
      if (v === "unseen") return drawUnseen(t, k);
      const d = D[v], s = at(d, t, k);
      stats($("#h2-stats"), [
        [pct(s.coverage), "answered automatically (coverage)"],
        [isNaN(s.acc) ? "–" : pct(s.acc), "accuracy on the answered messages"],
        [pct(1 - d.err), "accuracy if it answers everything"],
        [s.cost.toFixed(3), `cost per message (answer all: ${s.answerAll.toFixed(3)}, hand off all: 1)`],
        [s.randomCost.toFixed(3), "cost if the same number were handed off at random"],
      ]);
      // Risk-coverage curve
      const pts = [];
      for (let m = Math.max(1, Math.round(0.02 * d.n)); m <= d.n; m += Math.max(1, Math.floor(d.n / 300))) pts.push({ x: m / d.n, y: d.cumW[m] / m });
      pts.push({ x: 1, y: d.err });
      chart("h2-risk", {
        type: "scatter",
        data: { datasets: [
          { label: "Hand off least confident", data: pts, showLine: true, pointRadius: 0, borderColor: S(1), borderWidth: 2 },
          { label: "Hand off at random (B2)", data: [{ x: 0.02, y: d.err }, { x: 1, y: d.err }], showLine: true, pointRadius: 0, borderColor: S(5), borderDash: [6, 4], borderWidth: 2 },
          { label: "Current threshold", data: s.m ? [{ x: s.coverage, y: 1 - s.acc }] : [], pointRadius: 7, pointBackgroundColor: S(2), borderColor: S(2) }] },
        options: { scales: { x: { min: 0, max: 1, title: { display: true, text: "Coverage (share answered)" }, ticks: { callback: (x) => pct(x, 0) } },
                             y: { min: 0, title: { display: true, text: "Error rate on answered messages" }, ticks: { callback: (x) => pct(x, 0) } } },
          plugins: { title: { display: true, text: "Risk–coverage" }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: coverage ${pct(c.raw.x)}, error ${pct(c.raw.y)}` } } } },
      });
      // Cost vs threshold
      const ts = [], conf = [], rnd = [];
      for (let x = 0; x <= 1.0001; x += 0.01) { const a = at(d, x, k); ts.push(x); conf.push({ x, y: a.cost }); rnd.push({ x, y: a.randomCost }); }
      chart("h2-cost", {
        type: "scatter",
        data: { datasets: [
          { label: "Confidence rule", data: conf, showLine: true, pointRadius: 0, borderColor: S(1), borderWidth: 2 },
          { label: "Random handoff, same coverage", data: rnd, showLine: true, pointRadius: 0, borderColor: S(5), borderDash: [6, 4], borderWidth: 2 },
          { label: "Current threshold", data: [{ x: t, y: s.cost }], pointRadius: 7, pointBackgroundColor: S(2), borderColor: S(2) },
          ...(frozen ? [{ label: "Frozen (chosen on CV)", data: [{ x: frozen.threshold, y: at(d, frozen.threshold, k).cost }], pointRadius: 6, pointStyle: "rectRot", pointBackgroundColor: S(3), borderColor: S(3) }] : [])] },
        options: { scales: { x: { min: 0, max: 1, title: { display: true, text: "Threshold" } }, y: { min: 0, title: { display: true, text: `Cost per message (wrong = ${k}, handoff = 1)` } } },
          plugins: { title: { display: true, text: "Expected cost" }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: t = ${c.raw.x.toFixed(2)}, cost ${c.raw.y.toFixed(3)}` } } } },
      });
      $("#h2-note").textContent = v === "test"
        ? `Test set: ${d.n.toLocaleString()} messages, scored once by the frozen model. The thresholds were chosen on cross-validation, so the diamond is an honest out-of-sample choice.`
        : `Cross-validation: out-of-fold predictions for all ${d.n.toLocaleString()} training messages. This is where the thresholds were chosen.`;
    }

    function drawUnseen(t, k) {
      const share = (arr) => 1 - answeredCount({ conf: arr, n: arr.length }, t) / arr.length;
      const ref = R.unseen.cv.at_cost_thresholds[String(k)];
      stats($("#h2-stats"), [
        [pct(share(U.unseen)), "of unseen-intent messages handed off (good)"],
        [pct(share(U.seen)), "of seen-intent messages handed off (the price)"],
        [f3(R.unseen.cv.auroc_seen_vs_unseen), "AUROC, seen vs unseen, from confidence alone"],
        [ref ? ref.threshold.toFixed(3) : "–", ref ? `threshold for k = ${k}, chosen on seen intents only` : "pick k = 2, 5 or 10 for the reference threshold"],
      ]);
      const bins = 25, hist = (arr) => {
        const h = new Array(bins).fill(0);
        for (const x of arr) h[Math.min(bins - 1, Math.floor(x * bins))]++;
        return h.map((c, i) => ({ x: (i + 0.5) / bins, y: c / arr.length }));
      };
      const hs = hist(U.seen), hu = hist(U.unseen);
      const ymax = Math.max(...hs.map((p) => p.y), ...hu.map((p) => p.y)) * 1.05;
      chart("h2-risk", {
        type: "bar",
        data: { datasets: [
          { type: "bar", label: "Seen intents (67)", data: hs, backgroundColor: alpha(S(1), 0.55), barPercentage: 1, categoryPercentage: 1, grouped: false },
          { type: "bar", label: "Unseen intents (10)", data: hu, backgroundColor: alpha(S(2), 0.55), barPercentage: 1, categoryPercentage: 1, grouped: false },
          { type: "line", label: "Threshold", data: [{ x: t, y: 0 }, { x: t, y: ymax }], borderColor: S(5), borderWidth: 2, pointRadius: 0 }] },
        options: { scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Top probability" } }, y: { stacked: false, title: { display: true, text: "Share of messages" }, ticks: { callback: (x) => pct(x, 0) } } },
          plugins: { title: { display: true, text: "Confidence: seen vs unseen intents" } } },
      });
      const cs = [], cu = [];
      for (let x = 0; x <= 1.0001; x += 0.01) { cs.push({ x, y: 1 - answeredCount({ conf: U.seen, n: U.seen.length }, x) / U.seen.length }); cu.push({ x, y: 1 - answeredCount({ conf: U.unseen, n: U.unseen.length }, x) / U.unseen.length }); }
      chart("h2-cost", {
        type: "scatter",
        data: { datasets: [
          { label: "Unseen handed off", data: cu, showLine: true, pointRadius: 0, borderColor: S(2), borderWidth: 2 },
          { label: "Seen handed off", data: cs, showLine: true, pointRadius: 0, borderColor: S(1), borderWidth: 2 },
          { label: "Current threshold", data: [{ x: t, y: share(U.unseen) }, { x: t, y: share(U.seen) }], pointRadius: 6, pointBackgroundColor: S(5), borderColor: S(5) }] },
        options: { scales: { x: { min: 0, max: 1, title: { display: true, text: "Threshold" } }, y: { min: 0, max: 1, title: { display: true, text: "Share handed off" }, ticks: { callback: (x) => pct(x, 0) } } },
          plugins: { title: { display: true, text: "Handoff rate by threshold" } } },
      });
      $("#h2-note").textContent = `Unseen-intent mode: a model trained on 67 intents, out-of-fold confidences for ${U.seen.length.toLocaleString()} seen and ${U.unseen.length.toLocaleString()} unseen training messages. On the test set, AUROC is ${f3(R.unseen.test.auroc_seen_vs_unseen)} and the k = 5 threshold hands off ${pct(R.unseen.test.at_frozen_thresholds["5"].unseen_handed_off)} of unseen vs ${pct(R.unseen.test.at_frozen_thresholds["5"].seen_handed_off)} of seen messages.`;
    }
    renderers.push(draw);
    draw();
  }

  function renderE1() {
    const cv = R.e1.cv, te = R.e1.test;
    $("#e1-table").innerHTML = `<thead><tr><th></th><th class="n">Max probability</th><th class="n">E1</th></tr></thead><tbody>
      <tr><td>Error-detection AUROC (CV)</td><td class="n">${f3(cv.auroc_error_detection.max_prob)}</td><td class="n">${f3(cv.auroc_error_detection.e1)}</td></tr>
      <tr><td>Error-detection AUROC (test)</td><td class="n">${f3(te.auroc.max_prob)}</td><td class="n">${f3(te.auroc.e1)}</td></tr>
      <tr><td>AURC, lower is better (CV)</td><td class="n">${cv.aurc.max_prob.toFixed(4)}</td><td class="n">${cv.aurc.e1.toFixed(4)}</td></tr>
      <tr><td>AURC (test)</td><td class="n">${te.aurc.max_prob.toFixed(4)}</td><td class="n">${te.aurc.e1.toFixed(4)}</td></tr>
      <tr><td>Gain in AUROC, 95% CI (CV)</td><td class="n" colspan="2">${f3(cv.e1_minus_max_prob.auroc_diff)} ${ci(cv.e1_minus_max_prob.auroc_diff_ci95)}</td></tr></tbody>`;
  }

  // ---------- 8. H3 reliability ----------

  function initH3() {
    const sel = $("#h3-model");
    const ids = Object.keys(R.h3.cv);
    sel.innerHTML = ids.map((id) => `<option value="${id}">${esc(R.h3.cv[id].name)}</option>`).join("");
    const mode = seg("h3-data", () => draw());
    sel.onchange = () => draw();
    function draw() {
      const src = R.h3[mode.get()], m = src[sel.value];
      const maxN = Math.max(...m.bins.map((b) => b.n));
      chart("h3-chart", {
        type: "scatter",
        data: { datasets: [
          { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderColor: S(6), borderDash: [5, 4], borderWidth: 1.5 },
          { label: m.name, data: m.bins.map((b) => ({ x: b.conf, y: b.acc, n: b.n })), showLine: true, borderColor: S(1), backgroundColor: alpha(S(1), 0.7),
            pointRadius: m.bins.map((b) => 3 + 9 * Math.sqrt(b.n / maxN)) }] },
        options: { scales: { x: { min: 0, max: 1, title: { display: true, text: "Mean confidence in bin" } }, y: { min: 0, max: 1, title: { display: true, text: "Accuracy in bin" } } },
          plugins: { title: { display: true, text: "Reliability diagram (dot size = messages in bin)" },
            tooltip: { callbacks: { label: (c) => c.raw.n ? `confidence ${f3(c.raw.x)}, accuracy ${f3(c.raw.y)}, ${c.raw.n} messages` : "" } } } },
      });
      stats($("#h3-stats"), [
        [f3(m.ece), `ECE, 95% CI ${ci(m.ece_ci)}`],
        [pct(m.mean_confidence), "average confidence"],
        [pct(m.accuracy), "actual accuracy"],
        [m.mean_confidence < m.accuracy ? "Under" : "Over", "confident overall"],
      ]);
      const order = ids.slice().sort((a, b) => src[b].macro_f1 - src[a].macro_f1);
      const bestEce = ids.reduce((a, b) => (src[a].ece <= src[b].ece ? a : b));
      $("#h3-table").innerHTML = `<thead><tr><th>Model (by macro-F1)</th><th class="n">Macro-F1</th><th class="n">ECE</th></tr></thead><tbody>` +
        order.map((id, i) => `<tr class="${id === sel.value ? "hl" : ""}"><td>${esc(src[id].name)}${i === 0 ? " · most accurate" : ""}${id === bestEce ? " · best calibrated" : ""}</td>
          <td class="n">${f3(src[id].macro_f1)}</td><td class="n">${f3(src[id].ece)}</td></tr>`).join("") + "</tbody>";
    }
    renderers.push(draw);
    draw();
  }

  // ---------- 9. H4 ----------

  function initH4() {
    const mode = seg("h4-data", () => draw());
    function draw() {
      const v = mode.get(), curve = R.h4[v], full = R.h4[v + "_full"];
      chart("h4-chart", {
        type: "scatter",
        data: { datasets: [
          { label: "SVD + logistic regression", data: curve.map((c) => ({ x: c.d, y: c.share, f1: c.macro_f1, ev: c.explained_variance })), showLine: true, borderColor: S(1), backgroundColor: S(1), pointRadius: 5 },
          { label: "90% of full TF-IDF (pre-registered bar)", data: [{ x: 20, y: 0.9 }, { x: 1200, y: 0.9 }], showLine: true, pointRadius: 0, borderColor: S(5), borderDash: [6, 4], borderWidth: 1.5 },
          { label: "d = 300 limit", data: [{ x: 300, y: 0.75 }, { x: 300, y: 1.0 }], showLine: true, pointRadius: 0, borderColor: S(6), borderDash: [2, 3], borderWidth: 1.5 }] },
        options: { scales: { x: { type: "logarithmic", min: 20, max: 1200, title: { display: true, text: "SVD dimensions d (log scale)" }, ticks: { callback: (x) => ([25, 50, 100, 200, 300, 500, 1000].includes(x) ? x : "") } },
                             y: { min: 0.75, max: 1.0, title: { display: true, text: `Share of full TF-IDF macro-F1 (${f3(full)})` }, ticks: { callback: (x) => pct(x, 0) } } },
          plugins: { tooltip: { callbacks: { label: (c) => c.datasetIndex === 0 ? `d = ${c.raw.x}: macro-F1 ${f3(c.raw.f1)} (${pct(c.raw.y)} of full)${c.raw.ev ? `, explains ${pct(c.raw.ev, 0)} of variance` : ""}` : c.dataset.label } } } },
      });
    }
    renderers.push(draw);
    draw();
  }

  // ---------- 10. H1 ----------

  function initH1() {
    const order = R.h1.confusion.order, groups = R.h1.groups;
    const mode = seg("h1-data", () => draw());
    const mergeEl = $("#h1-merge");
    mergeEl.onchange = () => draw();
    const gIndex = new Map();
    groups.forEach((g, gi) => g.forEach((c) => gIndex.set(c, gi)));

    function merged(M) {
      const labels = [], map = order.map((c) => {
        const key = gIndex.has(c) ? `G${gIndex.get(c)}` : c;
        let i = labels.indexOf(key);
        if (i < 0) { labels.push(key); i = labels.length - 1; }
        return i;
      });
      const K = labels.length, out = Array.from({ length: K }, () => new Array(K).fill(0));
      M.forEach((row, i) => row.forEach((v, j) => { out[map[i]][map[j]] += v; }));
      return { labels, M: out };
    }
    function metrics(M) {
      const K = M.length;
      let total = 0, diag = 0, f1 = 0;
      const colSum = new Array(K).fill(0);
      M.forEach((row) => row.forEach((v, j) => { colSum[j] += v; total += v; }));
      for (let i = 0; i < K; i++) {
        const tp = M[i][i], rowSum = M[i].reduce((a, b) => a + b, 0);
        diag += tp;
        const p = colSum[i] ? tp / colSum[i] : 0, r = rowSum ? tp / rowSum : 0;
        f1 += p + r ? (2 * p * r) / (p + r) : 0;
      }
      return { acc: diag / total, macroF1: f1 / K, errors: total - diag, K };
    }

    const canvas = $("#h1-matrix");
    let hover = null, current = null;
    canvas.addEventListener("mousemove", (e) => {
      if (!current) return;
      const rect = canvas.getBoundingClientRect(), K = current.labels.length;
      const size = Math.min(rect.width, rect.height), cell = size / K;
      const j = Math.floor((e.clientX - rect.left) / cell), i = Math.floor((e.clientY - rect.top) / cell);
      hover = i >= 0 && j >= 0 && i < K && j < K ? { i, j } : null;
      paint();
    });
    canvas.addEventListener("mouseleave", () => { hover = null; paint(); });

    const name = (key) => key.startsWith("G") && /^G\d+$/.test(key) ? `group ${+key.slice(1) + 1} (${groups[+key.slice(1)].length} intents)` : key;

    function paint() {
      const { labels, M, groupSpans } = current;
      const rect = canvas.parentElement.getBoundingClientRect();
      const size = Math.min(rect.width, rect.height), dpr = window.devicePixelRatio || 1;
      canvas.style.width = size + "px"; canvas.style.height = size + "px";
      canvas.width = size * dpr; canvas.height = size * dpr;
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      const K = labels.length, cell = size / K;
      ctx.fillStyle = css("--surface-2"); ctx.fillRect(0, 0, size, size);
      let max = 1;
      M.forEach((row, i) => row.forEach((v, j) => { if (i !== j) max = Math.max(max, v); }));
      const ink = css("--accent");
      M.forEach((row, i) => row.forEach((v, j) => {
        if (i === j || !v) return;
        ctx.fillStyle = alpha(ink, 0.15 + 0.85 * Math.log1p(v) / Math.log1p(max));
        ctx.fillRect(j * cell, i * cell, Math.max(cell, 1), Math.max(cell, 1));
      }));
      ctx.strokeStyle = css("--warn"); ctx.lineWidth = 1.2;
      for (const [a, b] of groupSpans) ctx.strokeRect(a * cell, a * cell, (b - a) * cell, (b - a) * cell);
      if (hover) {
        const { i, j } = hover;
        ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5; ctx.strokeRect(j * cell, i * cell, cell, cell);
        const v = M[i][j];
        const txt = i === j ? `${name(labels[i])}: ${v} correct` : `${v} × true ${name(labels[i])} → predicted ${name(labels[j])}`;
        ctx.font = `12px ${css("--sans")}`;
        const w = Math.min(size - 8, ctx.measureText(txt).width + 12);
        const x = Math.min(size - w - 4, Math.max(4, j * cell - w / 2)), y = i * cell > 40 ? i * cell - 30 : i * cell + cell + 6;
        ctx.fillStyle = css("--surface"); ctx.strokeStyle = css("--border"); ctx.lineWidth = 1;
        ctx.fillRect(x, y, w, 24); ctx.strokeRect(x, y, w, 24);
        ctx.fillStyle = css("--text"); ctx.fillText(txt, x + 6, y + 16, w - 12);
      }
    }

    function draw() {
      const v = mode.get(), M = R.h1.confusion[v], isMerged = mergeEl.checked;
      const base = metrics(M);
      if (isMerged) {
        const m = merged(M);
        let pos = 0;
        current = { labels: m.labels, M: m.M, groupSpans: [] };
        const after = metrics(m.M);
        stats($("#h1-stats"), [
          [`${pct(base.acc)} → ${pct(after.acc)}`, "accuracy after merging"],
          [`${f3(base.macroF1)} → ${f3(after.macroF1)}`, "macro-F1 after merging"],
          [`${base.errors} → ${after.errors}`, "errors left"],
          [`77 → ${after.K}`, "labels"],
        ]);
      } else {
        const spans = [];
        let pos = 0;
        for (const g of groups) { spans.push([pos, pos + g.length]); pos += g.length; }
        current = { labels: order, M, groupSpans: spans };
        let inside = 0;
        M.forEach((row, i) => row.forEach((x, j) => { if (i !== j && gIndex.has(order[i]) && gIndex.get(order[i]) === gIndex.get(order[j])) inside += x; }));
        stats($("#h1-stats"), [
          [pct(base.acc), "accuracy, 77 intents"],
          [f3(base.macroF1), "macro-F1"],
          [`${pct(inside / base.errors)}`, `of ${base.errors} errors fall inside a group (chance: ${pct(R.h1.edge["5"].chance_share)} in CV)`],
          [pct(v === "cv" ? R.h1.top3_cv : R.h1.test.top3_accuracy), "top-3 accuracy"],
        ]);
      }
      paint();
      // group list with error counts
      const pos = new Map(order.map((c, i) => [c, i]));
      $("#h1-groups").innerHTML = groups.map((g, gi) => {
        let n = 0;
        for (const a of g) for (const b of g) if (a !== b) n += M[pos.get(a)][pos.get(b)];
        return `<div class="group ${isMerged ? "merged" : ""}"><div class="head">Group ${gi + 1} · ${g.length} intents · ${n} errors inside${isMerged ? " → now correct" : ""}</div>
          ${g.map((c) => `<span class="pill">${esc(c)}</span>`).join("")}</div>`;
      }).join("");
    }
    window.addEventListener("resize", () => current && paint());
    renderers.push(draw);
    draw();

    // (a) edge-threshold table
    const e = R.h1.edge;
    $("#h1-edge").innerHTML = `<thead><tr><th class="n">Edge ≥</th><th class="n">Errors inside groups</th><th class="n">If random</th><th>Group sizes</th></tr></thead><tbody>` +
      ["3", "5", "8", "10"].map((k) => `<tr class="${k === "5" ? "hl" : ""}"><td class="n">${k}${k === "5" ? " (pre-reg.)" : ""}</td><td class="n">${pct(e[k].share)}</td><td class="n">${pct(e[k].chance_share)}</td>
        <td>${e[k].group_sizes.length > 6 ? `${e[k].n_groups} groups of ${Math.min(...e[k].group_sizes)}–${Math.max(...e[k].group_sizes)}` : e[k].group_sizes.join(", ")}</td></tr>`).join("") + "</tbody>";
  }

  function renderH1Labels() {
    const s = R.h1.hand_labels.shares;
    const keys = ["ambiguous_label", "vague_message", "possible_mislabel", "vocabulary_gap"];
    chart("h1-labels", {
      type: "bar",
      data: { labels: keys.map(pretty), datasets: [{ data: keys.map((k) => s[k]), backgroundColor: [S(1), S(2), S(4), S(6)] }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${pct(c.raw, 0)} of ${R.h1.hand_labels.n} errors` } } },
        scales: { x: { min: 0, max: 0.8, ticks: { callback: (x) => pct(x, 0) } } } },
    });
  }

  // ---------- 11. Unseen table ----------

  function renderUnseen() {
    const by = R.unseen.cv.by_unseen_intent;
    const rows = Object.entries(by).sort((a, b) => b[1].handed_off_at_k5 - a[1].handed_off_at_k5);
    $("#unseen-table").innerHTML = `<thead><tr><th>Unseen intent</th><th class="n">Messages</th><th class="n">Mean confidence</th><th class="n">Handed off at k = 5</th><th>Most often mistaken for</th></tr></thead><tbody>` +
      rows.map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td class="n">${v.n}</td><td class="n">${f3(v.mean_confidence)}</td><td class="n">${pct(v.handed_off_at_k5, 0)}</td>
        <td class="small">${Object.entries(v.most_common_predictions).slice(0, 2).map(([p, n]) => `${esc(p)} (${n})`).join(", ")}</td></tr>`).join("") + "</tbody>";
  }

  // ---------- 12. Learning curve ----------

  function initCurve() {
    const cv = R.learning_curve.cv, te = R.learning_curve.test, el = $("#lc-f");
    const b1 = R.scores.find((s) => s.id === "B1_keyword_k20");
    el.oninput = () => draw();
    function draw() {
      const i = +el.value, c = cv[i], t = te[i];
      $("#lc-f-val").textContent = pct(c.fraction, 0);
      stats($("#lc-stats"), [
        [Math.round(c.n).toLocaleString(), "training messages per fold (CV)"],
        [`${f3(c.mean)} ± ${f3(c.sd)}`, "CV macro-F1 (mean ± SD over 5 draws)"],
        [f3(t.mean), `test macro-F1 (trained on ${Math.round(c.fraction * R.dataset.n_train).toLocaleString()} messages)`],
        [f3(b1.cv.macro_f1), "keyword rule, trained on everything (CV)"],
      ]);
      chart("lc-chart", {
        type: "scatter",
        data: { datasets: [
          { label: "Cross-validation", data: cv.map((p) => ({ x: p.n, y: p.mean, sd: p.sd })), showLine: true, borderColor: S(2), backgroundColor: S(2), pointRadius: cv.map((_, j) => (j === i ? 8 : 4)) },
          { label: "Test set", data: te.map((p, j) => ({ x: cv[j].fraction * R.dataset.n_train, y: p.mean })), showLine: true, borderColor: S(1), backgroundColor: S(1), pointRadius: te.map((_, j) => (j === i ? 8 : 4)) },
          { label: "Keyword rule (all data, CV)", data: [{ x: 300, y: b1.cv.macro_f1 }, { x: 12000, y: b1.cv.macro_f1 }], showLine: true, pointRadius: 0, borderColor: S(5), borderDash: [6, 4], borderWidth: 1.5 }] },
        options: { scales: { x: { type: "logarithmic", min: 300, max: 12000, title: { display: true, text: "Training messages (log scale)" }, ticks: { callback: (x) => ([500, 1000, 2000, 5000, 10000].includes(x) ? x.toLocaleString() : "") } },
                             y: { min: 0.55, max: 1, title: { display: true, text: "Macro-F1" } } },
          plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${Math.round(c.raw.x).toLocaleString()} messages, macro-F1 ${f3(c.raw.y)}${c.raw.sd ? ` ± ${f3(c.raw.sd)}` : ""}` } } } },
      });
    }
    renderers.push(draw);
    draw();
  }

  // ---------- nav highlight ----------

  function initNav() {
    const links = [...document.querySelectorAll(".nav a[href^='#']")];
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("section[id]").forEach((s) => obs.observe(s));
  }

  async function main() {
    initNav();
    try {
      R = await fetch("assets/results.json").then((r) => r.json());
    } catch (e) {
      document.querySelectorAll(".chart").forEach((c) => (c.innerHTML = `<p class="muted">Could not load results.json (${esc(e)}). Open the page through a web server, not file://.</p>`));
      return;
    }
    theme();
    initDemo();
    renderScores(); renderE1(); renderH1Labels(); renderUnseen();
    initH2(); initH3(); initH4(); initH1(); initCurve();
    renderers.push(renderScores, renderH1Labels);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { theme(); renderers.forEach((f) => f()); });
    window.__ready = true;
  }
  main();
})();
