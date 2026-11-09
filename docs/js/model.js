/* In-browser copy of the main model: word 1-2 gram + char_wb 3-5 gram TF-IDF, then softmax LR.
 *
 * Mirrors src/features.py exactly:
 *   normalize_text  = lowercase, collapse whitespace runs to one space, strip
 *   TOKEN_PATTERN   = (?u)\b\w\w+\b   -> maximal runs of 2+ word characters [\p{L}\p{N}_]
 *                     (JS's \b is ASCII-only even with /u, so the run form is used; it is equivalent)
 *   char_wb n-grams = sklearn's _char_wb_ngrams: pad each whitespace-separated word with spaces
 *   sublinear_tf    = 1 + ln(count);  then x * idf;  then L2-normalise each vectoriser separately
 * If you change src/features.py, change this file too and rerun docs/parity.html.
 */
(function () {
  "use strict";

  const WORD_RUN = /[\p{L}\p{N}_]{2,}/gu;

  function normalizeText(text) {
    return text.toLowerCase().replace(/\s+/g, " ").trim();
  }

  // Word tokens with the index of the whitespace word that contains them (for highlighting).
  function wordTokens(norm) {
    const words = norm.split(" ").filter(Boolean);
    const starts = [];
    let pos = 0;
    for (const w of words) { pos = norm.indexOf(w, pos); starts.push(pos); pos += w.length; }
    const toks = [];
    for (const m of norm.matchAll(WORD_RUN)) {
      let wi = 0;
      while (wi + 1 < starts.length && starts[wi + 1] <= m.index) wi++;
      toks.push({ tok: m[0], word: wi });
    }
    return { words, toks };
  }

  function wordNgrams(toks, minN, maxN) {
    const out = [];
    for (let n = minN; n <= maxN; n++) {
      for (let i = 0; i + n <= toks.length; i++) {
        const span = toks.slice(i, i + n);
        out.push({ gram: span.map((t) => t.tok).join(" "), words: [...new Set(span.map((t) => t.word))] });
      }
    }
    return out;
  }

  function charWbNgrams(words, minN, maxN) {
    const out = [];
    words.forEach((word, wi) => {
      const w = Array.from(" " + word + " ");          // code points, like Python str indexing
      const L = w.length;
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        out.push({ gram: w.slice(0, n).join(""), words: [wi] });
        while (offset + n < L) {
          offset += 1;
          out.push({ gram: w.slice(offset, offset + n).join(""), words: [wi] });
        }
        if (offset === 0) break;                          // short word: counted once
      }
    });
    return out;
  }

  // grams -> sparse TF-IDF vector {index: value}, L2-normalised; plus which words each feature touches.
  function tfidf(grams, index, idf, offset) {
    const counts = new Map(), owners = new Map();
    for (const g of grams) {
      const j = index.get(g.gram);
      if (j === undefined) continue;
      counts.set(j, (counts.get(j) || 0) + 1);
      if (!owners.has(j)) owners.set(j, []);
      owners.get(j).push(...g.words);
    }
    let ss = 0;
    const vals = new Map();
    for (const [j, c] of counts) { const v = (1 + Math.log(c)) * idf[j]; vals.set(j, v); ss += v * v; }
    const norm = Math.sqrt(ss) || 1;
    const feats = [];
    for (const [j, v] of vals) feats.push({ j: j + offset, x: v / norm, words: owners.get(j) });
    return feats;
  }

  class IntentModel {
    constructor(meta, weights) {
      this.meta = meta;
      this.classes = meta.classes;
      this.C = meta.classes.length;
      this.q = new Int8Array(weights);
      this.scale = Float64Array.from(meta.scale);
      this.b = Float64Array.from(meta.intercept);
      this.wordIndex = new Map(meta.word.vocab.map((t, i) => [t, i]));
      this.charIndex = new Map(meta.char.vocab.map((t, i) => [t, i]));
      this.nWord = meta.word.vocab.length;
      if (this.q.length !== meta.n_features * this.C) throw new Error("weights.bin size does not match model.json");
    }

    static async load(base) {
      const [meta, buf] = await Promise.all([
        fetch(base + "model.json").then((r) => r.json()),
        fetch(base + "weights.bin").then((r) => r.arrayBuffer()),
      ]);
      return new IntentModel(meta, buf);
    }

    w(j, c) { return this.q[j * this.C + c] * this.scale[j]; }

    features(text) {
      const norm = normalizeText(text);
      const { words, toks } = wordTokens(norm);
      const [wMin, wMax] = this.meta.word.ngram_range, [cMin, cMax] = this.meta.char.ngram_range;
      const f = tfidf(wordNgrams(toks, wMin, wMax), this.wordIndex, this.meta.word.idf, 0)
        .concat(tfidf(charWbNgrams(words, cMin, cMax), this.charIndex, this.meta.char.idf, this.nWord));
      return { norm, words, feats: f };
    }

    predict(text) {
      const { norm, words, feats } = this.features(text);
      const logits = Float64Array.from(this.b);
      for (const { j, x } of feats) for (let c = 0; c < this.C; c++) logits[c] += x * this.w(j, c);
      const mx = Math.max(...logits);
      let s = 0;
      const p = Array.from(logits, (z) => { const e = Math.exp(z - mx); s += e; return e; }).map((e) => e / s);
      const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
      const top = order[0];

      // Explanation: contribution of each feature to the top class's logit, relative to the
      // average class (softmax only cares about differences between classes).
      const contrib = feats.map(({ j, x, words: ws }) => {
        let mean = 0;
        for (let c = 0; c < this.C; c++) mean += this.w(j, c);
        mean /= this.C;
        return { j, ws, value: x * (this.w(j, top) - mean), label: this.featureLabel(j) };
      });
      const wordScore = new Array(words.length).fill(0);
      for (const c of contrib) {
        const share = c.value / c.ws.length;
        for (const wi of c.ws) wordScore[wi] += share;
      }
      return { norm, words, proba: p, order, top, contrib: contrib.sort((a, b) => b.value - a.value), wordScore, nFeatures: feats.length };
    }

    featureLabel(j) {
      return j < this.nWord
        ? { kind: "word", text: this.meta.word.vocab[j] }
        : { kind: "char", text: this.meta.char.vocab[j - this.nWord] };
    }
  }

  window.IntentModel = IntentModel;
  window.normalizeText = normalizeText;
})();
