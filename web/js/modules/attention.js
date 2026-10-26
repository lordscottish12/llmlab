import { h, intro, notes, box, presets, errorLine, slider, showTok, pct, busy, cssVar, debounce, tip, esc, svgEl } from "../ui.js";
import { post } from "../api.js";

const EXAMPLES = [
  { name: "Trophy", text: "The trophy didn't fit into the suitcase because it was too big.", query: " it", key: " trophy", alt: " suitcase" },
  { name: "Animal", text: "The animal didn't cross the street because it was too tired.", query: " it", key: " animal", alt: " street" },
  { name: "Agreement", text: "The keys to the old wooden cabinet in the hall are", query: " are", key: " keys", alt: " cabinet" },
  { name: "Repeats", text: "orange piano seven cloud bicycle river lamp quietly. orange piano seven cloud bicycle river lamp", query: " cloud", key: " bicycle", kind: "induction" },
  { name: "Names", text: "When Mary and John went to the store, John gave a drink to", query: " to", key: " Mary", alt: " John" },
];

export default {
  id: "attention",
  title: "Attention",
  mount(main) {
    const toy = toyPanel();
    const real = realPanel();
    main.append(
      intro("Chapter 5 · Attention",
        "How a word gathers information from the words before it",
        "Attention is the one place where tokens exchange information. Each token asks a question (its query), every earlier token offers a label (its key), and the better the match, the more of that token's content (its value) flows in. A model runs many of these in parallel, called heads, in every layer."),
      toy.el,
      real.el,
      notes({
        math: [
          "Each token's vector is turned into three smaller vectors by multiplying it with three learned tables: a <b>query</b> q, a <b>key</b> k and a <b>value</b> v.",
          { f: "score(i → j) = q_i · k_j      (multiply and add)" },
          "The scores of one token against all earlier tokens go through softmax, giving weights that sum to 100%. The token then receives the weighted mix of the values:",
          { f: "output_i = Σ_j weight(i → j) × v_j" },
          "Tokens may only look backwards, never at what comes later, because the model is trained to predict what comes next. Every head has its own three tables, so different heads learn to look for different things.",
        ],
        discuss: [
          "In “…because it was too big” (or “too small”), the word that settles what “it” refers to comes <em>after</em> “it”, and attention only looks backwards. So when the model is at “it”, it cannot yet know. Where, then, can the sentence be understood?",
          "Many heads put most of their weight on the very first token when there is nothing useful to look at (an “attention sink”). What does it mean for a mechanism to need somewhere to look away to?",
          "Is attention in this technical sense related to attention in the psychological sense, or is the shared word a coincidence?",
          "A head that reliably links pronouns to their referents was never told to do so. Did it learn a rule, or something else?",
        ],
      }),
    );
    real.start();
    return () => { tip(null); };
  },
};

// ------------------------------------------------------------------ the toy: one query, five keys, in 2D

function toyPanel() {
  const W = 560, H = 420, C = 210, CY = H / 2, S = 150;
  const keys = [
    { word: "trophy", k: [0.9, 0.35], color: "--c1" },
    { word: "suitcase", k: [0.55, -0.8], color: "--c2" },
    { word: "fit", k: [-0.75, 0.55], color: "--c3" },
    { word: "because", k: [-0.6, -0.55], color: "--c4" },
  ];
  let q = [0.85, 0.15];
  let sharp = 3;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Query and key vectors" });
  const bars = h("div", { class: "probs" });
  const mix = h("div", { style: { display: "flex", height: "1.4rem", border: "1px solid var(--rule)" } });
  const scoresTable = h("div", { class: "small muted mono" });

  const toXY = ([x, y]) => [C + x * S, CY - y * S];
  const fromXY = (px, py) => [Math.max(-1.3, Math.min(1.3, (px - C) / S)), Math.max(-1.3, Math.min(1.3, (CY - py) / S))];

  function render() {
    const scores = keys.map((k) => q[0] * k.k[0] + q[1] * k.k[1]);
    const ex = scores.map((s) => Math.exp(s * sharp));
    const tot = ex.reduce((a, b) => a + b, 0);
    const w = ex.map((e) => e / tot);
    const ink = cssVar("--ink"), line = cssVar("--line"), acc = cssVar("--accent"), muted = cssVar("--muted");
    svg.replaceChildren(
      svgEl("circle", { cx: C, cy: CY, r: S, fill: "none", stroke: line }),
      svgEl("line", { x1: 10, y1: CY, x2: 2 * C - 10, y2: CY, stroke: line }),
      svgEl("line", { x1: C, y1: 10, x2: C, y2: H - 10, stroke: line }),
      ...keys.flatMap((k, i) => {
        const [x, y] = toXY(k.k);
        const col = cssVar(k.color);
        const handle = svgEl("circle", { cx: x, cy: y, r: 9, fill: col, class: "key", "data-i": i });
        return [
          svgEl("line", { x1: C, y1: CY, x2: x, y2: y, stroke: col, "stroke-width": 2 + w[i] * 6, "stroke-linecap": "round" }),
          handle,
          svgEl("text", { x: x + 12, y: y - 10, fill: ink, "font-size": 14, "font-weight": 600, "font-family": "Archivo, sans-serif" }, `${k.word} ${pct(w[i])}`),
        ];
      }),
      (() => {
        const [x, y] = toXY(q);
        return svgEl("g", {},
          svgEl("line", { x1: C, y1: CY, x2: x, y2: y, stroke: acc, "stroke-width": 3, "stroke-dasharray": "6 4" }),
          svgEl("rect", { x: x - 9, y: y - 9, width: 18, height: 18, fill: acc, class: "key", "data-i": "q" }),
          svgEl("text", { x: x + 13, y: y + 18, fill: acc, "font-size": 14, "font-weight": 700, "font-family": "Archivo, sans-serif" }, "query of “it”"));
      })(),
      svgEl("text", { x: 12, y: H - 12, fill: muted, "font-size": 12, "font-family": "Archivo, sans-serif" }, "Drag the red square, or any of the dots."),
    );
    bars.replaceChildren(...keys.map((k, i) => h("div", { class: "prob" },
      h("span", { class: "word" }, k.word),
      h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${w[i] * 100}%`, background: `var(${k.color})` } })),
      h("span", { class: "pct" }, pct(w[i])))));
    mix.replaceChildren(...keys.map((k, i) => h("span", { style: { width: `${w[i] * 100}%`, background: `var(${k.color})`, transition: "width .1s" }, title: `${k.word}: ${pct(w[i])}` })));
    scoresTable.textContent = keys.map((k, i) => `q·k(${k.word}) = ${scores[i].toFixed(2)}`).join("   ");
  }

  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    const t = e.target.closest(".key");
    if (!t) return;
    drag = t.dataset.i;
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", (e) => {
    if (drag == null) return;
    const r = svg.getBoundingClientRect();
    const p = fromXY(((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H);
    if (drag === "q") q = p; else keys[+drag].k = p;
    render();
  });
  svg.addEventListener("pointerup", () => { drag = null; });

  const sharpS = slider("Sharpness", { min: 0.5, max: 10, step: 0.5, value: sharp, fmt: (v) => `×${v}`, oninput: (v) => { sharp = v; render(); } });
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  render();

  return {
    el: h("div", { class: "grid2 even" },
      box("A toy version: one question, four answers", { body: [h("div", { class: "toy" }, svg)] }),
      box("How much “it” takes from each word", {
        body: [
          h("p", { class: "muted" }, "The query of “it” asks something like “which thing am I?”. Each earlier word's key is an arrow too. The closer a key points in the query's direction, the higher its score, and the more of that word's information flows into “it”."),
          bars,
          h("span", { class: "label" }, "What “it” receives: a blend of the words' values"),
          mix,
          scoresTable,
          sharpS,
          h("p", { class: "small muted" }, "In a real model these arrows have 128 dimensions instead of 2, and are computed from the words rather than dragged."),
        ],
      })),
  };
}

// ------------------------------------------------------------------ the real model

function realPanel() {
  let ex = EXAMPLES[0];
  let tokens = [], layers = 0, heads = 0, layer = 0, head = 0, weights = null;
  let query = 0, hideFirst = true;

  const input = h("input", { type: "text", value: ex.text, "aria-label": "Sentence" });
  const msg = h("div");
  const layerS = slider("Layer", { min: 0, max: 1, step: 1, value: 0, fmt: (v) => `${v + 1}`, oninput: debounce((v) => { layer = v; loadLayer(); }, 120) });
  const headGrid = h("div", { class: "heads" });
  const arcs = h("div", { class: "arcs" });
  const arcsTitle = h("h3", {});
  const hideBox = h("input", { type: "checkbox", checked: true });
  hideBox.addEventListener("change", () => { hideFirst = hideBox.checked; renderHeads(); renderArcs(); });
  const keySel = h("select", { "aria-label": "Word it should look at" });
  const findOut = h("div", { class: "probs" });

  async function start() {
    tokens = [];
    try {
      const r = await post("/api/attention", { text: input.value, layer: 0 });
      tokens = r.tokens; layers = r.layers; heads = r.heads;
      layerS.input.max = layers - 1;
      msg.replaceChildren();
      query = Math.max(0, tokens.lastIndexOf(ex.query ?? ""));
      if (!ex.query || tokens[query] !== ex.query) query = tokens.length - 1;
      renderKeySel();
      // Jump straight to the head that best shows the example's point.
      const best = await search(ex.kind || "pair", ex.key);
      if (best) { layer = best.layer; head = best.head; layerS.set(layer); }
      await loadLayer();
    } catch (e) { errorLine(msg, e); }
  }

  async function loadLayer() {
    try {
      const r = await post("/api/attention", { text: input.value, layer });
      weights = r.weights;
      renderHeads();
      renderArcs();
    } catch (e) { errorLine(msg, e); }
  }

  const row = (hd, i) => {
    const w = weights[hd][i].slice(0, i + 1);
    if (!hideFirst || i === 0) return w;
    const rest = w.slice(1).reduce((a, b) => a + b, 0) || 1;
    return [0, ...w.slice(1).map((x) => x / rest)];
  };

  function renderHeads() {
    headGrid.replaceChildren(...Array.from({ length: heads }, (_, hd) => {
      const n = tokens.length;
      const cv = h("canvas", { width: n, height: n });
      const ctx = cv.getContext("2d");
      const img = ctx.createImageData(n, n);
      const bg = hex(cssVar("--sheet")), fg = hex(cssVar("--ink"));
      for (let i = 0; i < n; i++) {
        const r = row(hd, i);
        const m = Math.max(...r, 1e-6);
        for (let j = 0; j < n; j++) {
          const a = j <= i ? Math.sqrt((r[j] || 0) / m) : 0;
          const o = (i * n + j) * 4;
          for (let c = 0; c < 3; c++) img.data[o + c] = bg[c] + (fg[c] - bg[c]) * a;
          img.data[o + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      return h("button", { class: "head", type: "button", "aria-pressed": String(hd === head), title: `Layer ${layer + 1}, head ${hd + 1}`, onclick: () => { head = hd; renderHeads(); renderArcs(); } }, cv, h("span", {}, `head ${hd + 1}`));
    }));
  }

  function renderArcs() {
    const n = tokens.length;
    const widths = tokens.map((t) => Math.max(28, showTok(t).length * 8.2 + 12));
    const xs = [];
    let x = 10;
    widths.forEach((w) => { xs.push(x + w / 2); x += w + 4; });
    const W = x + 10, base = 190, H = 230;
    const r = row(head, query);
    const ink = cssVar("--ink"), acc = cssVar("--accent");
    const max = Math.max(...r, 1e-6);
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    r.forEach((w, j) => {
      if (j > query || w < 0.01) return;
      const x1 = xs[query], x2 = xs[j];
      const lift = Math.min(170, 30 + Math.abs(x1 - x2) * 0.45);
      svg.append(svgEl("path", { d: `M ${x1} ${base - 16} C ${x1} ${base - 16 - lift}, ${x2} ${base - 16 - lift}, ${x2} ${base - 16}`, fill: "none", stroke: acc, "stroke-opacity": 0.15 + 0.85 * (w / max), "stroke-width": 1 + 9 * w, "stroke-linecap": "round" }));
    });
    tokens.forEach((t, j) => {
      const txt = svgEl("text", { x: xs[j], y: base, "text-anchor": "middle", class: j === query ? "q" : "", opacity: j > query ? 0.3 : 1 }, showTok(t));
      txt.addEventListener("click", () => { query = j; renderKeySel(); renderArcs(); });
      txt.addEventListener("pointermove", (e) => tip(`<span class="mono">${esc(showTok(t))}</span>: ${j <= query ? `${pct(r[j] || 0)} of the attention of <span class="mono">${esc(showTok(tokens[query]))}</span>` : "comes later, so it cannot be seen"}`, e));
      txt.addEventListener("pointerleave", () => tip(null));
      svg.append(txt);
      if (j <= query && r[j] >= 0.05) svg.append(svgEl("text", { x: xs[j], y: base + 22, "text-anchor": "middle", fill: ink, style: "font-size:11px;fill:var(--muted)" }, pct(r[j])));
    });
    arcs.replaceChildren(svg);
    arcsTitle.textContent = `Layer ${layer + 1}, head ${head + 1}: where “${showTok(tokens[query]).trim()}” looks`;
  }

  function renderKeySel() {
    keySel.replaceChildren(...tokens.slice(0, query).map((t, j) => h("option", { value: j, selected: ex.key && t === ex.key ? true : null }, `${j + 1}: ${showTok(t)}`)));
  }

  async function search(kind, keyText) {
    const body = { text: input.value, kind, query };
    if (kind === "pair") {
      const k = keyText != null ? tokens.indexOf(keyText) : +keySel.value;
      if (k < 0) return null;
      body.key = k;
    }
    const r = await post("/api/attention/search", body);
    findOut.replaceChildren(...r.heads.slice(0, 8).map((x) => h("button", { class: "prob clickable", type: "button", onclick: async () => { layer = x.layer; head = x.head; layerS.set(layer); await loadLayer(); } },
      h("span", { class: "word" }, `L${x.layer + 1} · H${x.head + 1}`),
      h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${x.score * 100}%` } })),
      h("span", { class: "pct" }, pct(x.score)))));
    return r.heads[0];
  }

  const findPair = h("button", { class: "btn small", type: "button", onclick: () => search("pair").catch((e) => errorLine(findOut, e)) }, "Find heads");
  const findPrev = h("button", { class: "chip-btn", type: "button", onclick: () => search("previous").catch((e) => errorLine(findOut, e)) }, "look at the previous word");
  const findInd = h("button", { class: "chip-btn", type: "button", onclick: () => search("induction").catch((e) => errorLine(findOut, e)) }, "copy from an earlier repeat");
  const findFirst = h("button", { class: "chip-btn", type: "button", onclick: () => search("first").catch((e) => errorLine(findOut, e)) }, "look at the first token");

  const form = h("form", { class: "row", style: { flexWrap: "nowrap" }, onsubmit: (e) => { e.preventDefault(); ex = { text: input.value }; start(); } }, input, h("button", { class: "btn primary", type: "submit" }, "Analyse"));

  const arcsPanel = box("", {
    body: [
      h("p", { class: "small muted" }, "Click any word to see where it looks. Thicker, darker arcs carry more weight."),
      arcs,
      h("label", { class: "check" }, hideBox, "Hide attention to the first token (where heads park their attention when nothing else is relevant)")],
  });
  arcsPanel.querySelector("h3").replaceWith(arcsTitle); // the title changes with the selected head and word

  return {
    start,
    el: h("div", { class: "stack" },
      box("The real model: every head, every layer", {
        right: presets(EXAMPLES, (p) => { ex = p; input.value = p.text; start(); }, { active: EXAMPLES[0] }),
        body: [form, msg],
      }),
      h("div", { class: "grid2" },
        arcsPanel,
        box("Find heads that…", {
          body: [
            h("div", { class: "row" }, h("span", { class: "small" }, "link the selected word to"), keySel, findPair),
            h("div", { class: "chips" }, findPrev, findInd, findFirst),
            findOut,
          ],
        })),
      box("All heads in one layer", {
        body: [layerS, h("p", { class: "small muted" }, "Each square is one head. Row: the word that looks; column: the word it looks at. Dark = much attention. The diagonal stripe just below the main diagonal is a head that looks at the previous word."), headGrid],
      })),
  };
}

function hex(c) {
  const m = c.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}
