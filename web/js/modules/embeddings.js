import { h, intro, notes, box, presets, errorLine, esc, busy, cssVar, divergingColor, svgEl } from "../ui.js";
import { post, info } from "../api.js";
import { createStage, arrow } from "../scene.js";

const SETS = [
  {
    name: "Gender & royalty",
    groups: [
      ["man", "king", "prince", "boy", "father", "uncle", "husband", "actor"],
      ["woman", "queen", "princess", "girl", "mother", "aunt", "wife", "actress"],
    ],
    expr: "king - man + woman",
    axes: [["man", "woman"], ["boy", "man"], ["peasant", "king"]],
  },
  {
    name: "Countries & capitals",
    groups: [
      ["france", "germany", "italy", "spain", "japan", "china", "russia", "egypt"],
      ["paris", "berlin", "rome", "madrid", "tokyo", "beijing", "moscow", "cairo"],
    ],
    expr: "paris - france + japan",
    axes: [["country", "city"], ["europe", "asia"], ["north", "south"]],
  },
  {
    name: "Verb tenses",
    groups: [
      ["walk", "swim", "run", "eat", "go", "see", "write", "think"],
      ["walked", "swam", "ran", "ate", "went", "saw", "wrote", "thought"],
    ],
    expr: "swam - swim + write",
    axes: [["now", "then"], ["walk", "think"], ["slow", "fast"]],
  },
  {
    name: "Philosophers",
    groups: [
      ["plato", "aristotle", "kant", "hegel", "hume", "descartes", "nietzsche", "wittgenstein"],
      ["reason", "ethics", "logic", "metaphysics", "virtue", "mind", "knowledge", "truth"],
      ["newton", "darwin", "einstein", "galileo"],
    ],
    expr: "kant - germany + france",
    axes: [["ancient", "modern"], ["science", "philosophy"], ["mind", "body"]],
  },
  {
    name: "Bias check",
    groups: [
      ["doctor", "engineer", "programmer", "scientist", "boss", "pilot"],
      ["nurse", "receptionist", "secretary", "teacher", "librarian", "housekeeper"],
      ["he", "she"],
    ],
    expr: "doctor - man + woman",
    axes: [["he", "she"], ["poor", "rich"], ["weak", "strong"]],
  },
];

const CLASSES = ["c1", "c2", "c3", "c4"];

const GLOVE = "GloVe (“Global Vectors”, Stanford, 2014) is a much older and simpler model than an LLM. " +
  "It gives every whole word one fixed list of 300 numbers, learned from how often words appear near each other in Wikipedia and news text. " +
  "An LLM works on tokens instead, and its vectors change with the context (see the panel further down).";

export default {
  id: "embeddings",
  title: "Embeddings",
  mount(main) {
    let set = SETS[0];
    let extra = []; // words added by hand
    let result = null; // {expr, terms, neighbors}
    let method = "pca";

    // ---------------------------------------------------------- 3D stage
    const stageEl = h("div", { class: "stage" }, h("span", { class: "hint" }, "Drag to turn · scroll to zoom · click a word for its neighbours, again to clear"));
    const stage = createStage(stageEl, { camera: [2.1, 1.3, 2.6] });
    const { THREE } = stage;
    const world = new THREE.Group();
    stage.scene.add(world);
    const axesGroup = new THREE.Group();
    stage.scene.add(axesGroup);
    let spin = true;
    stage.controls.addEventListener("start", () => { spin = false; });
    stageEl.addEventListener("pointerenter", () => { spin = false; }); // hold still so words can be clicked
    // A click on empty space (not the end of a drag) clears the selected word.
    let downAt = null;
    stageEl.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
    stageEl.addEventListener("click", (e) => {
      const still = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5;
      if (still && selected && !e.target.closest(".wlabel, button")) clearNeighbors();
    });
    stage.onFrame.push(() => { if (spin) { world.rotation.y += 0.0015; axesGroup.rotation.y = world.rotation.y; } });

    function drawAxes(names) {
      stage.clear(axesGroup);
      const line = cssVar("--line");
      const mat = new THREE.LineBasicMaterial({ color: line });
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1.2, 0, 0), new THREE.Vector3(1.2, 0, 0),
        new THREE.Vector3(0, -1.2, 0), new THREE.Vector3(0, 1.2, 0),
        new THREE.Vector3(0, 0, -1.2), new THREE.Vector3(0, 0, 1.2),
      ]);
      axesGroup.add(new THREE.LineSegments(g, mat));
      const grid = new THREE.GridHelper(2.4, 12, line, line);
      grid.position.y = -1.2;
      grid.material.transparent = true;
      grid.material.opacity = 0.6;
      axesGroup.add(grid);
      if (names) {
        const ends = [[1.3, 0, 0], [0, 1.3, 0], [0, 0, 1.3]];
        names.forEach((n, i) => {
          axesGroup.add(stage.label(`${n[1]} →`, "llabel", ends[i]));
          axesGroup.add(stage.label(`← ${n[0]}`, "llabel", ends[i].map((v) => -v)));
        });
      }
    }

    function items() {
      const list = [];
      set.groups.forEach((g, gi) => g.forEach((w) => list.push({ label: w, cls: CLASSES[gi] })));
      extra.forEach((w) => list.push({ label: w, cls: "c4" }));
      if (result) {
        list.push({ label: result.expr, expr: result.expr, cls: "result" });
        const best = result.neighbors[0]?.[0];
        if (best && !list.some((i) => i.label === best)) list.push({ label: best, cls: "c4" });
      }
      return list;
    }

    async function draw() {
      const list = items();
      if (!list.length) { msg.replaceChildren(); stage.clear(world); drawAxes(method === "axes" ? set.axes : null); return; }
      try {
        const { xyz } = await post("/api/words/project", { items: list.map(({ label, expr }) => ({ label, expr })), method, axes: method === "axes" ? set.axes : null });
        msg.replaceChildren();
        stage.clear(world);
        drawAxes(method === "axes" ? set.axes : null);
        const pos = {};
        list.forEach((it, i) => {
          const p = xyz[i];
          pos[it.label] = p;
          const isRes = it.cls === "result";
          const color = isRes ? cssVar("--accent") : cssVar(`--${it.cls}`) || cssVar("--ink");
          const m = new THREE.Mesh(new THREE.SphereGeometry(isRes ? 0.035 : 0.022, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
          m.position.set(...p);
          world.add(m);
          const lab = stage.label(isRes ? "= ?" : it.label, `wlabel ${it.cls}`, [p[0], p[1] + 0.06, p[2]]);
          if (!isRes) { lab.element.dataset.word = it.label; lab.element.addEventListener("click", () => showNeighbors(it.label)); }
          lab.element.title = isRes ? result.expr : `${it.label}: click for nearest words`;
          world.add(lab);
        });
        // Arithmetic as arrows: the step from b to c, then the same step taken from a.
        if (result) {
          const t = result.terms;
          const plus = t.filter(([s]) => s > 0).map(([, w]) => w), minus = t.filter(([s]) => s < 0).map(([, w]) => w);
          const a = plus[0], c = plus[1], b = minus[0];
          const ink = cssVar("--muted"), acc = cssVar("--accent");
          if (a && b && c && pos[a] && pos[b] && pos[c]) {
            world.add(arrow(THREE, pos[b], pos[c], ink, { radius: 0.005, head: 0.05 }));
            world.add(arrow(THREE, pos[a], pos[result.expr], acc, { radius: 0.007, head: 0.06 }));
          }
          const best = result.neighbors[0]?.[0];
          if (best && pos[best]) {
            const dashed = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...pos[result.expr]), new THREE.Vector3(...pos[best])]), new THREE.LineDashedMaterial({ color: acc, dashSize: 0.02, gapSize: 0.015 }));
            dashed.computeLineDistances();
            world.add(dashed);
          }
        }
        markSelected();
      } catch (e) { errorLine(msg, e); }
    }
    stage.onTheme.push(() => draw());

    // ---------------------------------------------------------- side panel
    const msg = h("div");
    const wordsBox = h("div", { class: "chips" });
    const addInput = h("input", { type: "text", placeholder: "add a word…", style: { width: "11rem" } });
    const addForm = h("form", { class: "row", onsubmit: (e) => { e.preventDefault(); const w = addInput.value.trim().toLowerCase(); if (w) { extra.push(w); addInput.value = ""; renderWords(); draw(); } } }, addInput, h("button", { class: "btn small", type: "submit" }, "Add"));

    // A word in the list: click it for its neighbours, × takes it out of the picture.
    function wordChip(w, cls, remove) {
      return h("span", { class: `wlabel ${cls}`, "data-word": w },
        h("span", { title: `${w}: click for nearest words`, onclick: () => showNeighbors(w) }, w),
        h("button", { type: "button", class: "x", title: `Remove “${w}”`, "aria-label": `Remove ${w}`, onclick: () => { remove(); if (w === selected) clearNeighbors(); renderWords(); draw(); } }, "×"));
    }

    function renderWords() {
      wordsBox.replaceChildren(
        ...set.groups.flatMap((g, gi) => g.map((w) => wordChip(w, CLASSES[gi], () => { set.groups[gi] = set.groups[gi].filter((x) => x !== w); }))),
        ...extra.map((w) => wordChip(w, "c4", () => { extra = extra.filter((x) => x !== w); })));
      if (!wordsBox.childElementCount) wordsBox.append(h("span", { class: "small muted" }, "No words yet. Add your own, or pick a set."));
      markSelected();
    }

    const exprInput = h("input", { type: "text", class: "mono", value: set.expr, "aria-label": "Word arithmetic" });
    const hideInputs = h("input", { type: "checkbox", checked: true });
    const exprOut = h("div", { class: "probs" });
    const computeBtn = h("button", { class: "btn primary", type: "submit" }, "Compute");
    const exprForm = h("form", { class: "row", style: { flexWrap: "nowrap" }, onsubmit: (e) => { e.preventDefault(); compute(); } }, exprInput, computeBtn);

    async function compute() {
      const expr = exprInput.value.trim();
      if (!expr) { result = null; exprOut.replaceChildren(); draw(); return; }
      busy(computeBtn, true);
      try {
        const r = await post("/api/words/expr", { expr, k: 8, exclude_inputs: hideInputs.checked });
        result = { expr, ...r };
        exprOut.replaceChildren(
          ...r.neighbors.map(([w, s], i) => simRow(w, s, i === 0)),
          h("p", { class: "small muted" }, hideInputs.checked
            ? `Nearest of all, if the input words are allowed: ${r.with_inputs.slice(0, 3).map(([w, s]) => `${w} (${s.toFixed(2)})`).join(", ")}.`
            : "Input words included: often the answer is just one of the words you typed."));
        draw();
      } catch (e) { errorLine(exprOut, e); }
      busy(computeBtn, false);
    }
    hideInputs.addEventListener("change", compute);

    function simRow(word, s, hl) {
      return h("div", { class: `prob${hl ? " hl" : ""}` },
        h("span", { class: "word" }, word),
        h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${Math.max(0, s) * 100}%`, background: hl ? "" : "var(--ink)" } })),
        h("span", { class: "pct" }, s.toFixed(2)));
    }

    const nbOut = h("div", { class: "probs" });
    const nbBox = box("Nearest words", { body: [nbOut], cls: "open" });
    const nbTitle = nbBox.querySelector("h3");
    let selected = null;

    function markSelected() {
      for (const el of [...stageEl.querySelectorAll(".wlabel"), ...wordsBox.querySelectorAll(".wlabel")]) el.classList.toggle("sel", el.dataset.word === selected);
    }

    function clearNeighbors() {
      selected = null;
      nbTitle.textContent = "Nearest words";
      nbOut.replaceChildren(h("p", { class: "small muted" }, "Click any word in the picture or the list. Click it again, or empty space in the picture, to clear."));
      markSelected();
    }

    // Clicking the selected word again clears it.
    async function showNeighbors(word) {
      if (word === selected) { clearNeighbors(); return; }
      selected = word;
      markSelected();
      nbTitle.textContent = `Nearest to “${word}”`;
      try {
        const r = await post("/api/words/expr", { expr: word, k: 8, exclude_inputs: true });
        if (selected !== word) return;
        nbOut.replaceChildren(...r.neighbors.map(([w, s]) => {
          const row = simRow(w, s, false);
          row.title = "Click to add to the picture";
          row.style.cursor = "pointer";
          row.addEventListener("click", () => { if (!extra.includes(w)) { extra.push(w); renderWords(); draw(); } });
          return row;
        }));
      } catch (e) { errorLine(nbOut, e); }
    }

    const methodSeg = h("div", { class: "seg", role: "group", "aria-label": "Projection" },
      h("button", { type: "button", "aria-pressed": "true", "data-m": "pca" }, "Best view (PCA)"),
      h("button", { type: "button", "aria-pressed": "false", "data-m": "axes" }, "Meaning axes"));
    methodSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      method = b.dataset.m;
      methodSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      axesInfo.hidden = method !== "axes";
      renderAxes();
      draw();
    });
    const axesInfo = h("div", { class: "small muted", hidden: true });
    function renderAxes() {
      axesInfo.replaceChildren(h("p", {}, "Each axis is the direction from one word to another. A word's position on it says how far it leans that way."),
        ...set.axes.map((ax, i) => h("form", { class: "row", style: { marginTop: ".35rem" }, onsubmit: (e) => { e.preventDefault(); set.axes[i] = [e.target.a.value.trim().toLowerCase(), e.target.b.value.trim().toLowerCase()]; draw(); } },
          h("b", { class: "mono" }, "XYZ"[i]),
          h("input", { type: "text", name: "a", value: ax[0], style: { width: "7.5rem" } }), "→",
          h("input", { type: "text", name: "b", value: ax[1], style: { width: "7.5rem" } }),
          h("button", { class: "btn small", type: "submit" }, "Set"))));
    }

    // null clears everything; the meaning axes stay as they were.
    function pickSet(s) {
      set = s ? structuredClone(s) : { name: null, groups: [], expr: "", axes: set.axes };
      extra = [];
      exprInput.value = set.expr;
      clearNeighbors();
      renderWords();
      renderAxes();
      compute();
    }

    // ---------------------------------------------------------- contextual vectors
    const ctx = contextualPanel();
    const dot = dotPanel();

    main.append(
      intro("Chapter 2 · Embeddings",
        "Words as vectors in a space of meaning",
        "A language model turns each token into a vector: a long list of numbers, which you can picture as an arrow in a space with hundreds or thousands of dimensions. Training makes tokens that occur in similar contexts point in similar directions, and directions in that space can come to stand for things like gender, tense or capital-of. The picture shows this with GloVe, a simpler model that gives each whole word one vector, and marks only where each arrow ends. Further down: the language model's own vectors."),
      h("div", { class: "grid2 spaced" },
        box([h("span", { class: "term", title: GLOVE }, "GloVe"), ": 300 dimensions, projected onto 3"], { right: methodSeg, body: [stageEl], cls: "flush open" }),
        h("div", { class: "stack" },
          box("Word sets", { body: [presets(SETS, pickSet, { active: SETS[0], toggle: true }), wordsBox, addForm, axesInfo, msg], cls: "open" }),
          box("Word arithmetic", {
            cls: "open",
            body: [exprForm, h("label", { class: "check" }, hideInputs, "Leave out the words I typed"), exprOut],
          }),
          nbBox)),
      dot.el,
      ctx.el,
      notes({
        math: [
          "A word is a <em>vector</em>: a list of numbers, here 300 of them. The basic way to compare two vectors is the <em>dot product</em>: multiply them number by number and add up. It is large when the vectors point the same way, <em>and</em> when they are long.",
          { f: "a · b = a₁×b₁ + a₂×b₂ + … = |a| × |b| × cos(angle)" },
          "To compare only the direction, divide by the two lengths. That is the <em>cosine similarity</em>: 1 for the same direction, 0 at right angles (unrelated), −1 for opposite. We scaled all GloVe vectors here to length 1, so for them the dot product <em>is</em> the cosine. Attention (chapter 5) uses the plain dot product, where length counts too.",
          "Adding and subtracting vectors works number by number. The arrow from <b>man</b> to <b>woman</b> is roughly the same arrow as from <b>king</b> to <b>queen</b>, so <b>king − man + woman</b> lands near <b>queen</b>.",
          "No one can look at 300 dimensions, so the picture projects them onto 3 directions, like a shadow. “Best view” picks the 3 directions in which these particular words differ most (principal component analysis). Distances in the picture are only a rough guide; the similarity numbers on the right are exact.",
          "A language model starts with an <em>embedding table</em>: one learned row of numbers for each token in its vocabulary<span class=\"embsize\"></span>. Its first step only looks up the row for each token, so a token starts out with the same vector in every sentence. Context comes in later, as each layer reshapes the vectors.",
        ],
        discuss: [
          "These vectors were learned only from which words appear near which others. Is that a theory of meaning (“meaning is use”), or only a statistical shadow of one?",
          "Try the analogy with “Leave out the words I typed” switched off. How much of the famous result is in the method rather than in the data?",
          "The “Bias check” set shows occupations leaning towards “he” or “she”. Does the space describe the world, the texts, or the people who wrote them?",
          "GloVe gives “bank” a single vector. Is a word with two senses one word or two?",
        ],
      }),
    );

    clearNeighbors();
    renderWords();
    renderAxes();
    drawAxes(null);
    compute();

    info(true).then((d) => {
      const m = d.models.base;
      const size = main.querySelector(".embsize");
      if (m?.vocab && size) size.textContent = ` (for ${m.label}: ${m.vocab.toLocaleString("en")} rows of ${m.hidden.toLocaleString("en")} numbers)`;
    }).catch(() => {});

    return () => { stage.dispose(); dot.dispose(); };
  },
};

const CTX_SETS = [
  {
    name: "bank",
    word: "bank",
    sentences: [
      "She sat on the bank of the river and watched the water.",
      "The fishermen pulled their boat onto the muddy bank.",
      "Wildflowers grew along the bank of the stream.",
      "He deposited his salary at the bank on Friday.",
      "The bank refused to give them a loan.",
      "Interest rates at the central bank went up again.",
    ],
  },
  {
    name: "bat",
    word: "bat",
    sentences: [
      "A bat flew out of the cave at dusk.",
      "The bat hung upside down from the branch, asleep.",
      "He swung the bat and hit the ball over the fence.",
      "She bought a new cricket bat for the season.",
      "The vampire turned into a bat and vanished.",
    ],
  },
  {
    name: "light",
    word: "light",
    sentences: [
      "Turn on the light, it is getting dark.",
      "The light from the window fell across the floor.",
      "This suitcase is very light; I can carry it with one hand.",
      "She prefers a light meal in the evening.",
      "In the light of these arguments, we should reconsider.",
    ],
  },
];

function contextualPanel() {
  let current = structuredClone(CTX_SETS[0]);
  const wordIn = h("input", { type: "text", value: current.word, style: { width: "9rem" } });
  const sentIn = h("textarea", { rows: 6, class: "sent" }, current.sentences.join("\n"));
  const out = h("div", { class: "scroll" });
  const btn = h("button", { class: "btn primary", type: "button" }, "Compare");
  const layerOut = h("span", { class: "small muted" });

  async function run() {
    const sentences = sentIn.value.split("\n").map((s) => s.trim()).filter(Boolean);
    busy(btn, true, "Reading…");
    try {
      const r = await post("/api/contextual", { sentences, word: wordIn.value.trim() });
      const kept = sentences.filter((_, i) => r.found[i]);
      layerOut.textContent = `Vectors taken after layer ${r.layer}.`;
      const head = h("tr", {}, h("th", {}, "sentence"), ...kept.map((_, i) => h("th", { class: "num" }, `#${i + 1}`)));
      const rows = kept.map((s, i) => h("tr", {},
        h("td", { class: "sent" }, h("b", { class: "mono" }, `#${i + 1} `), highlight(s, wordIn.value.trim())),
        ...r.sim[i].map((v, j) => {
          const [rr, gg, bb] = divergingColor((v - 0.5) * 2, 1);
          return h("td", { class: "num", style: { background: i === j ? "" : `rgb(${rr},${gg},${bb})`, color: i === j ? "var(--faint)" : "" } }, i === j ? "—" : v.toFixed(2));
        })));
      out.replaceChildren(h("table", {}, h("thead", {}, head), h("tbody", {}, rows)));
    } catch (e) { errorLine(out, e); }
    busy(btn, false);
  }

  function highlight(s, w) {
    const i = s.toLowerCase().indexOf(w.toLowerCase());
    if (i < 0) return s;
    return [s.slice(0, i), h("b", { style: { color: "var(--accent)" } }, s.slice(i, i + w.length)), s.slice(i + w.length)];
  }

  btn.addEventListener("click", run);
  const modelName = h("span", {}, "the LLM");
  info().then((d) => { if (d.models.base?.label) modelName.textContent = d.models.base.label; }).catch(() => {});
  const el = box(["The same token, in different sentences, inside ", modelName], {
    cls: "open",
    right: presets(CTX_SETS, (p) => { current = structuredClone(p); wordIn.value = p.word; sentIn.value = p.sentences.join("\n"); run(); }, { active: CTX_SETS[0] }),
    body: [
      h("p", { class: "muted" }, "Above, each word has one fixed vector. A language model starts the same way: its first step looks up one fixed vector per token, so the token for “bank” starts out identical in every sentence. Then every layer rebuilds each token's vector from its surroundings. Here we read the vector halfway up the model, in each sentence, and compare. Warm cells: the model treats the two uses as similar. A word that is split into several tokens gets the average of their vectors."),
      h("div", { class: "grid2 even", style: { gap: "1rem" } },
        h("div", { style: { display: "grid", gap: ".6rem", alignContent: "start" } },
          h("label", { class: "field" }, h("span", {}, "Word"), wordIn),
          h("label", { class: "field" }, h("span", {}, "Sentences, one per line"), sentIn),
          h("div", { class: "row" }, btn, layerOut)),
        out),
    ],
  });
  return { el };
}

// ------------------------------------------------------------------ the dot product, in 2D

function dotPanel() {
  const W = 520, H = 400, CX = W / 2, CY = H / 2, S = 60, MAX = 3;
  let a = [2, 0.5], b = [1, 2];

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Two vectors and their dot product" });
  const calc = h("div", { class: "dotcalc" });

  const toXY = ([x, y]) => [CX + x * S, CY - y * S];
  const snap = (v) => Math.max(-MAX, Math.min(MAX, Math.round(v * 10) / 10));
  const fromXY = (px, py) => [snap((px - CX) / S), snap((CY - py) / S)];
  const len = (v) => Math.hypot(v[0], v[1]);
  const num = (v, d = 2) => `${v < -0.5 * 10 ** -d ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  const font = { "font-family": "Archivo, sans-serif", "font-size": 14, "font-weight": 700 };

  function render() {
    const line = cssVar("--line"), faint = cssVar("--faint"), ink = cssVar("--ink"), ca = cssVar("--c1"), cb = cssVar("--c2");
    const dot = a[0] * b[0] + a[1] * b[1], la = len(a), lb = len(b);
    const cos = la && lb ? dot / (la * lb) : null;
    const deg = cos == null ? null : (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;

    const grid = [];
    for (let i = -MAX; i <= MAX; i++) {
      grid.push(svgEl("line", { x1: CX + i * S, y1: CY - MAX * S - 10, x2: CX + i * S, y2: CY + MAX * S + 10, stroke: i ? line : faint }));
      grid.push(svgEl("line", { x1: CX - MAX * S - 10, y1: CY - i * S, x2: CX + MAX * S + 10, y2: CY - i * S, stroke: i ? line : faint }));
    }

    // The angle between the arrows, drawn as an arc at the origin.
    const extra = [], onTop = [];
    if (cos != null) {
      const ta = Math.atan2(a[1], a[0]);
      let d = Math.atan2(b[1], b[0]) - ta;
      if (d > Math.PI) d -= 2 * Math.PI;
      if (d <= -Math.PI) d += 2 * Math.PI;
      const pt = (t, r) => [CX + r * Math.cos(t), CY - r * Math.sin(t)];
      const [x0, y0] = pt(ta, 30), [x1, y1] = pt(ta + d, 30), [lx, ly] = pt(ta + d / 2, 50);
      extra.push(svgEl("path", { d: `M ${x0} ${y0} A 30 30 0 0 ${d > 0 ? 0 : 1} ${x1} ${y1}`, fill: "none", stroke: ink, "stroke-width": 1.5 }));
      // Drawn last, with a halo in the background colour, so the arrows never cover it.
      onTop.push(svgEl("text", { x: lx, y: ly + 5, fill: ink, stroke: cssVar("--sheet"), "stroke-width": 4, "stroke-linejoin": "round", "paint-order": "stroke", "text-anchor": "middle", ...font, "font-size": 13 }, `${Math.round(deg)}°`));
      // b's shadow on the line through a: the dot product is |a| times its (signed) length.
      const k = dot / (la * la), p = [a[0] * k, a[1] * k];
      const [bx, by] = toXY(b), [px, py] = toXY(p);
      extra.push(svgEl("line", { x1: CX, y1: CY, x2: px, y2: py, stroke: cb, "stroke-width": 9, "stroke-opacity": 0.3 }));
      extra.push(svgEl("line", { x1: bx, y1: by, x2: px, y2: py, stroke: cb, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    }

    // a's label sits below its dot and b's above, so they stay apart when the arrows overlap.
    const arrowTo = (v, col, name, dy) => {
      const [x, y] = toXY(v);
      return [
        svgEl("line", { x1: CX, y1: CY, x2: x, y2: y, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }),
        svgEl("circle", { cx: x, cy: y, r: 10, fill: col, class: "key", "data-v": name }),
        // Near the right edge the label goes to the left of the dot instead.
        svgEl("text", { x: x > W - 130 ? x - 14 : x + 14, y: y + dy, fill: col, "text-anchor": x > W - 130 ? "end" : "start", ...font }, `${name} (${num(v[0], 1)}, ${num(v[1], 1)})`),
      ];
    };
    svg.replaceChildren(...grid, ...extra, ...arrowTo(a, ca, "a", 24), ...arrowTo(b, cb, "b", -12), ...onTop,
      svgEl("text", { x: 12, y: H - 12, fill: cssVar("--muted"), "font-size": 12, "font-family": "Archivo, sans-serif" }, "Drag the two dots."),
      svgEl("text", { x: W - 12, y: H - 12, fill: ink, stroke: cssVar("--sheet"), "stroke-width": 5, "stroke-linejoin": "round", "paint-order": "stroke", "text-anchor": "end", ...font, "font-size": 20 }, `a · b = ${num(dot)}`));

    const row = (label, text) => [h("span", { class: "label" }, label), h("div", { class: "mono" }, text)];
    calc.replaceChildren(
      ...row("The dot product: multiply and add", `a · b = ${num(a[0], 1)} × ${num(b[0], 1)} + ${num(a[1], 1)} × ${num(b[1], 1)} = ${num(dot)}`),
      ...row("Lengths and angle", cos == null ? `|a| = ${num(la)}   |b| = ${num(lb)}` : `|a| = ${num(la)}   |b| = ${num(lb)}   angle = ${Math.round(deg)}°`),
      ...row("The same number, from the picture", cos == null ? "—" : `|a| × |b| × cos(${Math.round(deg)}°) = ${num(la)} × ${num(lb)} × ${num(cos)} = ${num(dot)}`),
      ...row("Direction only: the cosine similarity", cos == null ? "— (an arrow of length 0 has no direction)" : `cos(${Math.round(deg)}°) = a · b ÷ (|a| × |b|) = ${num(cos)}`),
    );
  }

  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    const t = e.target.closest(".key");
    if (!t) return;
    drag = t.dataset.v;
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = svg.getBoundingClientRect();
    const p = fromXY(((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H);
    if (drag === "a") a = p; else b = p;
    render();
  });
  svg.addEventListener("pointerup", () => { drag = null; });

  // Turn b to a given angle from a, keeping its length.
  const turnB = (angle) => () => {
    const lb = len(b) || 1, t = Math.atan2(a[1], a[0]) + angle;
    b = [lb * Math.cos(t), lb * Math.sin(t)];
    render();
  };
  const unit = () => { if (len(a)) a = a.map((x) => x / len(a)); if (len(b)) b = b.map((x) => x / len(b)); render(); };
  const action = (label, fn) => h("button", { class: "chip-btn", type: "button", onclick: fn }, label);

  const observer = new MutationObserver(render);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  render();

  return {
    el: h("div", { class: "grid2 even spaced" },
      box("Measuring similarity: the dot product", { body: [h("div", { class: "toy" }, svg)], cls: "open" }),
      box("Two ways to get the same number", {
        cls: "open",
        right: h("div", { class: "chips" }, action("Same direction", turnB(0)), action("At right angles", turnB(Math.PI / 2)), action("Opposite", turnB(Math.PI)), action("Make both length 1", unit)),
        body: [
          h("p", { class: "muted" }, "The dot product multiplies two vectors number by number and adds up. It grows when the arrows point the same way, and also when they get longer. In the picture, the dashed line drops b onto the line through a: the dot product is the length of a times the length of that shadow (negative when the shadow falls behind). Divide by both lengths and only the direction is left: the cosine similarity."),
          calc,
          h("p", { class: "small muted" }, "Word vectors work the same way, with 300 numbers instead of 2. The GloVe vectors above all have length 1, so for them the dot product and the cosine are the same number."),
        ],
      })),
    dispose: () => observer.disconnect(),
  };
}
