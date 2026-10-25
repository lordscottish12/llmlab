import { h, intro, notes, box, presets, errorLine, esc, busy, cssVar, divergingColor } from "../ui.js";
import { post } from "../api.js";
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
      ["plato", "aristotle", "kant", "hegel", "hume", "descartes", "nietzsche", "spinoza"],
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

export default {
  id: "embeddings",
  title: "Embeddings",
  mount(main) {
    let set = SETS[0];
    let extra = []; // words added by hand
    let result = null; // {expr, terms, neighbors}
    let method = "pca";

    // ---------------------------------------------------------- 3D stage
    const stageEl = h("div", { class: "stage" }, h("span", { class: "hint" }, "Drag to turn · scroll to zoom · click a word for its neighbours"));
    const stage = createStage(stageEl, { camera: [2.1, 1.3, 2.6] });
    const { THREE } = stage;
    const world = new THREE.Group();
    stage.scene.add(world);
    const axesGroup = new THREE.Group();
    stage.scene.add(axesGroup);
    let spin = true;
    stage.controls.addEventListener("start", () => { spin = false; });
    stageEl.addEventListener("pointerenter", () => { spin = false; }); // hold still so words can be clicked
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
          if (!isRes) lab.element.addEventListener("click", () => showNeighbors(it.label));
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
      } catch (e) { errorLine(msg, e); }
    }
    stage.onTheme.push(() => draw());

    // ---------------------------------------------------------- side panel
    const msg = h("div");
    const wordsBox = h("div", { class: "chips" });
    const addInput = h("input", { type: "text", placeholder: "add a word…", style: { width: "11rem" } });
    const addForm = h("form", { class: "row", onsubmit: (e) => { e.preventDefault(); const w = addInput.value.trim().toLowerCase(); if (w) { extra.push(w); addInput.value = ""; renderWords(); draw(); } } }, addInput, h("button", { class: "btn small", type: "submit" }, "Add"));

    function renderWords() {
      wordsBox.replaceChildren(...set.groups.flatMap((g, gi) => g.map((w) => h("span", { class: `wlabel ${CLASSES[gi]}`, onclick: () => showNeighbors(w) }, w))),
        ...extra.map((w) => h("span", { class: "wlabel c4", title: "Click to remove", onclick: () => { extra = extra.filter((x) => x !== w); renderWords(); draw(); } }, `${w} ×`)));
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

    const nbOut = h("div", { class: "probs" }, h("p", { class: "small muted" }, "Click any word in the picture or the list."));
    const nbTitle = h("h3", {}, "Nearest words");
    async function showNeighbors(word) {
      nbTitle.textContent = `Nearest to “${word}”`;
      try {
        const r = await post("/api/words/expr", { expr: word, k: 8, exclude_inputs: true });
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

    function pickSet(s) {
      set = structuredClone(s);
      extra = [];
      exprInput.value = set.expr;
      renderWords();
      renderAxes();
      compute();
    }

    // ---------------------------------------------------------- contextual vectors
    const ctx = contextualPanel();

    main.append(
      intro("Step 1 · Embeddings",
        "Words as points in a space of meaning",
        "Each token is turned into a long list of numbers: a point in a space with hundreds of dimensions. Training places words that occur in similar contexts close together. Directions in that space can come to stand for things like gender, tense or capital-of."),
      h("div", { class: "grid2" },
        box("300 dimensions, squeezed into 3", { right: methodSeg, body: [stageEl], cls: "flush" }),
        h("div", { class: "stack" },
          box("Word sets", { body: [presets(SETS, pickSet, { active: SETS[0] }), wordsBox, addForm, axesInfo, msg] }),
          box("Word arithmetic", {
            body: [exprForm, h("label", { class: "check" }, hideInputs, "Leave out the words I typed"), exprOut],
          }),
          box(null, { body: [nbTitle, nbOut] }))),
      ctx.el,
      notes({
        math: [
          "A word is a <em>vector</em>: a list of numbers, here 300 of them. Two words are similar when their vectors point the same way. We measure that with the <em>cosine similarity</em>: 1 means the same direction, 0 unrelated.",
          { f: "similarity(a, b) = cos(angle between a and b)" },
          "Adding and subtracting vectors works number by number. The arrow from <b>man</b> to <b>woman</b> is roughly the same arrow as from <b>king</b> to <b>queen</b>, so <b>king − man + woman</b> lands near <b>queen</b>.",
          "No one can look at 300 dimensions, so the picture flattens them to 3. “Best view” keeps the directions in which these particular words differ most (principal component analysis). Distances in the picture are only a rough guide; the similarity numbers on the right are exact.",
        ],
        discuss: [
          "These vectors were learned only from which words appear near which others. Is that a theory of meaning (“meaning is use”), or only a statistical shadow of one?",
          "Try the analogy with “Leave out the words I typed” switched off. How much of the famous result is in the method rather than in the data?",
          "The “Bias check” set shows occupations leaning towards “he” or “she”. Does the space describe the world, the texts, or the people who wrote them?",
          "Static vectors give “bank” one point. Is a word with two senses one word or two?",
        ],
      }),
    );

    renderWords();
    renderAxes();
    drawAxes(null);
    compute();

    return () => stage.dispose();
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
  const el = box("The same word, in different sentences, inside the LLM", {
    right: presets(CTX_SETS, (p) => { current = structuredClone(p); wordIn.value = p.word; sentIn.value = p.sentences.join("\n"); run(); }, { active: CTX_SETS[0] }),
    body: [
      h("p", { class: "muted" }, "The word vectors above give each word a single fixed point. Inside the language model, the vector for a word is rebuilt at every layer from its surroundings. Here we read it halfway up the model, for the word in each sentence, and compare. Warm cells: the model treats the two uses as similar."),
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
