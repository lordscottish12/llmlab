import { h, intro, notes, box, errorLine, esc, busy, cssVar, divergingColor, probBars, showTok, tip, pct, svgEl } from "../ui.js";
import { post } from "../api.js";
import { createStage } from "../scene.js";

const PROMPT = "The capital of France is";

const DX = 1.15; // distance between token columns
const DY = 0.32; // distance between layers
const BAR = 1.8; // height of a 100% output bar
const TILE = 0.82;

export default {
  id: "forward",
  title: "One prediction",
  mount(main) {
    let data = null;
    let selected = null; // [layer, pos]
    let timer = null;

    const stageEl = h("div", { class: "stage", style: { height: "clamp(30rem, 72vh, 52rem)" } },
      h("span", { class: "hint" }, "Drag to turn · scroll to zoom · hover a tile to read it as a prediction · click a tile or token to select it"));
    const caption = h("p", { class: "small muted" });
    const legend = h("div", { class: "legend" });
    const stage = createStage(stageEl, { camera: [6, 7, 12], target: [3, 5, 0], fov: 40 });
    const { THREE } = stage;
    const world = new THREE.Group();
    stage.scene.add(world);
    let tiles = []; // meshes with userData {l, pos}
    let floors = [];
    let floorLabels = [], tokenLabels = [];
    const selGroup = new THREE.Group(); // everything that follows the selected token

    const input = h("input", { type: "text", value: PROMPT, "aria-label": "Prompt" });
    const runBtn = h("button", { class: "btn primary", type: "submit" }, "Run the model");
    const form = h("form", { class: "row", style: { flexWrap: "nowrap" }, onsubmit: (e) => { e.preventDefault(); run(); } }, input, runBtn);
    const msg = h("div");
    const finalBox = h("div", {}, h("p", { class: "small muted" }, "Run the model to see its prediction."));
    const finalPanel = box("The output: next-token probabilities", { body: [finalBox], cls: "open" });
    const finalTitle = finalPanel.querySelector("h3");
    const lensBox = h("div", { class: "scroll fit" });
    const lensPanel = box("How the guess forms, layer by layer", { body: [h("p", { class: "small muted" }, "Click any tile in the picture to read that vector instead."), lensBox], cls: "open grow" });
    const lensTitle = lensPanel.querySelector("h3");
    const flow = h("p", { class: "small muted" });
    const arrowSvg = svgEl("svg", { viewBox: "0 0 320 320", role: "img", "aria-label": "The selected token's vector after every layer, as arrows" });
    arrowSvg.addEventListener("click", (e) => {
      const t = e.target.closest("[data-l]");
      if (t && data) select(+t.dataset.l, selected[1]);
    });
    const arrowNote = h("p", { class: "small muted" });
    const arrowPanel = box("Seen from two directions", { body: [h("div", { class: "split" }, h("div", { class: "toy" }, arrowSvg), arrowNote)], cls: "open" });
    const arrowTitle = arrowPanel.querySelector("h3");

    async function run() {
      busy(runBtn, true, "Running…");
      try {
        data = await post("/api/forward", { text: input.value });
        msg.replaceChildren();
        selected = [data.layers, data.tokens.length - 1];
        build();
        renderFinal();
        renderLens();
        animate();
      } catch (e) { errorLine(msg, e); }
      busy(runBtn, false);
    }

    function tileTexture(values) {
      const n = Math.round(Math.sqrt(values.length));
      const buf = new Uint8Array(n * n * 4);
      values.forEach((v, i) => {
        const [r, g, b] = divergingColor(v, 2);
        buf.set([r, g, b, 255], i * 4);
      });
      const tex = new THREE.DataTexture(buf, n, n);
      tex.magFilter = THREE.NearestFilter;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      return tex;
    }

    function build() {
      stage.clear(world);
      tiles = [];
      floors = [];
      floorLabels = [];
      const n = data.tokens.length, L = data.layers;
      const width = (n - 1) * DX;
      const line = cssVar("--line"), ink = cssVar("--ink"), muted = cssVar("--muted");

      // Residual stream: one vertical line per token, from embedding to output.
      const pipeMat = new THREE.LineBasicMaterial({ color: line });
      for (let p = 0; p < n; p++) {
        const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p * DX, 0, 0), new THREE.Vector3(p * DX, (L + 1) * DY, 0)]);
        world.add(new THREE.Line(g, pipeMat));
      }

      for (let l = 0; l <= L; l++) {
        const floor = new THREE.Group();
        floor.position.y = l * DY;
        // A thin glass plate per layer: the block that reads the whole row and writes back into it.
        if (l > 0) {
          const plate = new THREE.Mesh(
            new THREE.BoxGeometry(width + DX, 0.012, TILE + 0.25),
            new THREE.MeshBasicMaterial({ color: ink, transparent: true, opacity: 0.05, depthWrite: false }));
          plate.position.set(width / 2, -DY / 2, 0);
          floor.add(plate);
        }
        for (let p = 0; p < n; p++) {
          const c = data.cells[l][p];
          const m = new THREE.Mesh(new THREE.PlaneGeometry(TILE, TILE), new THREE.MeshBasicMaterial({ map: tileTexture(c.v), side: THREE.DoubleSide, transparent: true }));
          m.rotation.x = -Math.PI / 2;
          m.position.set(p * DX, 0, 0);
          m.userData = { l, pos: p };
          floor.add(m);
          tiles.push(m);
        }
        const name = l === 0 ? "embedding" : l === L ? `layer ${l} · output` : `layer ${l}`;
        if (l === 0 || l === L || l % 4 === 0) floor.add(stage.label(name, "llabel", [-0.9, 0, 0]));
        // What this layer would predict for the selected token if the model stopped here (filled in by renderSelection).
        const lab = stage.label("", "llabel", [width + 1.15, 0, 0]);
        floor.add(lab);
        floorLabels.push(lab);
        world.add(floor);
        floors.push(floor);
      }

      // Token labels under the embedding floor. Clicking one selects that token's top tile.
      tokenLabels = data.tokens.map((t, p) => {
        const lab = stage.label(showTok(t), "tlabel", [p * DX, -0.35, TILE / 2 + 0.1]);
        lab.element.style.pointerEvents = "auto";
        lab.element.style.cursor = "pointer";
        lab.element.addEventListener("click", () => select(data.layers, p));
        world.add(lab);
        return lab.element;
      });
      world.add(selGroup);

      // Fit the whole tower, from the token labels to the output bars, into view.
      // Room above the tower for a full-height bar and its label, whichever token is selected.
      const yTop = (L + 1) * DY + BAR + 0.9, yBottom = -0.6;
      const mid = new THREE.Vector3(width / 2 + 0.6, (yTop + yBottom) / 2, 0);
      stage.controls.target.copy(mid);
      const vfov = (stage.camera.fov * Math.PI) / 180 / 2;
      const hfov = Math.atan(Math.tan(vfov) * stage.camera.aspect);
      const dist = 1.15 * Math.max((yTop - yBottom) / 2 / Math.tan(vfov), (width + 4) / 2 / Math.tan(hfov));
      stage.camera.position.set(mid.x + dist * 0.28, mid.y + dist * 0.18, dist * 0.95);
      renderSelection();
      flow.textContent = `${data.tokens.length} tokens enter at the bottom. Each becomes a vector of ${data.hidden.toLocaleString()} numbers. Each of the ${data.layers} layers reads all the vectors and adds a little to each. At the top, the last vector is turned into a probability for every token in the vocabulary.`;
    }

    // The selected token's column at full strength, the others faded; the labels on the right and the
    // bars on top show what the selected token's vector predicts.
    function renderSelection() {
      const [sl, pos] = selected;
      const n = data.tokens.length, L = data.layers, width = (n - 1) * DX;
      const ink = cssVar("--ink"), muted = cssVar("--muted"), acc = cssVar("--accent");
      tiles.forEach((t) => { t.material.opacity = t.userData.pos === pos ? 1 : 0.3; });
      tokenLabels.forEach((el, p) => el.classList.toggle("sel", p === pos));
      floorLabels.forEach((lab, l) => {
        const [t, p] = data.cells[l][pos].lens[0];
        lab.element.textContent = `${showTok(t)}  ${pct(p)}`;
        lab.element.style.color = p > 0.25 ? acc : muted;
        lab.element.style.fontWeight = p > 0.25 ? "700" : "500";
      });

      stage.clear(selGroup);
      const r = (TILE / 2) * 1.08;
      selGroup.add(new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, z]) => new THREE.Vector3(pos * DX + x * r, sl * DY + 0.003, z * r))),
        new THREE.LineBasicMaterial({ color: acc })));
      const tok = `“${showTok(data.tokens[pos])}”`;
      selGroup.add(stage.label(pos === n - 1 ? "if it stopped here, it would say:" : `${tok}, if it stopped here:`, "llabel", [width + 1.15, (L + 1) * DY, 0]));

      // Bars above the selected token: what the model expects right after it.
      const top = data.cells[L][pos].lens.slice(0, 5);
      const y0 = (L + 1) * DY + 0.25;
      top.forEach(([t, p], i) => {
        const hgt = Math.max(0.02, p * BAR);
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.26, hgt, 0.26), new THREE.MeshStandardMaterial({ color: i === 0 ? acc : ink, roughness: 0.7 }));
        const x = pos * DX + (i - (top.length - 1) / 2) * 0.42;
        bar.position.set(x, y0 + hgt / 2, 0);
        selGroup.add(bar);
        if (i < 3) selGroup.add(stage.label(h("span", {}, showTok(t), h("small", {}, pct(p))), "plabel", [x, y0 + hgt + 0.25, 0]));
      });
      caption.textContent = pos === n - 1
        ? `Bars on top: the model's output, what it expects after ${tok}.`
        : `Bars on top: what the model expects after ${tok}. The text actually goes on with “${showTok(data.tokens[pos + 1])}”.`;
      legend.replaceChildren(h("span", {}, h("span", { class: "ramp" }), "  negative · zero · positive value"), h("span", {}, `Each tile shows 64 of the vector's ${data.hidden.toLocaleString("en")} numbers, laid out in a square.`));
      renderArrows();
    }

    // One arrow per layer for the selected token: its vector as the logit lens reads it, projected onto the
    // output directions of its two final top answers (the second made perpendicular to the first; see
    // forward_pass on the server). Higher layers are darker; red where that layer's top guess is already the
    // final answer. Clicking a dot selects that layer.
    function renderArrows() {
      const [sl, pos] = selected;
      const S = 320, C = S / 2, R = C - 30;
      const pts = data.cells.map((row) => row[pos].arrow);
      const scale = Math.max(...pts.map(([x, y]) => Math.hypot(x, y))) || 1;
      const [ax, ay] = data.axes[pos].map((t) => `“${showTok(t)}”`);
      const tok = `“${showTok(data.tokens[pos])}”`;
      const ink = cssVar("--ink"), acc = cssVar("--accent"), line = cssVar("--line"), muted = cssVar("--muted");
      const font = { "font-family": "Archivo, sans-serif", "font-size": 18, "font-weight": 600 };
      const at = ([x, y]) => [C + (x / scale) * R, C - (y / scale) * R];
      const arrow = (l) => {
        const [x, y] = at(pts[l]), sel = l === sl;
        const col = data.cells[l][pos].lens[0][0] === data.axes[pos][0] ? acc : ink;
        return svgEl("g", { opacity: sel ? 1 : 0.15 + 0.6 * (l / data.layers) },
          svgEl("line", { x1: C, y1: C, x2: x, y2: y, stroke: col, "stroke-width": sel ? 3.5 : 1.5, "stroke-linecap": "round" }),
          svgEl("circle", { cx: x, cy: y, r: sel ? 6 : 4, fill: col, class: "key", "data-l": l, title: l === 0 ? "embedding" : `after layer ${l}` }));
      };
      const [sx, sy] = at(pts[sl]);
      const flip = sx > C + R / 2;
      arrowSvg.replaceChildren(
        svgEl("line", { x1: 10, y1: C, x2: S - 10, y2: C, stroke: line }),
        svgEl("line", { x1: C, y1: 10, x2: C, y2: S - 10, stroke: line }),
        svgEl("text", { x: S - 10, y: C - 8, fill: acc, "text-anchor": "end", ...font }, `→ ${ax}`),
        svgEl("text", { x: C + 8, y: 24, fill: muted, ...font }, `↑ ${ay}`),
        ...pts.map((_, l) => l).filter((l) => l !== sl).map(arrow),
        arrow(sl),
        svgEl("text", { x: sx + (flip ? -10 : 10), y: sy + (sy > C ? 18 : -10), fill: ink, "text-anchor": flip ? "end" : "start", stroke: cssVar("--sheet"), "stroke-width": 4, "paint-order": "stroke", ...font }, sl === 0 ? "embedding" : `layer ${sl}`));
      arrowTitle.textContent = `The vector of ${tok}, seen from two directions`;
      arrowNote.textContent = `One arrow per layer: the vector of ${tok}, all ${data.hidden.toLocaleString("en")} numbers, seen from the directions of its two final top answers, ${ax} (→) and ${ay} (↑). Darker arrows come from higher layers; red where that layer's top guess is already ${ax}. Click a dot to select its layer.`;
    }

    function select(l, pos) {
      selected = [l, pos];
      renderSelection();
      renderFinal();
      renderLens();
    }

    function animate() {
      clearInterval(timer);
      floors.forEach((f) => (f.visible = false));
      let l = 0;
      timer = setInterval(() => {
        if (l < floors.length) floors[l++].visible = true;
        else clearInterval(timer);
      }, 70);
    }

    // The probabilities for the selected tile: the real output for the top tile of the last token, otherwise
    // what that vector would predict if the model stopped there.
    function renderFinal() {
      const [sl, pos] = selected;
      const L = data.layers, last = data.tokens.length - 1, c = data.cells[sl][pos];
      const tok = `“${showTok(data.tokens[pos])}”`;
      finalTitle.textContent = sl === L
        ? (pos === last ? "The output: next-token probabilities" : `The output at ${tok}: what it expects next`)
        : `If it stopped ${sl === 0 ? "at the embedding" : `after layer ${sl}`}: what ${tok} would predict`;
      const top = c.lens.slice(0, 10);
      finalBox.replaceChildren(...[
        probBars(top.map(([text, p]) => ({ text, p })), { rest: 1 - top.reduce((a, [, p]) => a + p, 0) }),
        h("p", { class: "small muted" }, `Uncertainty (entropy): ${c.entropy.toFixed(2)} bits, as uncertain as choosing among about ${Math.max(1, 2 ** c.entropy).toFixed(1)} equally likely options.`),
        sl === L && pos === last ? null : h("button", { class: "btn small", type: "button", onclick: () => select(L, last) }, "Show the real output")].filter(Boolean));
    }

    function renderLens() {
      const [, pos] = selected;
      const final = data.final[0][0];
      lensTitle.textContent = `What the vector for “${showTok(data.tokens[pos])}” would predict, layer by layer`;
      const rows = [];
      for (let l = data.layers; l >= 0; l--) {
        const lens = data.cells[l][pos].lens;
        rows.push(h("tr", { class: selected[0] === l ? "hl" : "" },
          h("td", { class: "num" }, l === 0 ? "emb" : l),
          ...lens.slice(0, 3).map(([t, p]) => h("td", { class: "tokcell", style: { color: t === final && pos === data.tokens.length - 1 ? "var(--accent)" : "", fontWeight: p > 0.25 ? 700 : 400 } }, `${showTok(t)} `, h("span", { class: "muted" }, pct(p))))));
      }
      lensBox.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "layer"), h("th", {}, "1st"), h("th", {}, "2nd"), h("th", {}, "3rd"))), h("tbody", {}, rows)));
    }

    // Hover and click on tiles.
    const ray = new THREE.Raycaster();
    const ptr = new THREE.Vector2();
    function hit(ev) {
      const r = stage.renderer.domElement.getBoundingClientRect();
      ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ptr, stage.camera);
      return ray.intersectObjects(tiles.filter((t) => t.parent.visible))[0]?.object;
    }
    stage.renderer.domElement.addEventListener("pointermove", (ev) => {
      if (!data) return;
      const o = hit(ev);
      if (!o) { tip(null); return; }
      const { l, pos } = o.userData;
      const c = data.cells[l][pos];
      tip(`<b>${l === 0 ? "Embedding" : `After layer ${l}`}</b> · token <span class="mono">${esc(showTok(data.tokens[pos]))}</span><br>
        <span class="muted">Read as a prediction:</span> ${c.lens.slice(0, 4).map(([t, p]) => `<span class="mono">${esc(showTok(t))}</span> ${pct(p)}`).join(", ")}<br>
        <span class="muted">Length of the vector: ${c.norm}</span>`, ev);
    });
    stage.renderer.domElement.addEventListener("pointerleave", () => tip(null));
    stage.renderer.domElement.addEventListener("click", (ev) => {
      const o = hit(ev);
      if (!o) return;
      select(o.userData.l, o.userData.pos);
    });
    stage.onTheme.push(() => { if (data) { build(); } });

    main.append(
      intro("Chapter 3 · One forward pass",
        "From a sentence to a probability for every possible next token",
        "This is the only thing the model does: read the text so far, and output how likely each token is to come next. Below you see the real numbers from one pass, bottom to top."),
      h("div", { class: "grid2 spaced" },
        h("div", { class: "stack" },
          box("Inside the model", { cls: "open", body: [caption, stageEl, legend, flow] }),
          arrowPanel),
        h("div", { class: "stack fill" },
          box("Prompt", { cls: "open", body: [form, msg] }),
          finalPanel,
          lensPanel)),
      notes({
        math: [
          "Each layer does two things. <b>Attention</b> lets each token's vector take in information from earlier tokens (see chapter 5). An <b>MLP</b> then transforms each vector on its own: multiply by a big table of numbers, bend the result, multiply again. Each layer <em>adds</em> its output to the vector it read, so the vector is gradually rewritten as it rises.",
          { f: "vector ← vector + attention(all vectors) ;  vector ← vector + mlp(vector)" },
          "At the top, the last vector is compared with the vector of every token in the vocabulary (a dot product: multiply and add). The scores are turned into percentages that sum to 100% by <em>softmax</em>.",
          { f: "p(token) = e^score(token) / Σ e^score(all tokens)" },
          "The column on the right uses the “logit lens”: we apply that last step early, to read what each intermediate vector would predict if the model stopped there.",
        ],
        discuss: [
          "The model outputs a distribution, not an answer. Where does the “answer” come from? (Chapter 4.)",
          "In the logit lens the right answer often appears suddenly in a middle layer. Is there a moment at which the model “knows” the capital of France?",
          "Training only ever rewards predicting the next token well. Can understanding grow out of that goal, or only an imitation of it?",
          "Billions of numbers, all visible, and still hard to interpret. Is this a better or a worse position than we are in with human brains?",
        ],
      }),
    );
    run();
    return () => { clearInterval(timer); tip(null); stage.dispose(); };
  },
};
