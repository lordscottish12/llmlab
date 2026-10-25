import { h, intro, notes, box, presets, errorLine, esc, busy, cssVar, divergingColor, probBars, showTok, tip, pct } from "../ui.js";
import { post } from "../api.js";
import { createStage } from "../scene.js";

const PROMPTS = [
  { name: "Capital", text: "The capital of France is" },
  { name: "Eiffel Tower", text: "The Eiffel Tower is located in the city of" },
  { name: "Socrates", text: "All men are mortal. Socrates is a man. Therefore, Socrates is" },
  { name: "Hamlet", text: "To be, or not to" },
  { name: "Opposites", text: "The opposite of hot is" },
  { name: "Arithmetic", text: "12 + 7 =" },
  { name: "Open-ended", text: "My favourite philosopher is" },
];

const DX = 1.15; // distance between token columns
const DY = 0.32; // distance between layers
const TILE = 0.82;

export default {
  id: "forward",
  title: "One prediction",
  mount(main) {
    let data = null;
    let selected = null; // [layer, pos]
    let timer = null;

    const stageEl = h("div", { class: "stage", style: { height: "clamp(30rem, 72vh, 52rem)" } },
      h("span", { class: "hint" }, "Drag to turn · scroll to zoom · hover a tile to read it as a prediction"));
    const stage = createStage(stageEl, { camera: [6, 7, 12], target: [3, 5, 0], fov: 40 });
    const { THREE } = stage;
    const world = new THREE.Group();
    stage.scene.add(world);
    let tiles = []; // meshes with userData {l, pos}
    let floors = [];

    const input = h("input", { type: "text", value: PROMPTS[0].text, "aria-label": "Prompt" });
    const runBtn = h("button", { class: "btn primary", type: "submit" }, "Run the model");
    const form = h("form", { class: "row", style: { flexWrap: "nowrap" }, onsubmit: (e) => { e.preventDefault(); run(); } }, input, runBtn);
    const msg = h("div");
    const finalBox = h("div", {}, h("p", { class: "small muted" }, "Run the model to see its prediction."));
    const lensBox = h("div", { class: "scroll", style: { maxHeight: "26rem" } });
    const lensTitle = h("h3", {}, "How the guess forms, layer by layer");
    const flow = h("p", { class: "small muted" });

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
        // What this layer would predict for the last token if the model stopped here.
        const top = data.cells[l][n - 1].lens[0];
        const lab = stage.label(`${showTok(top[0])}  ${pct(top[1])}`, "llabel", [width + 1.15, 0, 0]);
        lab.element.style.color = top[1] > 0.25 ? cssVar("--accent") : muted;
        lab.element.style.fontWeight = top[1] > 0.25 ? "700" : "500";
        floor.add(lab);
        world.add(floor);
        floors.push(floor);
      }

      // Token labels under the embedding floor.
      data.tokens.forEach((t, p) => world.add(stage.label(showTok(t), "tlabel", [p * DX, -0.35, TILE / 2 + 0.1])));
      world.add(stage.label("if it stopped here, it would say:", "llabel", [width + 1.15, (L + 1) * DY, 0]));

      // Output bars above the last token.
      const top = data.final.slice(0, 6);
      const barMat = (i) => new THREE.MeshStandardMaterial({ color: i === 0 ? cssVar("--accent") : ink, roughness: 0.7 });
      top.forEach(([t, p], i) => {
        const hgt = Math.max(0.02, p * 3.2);
        const bar = new THREE.Mesh(new THREE.BoxGeometry(0.26, hgt, 0.26), barMat(i));
        const x = (n - 1) * DX + (i - (top.length - 1) / 2) * 0.42;
        bar.position.set(x, (L + 1) * DY + hgt / 2 + 0.25, 0);
        world.add(bar);
        if (i < 3) world.add(stage.label(h("span", {}, showTok(t), h("small", {}, pct(p))), "plabel", [x, (L + 1) * DY + hgt + 0.5, 0]));
      });

      // Fit the whole tower, from the token labels to the output bars, into view.
      const yTop = (L + 1) * DY + 1.6, yBottom = -0.6;
      const mid = new THREE.Vector3(width / 2 + 0.6, (yTop + yBottom) / 2, 0);
      stage.controls.target.copy(mid);
      const vfov = (stage.camera.fov * Math.PI) / 180 / 2;
      const hfov = Math.atan(Math.tan(vfov) * stage.camera.aspect);
      const dist = 1.15 * Math.max((yTop - yBottom) / 2 / Math.tan(vfov), (width + 4) / 2 / Math.tan(hfov));
      stage.camera.position.set(mid.x + dist * 0.28, mid.y + dist * 0.18, dist * 0.95);
      flow.textContent = `${data.tokens.length} tokens enter at the bottom. Each becomes a vector of ${data.hidden.toLocaleString()} numbers (each tile shows 64 of them). Each of the ${data.layers} layers reads all the vectors and adds a little to each. At the top, the last vector is turned into a probability for every token in the vocabulary.`;
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

    function renderFinal() {
      finalBox.replaceChildren(
        probBars(data.final.slice(0, 10).map(([text, p]) => ({ text, p })), { rest: 1 - data.final.slice(0, 10).reduce((a, [, p]) => a + p, 0) }),
        h("p", { class: "small muted" }, `Uncertainty (entropy): ${data.entropy.toFixed(2)} bits, as uncertain as choosing among about ${Math.max(1, 2 ** data.entropy).toFixed(1)} equally likely options.`));
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
      selected = [o.userData.l, o.userData.pos];
      renderLens();
    });
    stage.onTheme.push(() => { if (data) { build(); } });

    // ------------------------------------------------------- guessing game
    const guessIn = h("input", { type: "text", placeholder: "the class's guesses, separated by commas", "aria-label": "Guesses" });
    const guessBtn = h("button", { class: "btn", type: "submit" }, "Score");
    const guessOut = h("div", { class: "scroll" });
    const guessForm = h("form", { class: "row", style: { flexWrap: "nowrap" }, onsubmit: async (e) => {
      e.preventDefault();
      const guesses = guessIn.value.split(",").map((s) => s.trim()).filter(Boolean);
      if (!guesses.length) return;
      busy(guessBtn, true);
      try {
        const r = await post("/api/guess", { prefix: input.value, guesses });
        r.guesses.sort((a, b) => b.logprob - a.logprob);
        guessOut.replaceChildren(h("table", {},
          h("thead", {}, h("tr", {}, h("th", {}, "guess"), h("th", { class: "num" }, "probability"), h("th", { class: "num" }, "surprise"), h("th", { class: "num" }, "rank"))),
          h("tbody", {}, r.guesses.map((g) => h("tr", {},
            h("td", {}, g.guess, g.tokens.length > 1 ? h("span", { class: "small muted" }, ` (${g.tokens.length} tokens)`) : null),
            h("td", { class: "num" }, pct(g.p)),
            h("td", { class: "num" }, `${g.bits.toFixed(1)} bits`),
            h("td", { class: "num" }, g.tokens[0].rank.toLocaleString()))))));
      } catch (er) { errorLine(guessOut, er); }
      busy(guessBtn, false);
    } }, guessIn, guessBtn);

    main.append(
      intro("Step 2 · One forward pass",
        "From a sentence to a probability for every possible next token",
        "This is the only thing the model does: read the text so far, and output how likely each token is to come next. Below you see the real numbers from one pass, bottom to top."),
      h("div", { class: "grid2" },
        box("Inside the model", { body: [stageEl, h("div", { class: "legend" }, h("span", {}, h("span", { class: "ramp" }), "  negative · zero · positive value"), flow)] }),
        h("div", { class: "stack" },
          box("Prompt", { body: [presets(PROMPTS, (p) => { input.value = p.text; run(); }, { active: PROMPTS[0] }), form, msg] }),
          box("The output: next-token probabilities", { body: [finalBox] }),
          box(null, { body: [lensTitle, h("p", { class: "small muted" }, "Click any tile in the picture to read that vector instead."), lensBox] }))),
      box("Game: guess the next word before the model does", {
        body: [h("p", { class: "muted" }, "Collect guesses from the room, then see how likely the model thought each one was. “Surprise” is how the model is scored during training: the less likely it thought the right answer, the more bits of surprise, and the more its numbers get adjusted."), guessForm, guessOut],
      }),
      notes({
        math: [
          "Each layer does two things. <b>Attention</b> lets each token's vector take in information from earlier tokens (see step 4). An <b>MLP</b> then transforms each vector on its own: multiply by a big table of numbers, bend the result, multiply again. Each layer <em>adds</em> its output to the vector it read, so the vector is gradually rewritten as it rises.",
          { f: "vector ← vector + attention(all vectors) ;  vector ← vector + mlp(vector)" },
          "At the top, the last vector is compared with the vector of every token in the vocabulary (a dot product: multiply and add). The scores are turned into percentages that sum to 100% by <em>softmax</em>.",
          { f: "p(token) = e^score(token) / Σ e^score(all tokens)" },
          "The column on the right uses the “logit lens”: we apply that last step early, to read what each intermediate vector would predict if the model stopped there.",
        ],
        discuss: [
          "The model outputs a distribution, not an answer. Where does the “answer” come from? (Step 3.)",
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
