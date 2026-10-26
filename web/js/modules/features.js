import { h, intro, notes, box, presets, errorLine, slider, busy, tokSpan, showTok } from "../ui.js";
import { post, get, stream } from "../api.js";

const TEXTS = [
  { name: "Paris", text: "The Eiffel Tower is in Paris, the capital of France." },
  { name: "Kant", text: "Kant argued that the moral law is given by reason itself." },
  { name: "Grief", text: "She felt a deep sadness after her grandmother's funeral." },
  { name: "Code", text: "def add(a, b):\n    return a + b" },
  { name: "Numbers", text: "In 1789 the French Revolution began, and by 1799 it was over." },
];

const PROMPTS = [
  { name: "My holiday", text: "Last summer, on my holiday, I" },
  { name: "A recipe", text: "To make a good soup, you need" },
  { name: "The news", text: "Breaking news this morning:" },
];

export default {
  id: "features",
  title: "Features",
  mount(main) {
    let layers = [], layer = null, data = null, pos = null, feature = null;
    let strength = 0.6;
    const ctrls = [];

    const textIn = h("textarea", { rows: 2, "aria-label": "Text to take apart" }, TEXTS[0].text);
    const runBtn = h("button", { class: "btn primary", type: "button" }, "Take it apart");
    const layerSeg = h("div", { class: "seg", role: "group", "aria-label": "Layer" });
    const msg = h("div");
    const chips = h("div", { class: "tokens" });
    const featList = h("div", { class: "featlist" });
    const featTitle = h("h3", {}, "Active features");
    const detail = h("div", { class: "featdetail" }, h("p", { class: "small muted" }, "Click a feature to see where it fires."));

    // ---------------------------------------------------------- reading a text
    async function run() {
      if (layer == null) return;
      busy(runBtn, true, "Reading…");
      try {
        data = await post("/api/sae/encode", { text: textIn.value, layer });
        msg.replaceChildren();
        // Start on the token with the most clearly active feature, skipping the first token (see notes).
        pos = data.tokens.length > 1 ? 1 + data.acts.slice(1).reduce((best, a, i, all) => ((a[0]?.[1] ?? 0) > (all[best][0]?.[1] ?? 0) ? i : best), 0) : 0;
        feature = null;
        render();
      } catch (e) { errorLine(msg, e); }
      busy(runBtn, false);
    }

    // The tokens as chips. With a feature selected, each chip is shaded by how strongly that feature fires on it.
    function renderChips() {
      const of = (p) => (feature == null ? 0 : (data.acts[p].find(([f]) => f === feature)?.[1] ?? 0));
      const max = Math.max(1e-9, ...data.tokens.map((_, p) => of(p)));
      chips.replaceChildren(...data.tokens.map((t, p) => {
        const a = of(p) / max;
        const el = tokSpan(t, { class: `tok${p === pos ? " sel" : ""}`, title: feature == null ? "Click to see this token's features" : `feature #${feature}: ${of(p).toFixed(1)}` });
        if (feature != null) el.style.background = a > 0 ? `color-mix(in srgb, var(--accent) ${Math.round(15 + 60 * a)}%, var(--sheet))` : "var(--wash)";
        el.style.cursor = "pointer";
        el.addEventListener("click", () => { pos = p; render(); });
        return el;
      }));
    }

    const label = (f) => {
      const l = data.labels[f];
      return l?.tokens?.length ? l.tokens.map((t) => `“${t}”`).join(" · ") : "(rarely fires on web text)";
    };

    function renderList() {
      featTitle.textContent = `Active features at “${showTok(data.tokens[pos])}”`;
      const acts = data.acts[pos].slice(0, 12);
      const max = acts[0]?.[1] || 1;
      featList.replaceChildren(
        h("p", { class: "small muted" }, `${data.acts[pos].length} of 32,768 features are active here; the strongest 12, with the tokens each one fires on most in web text:`),
        ...acts.map(([f, a]) => h("button", { class: `prob clickable feat${f === feature ? " hl" : ""}`, type: "button", onclick: () => pick(f) },
          h("span", { class: "word" }, `#${f}`),
          h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${(a / max) * 100}%` } })),
          h("span", { class: "pct" }, a.toFixed(1)),
          h("span", { class: "flabel" }, label(f)))));
    }

    function render() {
      renderChips();
      renderList();
    }

    // ---------------------------------------------------------- one feature
    async function pick(f) {
      feature = feature === f ? null : f;
      render();
      if (feature == null) { detail.replaceChildren(h("p", { class: "small muted" }, "Click a feature to see where it fires.")); return; }
      try {
        const d = await post("/api/sae/feature", { layer, feature: f });
        detail.replaceChildren(
          h("p", {}, h("b", {}, `Feature #${f}`), ` · fires on ${(100 * d.freq).toFixed(d.freq < 0.001 ? 3 : 2)}% of tokens in web text · most often on ${d.tokens.length ? d.tokens.map((t) => `“${t}”`).join(", ") : "nothing in our sample"}`),
          h("span", { class: "label" }, "Where it fires most strongly in web text"),
          d.examples.length
            ? h("ol", { class: "snips" }, ...d.examples.map((e) => h("li", {}, h("span", { class: "muted" }, e.pre), h("mark", {}, e.tok), h("span", { class: "muted" }, e.post), h("span", { class: "mono small" }, `  ${e.act.toFixed(1)}`))))
            : h("p", { class: "small muted" }, "It never fired in our sample of web text."),
          steerBox);
      } catch (e) { errorLine(detail, e); }
    }

    // ---------------------------------------------------------- turning a feature up
    const promptIn = h("input", { type: "text", value: PROMPTS[0].text, "aria-label": "Start of a text" });
    const outA = h("div", { class: "answer", style: { minHeight: "5rem" } });
    const outB = h("div", { class: "answer", style: { minHeight: "5rem" } });
    const genBtn = h("button", { class: "btn primary", type: "button" }, "Write both");
    const strengthS = slider("Strength", { min: 0, max: 1.5, step: 0.05, value: strength, fmt: (v) => v.toFixed(2), oninput: (v) => (strength = v) });
    async function generate() {
      if (ctrls.length) { ctrls.splice(0).forEach((c) => c.abort()); return; }
      genBtn.textContent = "Stop";
      const seed = Math.floor(Math.random() * 1e6);
      const one = async (el, s) => {
        const c = new AbortController();
        ctrls.push(c);
        el.textContent = promptIn.value;
        el.classList.add("cursor");
        await stream({ prompt: promptIn.value, role: "base", max_new_tokens: 60, temperature: 0.7, seed, feature: s ? { layer, index: feature, strength: s } : null }, (ev) => { if (ev.text) el.textContent += ev.text; }, c.signal);
        el.classList.remove("cursor");
      };
      try { await Promise.all([one(outA, 0), one(outB, strength)]); } catch (e) { errorLine(outB, e); }
      outA.classList.remove("cursor"); outB.classList.remove("cursor");
      ctrls.length = 0;
      genBtn.textContent = "Write both";
    }
    genBtn.addEventListener("click", generate);
    const steerBox = h("div", { class: "stack" },
      box("Turn it up while the model writes", {
        right: presets(PROMPTS, (p) => { promptIn.value = p.text; }, { active: PROMPTS[0] }),
        body: [
          h("p", { class: "small muted" }, "The base model continues the text twice, with the same dice: once as it is, once with this feature's direction added at every step, as in chapter 8."),
          h("div", { class: "row", style: { flexWrap: "nowrap" } }, promptIn, genBtn),
          strengthS,
          h("div", { class: "grid2 even" }, box("Normal", { body: [outA] }), box("With the feature turned up", { body: [outB] })),
        ],
      }));

    // ---------------------------------------------------------- layers
    get("/api/sae").then((r) => {
      layers = r.layers;
      if (!layers.length) {
        msg.replaceChildren(h("p", { class: "small" }, `No SAEs are prepared for ${r.model}. Qwen's SAEs exist for Qwen3-1.7B-Base, the laptop profile's base model: run scripts/prepare_sae.py.`));
        return;
      }
      layer = layers[Math.floor(layers.length / 2)];
      layerSeg.replaceChildren(...layers.map((l) => h("button", { type: "button", "aria-pressed": String(l === layer), "data-l": l }, `Layer ${l + 1}`)));
      run();
    }).catch((e) => errorLine(msg, e));
    layerSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      layer = +b.dataset.l;
      layerSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      detail.replaceChildren(h("p", { class: "small muted" }, "Click a feature to see where it fires."));
      run();
    });

    const listPanel = box("", { body: [featList] });
    listPanel.querySelector("h3").replaceWith(featTitle);

    main.append(
      intro("Chapter 10 · Features",
        "Taking a vector apart",
        "A vector of 2,048 numbers mixes many things at once. A sparse autoencoder rewrites it as a few active “features” out of 32,768, each a direction that tends to stand for one thing. Qwen trained these for its 1.7B base model; we describe each feature by the web text that switches it on most."),
      box("Text", {
        right: presets(TEXTS, (p) => { textIn.value = p.text; run(); }, { active: TEXTS[0] }),
        body: [textIn, h("div", { class: "row" }, runBtn, layerSeg), msg],
      }),
      h("div", { class: "grid2 even" },
        box("The text, token by token", { body: [h("p", { class: "small muted" }, "Click a token to see its features. With a feature selected, darker means it fires more strongly there."), chips, detail] }),
        listPanel),
      notes({
        math: [
          "The autoencoder has two tables of 32,768 directions. To read a vector, it measures how well each direction fits (a dot product), keeps only the 50 best, and sets all others to zero. Those 50 numbers, put back together with the second table, rebuild the vector:",
          { f: "features = top50( W_enc · h + b_enc )        h ≈ W_dec · features + b_dec" },
          "It is trained to rebuild the model's vectors as well as possible from as few features as possible. The pressure to be sparse is what tends to make each feature mean one thing.",
          "Nobody tells a feature what it means. We label each one by its evidence: the tokens and passages, from about 120,000 tokens of web text, where it fires most strongly. Many features are clear; some are puzzling; a few are both.",
          "The first token of every text carries huge, text-independent values (chapter 3), so its features are not very telling.",
        ],
        discuss: [
          "A feature that fires on “Paris”, “Eiffel” and “croissant”: is that the concept France, or a word-association habit? What would decide it?",
          "Some features are hard to name, even with their examples in front of us. Is that a failure of the method, or a sign that the model's concepts are not ours?",
          "Turning a feature up changes what the model writes. Does that show the feature is what the model “uses”, or only that adding a direction pushes the text somewhere?",
          "We understand the model better through a second model (the autoencoder) that we trained to imitate it. How far can such an explanation be trusted?",
        ],
      }),
    );
    return () => ctrls.forEach((c) => c.abort());
  },
};
