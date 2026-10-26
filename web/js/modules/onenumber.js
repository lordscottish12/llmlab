import { h, intro, notes, box, presets, errorLine, slider } from "../ui.js";
import { get, stream } from "../api.js";

const STORIES = [
  { name: "Story", text: "Once upon a time, in a small village by the sea, there lived" },
  { name: "Recipe", text: "To make a good cup of tea, first" },
  { name: "Philosophy", text: "The question of free will has puzzled philosophers because" },
];

const STARTS = [
  { name: "Holiday", text: "Last summer, on my holiday, I" },
  { name: "Weather", text: "This morning the weather was" },
  { name: "Dinner", text: "For dinner yesterday we cooked" },
  { name: "Auf Deutsch", text: "Letzten Sommer waren wir in den Ferien und" },
];

export default {
  id: "onenumber",
  title: "One number",
  mount(main) {
    let info = null, language = "zh", strength = 2.5;
    const ctrls = [];
    const msg = h("div");

    // Write the same text twice from the same start, once as it is and once changed. Greedy (temperature 0)
    // gives the same text every time; otherwise both get the same dice, and a new click gives a new pair.
    function pair(prompt, outA, outB, change, btn, label, temperature = 0) {
      return async () => {
        if (ctrls.length) { ctrls.splice(0).forEach((c) => c.abort()); return; }
        btn.textContent = "Stop";
        const seed = Math.floor(Math.random() * 1e6);
        const one = async (el, neurons) => {
          const c = new AbortController();
          ctrls.push(c);
          el.textContent = prompt.value;
          el.classList.add("cursor");
          await stream({ prompt: prompt.value, role: "base", max_new_tokens: 60, temperature, seed, neurons }, (ev) => { if (ev.text) el.textContent += ev.text; }, c.signal);
          el.classList.remove("cursor");
        };
        try { await Promise.all([one(outA, null), one(outB, change())]); } catch (e) { errorLine(outB, e); }
        outA.classList.remove("cursor"); outB.classList.remove("cursor");
        ctrls.length = 0;
        btn.textContent = label;
      };
    }

    // ---------------------------------------------------------- 1 · the super weight
    const swFacts = h("p", {});
    const swPrompt = h("input", { type: "text", value: STORIES[0].text, "aria-label": "Start of a text" });
    const swA = h("div", { class: "answer", style: { minHeight: "6rem" } });
    const swB = h("div", { class: "answer", style: { minHeight: "6rem" } });
    const swBtn = h("button", { class: "btn primary", type: "button" }, "Write both");
    swBtn.addEventListener("click", pair(swPrompt, swA, swB, () => ({ zero_super: true }), swBtn, "Write both"));

    // ---------------------------------------------------------- 2 · language neurons
    const langSeg = h("div", { class: "seg", role: "group", "aria-label": "Language" });
    const langFacts = h("p", { class: "small muted" });
    const muteBox = h("input", { type: "checkbox", checked: true });
    const strengthS = slider("Strength", { min: 0, max: 4, step: 0.25, value: strength, fmt: (v) => `×${v.toFixed(2)}`, oninput: (v) => (strength = v) });
    const lnPrompt = h("input", { type: "text", value: STARTS[0].text, "aria-label": "Start of a text" });
    const lnA = h("div", { class: "answer", style: { minHeight: "6rem" } });
    const lnB = h("div", { class: "answer", style: { minHeight: "6rem" } });
    const lnBtn = h("button", { class: "btn primary", type: "button" }, "Write both");
    // Swiss German shares many neurons with German; pushing only its own ones adds "dialect" without pushing German.
    lnBtn.addEventListener("click", pair(lnPrompt, lnA, lnB, () => ({ language, strength, mute: muteBox.checked ? "en" : null, exclude: language === "gsw" ? "de" : null }), lnBtn, "Write both", 0.7));
    const lnTitle = h("span", {}, "Pushed");

    function showLanguage() {
      const d = info.languages[language];
      langSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.c === language)));
      langFacts.textContent = `${d.name}: ${d.count} of the model's ${info.neurons_total.toLocaleString("en")} neurons (${((100 * d.count) / info.neurons_total).toFixed(2)}%) fire mostly on ${d.name} text.` +
        (language === "gsw" ? " Only those it does not share with German are pushed. This model cannot write Swiss German, so watch where it drifts instead: from German, often to Dutch; from English, sometimes to Scots-like dialect at higher strength. Each click is different." : "");
      lnTitle.textContent = `With the ${d.name} neurons pushed`;
    }

    get("/api/neurons").then((r) => {
      info = r;
      if (!r.available) {
        msg.replaceChildren(h("p", { class: "small" }, `Nothing found for ${r.model} yet: run scripts/find_neurons.py.`));
        return;
      }
      const sw = r.super_weight;
      swFacts.replaceChildren(
        "In layer ", h("b", {}, String(sw.layer + 1)), ", the MLP's output table has a weight at row ", h("b", {}, sw.row.toLocaleString("en")),
        ", column ", h("b", {}, sw.col.toLocaleString("en")), " of ", h("b", {}, sw.value.toFixed(3)),
        `, about ${Math.round(Math.abs(sw.value) / sw.typical)} times a typical weight there. It is one of the model's ${(r.params / 1e9).toFixed(1)} billion numbers. On the first token it produces a huge value (around ${Math.round(sw.spike).toLocaleString("en")}): the giant first vector of chapter 3.`);
      langSeg.replaceChildren(...Object.entries(r.languages).filter(([c]) => c !== "en").map(([c, d]) => h("button", { type: "button", "data-c": c }, d.name)));
      showLanguage();
    }).catch((e) => errorLine(msg, e));
    langSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || !info) return;
      language = b.dataset.c;
      showLanguage();
    });

    main.append(
      intro("Chapter 11 · One number",
        "How much can a single number do?",
        "A model is billions of numbers, and most of them matter only a little. A few matter enormously. Set one weight to zero and the model loses its way; push a few hundred of its neurons and it starts writing in another language."),
      msg,
      box("1 · The super weight", {
        right: presets(STORIES, (p) => { swPrompt.value = p.text; }, { active: STORIES[0] }),
        body: [swFacts,
          h("div", { class: "row", style: { flexWrap: "nowrap" } }, swPrompt, swBtn),
          h("div", { class: "grid2 even" }, box("As it is", { body: [swA] }), box("With this one weight set to 0", { body: [swB] }))],
      }),
      box("2 · Neurons for a language", {
        cls: "gap-above",
        right: presets(STARTS, (p) => { lnPrompt.value = p.text; }, { active: STARTS[0] }),
        body: [
          h("p", { class: "small muted" }, "A neuron here is one of the numbers inside a layer's MLP. Some fire mostly on one language. Push them to (a multiple of) their usual level, and the model writes as if the text were in that language. Each click rolls new dice, the same for both texts; too much strength makes it stammer."),
          h("div", { class: "row" }, langSeg, h("label", { class: "check" }, muteBox, "switch off the English neurons")),
          langFacts,
          strengthS,
          h("div", { class: "row", style: { flexWrap: "nowrap" } }, lnPrompt, lnBtn),
          h("div", { class: "grid2 even" }, box("As it is", { body: [lnA] }), box(lnTitle, { body: [lnB] }))],
      }),
      notes({
        math: [
          "<b>The super weight</b> (Yu et al., 2024). The MLP's last step multiplies by a table of weights. One weight in an early layer meets one neuron that, on the first token, becomes enormous; their product is the giant value that later layers rely on to “park” attention. Set that weight to zero and the giant value disappears.",
          "It is found with a single run of the model: look for the layer where one MLP neuron spikes, and take the weight that connects it to the spiking output. Setting a random weight to zero instead changes nothing measurable.",
          "<b>Language neurons</b> (Tang et al., 2024). Count, for every neuron, how often it fires on text in each of eight languages. Keep those that fire often, but almost only for one language. Pushing them to a multiple of their usual level, and switching off the English ones, tips the model into that language.",
          { f: "neuron ← strength × (its typical level for the language)" },
        ],
        discuss: [
          "If one number out of 1.7 billion can break the model, what does that say about how its knowledge is stored: spread out, or concentrated?",
          "The super weight's job seems to be producing a value that means nothing in itself, a place to park attention. Is that a “thought”, or plumbing?",
          "Chinese, Japanese and Russian switch easily; French, German and Spanish less so. Why might languages that share an alphabet with English share more of its neurons?",
          "When the language neurons are pushed, the model often keeps the same story, only in another language. What does that suggest about where the meaning lives?",
        ],
      }),
    );
    return () => ctrls.forEach((c) => c.abort());
  },
};
