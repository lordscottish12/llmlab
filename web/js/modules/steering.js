import { h, intro, notes, box, presets, errorLine, slider, busy } from "../ui.js";
import { post, stream, info } from "../api.js";
import { LARGE } from "./steering-sets.js";

const NEUTRAL = [
  "The meeting has been moved to Thursday afternoon.",
  "Please remember to bring your own laptop.",
  "The train leaves from platform four at nine.",
  "He wrote the numbers down in a small notebook.",
  "The report contains three tables and a summary.",
  "She bought bread, cheese and a bottle of water.",
];

const CONCEPTS = [
  {
    name: "Love ↔ hate",
    positive: ["I love you so much.", "Love is the most wonderful feeling.", "I adore my friends; they fill my heart with warmth.", "What a beautiful, joyful, loving day.", "I cherish every moment we spend together."],
    negative: ["I hate you so much.", "Hate is the most terrible feeling.", "I despise my enemies; they fill my heart with rage.", "What an ugly, miserable, hateful day.", "I resent every moment we spend together."],
    tests: ["Thank you for the lovely flowers!", "Get out of my house and never come back.", "The train leaves at nine.", "I can't wait to see you again.", "This is the worst meal I have ever had."],
  },
  {
    name: "The ocean",
    positive: ["The waves crashed against the rocks of the shore.", "We sailed across the deep blue sea.", "Whales and dolphins swim in the open ocean.", "The tide came in over the sandy beach.", "Salt spray and seagulls filled the harbour air."],
    negative: NEUTRAL.slice(0, 5),
    tests: ["The ship left port at dawn.", "I need to file my tax return.", "Fish swam between the coral.", "The desert was dry and hot.", "He bought a new pair of shoes."],
  },
  {
    name: "Sad ↔ happy",
    positive: ["I feel so sad and alone today.", "Tears ran down her face at the funeral.", "Everything seems grey and hopeless.", "He missed her terribly after she left.", "Nothing brings me joy any more."],
    negative: ["I feel so happy and loved today.", "She laughed with delight at the party.", "Everything seems bright and full of hope.", "He was thrilled when she came back.", "Everything brings me joy these days."],
    tests: ["My dog died last week.", "We won the championship!", "The meeting is at three.", "I miss my old friends.", "What a gorgeous sunny morning."],
  },
  {
    name: "Philosophy",
    positive: ["Kant argued that the moral law is given by reason itself.", "What is the nature of being, asked Heidegger.", "Descartes doubted everything except that he was thinking.", "The problem of universals divides realists and nominalists.", "Is knowledge justified true belief?"],
    negative: NEUTRAL.slice(0, 5),
    tests: ["Hume denied that we perceive causation.", "Please pass the salt.", "What is it like to be a bat?", "The bus was late again.", "Plato's forms exist outside space and time."],
  },
];

const PROMPTS = [
  { name: "Weekend", text: "Tell me about your weekend." },
  { name: "A poem", text: "Write a four-line poem about a cat." },
  { name: "Advice", text: "Give me one piece of advice for my exam tomorrow." },
  { name: "About me", text: "What do you think of me?" },
];

export default {
  id: "steering",
  title: "Steering",
  mount(main) {
    let concept = structuredClone(CONCEPTS[0]);
    let size = "small"; // which example set is loaded: the preset's few examples, or its large set of matched pairs
    let layer = 12, strength = 0.8, nLayers = 28;
    const ctrls = [];

    const posIn = h("textarea", { rows: 5, class: "sent" }, concept.positive.join("\n"));
    const negIn = h("textarea", { rows: 5, class: "sent" }, concept.negative.join("\n"));
    const promptIn = h("textarea", { rows: 2 }, PROMPTS[0].text);
    const outA = h("div", { class: "answer", style: { minHeight: "8rem" } });
    const outB = h("div", { class: "answer", style: { minHeight: "8rem" } });
    const msg = h("div");
    const testsIn = h("textarea", { rows: 5, class: "sent" }, concept.tests.join("\n"));
    const meter = h("div", { class: "meters" });
    const meterBtn = h("button", { class: "btn", type: "button" }, "Measure");

    const lines = (el) => el.value.split("\n").map((s) => s.trim()).filter(Boolean);

    // Every line is one example; the counts make that visible.
    const posCount = h("span", {}), negCount = h("span", {});
    const count = () => {
      const n = (el) => lines(el).length;
      posCount.textContent = ` · ${n(posIn)} example${n(posIn) === 1 ? "" : "s"}`;
      negCount.textContent = ` · ${n(negIn)} example${n(negIn) === 1 ? "" : "s"}`;
    };
    posIn.addEventListener("input", count);
    negIn.addEventListener("input", count);

    const sizeNote = h("p", { class: "small muted" });
    const sizeSeg = h("div", { class: "seg", role: "group", "aria-label": "Which examples" },
      h("button", { type: "button", "aria-pressed": "true", "data-s": "small" }, "Small set"),
      h("button", { type: "button", "aria-pressed": "false", "data-s": "large" }, "Large set"));
    // Fill both lists from the current preset, with its few examples or its large set of matched pairs.
    function load() {
      const large = size === "large" && LARGE[concept.name];
      posIn.value = large ? large.map((p) => p[0]).join("\n") : concept.positive.join("\n");
      negIn.value = large ? large.map((p) => p[1]).join("\n") : concept.negative.join("\n");
      sizeSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === size)));
      sizeNote.textContent = large
        ? `${large.length} matched pairs: line n of the second list is line n of the first, with the concept taken out. Averaging over many pairs leaves little but the concept itself.`
        : "A few examples, to read and to change. Try your own, or switch to the large set for a cleaner direction.";
      count();
      meter.replaceChildren();
      showScale();
    }
    sizeSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      size = b.dataset.s;
      load();
    });
    const steer = (s) => ({ positive: lines(posIn), negative: lines(negIn), layer, strength: s });

    const layerS = slider("Layer", { min: 0, max: nLayers - 1, step: 1, value: layer, fmt: (v) => `${v + 1}`, oninput: (v) => (layer = v) });
    const strengthS = slider("Strength", { min: -2, max: 2, step: 0.05, value: strength, fmt: (v) => v.toFixed(2), oninput: (v) => { strength = v; showScale(); } });
    // A scale above the slider's track: which list each end pushes towards, with the current side highlighted.
    const scaleL = h("span", {}), scaleM = h("span", {}, "neutral"), scaleR = h("span", {});
    strengthS.prepend(h("div", { class: "scale" }, scaleL, scaleM, scaleR));
    strengthS.classList.add("scaled");
    function showScale() {
      const [a, b] = concept.name.includes("↔") ? concept.name.split(/\s*↔\s*/) : [concept.name, "away from it"];
      scaleL.textContent = `← 2nd list: ${b}`;
      scaleR.textContent = `1st list: ${a} →`;
      scaleL.classList.toggle("on", strength < 0);
      scaleM.classList.toggle("on", strength === 0);
      scaleR.classList.toggle("on", strength > 0);
    }

    info().then((d) => {
      const m = d.models.chat;
      if (m.layers) {
        nLayers = m.layers;
        layer = Math.round(nLayers * 0.4);
        layerS.input.max = nLayers - 1;
        layerS.set(layer);
      }
    }).catch(() => {});

    const genBtn = h("button", { class: "btn primary", type: "button" }, "Generate both");
    async function generate() {
      if (ctrls.length) { ctrls.splice(0).forEach((c) => c.abort()); return; }
      genBtn.textContent = "Stop";
      try {
        const prompt = (await post("/api/template", { messages: [{ role: "user", content: promptIn.value }], enable_thinking: false })).text;
        const seed = Math.floor(Math.random() * 1e6);
        const one = async (el, s) => {
          const c = new AbortController();
          ctrls.push(c);
          el.textContent = "";
          el.classList.add("cursor");
          await stream({ prompt, role: "chat", max_new_tokens: 160, temperature: 0.7, seed, steer: s ? steer(s) : null }, (ev) => { if (ev.text) el.textContent += ev.text; }, c.signal);
          el.classList.remove("cursor");
        };
        await Promise.all([one(outA, 0), one(outB, strength)]);
        msg.replaceChildren();
      } catch (e) { errorLine(msg, e); }
      outA.classList.remove("cursor"); outB.classList.remove("cursor");
      ctrls.length = 0;
      genBtn.textContent = "Generate both";
    }
    genBtn.addEventListener("click", generate);

    async function measure() {
      busy(meterBtn, true, "Measuring…");
      try {
        const texts = lines(testsIn);
        const r = await post("/api/steer/project", { positive: lines(posIn), negative: lines(negIn), layer, texts });
        meter.replaceChildren(...texts.map((t, i) => {
          const s = Math.max(-1.5, Math.min(1.5, r.scores[i]));
          const left = s < 0 ? 50 + (s / 1.5) * 50 : 50, width = Math.abs(s / 1.5) * 50;
          // One line per sentence: the sentence, its bar, its score.
          return h("div", { class: "hbar" },
            h("span", { class: "sent" }, t),
            h("div", { class: "meter" }, h("span", { class: "mid" }), h("span", { class: "fill", style: { left: `${left}%`, width: `${width}%`, background: s >= 0 ? "var(--pos)" : "var(--neg)" } })),
            h("span", { class: "mono small", style: { textAlign: "right" } }, r.scores[i].toFixed(2)));
        }));
      } catch (e) { errorLine(meter, e); }
      busy(meterBtn, false);
    }
    meterBtn.addEventListener("click", measure);

    main.append(
      intro("Chapter 8 · Steering",
        "Finding a concept inside the model, and turning it up",
        "Feed the model texts that differ in one respect, such as loving versus hateful sentences, and compare its internal vectors. The average difference is a direction in its space. Adding that direction while the model writes pushes everything it says that way, whatever it was asked."),
      h("div", { class: "grid2 even" },
        box("1 · Define the concept by examples", {
          right: presets(CONCEPTS, (p) => { concept = structuredClone(p); testsIn.value = p.tests.join("\n"); load(); }, { active: CONCEPTS[0] }),
          body: [
            h("label", { class: "field" }, h("span", {}, "Texts that have it, one example per line", posCount), posIn),
            h("label", { class: "field" }, h("span", {}, "Texts that lack it (or have the opposite)", negCount), negIn),
            sizeSeg,
            sizeNote,
            layerS,
          ],
        }),
        box("2 · Read it: the model's own concept meter", {
          right: meterBtn,
          body: [
            h("p", { class: "small muted" }, "How far each sentence's vector lies along the direction. +1: like the first list, −1: like the second list."),
            h("label", { class: "field" }, h("span", {}, "Test sentences"), testsIn),
            meter,
          ],
        })),
      box("3 · Write with it: add the direction while the model answers", {
        cls: "gap-above",
        right: presets(PROMPTS, (p) => { promptIn.value = p.text; }, { active: PROMPTS[0] }),
        body: [
          h("div", { class: "grid2 even", style: { gap: "1rem" } }, h("label", { class: "field" }, h("span", {}, "Ask the chat model"), promptIn), h("div", { style: { display: "grid", gap: ".6rem", alignContent: "end" } }, strengthS, h("div", { class: "row" }, genBtn))),
          msg,
          h("div", { class: "grid2 even" },
            box("Normal", { body: [outA] }),
            box("Steered", { body: [outB] })),
        ],
      }),
      notes({
        math: [
          "Take the vector after one layer for every token of every example, and average them: one average for the first list, one for the second. Their difference is the <em>steering vector</em>.",
          { f: "direction = mean(vectors | first list) − mean(vectors | second list)" },
          "With five examples a side, the direction also picks up whatever else those few sentences happen to share: certain words, a length, a topic. The large sets use about 100 matched pairs, sentences that differ only in the concept, so those accidents average out.",
          "The concept meter projects a new sentence's average vector onto that direction (a dot product), rescaled so the two lists sit at +1 and −1.",
          "To steer, we add the direction, times the strength, to the vector of every token at that layer, at every step of writing. Strength is measured against the typical length of the vectors there: 1.0 adds an arrow as long as the vector itself. Too much, and the text falls apart.",
          { f: "vector_layer ← vector_layer + strength × direction" },
        ],
        discuss: [
          "The model was never taught the word “love” as a concept. Yet one direction among thousands seems to stand for it. Does the model <em>have</em> the concept?",
          "Steering changes what the model says without changing its input. Is this closer to persuading someone or to brain stimulation? What follows for responsibility?",
          "Try negative strength. Is “hate” simply “minus love” in this space? Is it in ours?",
          "The meter is a crude lie detector of sorts. What would it take for a reading like this to tell us what a model “really” believes?",
        ],
      }),
    );
    load();
    return () => ctrls.forEach((c) => c.abort());
  },
};
