import { h, intro, notes, box, presets, errorLine, probBars, slider, showTok, pct, busy } from "../ui.js";
import { post } from "../api.js";

const PROMPTS = [
  { name: "Story", text: "Once upon a time, in a small village by the sea, there lived" },
  { name: "Definition", text: "Philosophy is" },
  { name: "Argument", text: "The strongest argument against free will is" },
  { name: "Recipe", text: "To make a good cup of tea, first" },
  { name: "Dialogue", text: "Student: Can a machine think?\nTeacher:" },
];

const END = "<|endoftext|>";

export default {
  id: "generate",
  title: "Writing text",
  mount(main) {
    let prompt = PROMPTS[0].text;
    let gen = []; // [{text, candidates, forced}]
    let current = null; // candidates for the next position
    let temperature = 1.0;
    let mode = "weighted"; // how the next token is chosen: "uniform", "weighted" or "top"
    let selected = null; // index into gen whose alternatives are shown
    let running = false;

    const doc = h("div", { class: "doc", "aria-live": "polite" });
    const promptIn = h("textarea", { rows: 2 }, prompt);
    const nextBox = h("div");
    const nextPanel = box("Next token", { body: [nextBox] });
    const nextTitle = nextPanel.querySelector("h3");
    const msg = h("div");
    const count = h("span", { class: "small muted" });

    const text = (upto = gen.length) => prompt + gen.slice(0, upto).map((g) => g.text).join("");

    async function fetchNext() {
      const r = await post("/api/next", { text: text(), temperature: mode === "weighted" ? temperature : 1, top: 15 });
      current = r;
      return r;
    }

    function renderDoc() {
      doc.replaceChildren(h("span", { class: "given" }, prompt),
        ...gen.map((g, i) => h("span", {
          class: `gtok${selected === i ? " sel" : ""}${g.forced ? " forced" : ""}${g.rank ? " blind" : ""}`,
          title: g.rank
            ? `Picked blindly. The model gave it ${pct(g.p)}: number ${g.rank.toLocaleString("en")} of ${g.of.toLocaleString("en")}. Click to see the alternatives here.`
            : `Chosen with ${pct(g.p)} probability. Click to see the alternatives here.`,
          onclick: () => { selected = selected === i ? null : i; renderDoc(); renderNext(); },
        }, g.text === END ? "⟨end⟩" : g.text)),
        h("span", { class: "cursor" }));
      if (selected == null) doc.scrollTop = doc.scrollHeight; // follow the end, unless a token is being looked at
      count.textContent = gen.length ? `${gen.length} tokens added. Underlined in yellow: picked by hand. Dotted: picked blindly, like Llull's machine.` : "No tokens added yet."; // never empty, so the line keeps its space
    }

    function renderNext() {
      if (selected != null) {
        const g = gen[selected];
        nextTitle.textContent = `Alternatives at token ${selected + 1}`;
        nextBox.replaceChildren(
          h("p", { class: "small muted" }, "What the model could have written here. Pick another option to rewrite the text from this point on."),
          probBars(g.candidates.candidates.map((c) => ({ ...c, text: c.text === END ? "⟨end of text⟩" : c.text })), {
            rest: g.candidates.rest,
            highlight: g.candidates.candidates.findIndex((c) => c.text === g.text),
            onPick: (c, k) => branch(selected, g.candidates.candidates[k]),
          }),
          h("button", { class: "btn small", type: "button", onclick: () => { selected = null; renderDoc(); renderNext(); } }, "Back to the end of the text"));
        return;
      }
      nextTitle.textContent = "Next token";
      if (!current) { nextBox.replaceChildren(); return; }
      // The bars show the chances as the selected mode sees them: Llull's machine gives every token the
      // same tiny chance, "Most likely" only ever takes the first.
      const items = current.candidates.map((c) => ({ ...c, text: c.text === END ? "⟨end of text⟩" : c.text }));
      const pool = current.uniform.pool, even = 1 / pool;
      const bars = mode === "uniform"
        ? probBars(items.map((c) => ({ ...c, p: even, label: `${(100 * even).toPrecision(2)}%` })), { rest: 1 - items.length * even, scale: 1, onPick: (c, k) => append(current.candidates[k], true) })
        : probBars(items, { rest: current.rest, dim: mode === "top" ? (k) => k > 0 : null, onPick: (c, k) => append(current.candidates[k], true) });
      nextBox.replaceChildren(bars,
        h("p", { class: "small muted" }, mode === "uniform"
          ? `Llull's machine weighs every token the same: 1 in ${pool.toLocaleString("en")}. Click a bar to choose that token yourself.`
          : `Click a bar to choose that token yourself. Entropy ${current.entropy.toFixed(2)} bits.`));
    }

    function sample(cands) {
      const total = cands.reduce((a, c) => a + c.p, 0);
      let r = Math.random() * total;
      for (const c of cands) { r -= c.p; if (r <= 0) return c; }
      return cands[0];
    }

    async function append(c, forced = false) {
      gen.push({ text: c.text, p: c.p, rank: c.rank, of: c.of, candidates: current, forced });
      selected = null;
      renderDoc();
      if (c.text === END) { current = null; renderNext(); return false; }
      await fetchNext();
      renderNext();
      return true;
    }

    async function branch(i, c) {
      current = gen[i].candidates;
      gen = gen.slice(0, i);
      try { await append(c, true); } catch (e) { errorLine(msg, e); }
    }

    async function step(n) {
      if (running) return;
      running = true;
      buttons.forEach((b) => (b.disabled = true));
      try {
        if (!current) await fetchNext();
        for (let k = 0; k < n; k++) {
          const c = mode === "top" ? current.candidates[0] : mode === "uniform" ? current.uniform : sample(current.candidates);
          if (!(await append(c))) break;
        }
        msg.replaceChildren();
      } catch (e) { errorLine(msg, e); }
      buttons.forEach((b) => (b.disabled = false));
      running = false;
    }

    async function reset() {
      prompt = promptIn.value;
      gen = [];
      selected = null;
      current = null;
      renderDoc();
      try { await fetchNext(); msg.replaceChildren(); } catch (e) { errorLine(msg, e); }
      renderNext();
    }

    const temp = slider("Temperature", {
      min: 0, max: 2, step: 0.05, value: temperature, fmt: (v) => v.toFixed(2),
      oninput: debounced((v) => { temperature = v; if (selected == null) fetchNext().then(renderNext).catch((e) => errorLine(msg, e)); }),
    });
    const MODES = {
      uniform: ["Random, like Llull", "Every token in the vocabulary is equally likely, whatever the text says so far: Llull's disks, with a very large alphabet."],
      weighted: ["Weighted random", "A roll of the dice, weighted by the model's percentages: a 30% token is picked about 3 times in 10. Temperature reshapes the percentages first."],
      top: ["Most likely", "Always the top token (greedy decoding): the same text every time."],
    };
    // All three descriptions share one spot and only the selected one shows, so the space always fits the
    // longest and nothing below moves when the mode changes.
    const notesByMode = Object.fromEntries(Object.entries(MODES).map(([m, [, note]]) => [m, h("p", { class: "small muted" }, note)]));
    const modeNote = h("div", { class: "overlay" }, ...Object.values(notesByMode));
    const showNote = () => Object.entries(notesByMode).forEach(([m, el]) => { el.style.visibility = m === mode ? "" : "hidden"; });
    showNote();
    const modeSeg = h("div", { class: "seg", role: "group", "aria-label": "How to choose the next token" },
      ...Object.entries(MODES).map(([m, [label]]) => h("button", { type: "button", "aria-pressed": String(m === mode), "data-m": m }, label)));
    modeSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || running) return;
      mode = b.dataset.m;
      modeSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      showNote();
      temp.style.visibility = mode === "weighted" ? "" : "hidden"; // temperature only reshapes weighted picks; keep its space
      if (selected == null) fetchNext().then(renderNext).catch((er) => errorLine(msg, er));
    });
    const add1 = h("button", { class: "btn primary", type: "button", onclick: () => step(1) }, "+1 token");
    const add10 = h("button", { class: "btn", type: "button", onclick: () => step(10) }, "+10");
    const add20 = h("button", { class: "btn", type: "button", onclick: () => step(20) }, "+20");
    const undo = h("button", { class: "btn", type: "button", onclick: async () => { if (!gen.length) return; const last = gen.pop(); current = last.candidates; selected = null; renderDoc(); renderNext(); } }, "Undo");
    const restart = h("button", { class: "btn", type: "button", onclick: reset }, "Start again");
    const buttons = [add1, add10, add20, undo, restart];
    promptIn.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); reset(); } });

    main.append(
      intro("Chapter 4 · From predictor to writer",
        "Pick a token, add it, predict again",
        "A model that only predicts the next token becomes a writer by running in a loop: choose one token from the distribution, append it to the text, and feed the longer text back in. Every chatbot answer is made this way, one token at a time."),
      h("div", { class: "grid2" },
        h("div", { class: "stack" },
          box("Starting text", { right: presets(PROMPTS, (p) => { promptIn.value = p.text; reset(); }, { active: PROMPTS[0] }), body: [promptIn, h("div", { class: "row" }, h("button", { class: "btn", type: "button", onclick: reset }, "Use this text"), h("span", { class: "small muted" }, "or press Enter"))] }),
          box("The text so far", { body: [doc, count, modeSeg, modeNote, h("div", { class: "row" }, ...buttons), temp, msg] })),
        nextPanel),
      notes({
        math: [
          "Three ways to choose. <b>Random, like Llull</b> ignores the model: each of roughly 150,000 tokens has the same chance. <b>Weighted random</b> rolls a die weighted by the model's percentages: a 30% token is chosen about 3 times in 10. <b>Most likely</b> always takes the top token (greedy decoding): the same text every time.",
          "<b>Temperature</b> reshapes the percentages before the roll. It divides every score by T before the softmax:",
          { f: "p(token) ∝ e^(score / T)" },
          "Low T sharpens the distribution (T → 0 is greedy); high T flattens it, so unlikely tokens get picked more often. Chat products typically use something like 0.7.",
          "The model also predicts a special “end of text” token. When it is picked, generation stops.",
        ],
        discuss: [
          "With “Random, like Llull”, every token is as likely as any other, and the text falls apart at once. What, exactly, does the model add to Llull's machine?",
          "Run “+10” with weighted randomness several times from the same start. Which of these texts is what the model “meant to say”?",
          "Force an unlikely token by clicking it, then let the model continue. It will go along with it and make sense of it. What does that suggest about how its later text relates to what came before?",
          "Is the randomness a defect, or part of what makes the output useful? Is there an analogy with human creativity?",
          "Who is the author of a text produced this way: the model, the person who wrote the start, or the people whose writing it learned from?",
        ],
      }),
    );
    renderDoc();
    reset();
  },
};

function debounced(fn, ms = 200) {
  let t;
  return (v) => { clearTimeout(t); t = setTimeout(() => fn(v), ms); };
}
