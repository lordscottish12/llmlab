import { h, intro, notes, box, presets, busy, svgEl } from "../ui.js";

// Ways to load the machine. Disks are listed from the outside in; read() turns the words in the reading
// window into a sentence. Llull's alphabet follows his Ars brevis (simplified: one meaning per disk).
const LLULL = {
  name: "Llull's alphabet",
  rings: ["questions", "principles", "subjects"],
  disks: [
    ["whether?", "what?", "of what?", "why?", "how much?", "of what kind?", "when?", "where?", "how? with what?"],
    ["goodness", "greatness", "eternity", "power", "wisdom", "will", "virtue", "truth", "glory"],
    ["God", "the angel", "heaven", "man", "imagination", "the animal", "the plant", "matter", "the tool"],
  ],
  letters: true,
  read([q, p, s]) {
    const adj = ["good", "great", "eternal", "powerful", "wise", "free", "virtuous", "true", "glorious"][p.i];
    const noun = p.w;
    return [
      `Is ${s.w} ${adj}?`,
      `What is the ${noun} of ${s.w}?`,
      `Of what is the ${noun} of ${s.w} made?`,
      `Why is ${s.w} ${adj}?`,
      `How ${adj} is ${s.w}?`,
      `What kind of ${noun} does ${s.w} have?`,
      `When is ${s.w} ${adj}?`,
      `Where is ${s.w} ${adj}?`,
      `How, and by what means, is ${s.w} ${adj}?`,
    ][q.i];
  },
};

const TIGER = {
  name: "Borges's tiger",
  rings: ["shades", "colours"],
  disks: [
    ["yellowishly", "blackly", "whitely", "greenly", "purplishly", "bluely", "orangely", "greyly", "redly"],
    ["blue", "yellow", "black", "white", "green", "purple", "orange", "grey", "red"],
  ],
  read: ([a, c]) => `The tiger is ${a.w} ${c.w}.`,
};

const MODERN = {
  name: "Loaded like Borges, 1937",
  rings: ["politics", "Llull's relations", "physics"],
  disks: [
    ["Surplus Value", "Proletariat", "Capitalism", "Class Struggle", "Dialectical Materialism", "Engels"],
    ["difference", "concord", "contrariety", "beginning", "middle", "end", "majority", "equality", "minority"],
    ["Entropy", "Time", "Electrons", "Potential Energy", "Fourth Dimension", "Relativity", "Protons", "Einstein"],
  ],
  read([m, r, p]) {
    const rel = ["differs from", "agrees with", "contradicts", "is the beginning of", "is the middle of", "is the end of", "is greater than", "equals", "is less than"][r.i];
    return `${p.w} ${rel} ${m.w}.`;
  },
};

const LOADS = [LLULL, TIGER, MODERN];
const ESSAY = "https://gwern.net/doc/borges/1937-borges-raymondllullsthinkingmachine.pdf";
const LETTERS = ["B", "C", "D", "E", "F", "G", "H", "I", "K"];

export default {
  id: "llull",
  title: "Llull's machine",
  num: "Intro",
  mount(main) {
    const machine = machinePanel();
    main.append(
      intro("Prologue · Ramon Llull, read by Borges",
        "A thinking machine that works all too well",
        ["Around 1300 the Majorcan philosopher Ramon Llull built a “thinking machine” from paper disks: turn them, and they combine a fixed set of concepts into statements and questions. In ",
          h("a", { href: ESSAY, target: "_blank", rel: "noopener", title: "Jorge Luis Borges, “Ramon Llull's Thinking Machine” (1937), translated by Esther Allen. Opens the PDF on gwern.net in a new tab." }, "a short essay from 1937"),
          ", Jorge Luis Borges argues that it cannot think, and then that it works all too well. Seven centuries later we have built another machine that combines a fixed set of pieces. Try Llull's first."]),
      machine.el,
      notes({
        math: [
          "Llull's first figure joins nine attributes of God in pairs. Every pair is one line in the diagram:",
          { f: "9 × 8 ÷ 2 = 36 pairs" },
          "Three disks of nine words give 9 × 9 × 9 = 729 readings, and each new disk multiplies the count again. A machine that produces all of them “works all too well”: it gives every answer, so it cannot tell you which one is right.",
          "A language model also builds text from a fixed set of pieces: 151,936 tokens for the Qwen3 models used here. Ten tokens in a row already allow about 6 × 10<sup>51</sup> texts. What it adds to Llull is a weight on every continuation: how likely a person would be to write it.",
        ],
        discuss: [
          "Borges first says the machine does not work, then that it works all too well. Which of the two describes a language model better?",
          "Swift mocks a similar machine in Gulliver's Travels: students turn its handles and copy down the fragments that happen to make sense. Where does the meaning come from: the machine, or the people reading it?",
          "Borges defends the machine as a poet's tool: the poet tries adjectives for “tiger” until one startles. Is that thinking? Is it what a language model does?",
          "Borges would load the machine with Entropy and Class Struggle instead of goodness and glory. What would we load it with today, and what does that choice say about us?",
        ],
      }),
    );
    return () => machine.dispose();
  },
};

// ------------------------------------------------------------------ the machine: concentric disks you can turn

function machinePanel() {
  const S = 560, C = S / 2, HUB = 70, OUT = 272;
  let load = LLULL;
  let rot = []; // degrees per disk
  let rings = []; // {g, n, step, r0, r1, labels}
  let seen = new Set();
  let raf = 0;

  const svg = svgEl("svg", { viewBox: `0 0 ${S} ${S}`, role: "img", "aria-label": "Llull's machine: concentric disks that can be turned" });
  const sentence = h("p", { class: "llull-out", "aria-live": "polite" });
  const letters = h("p", { class: "mono muted" });
  const count = h("p", { class: "small muted" });
  const legend = h("p", { class: "small muted" });
  const spinBtn = h("button", { class: "btn primary", type: "button" }, "Spin");

  const mod = (a, n) => ((a % n) + n) % n;
  const index = (k) => mod(Math.round(-rot[k] / rings[k].step), rings[k].n);

  function build() {
    cancelAnimationFrame(raf);
    const n = load.disks.length, band = (OUT - HUB) / n;
    rot = load.disks.map(() => 0);
    seen = new Set();
    rings = load.disks.map((words, k) => {
      const r1 = OUT - k * band, r0 = r1 - band, step = 360 / words.length, rm = (r0 + r1) / 2;
      const g = svgEl("g", { class: "ring" },
        svgEl("circle", { cx: C, cy: C, r: r1, class: `band${k % 2 ? " alt" : ""}` }));
      const labels = words.map((w, i) => {
        const a = i * step;
        const spoke = (a + step / 2) * (Math.PI / 180);
        g.append(svgEl("line", { x1: C + r0 * Math.sin(spoke), y1: C - r0 * Math.cos(spoke), x2: C + r1 * Math.sin(spoke), y2: C - r1 * Math.cos(spoke), class: "spoke" }));
        // Long labels wrap onto two lines, at the first space after the first word.
        const parts = w.length > 11 && w.includes(" ") ? [w.slice(0, w.indexOf(" ")), w.slice(w.indexOf(" ") + 1)] : [w];
        const t = svgEl("text", { x: C, y: C - rm + (load.letters ? 12 : 4) - (parts.length - 1) * 6, transform: `rotate(${a} ${C} ${C})`, class: "word", "text-anchor": "middle" },
          ...parts.map((p, j) => svgEl("tspan", { x: C, dy: j ? 13 : 0 }, p)));
        g.append(t);
        let letter = null;
        if (load.letters) {
          letter = svgEl("text", { x: C, y: C - rm - 6, transform: `rotate(${a} ${C} ${C})`, class: "letter", "text-anchor": "middle" }, LETTERS[i]);
          g.append(letter);
        }
        return [t, letter];
      });
      return { g, n: words.length, step, r0, r1, labels };
    });
    const win = 30;
    svg.replaceChildren(
      ...rings.map((r) => r.g),
      svgEl("circle", { cx: C, cy: C, r: HUB, class: "hub" }),
      // In Llull's figure, A at the centre stands for God.
      load.letters ? svgEl("text", { x: C, y: C + 12, class: "letter hubletter", "text-anchor": "middle" }, "A") : "",
      svgEl("rect", { x: C - win, y: C - OUT + 2, width: 2 * win, height: OUT - HUB - 4, class: "window" }));
    legend.textContent = `Disks, from the outside in: ${load.rings.join(" · ")}. ${load.disks.map((d) => d.length).join(" × ")} = ${load.disks.reduce((a, d) => a * d.length, 1).toLocaleString("en")} possible readings. Drag a disk to turn it.`;
    rings.forEach((_, k) => place(k));
    read(true);
  }

  function place(k) {
    rings[k].g.setAttribute("transform", `rotate(${rot[k]} ${C} ${C})`);
  }

  // settled: the disks have come to rest, so this reading counts as read.
  function read(settled = false) {
    const picked = rings.map((r, k) => ({ i: index(k), w: load.disks[k][index(k)] }));
    rings.forEach((r, k) => r.labels.forEach(([t, l], i) => { t.classList.toggle("on", i === picked[k].i); l?.classList.toggle("on", i === picked[k].i); }));
    sentence.textContent = load.read(picked);
    letters.textContent = load.letters ? picked.map((p) => LETTERS[p.i]).join(" · ") : "";
    if (settled) seen.add(picked.map((p) => p.i).join());
    const total = load.disks.reduce((a, d) => a * d.length, 1);
    count.textContent = `You have read ${seen.size.toLocaleString("en")} of ${total.toLocaleString("en")} possible readings.`;
  }

  // Turn disks to target angles; done() runs when all have arrived.
  function animate(targets, ms, done) {
    cancelAnimationFrame(raf);
    const from = rot.slice(), t0 = performance.now();
    const tick = (t) => {
      const u = Math.min(1, (t - t0) / ms), e = 1 - (1 - u) ** 3;
      targets.forEach((to, k) => { if (to != null) { rot[k] = from[k] + (to - from[k]) * e; place(k); } });
      read(u >= 1);
      if (u < 1) raf = requestAnimationFrame(tick);
      else done?.();
    };
    raf = requestAnimationFrame(tick);
  }
  const snap = (k) => Math.round(rot[k] / rings[k].step) * rings[k].step;

  // Dragging: the disk under the pointer follows it, then settles on the nearest word.
  let drag = null;
  const at = (e) => {
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * S - C, y = ((e.clientY - r.top) / r.height) * S - C;
    return { r: Math.hypot(x, y), a: (Math.atan2(x, -y) * 180) / Math.PI };
  };
  svg.addEventListener("pointerdown", (e) => {
    const p = at(e);
    const k = rings.findIndex((r) => p.r >= r.r0 && p.r <= r.r1);
    if (k < 0) return;
    cancelAnimationFrame(raf);
    drag = { k, a: p.a, rot: rot[k] };
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    let d = at(e).a - drag.a;
    d = ((d + 540) % 360) - 180;
    rot[drag.k] = drag.rot + d;
    drag.rot = rot[drag.k];
    drag.a = at(e).a;
    place(drag.k);
    read();
  });
  const release = () => {
    if (!drag) return;
    const k = drag.k;
    drag = null;
    svg.classList.remove("dragging");
    animate(rings.map((_, j) => (j === k ? snap(k) : null)), 250);
  };
  svg.addEventListener("pointerup", release);
  svg.addEventListener("pointercancel", release);

  // Spinning: each disk turns a random amount, the outer ones first, and stops on a word.
  spinBtn.addEventListener("click", () => {
    const targets = rings.map((r, k) => {
      const turns = (1 + Math.random()) * 360 * (k % 2 ? -1 : 1);
      return Math.round((rot[k] + turns) / r.step) * r.step;
    });
    busy(spinBtn, true, "Spinning…");
    animate(targets, 1600, () => busy(spinBtn, false));
  });

  build();
  return {
    el: h("div", { class: "grid2" },
      box("The machine", {
        right: presets(LOADS, (p) => { load = p || LLULL; build(); }, { active: LLULL }),
        body: [h("div", { class: "llull" }, svg), legend],
      }),
      box("What it says", {
        body: [sentence, letters, h("div", { class: "row" }, spinBtn), count,
          h("p", { class: "small muted" }, "Every combination is equally possible, and the machine has no way to prefer one. That is Borges's point: it produces sense and nonsense with the same confidence, and leaves the choosing to us."),
          h("blockquote", { class: "pull" },
            h("p", { lang: "fr" }, "Elle ne fonctionne que trop."),
            h("p", {}, "It works all too well."),
            h("cite", {}, "Jorge Luis Borges, “Ramon Llull's Thinking Machine” (1937), translated by Esther Allen"))],
      })),
    dispose: () => cancelAnimationFrame(raf),
  };
}
