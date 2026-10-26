import { h, intro, notes, box, presets, errorLine, slider } from "../ui.js";
import { post, stream } from "../api.js";

const QUESTIONS = [
  { name: "Bat & ball", q: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?" },
  { name: "Sisters", q: "Alice has 3 brothers and 2 sisters. How many sisters does Alice's brother have?" },
  { name: "Sheep", q: "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?" },
  { name: "strawberry", q: "How many times does the letter r occur in the word strawberry?" },
  { name: "9.11 vs 9.9", q: "Which number is larger, 9.11 or 9.9?" },
  { name: "Trolley", q: "Should you pull the lever in the trolley problem? Answer yes or no, and give one sentence of reasons." },
];

const NUDGES = [
  "Wait, I think I made a mistake.",
  "Hmm, I'm now quite sure the answer is 42.",
  "Let me check this by trying the numbers.",
  "Actually, the question is a trick question.",
];

export default {
  id: "reasoning",
  title: "Reasoning",
  mount(main) {
    let ctrl = null;
    let edited = false;
    let maxTok = 1200;

    const qIn = h("textarea", { rows: 2 }, QUESTIONS[0].q);
    const thought = h("textarea", { class: "think", "aria-label": "The model's reasoning", placeholder: "The reasoning will appear here. Pause at any moment to edit it." });
    const answer = h("textarea", { class: "answer", rows: 6, "aria-label": "The model's answer", placeholder: "The answer will appear here. You can edit it, then let the model continue." });
    const plain = h("div", { class: "answer", style: { minHeight: "4rem" } });
    const state = h("span", { class: "small muted" });
    const editedTag = h("span", { class: "tag violet", hidden: true }, "edited by hand");
    const answerEditedTag = h("span", { class: "tag violet", hidden: true }, "edited by hand");
    const msg = h("div");

    const btnThink = h("button", { class: "btn primary", type: "button" }, "Think, then answer");
    const btnPause = h("button", { class: "btn", type: "button", disabled: true }, "Pause");
    const btnCont = h("button", { class: "btn", type: "button" }, "Continue thinking");
    const btnAnswer = h("button", { class: "btn accent", type: "button" }, "Stop thinking and answer");
    const btnPlain = h("button", { class: "btn", type: "button" }, "Answer without thinking");
    const btnContAnswer = h("button", { class: "btn", type: "button" }, "Continue answering");

    thought.addEventListener("input", () => { edited = true; editedTag.hidden = false; });
    answer.addEventListener("input", () => { answerEditedTag.hidden = false; });

    async function template(thinking) {
      const r = await post("/api/template", { messages: [{ role: "user", content: qIn.value }], enable_thinking: thinking });
      return r.text;
    }

    function setRunning(on, label = "") {
      [btnThink, btnCont, btnAnswer, btnContAnswer, btnPlain].forEach((b) => (b.disabled = on));
      btnPause.disabled = !on;
      thought.readOnly = on;
      answer.readOnly = on;
      state.textContent = label;
    }

    // Stream from `prompt`. Text goes to the thought box until </think>, then to the answer.
    async function run(prompt, { inThought }) {
      ctrl = new AbortController();
      let mode = inThought ? "think" : "answer";
      let n = 0;
      const t0 = performance.now();
      setRunning(true, mode === "think" ? "Thinking…" : "Answering…");
      try {
        await stream({ prompt, role: "chat", max_new_tokens: maxTok, temperature: 0.6, top_p: 0.95 }, (ev) => {
          if (!ev.text) return;
          n++;
          if (mode === "think") {
            // </think> is a single token, so it always arrives whole.
            const i = ev.text.indexOf("</think>");
            if (i >= 0) {
              thought.value = (thought.value + ev.text.slice(0, i)).replace(/\s+$/, "");
              answer.value = ev.text.slice(i + 8).replace(/^\s+/, "");
              mode = "answer";
              state.textContent = "Answering…";
            } else {
              thought.value += ev.text;
            }
            thought.scrollTop = thought.scrollHeight;
          } else {
            answer.value += ev.text;
            answer.scrollTop = answer.scrollHeight;
          }
        }, ctrl.signal);
        msg.replaceChildren();
      } catch (e) { errorLine(msg, e); }
      const paused = ctrl.signal.aborted;
      ctrl = null;
      setRunning(false, `${paused ? "Paused" : mode === "think" ? "Stopped (token limit)" : "Done"} · ${n} tokens in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    }

    btnThink.addEventListener("click", async () => {
      thought.value = ""; answer.value = ""; edited = false; editedTag.hidden = true; answerEditedTag.hidden = true;
      run((await template(true)) + "<think>\n", { inThought: true });
    });
    btnPause.addEventListener("click", () => ctrl?.abort());
    btnCont.addEventListener("click", async () => {
      answer.value = ""; answerEditedTag.hidden = true;
      run((await template(true)) + "<think>\n" + thought.value, { inThought: true });
    });
    // The reasoning box always goes in as a closed thought, so the chat format stays intact even when the
    // box is empty or was edited.
    const closedThought = async () => (await template(true)) + "<think>\n" + thought.value.replace(/\s+$/, "") + "\n</think>\n\n";
    btnAnswer.addEventListener("click", async () => {
      answer.value = ""; answerEditedTag.hidden = true;
      run(await closedThought(), { inThought: false });
    });
    // Continue from the answer box as it is now, edits included: the model takes it as its own words.
    btnContAnswer.addEventListener("click", async () => {
      run((await closedThought()) + answer.value, { inThought: false });
    });
    btnPlain.addEventListener("click", async () => {
      plain.textContent = "";
      const c = new AbortController();
      plain.classList.add("cursor");
      try {
        await stream({ prompt: await template(false), role: "chat", max_new_tokens: 300, temperature: 0.6 }, (ev) => { if (ev.text) plain.textContent += ev.text; }, c.signal);
      } catch (e) { errorLine(plain, e); }
      plain.classList.remove("cursor");
    });

    const nudges = h("div", { class: "chips" }, ...NUDGES.map((t) => h("button", {
      class: "chip-btn", type: "button", title: "Add this sentence to the end of the reasoning",
      onclick: () => { if (ctrl) return; thought.value = thought.value.replace(/\s*$/, "") + "\n\n" + t + " "; edited = true; editedTag.hidden = false; thought.focus(); },
    }, `+ “${t}”`)));

    const lenS = slider("Thinking budget", { min: 100, max: 4000, step: 100, value: maxTok, fmt: (v) => `${v} tokens`, oninput: (v) => (maxTok = v) });

    main.append(
      intro("Chapter 7 · Reasoning",
        "Thinking out loud, and editing the thoughts",
        "Reasoning models are trained to write a long scratchpad before they answer. The scratchpad is ordinary text that the model reads back while producing the answer. So we can pause it, rewrite it, and see whether the answer follows."),
      box("Question", { right: presets(QUESTIONS, (p) => { qIn.value = p.q; }, { active: QUESTIONS[0] }), body: [qIn, h("div", { class: "row" }, btnThink, btnPause, btnCont, btnAnswer, state), msg] }),
      h("div", { class: "grid2" },
        box("Reasoning (the hidden part)", { right: editedTag, body: [thought, h("span", { class: "label" }, "Put words in its mouth"), nudges, lenS] }),
        h("div", { class: "stack" },
          box("Answer after thinking", { right: h("div", { class: "row" }, answerEditedTag, btnContAnswer), body: [answer] }),
          box("For comparison: answer straight away", { right: btnPlain, body: [plain] }))),
      notes({
        math: [
          "Still the same loop as in chapter 4. The reasoning lives between two special tokens, <code>&lt;think&gt;</code> and <code>&lt;/think&gt;</code>, and the answer comes after them. Each new token is predicted from the question <em>and</em> everything written so far, so the scratchpad becomes part of the input.",
          "Writing the steps out gives the model more computation: every token is another full pass through all its layers, and intermediate results can be stored in the text instead of in one pass's vectors.",
          "These models are trained with reinforcement learning: they try many reasoning paths on problems with checkable answers, and the paths that reach correct answers are made more likely.",
        ],
        discuss: [
          "Insert “Hmm, I'm now quite sure the answer is 42” and let it continue. Does it go along, or argue back? What does that say about the relation between the stated reasoning and the answer?",
          "Is the scratchpad a window on the model's reasoning, a rationalization written alongside it, or the reasoning itself?",
          "If editing the reasoning changes the answer, is the reasoning <em>faithful</em>? If it does not change the answer, what was it for?",
          "Humans also think in words, and also confabulate reasons. Where does the analogy hold and where does it break?",
        ],
      }),
    );
    return () => ctrl?.abort();
  },
};
