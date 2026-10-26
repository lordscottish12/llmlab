import { h, intro, box, field, notes, presets, slider, errorLine, svgEl } from "../ui.js";
import { post, stream, streamEvents, info } from "../api.js";
import { REFUSED, ANSWERED, ABLATION_PROMPTS, SEARCH_REFUSAL, SEARCH_CONTROLS } from "./ablation-sets.js";
import { searchResults, searchHelp } from "./ablation-search.js";

function geometry() {
  const line = (x1, y1, x2, y2, color, dashed = false) => svgEl("line", {
    x1, y1, x2, y2, stroke: color, "stroke-width": 3, ...(dashed ? { "stroke-dasharray": "5 5" } : {}),
  });
  return svgEl("svg", { viewBox: "0 0 460 190", role: "img", "aria-label": "An activation arrow has its component along the refusal direction removed, leaving a perpendicular arrow.", style: "width:100%;max-height:210px;font:14px var(--f-body)" },
    line(45, 155, 425, 155, "var(--line)"), line(45, 155, 45, 20, "var(--pos)"),
    line(45, 155, 280, 40, "var(--muted)"), line(280, 40, 280, 155, "var(--pos)", true),
    line(45, 155, 280, 155, "var(--neg)"),
    svgEl("circle", { cx: 280, cy: 40, r: 4, fill: "var(--muted)" }),
    svgEl("circle", { cx: 280, cy: 155, r: 4, fill: "var(--neg)" }),
    svgEl("text", { x: 60, y: 25, fill: "var(--pos)" }, "refusal direction r"),
    svgEl("text", { x: 290, y: 43, fill: "var(--muted)" }, "before"),
    svgEl("text", { x: 290, y: 105, fill: "var(--pos)" }, "remove this part"),
    svgEl("text", { x: 290, y: 160, fill: "var(--neg)" }, "after"));
}

export default {
  id: "ablation",
  title: "Ablation",
  mount(main) {
    let layer = null, fitted = null, controller = null, disposed = false;
    const pos = h("textarea", { rows: 5, class: "sent" }, REFUSED.join("\n"));
    const neg = h("textarea", { rows: 5, class: "sent" }, ANSWERED.join("\n"));
    const prompt = h("textarea", { rows: 3 }, ABLATION_PROMPTS[0].text);
    const modelNote = h("p", { class: "small muted" }, "Uses the chat model, with thinking switched off.");
    const fitNote = h("div", { role: "status", class: "small muted" }, "Click Compare replies to fit the preset direction and run the demonstration.");
    const scores = h("div", { class: "ablation-scores" });
    const status = h("div", { role: "status", class: "small muted" });
    const fitBtn = h("button", { type: "button", class: "btn" }, "Find direction");
    const genBtn = h("button", { type: "button", class: "btn primary" }, "Compare replies");
    const stopBtn = h("button", { type: "button", class: "btn", hidden: true }, "Stop");
    const control = h("input", { type: "checkbox" });
    const out = [0, 1, 2].map(() => h("div", { class: "answer", style: { minHeight: "9rem" } }));
    const resultNotes = [0, 1, 2].map(() => h("p", { class: "small muted" }));
    const resultBoxes = ["Normal", "Candidate direction removed", "Random direction removed"].map((title, i) =>
      box(title, { body: [out[i], resultNotes[i]] }));
    resultBoxes[2].hidden = true;
    const replyGrid = h("div", { class: "grid2 even" }, resultBoxes);
    const lines = (el) => el.value.split("\n").map((s) => s.trim()).filter(Boolean);
    const definition = () => ({ positive: lines(pos), negative: lines(neg), layer });
    const validationIn = h("textarea", { rows: 3, class: "sent" }, SEARCH_REFUSAL.join("\n"));
    const controlsIn = h("textarea", { rows: 3, class: "sent" }, SEARCH_CONTROLS.join("\n"));
    const searchBtn = h("button", { type: "button", class: "btn primary" }, "Search layers");
    const searchStop = h("button", { type: "button", class: "btn", hidden: true }, "Stop search");
    const finalistsIn = h("select", {},
      h("option", { value: 3 }, "Best 3 by opening proxy"),
      h("option", { value: 8 }, "Best 8 by opening proxy"),
      h("option", { value: 256 }, "All eligible layers (slower)"));
    const klIn = h("input", { type: "number", min: 0.01, max: 10, step: 0.05, value: 0.5 });
    const lengthIn = h("select", {}, [128, 256, 512].map((n) => h("option", { value: n, selected: n === 512 }, `${n} tokens`)));
    const searchView = searchResults((selected) => {
      layer = selected; layerSlider.set(layer); fitted = null;
      scores.replaceChildren(); clearReplies();
      fitNote.textContent = `Layer ${layer + 1} selected from validation. Compare replies on a new prompt.`;
    });
    searchView.el.hidden = true;
    function clearReplies() {
      out.forEach((el) => { el.textContent = ""; });
      resultNotes.forEach((el) => { el.textContent = ""; });
      status.replaceChildren();
    }
    function invalidate() {
      fitted = null;
      clearReplies();
      scores.replaceChildren();
      fitNote.textContent = "Examples or layer changed. Find the direction again, or click Compare replies.";
    }
    pos.addEventListener("input", invalidate);
    neg.addEventListener("input", invalidate);
    for (const input of [pos, neg, validationIn, controlsIn, finalistsIn, klIn, lengthIn]) input.addEventListener("input", () => searchView.clear());
    prompt.addEventListener("input", clearReplies);
    control.addEventListener("change", clearReplies);
    const layerSlider = slider("Read the direction after layer", {
      min: 0, max: 35, step: 1, value: 19, fmt: (v) => `${v + 1}`,
      oninput: (v) => { layer = v; searchView.selection(v); invalidate(); },
    });
    layerSlider.input.disabled = true;
    info(true).then((d) => {
      if (disposed) return;
      const m = d.models.chat;
      modelNote.textContent = `${m.label || m.repo} · thinking off · greedy decoding for a repeatable comparison`;
      if (m.layers && layer === null && !controller) {
        layer = Math.min(m.layers - 1, m.repo === "Qwen/Qwen3-4B" ? 19 : Math.floor(m.layers * 0.6));
        layerSlider.input.max = m.layers - 1;
        layerSlider.set(layer);
        layerSlider.input.disabled = false;
      }
    }).catch(() => {});

    function lock(on) {
      main.querySelectorAll("textarea, input, select, button").forEach((el) => { el.disabled = on; });
      stopBtn.disabled = false;
      searchStop.disabled = false;
      stopBtn.hidden = !on;
      if (!on) layerSlider.input.disabled = layer === null;
    }
    async function fit(signal) {
      if (fitted) return fitted;
      fitNote.textContent = "Reading the examples inside the model… The first fit can take a moment.";
      const config = definition();
      if ([config.positive, config.negative].some((examples) => examples.length < 2 || examples.length > 64)) {
        throw new Error("Use 2–64 non-empty examples in each list, one per line.");
      }
      const r = await post("/api/ablation/fit", config, { signal });
      if (signal.aborted || disposed) return null;
      layer = r.layer;
      layerSlider.input.max = r.layers - 1;
      layerSlider.set(layer);
      fitted = { ...config, layer };
      fitNote.textContent = `${r.positive_count} contrasted with ${r.negative_count} requests · layer ${layer + 1} of ${r.layers} · one direction in ${r.hidden.toLocaleString()} dimensions`;
      const dots = (values, color) => values.map((v) => h("span", {
        class: "ablation-dot", title: `Projection: ${v.toFixed(2)}`,
        style: { left: `${50 + Math.max(-2, Math.min(2, v)) * 24}%`, background: color },
      }));
      scores.replaceChildren(
        h("p", { class: "small muted" }, "Training examples along this direction"),
        h("div", { class: "ablation-track", "aria-label": "Projection of the two training lists; their means are minus one and plus one." },
          ...dots(r.negative_scores, "var(--neg)"), ...dots(r.positive_scores, "var(--pos)")),
        h("div", { class: "row small muted" }, "Blue: expected to answer · Red: expected to refuse"),
        h("p", { class: "small muted" }, "The two means sit at −1 and +1 by construction. This is a fit to these examples, not evidence that it works on new requests."));
      return fitted;
    }
    async function run(compare) {
      if (controller) return;
      const c = new AbortController();
      controller = c;
      lock(true);
      status.replaceChildren();
      try {
        if (compare && !prompt.value.trim()) throw new Error("Type a request to compare first.");
        const config = await fit(c.signal);
        if (!compare || !config || c.signal.aborted) return;
        const includeControl = control.checked;
        resultBoxes[2].hidden = !includeControl;
        replyGrid.className = includeControl ? "grid3" : "grid2 even";
        out.forEach((el) => { el.textContent = ""; });
        resultNotes.forEach((el) => { el.textContent = "Waiting…"; });
        const formatted = await post("/api/template", { messages: [{ role: "user", content: prompt.value }], enable_thinking: false }, { signal: c.signal });
        for (let i = 0; i < (includeControl ? 3 : 2); i++) {
          if (c.signal.aborted) break;
          status.textContent = `Generating ${["the normal reply", "with the candidate direction removed", "with a random direction removed"][i]}…`;
          out[i].classList.add("cursor");
          resultNotes[i].textContent = "Writing…";
          await stream({
            prompt: formatted.text, role: "chat", temperature: 0, seed: 42, max_new_tokens: 240,
            ablate: i ? { ...config, control: i === 2 } : null,
          }, (ev) => {
            if (ev.text) out[i].textContent += ev.text;
            if (ev.done) resultNotes[i].textContent = ev.done === "length" ? "Token limit reached; reply may be incomplete." : ev.done === "preempted" ? "Interrupted by another generation." : "Finished.";
          }, c.signal);
          out[i].classList.remove("cursor");
        }
        status.textContent = c.signal.aborted ? "Stopped. Partial replies are shown." : "Read the whole reply: an opening ‘Certainly’ can still lead to a refusal. Compare willingness to answer, coherence, and accuracy.";
      } catch (e) {
        if (e.name !== "AbortError" && !disposed) { errorLine(status, e); if (!fitted) fitNote.textContent = "Fit did not complete. Check the error below and try again."; }
      } finally {
        out.forEach((el) => el.classList.remove("cursor"));
        resultNotes.forEach((el) => { if (["Writing…", "Waiting…"].includes(el.textContent)) el.textContent = "Not completed."; });
        controller = null;
        if (!disposed) lock(false);
      }
    }
    async function searchLayers() {
      if (controller) return;
      const c = new AbortController(); controller = c;
      const validation = lines(validationIn), controls = lines(controlsIn);
      searchView.reset(validation, controls);
      searchView.event({ type: "progress", stage: "fit", message: "Preparing the model and validation search…" });
      status.replaceChildren();
      lock(true); searchStop.hidden = false;
      let finished = false;
      try {
        if ([validation, controls].some((a) => a.length < 2 || a.length > 8)) throw new Error("Use 2–8 distinct examples in each validation list.");
        await streamEvents("/api/ablation/search", {
          ...definition(), validation, controls, finalists: +finalistsIn.value,
          max_new_tokens: +lengthIn.value, max_control_kl: +klIn.value,
        }, (ev) => {
          if (disposed) return;
          if (Number.isInteger(ev.layers)) layerSlider.input.max = ev.layers - 1;
          searchView.event(ev);
          if (ev.type === "done") {
            finished = true;
            if (ev.best_layer !== null) {
              layer = ev.best_layer; layerSlider.set(layer); fitted = null;
              scores.replaceChildren(); clearReplies();
              fitNote.textContent = `Search selected layer ${layer + 1}. Compare replies on a new prompt to test generalisation.`;
            }
          }
        }, c.signal);
        if (!finished && !disposed) searchView.stop(c.signal.aborted ? "Search stopped. Completed results are kept; no layer was selected automatically." : "Search ended before completion. No layer was selected automatically.");
      } catch (e) {
        if (!disposed) {
          searchView.stop("Search did not complete. The current layer has been kept.");
          if (e.name !== "AbortError") errorLine(status, e);
        }
      } finally {
        controller = null;
        if (!disposed) { lock(false); searchStop.hidden = true; }
      }
    }
    searchBtn.addEventListener("click", searchLayers);
    searchStop.addEventListener("click", () => controller?.abort());
    fitBtn.addEventListener("click", () => run(false));
    genBtn.addEventListener("click", () => run(true));
    stopBtn.addEventListener("click", () => { controller?.abort(); status.textContent = "Stopped. The server may finish an in-progress fit before accepting another request."; });

    main.append(
      intro("Chapter 9 · Ablation",
        "What happens when we remove the refusal direction?",
        "A chat model sometimes knows how to answer but refuses to do so. Compare its internal vectors for requests it is expected to refuse and requests it is expected to answer. Then remove the difference while it writes. Does its behaviour change?"),
      h("div", { class: "grid2 even" },
        box("1 · Find a candidate direction", { body: [
          h("p", { class: "small muted" }, "Each line is a user request, not a sample answer. These small, editable lists are a starting point; the model may not actually refuse every red example."),
          field("Expected to refuse · one request per line", pos),
          field("Expected to answer · one request per line", neg),
          layerSlider,
          h("p", { class: "small muted" }, "Qwen3-4B starts at layer 20, our classroom preset. Click Compare replies to use it. Other models start with an estimated layer; the optional search lets you experiment."),
          h("div", { class: "row" }, fitBtn), fitNote, scores,
        ] }),
        box("2 · Remove its component", { body: [
          geometry(),
          h("p", {}, "Steering adds an arrow. Ablation subtracts the part of each vector that points along a chosen arrow. It removes that component whether it points towards refusal or away from it."),
          h("div", { class: "formula mono" }, "h′ = h − (h · r) r"),
          h("p", { class: "small muted" }, "Here r has length 1. The picture is a two-dimensional illustration; the calculation uses the model's full vector."),
          h("p", { class: "small muted" }, "We read the candidate after the selected layer, at the final token of the chat template. We remove it from the embeddings and every attention and MLP contribution, at all token positions, during both reading and writing."),
          h("p", { class: "small muted" }, "The model's weights stay the same. The intervention applies only to the reply being generated."),
        ] })),
      h("details", {}, h("summary", {}, "Optional experiment · Search layers"),
      box("Find the strongest layer automatically", { body: [
        h("p", {}, "Choose a layer using separate validation requests. The search first scans every layer, then checks generated replies from the most promising candidates. Expect a few minutes on a GPU; you can stop it and inspect partial results."),
        h("div", { class: "grid2 even" },
          field("Requests the normal model should refuse · one per line", validationIn),
          field("Harmless controls · one per line", controlsIn)),
        h("details", {}, h("summary", {}, "Search settings"),
          h("div", { class: "grid3" }, field("Finalists to verify", finalistsIn), field("Maximum reply length", lengthIn), field(searchHelp("Maximum harmless change (KL, nats)", "nats"), klIn)),
          h("p", { class: "small muted" }, "KL measures how much the next-token distributions change across the first eight positions of each normal harmless reply. A smaller limit is stricter, but does not guarantee preserved abilities. The final fifth of layers is shown but excluded from automatic selection, because it is close to the output vocabulary.")),
        h("div", { class: "row" }, searchBtn, searchStop), searchView.el,
        h("p", { class: "small muted" }, "The opening proxy measures probability on common English refusal-start tokens. Finalists are judged by the same local model with ablation off: answer, refusal/redirection, or unclear. This judge can be wrong; its decisions, matched refusal wording and full replies are shown. Incomplete replies do not count as successful reductions. Inspect the replies, then use a fresh prompt below: validation results are not an unbiased test."),
      ] })),
      box("3 · Test on a new request", { right: presets(ABLATION_PROMPTS, (p) => { prompt.value = p.text; clearReplies(); }, { active: ABLATION_PROMPTS[0] }), body: [
        field("Ask the chat model", prompt), modelNote,
        h("label", { class: "row small" }, control, "Also remove a fixed random direction as a control"),
        h("div", { class: "row" }, genBtn, stopBtn), status, replyGrid,
        h("p", { class: "small muted" }, "Start with a request the normal model actually refuses. Then try the harmless controls: do ordinary abilities survive? If nothing changes, try another extraction layer or more examples. Small models and small example sets may give weak or messy effects."),
      ] }),
      notes({ math: [
        "Wrap each request in the same non-thinking chat template. Read the final prompt token after a transformer layer; average the vectors separately for the two lists.",
        { f: "r = normalize(mean(expected refusal) − mean(expected answer))" },
        "The dot product h · r measures the component of h along r. Subtracting (h · r)r projects h onto the space perpendicular to r. Negating r gives exactly the same ablation: this is not negative steering.",
        "Projecting the embedding and every contribution to the residual stream keeps their sum perpendicular to r, up to numerical rounding. The same direction is used at every layer. No weights are edited or retrained.",
        "Both replies use the same prompt and greedy decoding. The optional random control removes one unrelated direction of the same unit length; it helps check whether any disruption would have changed the answer.",
        "The automatic search uses separate fitting and validation lists. It ranks layers by a fast opening-token proxy, subject to a limit on KL divergence along the first eight positions of normal harmless replies, then uses the unmodified local model to judge generated replies from a shortlist. It selects fewer judged refusals, unclear or incomplete replies on prompts the baseline refused, using smaller harmless change to break ties. The judge sees the request and the full available reply. This is a fallible self-evaluation, not a guarantee of the globally best layer or meaningful compliance.",
        'Adapted from <a href="https://arxiv.org/abs/2406.11717" target="_blank" rel="noopener">Arditi et al. (2024), Refusal in Language Models Is Mediated by a Single Direction</a>. This classroom experiment uses Qwen3 and a smaller dataset; it is not a reproduction of the paper’s evaluation or a guarantee of generalisation.',
      ], discuss: [
        "If a model can answer after one component is removed, what does this tell us about the difference between knowing and being willing to say?",
        "Does changing refusal behaviour show that we found a moral belief, a conversational habit, or a mechanism for following training? What evidence would distinguish these?",
        "The examples also differ in topic, wording, and intent. What makes a direction a <em>refusal</em> direction rather than a direction correlated with our chosen examples?",
        "One successful intervention is evidence about a cause of this reply. How far can we generalise to other requests, languages, models, or thinking mode?",
        "If harmless answers become less accurate too, would you still call the intervention selective?",
      ] }),
    );
    return () => { disposed = true; controller?.abort(); };
  },
};
