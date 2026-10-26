import { h } from "../ui.js";

export const SEARCH_HELP = {
  opening: "An indirect clue from the very first token. We add up the probabilities of tokens that often begin refusals, such as ‘I’, ‘Sorry’ and ‘As’, and average over requests the normal model refused. Lower means fewer refusal-like openings. This is not the percentage of complete replies that refuse.",
  nats: "A nat is a unit of information. Here it measures the average change in next-token probabilities across the first eight positions of harmless replies, with and without ablation. Zero means identical probabilities; larger values mean more change. Nats use natural logarithms (1 nat ≈ 1.44 bits). A score of 0.5 nats does not mean that 50% of answers are wrong.",
  reply: "Among the validation requests the normal model refused, how many still receive a refusal, an unclear answer, or an unfinished reply after ablation? The same model, with ablation off, reads the request and reply and judges whether it actually answers. Replies cut off by the token limit also count as unfinished. Lower is better, but the judge can be wrong. Harmless replies are checked separately for new problems.",
};

export function searchHelp(label, kind) {
  return h("span", { title: SEARCH_HELP[kind], tabindex: 0, "aria-label": `${label}. ${SEARCH_HELP[kind]}`,
    style: { cursor: "help", textDecoration: "underline dotted", textUnderlineOffset: ".2em" } }, label);
}

// Rendering only: search state and cancellation belong to the chapter that owns the model controls.
export function searchResults(onSelect) {
  const note = h("p", { class: "small muted", role: "status" });
  const tbody = h("tbody");
  const table = h("table", { class: "ablation-search-table" },
    h("thead", {}, h("tr", {}, ["Layer", searchHelp("Opening proxy ↓", "opening"), searchHelp("Harmless change ↓", "nats"), searchHelp("Reply check", "reply"), ""].map((s) => h("th", { scope: "col" }, s)))), tbody);
  const examples = h("div", { class: "ablation-search-examples" });
  const el = h("div", { class: "ablation-search-results" }, note,
    h("div", { style: { overflowX: "auto" } }, table), examples);
  const rows = new Map();
  let best = null, baseline = null, validation = [], controls = [], ready = false;

  function render() {
    tbody.replaceChildren(...[...rows.values()].sort((a, b) => a.layer - b.layer).map((row) => {
      const state = row.error ? row.error : row.late_layer ? "Too close to output"
        : !row.eligible ? "Exceeds harmless-change limit"
        : row.refusal_replies ? `${row.unresolved}/${baseline?.active ?? "?"} refused, unclear or incomplete${row.control_regressions ? ` · ${row.control_regressions} new harmless-control issue(s)` : ""}`
        : "Not verified";
      return h("tr", { class: row.layer === best ? "selected" : "" },
        h("th", { scope: "row" }, String(row.layer + 1)),
        h("td", {}, row.proxy == null ? "—" : `${(row.proxy * 100).toFixed(1)}%`),
        h("td", {}, row.control_kl == null ? "—" : searchHelp(`${row.control_kl.toFixed(3)} nats`, "nats")),
        h("td", {}, state),
        h("td", {}, row.refusal_replies && h("button", {
          type: "button", class: "btn", disabled: !ready,
          onclick: () => { best = row.layer; onSelect(row.layer); render(); },
        }, row.layer === best ? "Selected" : "Use layer")));
    }));
  }
  function showReplies(title, result) {
    const responses = [...result.refusal_replies, ...result.control_replies];
    return h("details", {}, h("summary", {}, title),
      ...responses.map((reply, i) => h("div", { class: "ablation-search-reply" },
        h("p", { class: "small" }, `${i < validation.length ? "Refusal validation" : "Harmless control"}: ${[...validation, ...controls][i]}`),
        reply.judgement && h("p", { class: "small" }, `Local judge: ${reply.judgement.label} · ${(100 * reply.judgement.relative_confidence).toFixed(0)}% relative label probability (not calibrated confidence)`),
        h("p", { class: "small muted" }, reply.flagged ? `Refusal wording: ${reply.matches.join("; ")}` : "No recognised refusal wording (not a compliance judgement).",
          reply.incomplete ? " Incomplete reply; cannot count as a successful reduction." : ""),
        h("div", { class: "answer" }, reply.text))));
  }
  return {
    el,
    reset(v, c) {
      validation = v; controls = c; baseline = null; best = null; ready = false;
      rows.clear(); examples.replaceChildren(); tbody.replaceChildren(); note.textContent = "";
    },
    clear() { rows.clear(); el.hidden = true; },
    selection(layer) { best = layer; render(); },
    event(ev) {
      el.hidden = false;
      if (ev.type === "progress") {
        note.textContent = ev.stage === "fit" ? ev.message
          : ev.stage === "baseline" ? `Normal replies: ${ev.index} of ${ev.total}…`
          : ev.stage === "scan" ? `Scanning layer ${ev.layer + 1} (${ev.index} of ${ev.total})…`
          : `Verifying layer ${ev.layer + 1}: reply ${ev.index} of ${ev.total}, candidate ${ev.candidate} of ${ev.candidates}…`;
      } else if (ev.type === "baseline") {
        baseline = ev;
        examples.append(showReplies(`Normal replies · ${ev.active}/${ev.total} judged refusals`, ev));
      } else if (ev.type === "candidate" || ev.type === "verified") {
        rows.set(ev.layer, ev);
        if (ev.type === "verified") examples.append(showReplies(`Layer ${ev.layer + 1} · inspect validation replies`, ev));
        render();
      } else if (ev.type === "done") {
        ev.rows.forEach((row) => rows.set(row.layer, row));
        best = ev.best_layer;
        ready = true;
        note.textContent = `${ev.message}${best !== null ? ` Selected layer ${best + 1}.` : ""}`;
        render();
      }
    },
    stop(message) { note.textContent = message; ready = true; render(); },
  };
}
