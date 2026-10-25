// Small DOM helpers shared by the demos.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function debounce(fn, ms = 250) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// Visible whitespace for token pieces: a leading space shows as a faint dot, a newline as ↵.
export function showTok(s) {
  return s.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/ /g, "·");
}

export function tokSpan(text, attrs = {}) {
  const el = h("span", { class: "tok", ...attrs });
  for (const part of text.split(/( +|\n)/)) {
    if (!part) continue;
    if (/^ +$/.test(part)) el.append(h("span", { class: "sp" }, "·".repeat(part.length)));
    else if (part === "\n") el.append(h("span", { class: "sp" }, "↵"));
    else el.append(part);
  }
  return el;
}

export function intro(label, title, lede) {
  return h("section", { class: "intro" },
    h("span", { class: "label" }, label),
    h("h2", {}, title),
    lede && h("p", { class: "lede" }, lede));
}

// The two boxes under every demo: the math in plain words, and questions for discussion.
export function notes({ math, discuss }) {
  return h("div", { class: "notes" },
    h("section", {},
      h("span", { class: "label" }, "The math, roughly"),
      ...math.map((m) => typeof m === "string" ? h("p", { html: m }) : h("div", { class: "formula", html: m.f }))),
    h("section", {},
      h("span", { class: "label" }, "For discussion"),
      h("ul", {}, discuss.map((d) => h("li", { html: d })))));
}

export function box(title, { right, body, cls = "" } = {}) {
  const b = h("div", { class: "body" });
  if (body) b.append(...[].concat(body));
  const el = h("div", { class: `box ${cls}` }, title != null && h("header", {}, h("h3", {}, title), right), b);
  el.body = b;
  return el;
}

export function pct(p) {
  if (p >= 0.995) return "100%";
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`;
  if (p >= 0.01) return `${(p * 100).toFixed(1)}%`;
  if (p >= 0.001) return `${(p * 100).toFixed(2)}%`;
  return "<0.1%";
}

// A ranked list of next tokens with bars. items: [{text, p}]
export function probBars(items, { onPick, rest, scale, highlight } = {}) {
  const max = scale ?? Math.max(...items.map((i) => i.p), 1e-9);
  const rows = items.map((it, k) => {
    const el = h(onPick ? "button" : "div", {
      class: `prob${onPick ? " clickable" : ""}${highlight === k ? " hl" : ""}`,
      type: onPick ? "button" : null,
      title: onPick ? "Append this token" : null,
      onclick: onPick ? () => onPick(it, k) : null,
    },
      h("span", { class: "word" }, showTok(it.text)),
      h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${(it.p / max) * 100}%` } })),
      h("span", { class: "pct" }, pct(it.p)));
    return el;
  });
  if (rest != null && rest > 0.0005) {
    rows.push(h("div", { class: "prob rest" },
      h("span", { class: "word" }, "everything else"),
      h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${Math.min(1, rest / max) * 100}%` } })),
      h("span", { class: "pct" }, pct(rest))));
  }
  return h("div", { class: `probs${highlight != null ? " hlmode" : ""}` }, rows);
}

export function slider(label, { min, max, step, value, fmt = (v) => v, oninput }) {
  const out = h("output", {}, fmt(value));
  const input = h("input", { type: "range", min, max, step, value });
  input.addEventListener("input", () => { out.textContent = fmt(+input.value); oninput?.(+input.value); });
  const el = h("label", { class: "slider" }, h("span", {}, label), input, out);
  el.input = input;
  el.set = (v) => { input.value = v; out.textContent = fmt(+v); };
  return el;
}

export function field(label, control) {
  return h("label", { class: "field" }, h("span", {}, label), control);
}

export function presets(list, onPick, { active } = {}) {
  const wrap = h("div", { class: "chips" });
  for (const p of list) {
    const b = h("button", { class: "chip-btn", type: "button", "aria-pressed": p === active ? "true" : "false" }, p.name);
    b.addEventListener("click", () => {
      wrap.querySelectorAll(".chip-btn").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      onPick(p);
    });
    wrap.append(b);
  }
  return wrap;
}

let tipEl;
export function tip(html, ev) {
  if (!tipEl) { tipEl = h("div", { class: "tip", hidden: true }); document.body.append(tipEl); }
  if (!html) { tipEl.hidden = true; return; }
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const pad = 14, r = tipEl.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tipEl.style.left = `${x}px`;
  tipEl.style.top = `${y}px`;
}

export function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Diverging colour for a vector value: blue below zero, grey at zero, red above.
function hexRgb(hex) {
  const m = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}
export function divergingColor(v, lim = 2.5) {
  const t = Math.max(-1, Math.min(1, v / lim));
  const z = hexRgb(cssVar("--zero")), end = hexRgb(cssVar(t < 0 ? "--neg" : "--pos"));
  const a = Math.abs(t) ** 0.8;
  return z.map((c, i) => Math.round(c + (end[i] - c) * a));
}

export function errorLine(el, e) {
  el.replaceChildren(h("p", { class: "error" }, e.message || String(e)));
}

export function busy(btn, on, text) {
  if (on) { btn.dataset.label = btn.textContent; btn.disabled = true; if (text) btn.textContent = text; }
  else { btn.disabled = false; if (btn.dataset.label) btn.textContent = btn.dataset.label; }
}
