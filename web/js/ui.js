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

export const INVISIBLE = /(\p{Default_Ignorable_Code_Point})/u;
const hex = (n, w) => n.toString(16).toUpperCase().padStart(w, "0");

// Spaces as dots, newlines as ↵, invisible characters by their code point (U+FE0F).
function appendText(el, text) {
  for (const part of text.split(/( +|\n)/)) {
    if (!part) continue;
    if (/^ +$/.test(part)) el.append(h("span", { class: "sp" }, "·".repeat(part.length)));
    else if (part === "\n") el.append(h("span", { class: "sp" }, "↵"));
    else for (const bit of part.split(INVISIBLE)) {
      if (!bit) continue;
      if (INVISIBLE.test(bit)) el.append(h("span", { class: "cp" }, `U+${hex(bit.codePointAt(0), 4)}`));
      else el.append(bit);
    }
  }
}

// A token that holds only part of a character: its bytes in hex, any plain ASCII byte as text.
function appendBytes(el, bytes) {
  for (const b of bytes) {
    if (b >= 0x20 && b < 0x7f) appendText(el, String.fromCharCode(b));
    else el.append(h("span", { class: "byte" }, hex(b, 2)));
  }
}

export function tokSpan(text, attrs = {}, bytes = null) {
  const el = h("span", { class: "tok", ...attrs });
  if (bytes) appendBytes(el, bytes);
  else appendText(el, text);
  return el;
}

export function svgEl(tag, attrs = {}, ...children) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) el.append(c instanceof Node ? c : document.createTextNode(c));
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
// Optional: an item's `label` replaces its percentage; dim(k) greys out row k ("everything else" is row items.length).
export function probBars(items, { onPick, rest, scale, highlight, dim } = {}) {
  const max = scale ?? Math.max(...items.map((i) => i.p), 1e-9);
  const rows = items.map((it, k) => {
    const el = h(onPick ? "button" : "div", {
      class: `prob${onPick ? " clickable" : ""}${highlight === k ? " hl" : ""}${dim?.(k) ? " dim" : ""}`,
      type: onPick ? "button" : null,
      title: onPick ? "Append this token" : null,
      onclick: onPick ? () => onPick(it, k) : null,
    },
      h("span", { class: "word" }, showTok(it.text)),
      h("span", { class: "track" }, h("span", { class: "bar", style: { width: `${(it.p / max) * 100}%` } })),
      h("span", { class: "pct" }, it.label ?? pct(it.p)));
    return el;
  });
  if (rest != null && rest > 0.0005) {
    rows.push(h("div", { class: `prob rest${dim?.(items.length) ? " dim" : ""}` },
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

// With toggle, clicking the active preset again switches it off and calls onPick(null).
export function presets(list, onPick, { active, toggle = false } = {}) {
  const wrap = h("div", { class: "chips" });
  for (const p of list) {
    const b = h("button", { class: "chip-btn", type: "button", "aria-pressed": p === active ? "true" : "false" }, p.name);
    b.addEventListener("click", () => {
      const off = toggle && b.getAttribute("aria-pressed") === "true";
      wrap.querySelectorAll(".chip-btn").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", String(!off));
      onPick(off ? null : p);
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

// Native title tooltips are drawn in the operating system's theme, not the page's. This shows them
// in the page's own tooltip instead: each title moves to data-tip on first hover (and again whenever
// the code sets a new title).
export function themedTitles() {
  let el = null, timer = 0, at = null;
  const hide = () => { clearTimeout(timer); el = null; tip(null); };
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest?.("[title], [data-tip]");
    if (t?.hasAttribute("title")) { t.dataset.tip = t.getAttribute("title"); t.removeAttribute("title"); }
    if (t === el) return;
    hide();
    if (!t?.dataset.tip) return;
    el = t; at = e;
    timer = setTimeout(() => { if (el === t) tip(esc(t.dataset.tip).replace(/\n/g, "<br>"), at); }, 350);
  });
  document.addEventListener("pointermove", (e) => {
    if (!el) return;
    at = e;
    if (tipEl && !tipEl.hidden) tip(esc(el.dataset.tip).replace(/\n/g, "<br>"), e);
  });
  document.addEventListener("pointerout", (e) => { if (el && !el.contains(e.relatedTarget)) hide(); });
  document.addEventListener("pointerdown", hide);
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
