import { h, themedTitles } from "./ui.js";
import { info, serverUrl, setServerUrl } from "./api.js";
import llull from "./modules/llull.js";
import tokens from "./modules/tokens.js";
import embeddings from "./modules/embeddings.js";
import forward from "./modules/forward.js";
import generate from "./modules/generate.js";
import attention from "./modules/attention.js";
import chat from "./modules/chat.js";
import reasoning from "./modules/reasoning.js";
import steering from "./modules/steering.js";
import ablation from "./modules/ablation.js";
import features from "./modules/features.js";
import onenumber from "./modules/onenumber.js";
import reading from "./modules/reading.js";

const MODULES = [llull, tokens, embeddings, forward, generate, attention, chat, reasoning, steering, ablation, features, onenumber, reading];

const nav = document.getElementById("nav");
const main = document.getElementById("main");
MODULES.forEach((m, i) => {
  nav.append(h("li", {}, h("a", { href: `#${m.id}` }, h("b", {}, m.num ?? String(i).padStart(2, "0")), h("span", {}, m.title))));
});

// The landing page, shown when the address names no chapter: the image, the title, and a way in.
function cover() {
  document.body.classList.add("cover-mode");
  main.replaceChildren(h("section", { class: "cover" },
    h("img", { src: "img/library-of-babel.webp", alt: "Borges's Library of Babel: endless hexagonal galleries of books, joined by spiral stairs" }),
    h("h1", { class: "brand" }, "LLM Lab"),
    h("a", { class: "btn primary", href: `#${MODULES[0].id}` }, "Continue")));
  document.title = "LLM Lab";
}

let cleanup = null;
function route() {
  try { cleanup?.(); } catch (e) { console.error(e); }
  cleanup = null;
  if (!location.hash.slice(1)) { cover(); return; }
  document.body.classList.remove("cover-mode");
  const id = location.hash.slice(1);
  const m = MODULES.find((x) => x.id === id) || MODULES[0];
  nav.querySelectorAll("a").forEach((a) => a.setAttribute("aria-current", a.getAttribute("href") === `#${m.id}` ? "page" : "false"));
  main.replaceChildren();
  cleanup = m.mount(main) || null;
  document.title = `${m.title} · LLM Lab`;
  scrollTo({ top: 0 });
}
addEventListener("hashchange", route);
route();
themedTitles();

// ---------------------------------------------------------------- theme and projector mode
const root = document.documentElement;
const themeButtons = document.querySelectorAll("[data-set-theme]");
const currentTheme = () => root.dataset.theme || "light"; // Day unless Night was chosen
const syncTheme = () => themeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setTheme === currentTheme())));
themeButtons.forEach((b) => b.addEventListener("click", () => {
  root.dataset.theme = b.dataset.setTheme;
  try { localStorage.setItem("lab-theme", b.dataset.setTheme); } catch {}
  syncTheme();
}));
syncTheme();

const present = document.getElementById("present");
const syncPresent = () => present.setAttribute("aria-pressed", String(root.classList.contains("present")));
present.addEventListener("click", () => {
  root.classList.toggle("present");
  try { localStorage.setItem("lab-present", root.classList.contains("present") ? "1" : "0"); } catch {}
  syncPresent();
});
syncPresent();

// The "math, roughly" and "for discussion" boxes: hidden unless switched on (and remembered).
const notesBtn = document.getElementById("notes");
const syncNotes = () => notesBtn.setAttribute("aria-pressed", String(root.classList.contains("show-notes")));
notesBtn.addEventListener("click", () => {
  root.classList.toggle("show-notes");
  try { localStorage.setItem("lab-notes", root.classList.contains("show-notes") ? "1" : "0"); } catch {}
  syncNotes();
});
syncNotes();

// ---------------------------------------------------------------- server status
const status = document.getElementById("status");
const modelinfo = document.getElementById("modelinfo");
async function poll() {
  let next = 4000;
  try {
    const d = await info(true);
    const ms = Object.values(d.models);
    const ready = ms.every((m) => m.status.startsWith("ready"));
    const failed = ms.some((m) => m.status === "error");
    status.className = `status ${failed ? "bad" : ready ? "ok" : "busy"}`;
    const where = serverUrl() ? new URL(serverUrl()).host : "local";
    status.querySelector("span").textContent = failed ? "A model failed to load" : ready ? `Models ready · ${where}` : "Loading models…";
    status.title = ms.map((m) => `${m.role}: ${m.repo} — ${m.status}${m.error ? `\n${m.error}` : ""}`).join("\n") +
      (d.gpu ? `\nGPU: ${d.gpu.name}, ${d.gpu.used_gb} of ${d.gpu.total_gb} GB in use` : "\nNo GPU") + "\n\nClick to change server.";
    modelinfo.textContent = ms.map((m) => m.label).join(" · ");
    if (ready) next = 20000;
  } catch {
    status.className = "status bad";
    status.querySelector("span").textContent = "No server";
    status.title = `Cannot reach ${serverUrl() || "the local server"}. Click to change server.`;
  }
  setTimeout(poll, next);
}
poll();

status.addEventListener("click", () => {
  const url = prompt("Model server address (empty = the server this page came from):", serverUrl());
  if (url === null) return;
  setServerUrl(url.trim());
  location.reload();
});
