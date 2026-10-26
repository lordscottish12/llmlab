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

const MODULES = [llull, tokens, embeddings, forward, generate, attention, chat, reasoning, steering];

const nav = document.getElementById("nav");
const main = document.getElementById("main");
MODULES.forEach((m, i) => {
  nav.append(h("li", {}, h("a", { href: `#${m.id}` }, h("b", {}, m.num ?? String(i).padStart(2, "0")), h("span", {}, m.title))));
});

let cleanup = null;
function route() {
  const id = location.hash.slice(1) || MODULES[0].id;
  const m = MODULES.find((x) => x.id === id) || MODULES[0];
  nav.querySelectorAll("a").forEach((a) => a.setAttribute("aria-current", a.getAttribute("href") === `#${m.id}` ? "page" : "false"));
  try { cleanup?.(); } catch (e) { console.error(e); }
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
const dark = matchMedia("(prefers-color-scheme: dark)");
const currentTheme = () => root.dataset.theme || (dark.matches ? "dark" : "light");
const syncTheme = () => themeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setTheme === currentTheme())));
themeButtons.forEach((b) => b.addEventListener("click", () => {
  root.dataset.theme = b.dataset.setTheme;
  try { localStorage.setItem("lab-theme", b.dataset.setTheme); } catch {}
  syncTheme();
}));
dark.addEventListener("change", syncTheme);
syncTheme();

const present = document.getElementById("present");
const syncPresent = () => present.setAttribute("aria-pressed", String(root.classList.contains("present")));
present.addEventListener("click", () => {
  root.classList.toggle("present");
  try { localStorage.setItem("lab-present", root.classList.contains("present") ? "1" : "0"); } catch {}
  syncPresent();
});
syncPresent();

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
