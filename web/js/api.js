// Talks to the model server. By default that is the server that served this page; add
// ?server=https://… to the address (or set it from the status menu) to use a remote GPU.

const KEY = "lab-server";

export function serverUrl() {
  const q = new URLSearchParams(location.search).get("server");
  if (q) return q.replace(/\/$/, "");
  try { return (localStorage.getItem(KEY) || "").replace(/\/$/, ""); } catch { return ""; }
}

export function setServerUrl(url) {
  try { url ? localStorage.setItem(KEY, url) : localStorage.removeItem(KEY); } catch {}
}

export async function post(path, body, { signal } = {}) {
  const res = await fetch(serverUrl() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const data = await res.json().catch(() => ({ error: `Server answered ${res.status}` }));
  if (!res.ok || data.error) throw new Error(data.error || data.detail || `Server answered ${res.status}`);
  return data;
}

export async function get(path) {
  const res = await fetch(serverUrl() + path);
  if (!res.ok) throw new Error(`Server answered ${res.status}`);
  return res.json();
}

// Streams /api/generate. onEvent receives {text} pieces and finally {done}. Returns when the stream ends.
export async function stream(body, onEvent, signal) {
  const res = await fetch(serverUrl() + "/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || data.detail || `Server answered ${res.status}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).replace(/^data: /, "");
        buf = buf.slice(i + 2);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.error) throw new Error(ev.error);
        onEvent(ev);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") throw e;
  }
}

let infoCache = null;
export async function info(force = false) {
  if (!infoCache || force) infoCache = await get("/api/info");
  return infoCache;
}
