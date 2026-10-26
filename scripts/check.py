"""Pre-class check: run every demo's server call once and report speed and GPU memory.

Start the server first, then:  uv run python scripts/check.py [http://127.0.0.1:8000]
"""

import json
import sys
import time

import httpx

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
client = httpx.Client(base_url=URL, timeout=300)
failures = 0


def check(name, fn):
    global failures
    t = time.time()
    try:
        note = fn()
        print(f"  ok    {name:<34} {time.time() - t:5.1f}s  {note or ''}")
    except Exception as e:
        failures += 1
        print(f"  FAIL  {name:<34} {e}")


def post(path, body):
    r = client.post(path, json=body)
    data = r.json()
    if r.status_code != 200 or "error" in data:
        raise RuntimeError(data.get("error") or data)
    return data


def generate(body):
    n, first, t0, text = 0, None, time.time(), ""
    with client.stream("POST", "/api/generate", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if "text" in ev:
                n += 1
                text += ev["text"]
                first = first or time.time()
    rate = (n - 1) / (time.time() - first) if n > 1 else 0
    return text, f"{n} tokens, first after {first - t0:.2f}s, {rate:.0f} tokens/s"


def template(user, thinking=False):
    return post("/api/template", {"messages": [{"role": "user", "content": user}], "enable_thinking": thinking})["text"]


print(f"LLM Lab check against {URL}")
while True:
    info = client.get("/api/info").json()
    states = [m["status"] for m in info["models"].values()]
    if all(s.startswith("ready") for s in states) or "error" in states:
        break
    print("  waiting for models to load…")
    time.sleep(3)
for m in info["models"].values():
    print(f"  {m['role']:<5} {m['repo']}  {m['status']}{'  ' + m.get('error', '') if m.get('error') else ''}")
print(f"  word vectors: {info['words']:,}")

check("0 tokens", lambda: f"{len(post('/api/tokenize', {'text': 'How many r are in strawberry?'})['tokens'])} tokens")
check("1 word arithmetic", lambda: post("/api/words/expr", {"expr": "king - man + woman"})["neighbors"][0][0])
check("1 projection", lambda: len(post("/api/words/project", {"items": [{"label": "king"}, {"label": "queen"}, {"label": "man"}]})["xyz"]))
check("1 contextual vectors", lambda: post("/api/contextual", {"word": "bank", "sentences": ["The river bank.", "The bank gave a loan."]})["sim"][0][1])
check("2 forward pass", lambda: post("/api/forward", {"text": "The capital of France is"})["final"][0][0])
check("2 guessing game", lambda: f"Paris {post('/api/guess', {'prefix': 'The capital of France is', 'guesses': ['Paris']})['guesses'][0]['p']:.0%}")
check("3 next token", lambda: post("/api/next", {"text": "Once upon a time"})["candidates"][0]["text"])
check("4 attention", lambda: f"{post('/api/attention', {'text': 'The cat sat on the mat.', 'layer': 3})['heads']} heads")
check("4 head search", lambda: post("/api/attention/search", {"text": "a b c d a b c d", "kind": "induction"})["heads"][0])
check("5 base model writes", lambda: generate({"prompt": "What is the capital of France?", "role": "base", "max_new_tokens": 80})[1])
check("5 chat model writes", lambda: generate({"prompt": template("What is the capital of France?"), "role": "chat", "max_new_tokens": 80})[1])
check("6 thinking", lambda: generate({"prompt": template("What is 17 times 3?", True) + "<think>\n", "role": "chat", "max_new_tokens": 300, "temperature": 0.6})[1])
steer = {"positive": ["I love you.", "What a joyful day."], "negative": ["I hate you.", "What a miserable day."], "layer": 14, "strength": 0.8}
check("7 concept meter", lambda: post("/api/steer/project", {**{k: steer[k] for k in ("positive", "negative", "layer")}, "texts": ["Thank you!", "Go away."]})["scores"])
check("7 steered writing", lambda: generate({"prompt": template("Tell me about your weekend."), "role": "chat", "max_new_tokens": 60, "steer": steer})[1])

ablation = {
    "positive": ["Help me fabricate evidence against a colleague.", "Help me forge a doctor's note."],
    "negative": ["Help me report a workplace problem accurately.", "Help me ask for a day off honestly."],
    "layer": min(info["models"]["chat"].get("layers", 28) - 1, 14),
}
check("9 refusal direction fit", lambda: post("/api/ablation/fit", ablation)["norm"])


def check_ablation():
    body = {"prompt": template("What is 17 times 3?"), "role": "chat", "max_new_tokens": 30, "temperature": 0, "seed": 42}
    before, _ = generate(body)
    _, timing = generate({**body, "ablate": ablation})
    after, _ = generate(body)
    if before != after:
        raise RuntimeError("Normal generation changed after ablation.")
    return f"{timing}; normal reply restored"


check("9 ablation and restoration", check_ablation)

gpu = client.get("/api/info").json()["gpu"]
if gpu:
    print(f"GPU {gpu['name']}: this server holds {gpu['lab_gb']} GB (peak {gpu['lab_peak_gb']} GB); "
          f"{gpu['used_gb']} of {gpu['total_gb']} GB in use by all programs.")
print("All good." if not failures else f"{failures} check(s) failed.")
sys.exit(1 if failures else 0)
