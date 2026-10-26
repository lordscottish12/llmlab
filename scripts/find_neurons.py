"""Find the single numbers for the "One number" chapter, for a base model, and save them to data/neurons/.

1. The super weight (Yu et al. 2024, "The Super Weight in Large Language Models"): one weight in an early
   layer's MLP output projection that creates a huge activation on the first token. Found with one forward
   pass: the layer whose down_proj input spikes, and in it the input neuron and output dimension of the spike.

2. Language-specific neurons (Tang et al., ACL 2024, "Language-Specific Neurons: The Key to Multilingual
   Capabilities in Large Language Models"; their LAPE method): for every MLP neuron, how often it fires on
   text in each language. Neurons that fire often, but almost only for one or two languages, are kept per
   language, with their typical activation there, so that the chapter can push them to that level.

Usage:
    uv run python scripts/find_neurons.py                         # Qwen3-1.7B-Base
    uv run python scripts/find_neurons.py --model Qwen/Qwen3-4B-Base
"""

import argparse
import gzip
import json
import os
import time
import urllib.request

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "neurons")
# One validation shard of multilingual C4 (mC4) per language: {code: (name, number of shards)}. Swiss German
# is not in mC4; it comes from the Alemannic Wikipedia (mostly Swiss German dialects), shards = None.
LANGUAGES = {"en": ("English", 128), "fr": ("French", 16), "de": ("German", 16), "es": ("Spanish", 16),
             "it": ("Italian", 8), "ru": ("Russian", 32), "zh": ("Chinese", 2), "ja": ("Japanese", 8),
             "gsw": ("Swiss German", None)}


def wikipedia_als(n: int) -> list[str]:
    """Articles from the Alemannic Wikipedia, through the Hugging Face dataset viewer (plain JSON, 100 rows a
    page), cached in data/neurons/ so that a rerun does not fetch them again."""
    cache = os.path.join(OUT, "text-gsw.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    out = []
    for offset in range(0, 30_000, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.als&split=train&offset={offset}&length=100"
        for attempt in range(5):  # the viewer API sometimes answers 502 for a moment
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    rows = json.load(r)["rows"]
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        out += [row["row"]["text"] for row in rows if len(row["row"]["text"]) > 1500]
        if len(out) >= n * 2:
            break
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def mc4(code: str, shards: int):
    path = hf_hub_download("allenai/c4", f"multilingual/c4-{code}-validation.tfrecord-00000-of-{shards:05d}.json.gz", repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def texts(tok, code: str, shards: int | None, n: int, length: int) -> list[list[int]]:
    out = []
    for doc in (wikipedia_als(n) if shards is None else mc4(code, shards)):
        ids = tok(doc, add_special_tokens=False).input_ids
        if len(ids) >= length:
            out.append(ids[:length])
            if len(out) == n:
                break
    return out


@torch.no_grad()
def super_weight(model, tok) -> dict:
    spikes = []
    def hook(l):
        def f(mod, inp, out):
            x = inp[0][0].float().abs()  # (tokens, intermediate)
            y = out[0].float().abs()  # (tokens, hidden)
            spikes.append((float(x.max()), l, int(x.max(0).values.argmax()), int(y.max(0).values.argmax())))
        return f
    hooks = [layer.mlp.down_proj.register_forward_hook(hook(l)) for l, layer in enumerate(model.model.layers)]
    try:
        model(tok("My favourite philosopher is Immanuel Kant, who lived in Königsberg.", return_tensors="pt").input_ids.to(model.device))
    finally:
        for h in hooks:
            h.remove()
    spike, layer, col, row = max(spikes)  # the layer whose down_proj input spikes
    W = model.model.layers[layer].mlp.down_proj.weight
    return {"layer": layer, "row": row, "col": col, "value": round(float(W[row, col]), 5),
            "typical": round(float(W.float().abs().median()), 5), "spike": round(spike, 1)}


@torch.no_grad()
def language_neurons(model, tok, n: int, length: int, top_share: float, batch: int = 16) -> dict:
    layers = model.model.layers
    L, I = len(layers), model.config.intermediate_size
    fired = {}  # language -> (L, I) count of tokens on which each neuron fired
    level = {}  # language -> (L, I) sum of its down_proj input when it fired
    current = {}
    def hook(l):
        def f(mod, inp):
            x = inp[0].float()  # (batch, tokens, intermediate): what the neuron passes on
            on = x > 0
            current["fired"][l] += on.sum((0, 1))
            current["level"][l] += (x * on).sum((0, 1))
        return f
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(hook(l)) for l, layer in enumerate(layers)]
    tokens = {}
    try:
        for code, (name, shards) in LANGUAGES.items():
            current["fired"] = torch.zeros(L, I, device=model.device)
            current["level"] = torch.zeros(L, I, device=model.device)
            data = texts(tok, code, shards, n, length)
            for b in range(0, len(data), batch):
                model(torch.tensor(data[b : b + batch], device=model.device), use_cache=False)
            tokens[code] = len(data) * length
            fired[code], level[code] = current["fired"].clone(), current["level"].clone()
            print(f"  {name}: {len(data)} passages", flush=True)
    finally:
        for h in hooks:
            h.remove()

    codes = list(LANGUAGES)
    p = torch.stack([fired[c] / tokens[c] for c in codes])  # (languages, L, I): how often each neuron fires
    q = p / p.sum(0, keepdim=True).clamp_min(1e-12)
    entropy = -(q * q.clamp_min(1e-12).log()).sum(0)  # (L, I): low = fires for few languages
    # LAPE: among neurons that fire often for at least one language, the lowest-entropy share.
    often = p.max(0).values > p.flatten().quantile(0.95)
    cand = entropy.masked_fill(~often, float("inf"))
    k = int(top_share * L * I)
    chosen = cand.flatten().topk(k, largest=False).indices
    out = {}
    for i, code in enumerate(codes):
        thresh = p[i].flatten().quantile(0.95)
        rows = []
        for idx in chosen.tolist():
            l, j = divmod(idx, I)
            if p[i, l, j] > thresh:
                rows.append([l, j, round(float(level[code][l, j] / fired[code][l, j].clamp_min(1)), 4)])
        out[code] = {"name": LANGUAGES[code][0], "neurons": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--passages", type=int, default=300)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--share", type=float, default=0.01, help="share of all neurons kept as language-specific (LAPE: 1%%)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    sw = super_weight(model, tok)
    print(f"super weight: layers[{sw['layer']}].mlp.down_proj.weight[{sw['row']}, {sw['col']}] = {sw['value']} (typical {sw['typical']})")
    langs = language_neurons(model, tok, args.passages, args.length, args.share)
    for code, d in langs.items():
        print(f"  {d['name']}: {len(d['neurons'])} neurons")
    path = os.path.join(OUT, args.model.split("/")[-1] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "super_weight": sw, "languages": langs,
                   "corpus": f"allenai/c4 multilingual validation, {args.passages} x {args.length} tokens per language"}, f)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
