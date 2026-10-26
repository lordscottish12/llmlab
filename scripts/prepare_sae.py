"""Prepare sparse autoencoders (SAEs) for the features chapter.

Uses Qwen's own SAEs for Qwen3-1.7B-Base (Qwen-Scope: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50): one per
layer, reading the residual stream after that layer, 32,768 features of which 50 are active per token.
For each chosen layer this script stores the SAE at half precision in data/sae/<model>/layer<n>.pt, and
describes every feature by its own evidence on generic web text (C4): how often it fires, the tokens it
fires on most, and the snippets where it fires most strongly. Qwen-Scope ships no feature labels.

Usage:
    uv run python scripts/prepare_sae.py                       # layers 7, 14, 21; 2000 passages
    uv run python scripts/prepare_sae.py --layers 14 --passages 500
"""

import argparse
import collections
import functools
import gzip
import json
import os
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from fit_jlens import passages

MODEL = "Qwen/Qwen3-1.7B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
K = 50  # active features per token
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sae", MODEL.split("/")[-1])
EXAMPLES = 6  # snippets kept per feature
SKIP = 4  # leading tokens of each passage left out: the first ones carry outsized, text-independent values


class Stop(Exception):
    """Raised by the hook to end a forward pass once the layer we need has run."""


def describe(model, tok, sae, layer: int, texts: list[list[int]], batch: int = 16) -> dict:
    W_enc, b_enc = sae["W_enc"], sae["b_enc"]
    got = {}
    def hook(_, __, out):
        got["h"] = out[0] if isinstance(out, tuple) else out
        raise Stop
    handle = model.model.layers[layer].register_forward_hook(hook)
    feats, vals, where = [], [], []  # one entry per (token, active feature)
    try:
        for b in range(0, len(texts), batch):
            x = torch.tensor(texts[b : b + batch], device=W_enc.device)
            try:
                with torch.no_grad():
                    model(input_ids=x, use_cache=False)
            except Stop:
                pass
            h = got["h"][:, SKIP:].float()  # (batch, n, d)
            pre = h.to(W_enc.dtype) @ W_enc.T + b_enc
            v, i = pre.float().topk(K, dim=-1)  # (batch, n, K)
            keep = v > 0
            n = h.shape[1]
            tokpos = (torch.arange(b, b + x.shape[0], device=v.device)[:, None] * 10_000 + SKIP + torch.arange(n, device=v.device)[None, :])
            feats.append(i[keep].int().cpu().numpy())
            vals.append(v[keep].half().cpu().numpy())
            where.append(tokpos[:, :, None].expand_as(v)[keep].int().cpu().numpy())
    finally:
        handle.remove()
    feats, vals, where = np.concatenate(feats), np.concatenate(vals).astype(np.float32), np.concatenate(where)

    d_sae = W_enc.shape[0]
    n_tokens = len(texts) * (len(texts[0]) - SKIP)
    order = np.lexsort((-vals, feats))  # by feature, strongest first
    feats, vals, where = feats[order], vals[order], where[order]
    starts = np.searchsorted(feats, np.arange(d_sae + 1))
    piece = functools.lru_cache(maxsize=None)(lambda t: tok.decode([t]))  # the same few thousand tokens recur
    out = {"freq": [], "tokens": [], "examples": []}
    for f in range(d_sae):
        a, z = starts[f], starts[f + 1]
        out["freq"].append(round((z - a) / n_tokens, 6))
        top = where[a : min(z, a + 200)]
        common = collections.Counter(piece(texts[w // 10_000][w % 10_000]).strip() for w in top)
        out["tokens"].append([t for t, _ in common.most_common(4) if t])
        ex, used = [], set()
        for w, val in zip(where[a:z], vals[a:z]):
            p, q = w // 10_000, w % 10_000
            if p in used:  # one snippet per passage, for variety
                continue
            used.add(p)
            ids = texts[p]
            ex.append({"pre": tok.decode(ids[max(0, q - 14) : q]), "tok": piece(ids[q]), "post": tok.decode(ids[q + 1 : q + 5]), "act": round(float(val), 2)})
            if len(ex) == EXAMPLES:
                break
        out["examples"].append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    ap.add_argument("--passages", type=int, default=2000)
    ap.add_argument("--length", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    texts = passages(tok, args.passages, args.length)
    print(f"{len(texts)} passages of {args.length} tokens")
    for layer in args.layers:
        t0 = time.time()
        raw = torch.load(hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt"), map_location="cpu")
        sae = {k: v.to(torch.float16).cuda() for k, v in raw.items()}
        del raw
        torch.save({k: v.cpu() for k, v in sae.items()} | {"k": K, "layer": layer, "repo": SAE_REPO}, os.path.join(OUT, f"layer{layer}.pt"))
        info = describe(model, tok, sae, layer, texts)
        info |= {"layer": layer, "k": K, "repo": SAE_REPO, "corpus": f"allenai/c4 en validation, {len(texts)} x {args.length} tokens"}
        with gzip.open(os.path.join(OUT, f"layer{layer}.features.json.gz"), "wt", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
        alive = sum(1 for x in info["freq"] if x > 0)
        print(f"layer {layer}: {alive:,} of {len(info['freq']):,} features fire on the corpus; {time.time() - t0:.0f} s", flush=True)
        del sae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
