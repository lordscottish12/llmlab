"""Fit a Jacobian lens for a model and save it to data/jlens/.

The Jacobian lens (Anthropic, 2026, "Verbalizable Representations Form a Global Workspace in Language
Models") reads an intermediate residual vector h_l by first carrying it into the last layer's basis with
the average Jacobian, then decoding it with the model's own final norm and unembedding:

    lens_l(h) = unembed( J_l @ h ),   J_l = E[ d h_final / d h_l ]

The estimator follows the reference code (github.com/anthropics/jacobian-lens, jlens/fitting.py): for
each output dimension, a one-hot cotangent at every valid target position at once, one backward pass,
and the gradient averaged over source positions. So row i of J_l is how much a nudge to h_l moves
dimension i of the final vector, summed over the positions after it and averaged over positions. The
corpus is generic web text (a C4 validation shard), 128 tokens per passage.

Usage:
    uv run python scripts/fit_jlens.py                                  # Qwen3-1.7B-Base, 200 passages
    uv run python scripts/fit_jlens.py --model Qwen/Qwen3-4B-Base --prompts 300

Cost per passage: one forward pass and d_model / dim_batch backward passes. Progress is saved every few
passages, and a rerun resumes where it stopped.
"""

import argparse
import gzip
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jlens")


def passages(tok, n: int, length: int):
    """The first `n` C4 validation texts that are at least `length` tokens long, cut to `length` tokens."""
    path = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ids = tok(json.loads(line)["text"], add_special_tokens=False).input_ids
            if len(ids) >= length:
                out.append(ids[:length])
                if len(out) == n:
                    break
    return out


def jacobians_for(model, ids: list[int], dim_batch: int, skip_first: int) -> torch.Tensor:
    """J for one passage: (source states, d, d). Source state 0 is the embedding, state s the output of layer s-1."""
    layers = model.model.layers
    n_src = len(layers)  # embedding plus every layer but the last
    d = model.config.hidden_size
    dev = model.model.embed_tokens.weight.device

    states: list[torch.Tensor] = []
    def keep(_, __, out):
        h = out[0] if isinstance(out, tuple) else out
        states.append(h)
    def leaf(_, __, out):
        out = out.detach().requires_grad_(True)  # gradients flow from here on, without touching the weights
        states.append(out)
        return out
    hooks = [model.model.embed_tokens.register_forward_hook(leaf)] + [l.register_forward_hook(keep) for l in layers]
    try:
        x = torch.tensor([ids] * dim_batch, device=dev)
        model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    sources, target = states[:n_src], states[n_src]

    valid = torch.arange(skip_first, len(ids) - 1, device=dev)  # skip the attention-sink start and the last position
    J = torch.zeros(n_src, d, d)
    batch = torch.arange(dim_batch, device=dev)
    n_pass = (d + dim_batch - 1) // dim_batch
    for k in range(n_pass):
        start = k * dim_batch
        m = min(dim_batch, d - start)
        cot = torch.zeros_like(target)
        cot[batch[:m, None], valid[None, :], start + batch[:m, None]] = 1.0
        grads = torch.autograd.grad(target, sources, grad_outputs=cot, retain_graph=k < n_pass - 1)
        for s, g in enumerate(grads):
            J[s, start : start + m] = g[:m, valid].float().mean(1).cpu()
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--skip-first", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    name = args.model.split("/")[-1]
    final, ckpt = os.path.join(OUT, f"{name}.pt"), os.path.join(OUT, f"{name}.partial.pt")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    texts = passages(tok, args.prompts, args.length)
    print(f"{len(texts)} passages of {args.length} tokens")

    total, done = None, 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt)
        total, done = c["sum"], c["count"]
        print(f"resuming after {done} passages")
    t0, start = time.time(), done
    for i in range(done, len(texts)):
        J = jacobians_for(model, texts[i], args.dim_batch, args.skip_first)
        total = J if total is None else total + J
        done = i + 1
        per = (time.time() - t0) / (done - start)
        print(f"{done}/{len(texts)}  {per:.1f} s per passage, about {per * (len(texts) - done) / 60:.0f} min left", flush=True)
        if done % 10 == 0 or done == len(texts):
            torch.save({"sum": total, "count": done}, ckpt)

    torch.save({"model": args.model, "J": (total / done).half(), "passages": done, "length": args.length,
                "corpus": "allenai/c4 en validation"}, final)
    os.remove(ckpt)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
