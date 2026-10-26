"""Sparse autoencoders (SAEs) for the features chapter.

Qwen's own SAEs for Qwen3-1.7B-Base (Qwen-Scope), prepared by scripts/prepare_sae.py into data/sae/<model>/:
the weights at half precision, and a description of every feature (how often it fires, its most common
tokens, its strongest snippets of web text).

An SAE rewrites a residual vector h as a sum of learned directions ("features"), of which only the k that
fit best are active:

    acts = top_k( W_enc h + b_enc )        h ≈ W_dec acts + b_dec
"""

import gzip
import json
import os
import threading

import torch

from .config import DATA_DIR


class SAEs:
    def __init__(self, lm):
        self.lm = lm  # the model the SAEs were trained on
        self.dir = os.path.join(DATA_DIR, "sae", lm.spec.repo.split("/")[-1])
        self._weights: dict[int, dict] = {}
        self._features: dict[int, dict] = {}
        self._lock = threading.Lock()

    def layers(self) -> list[int]:
        if not os.path.isdir(self.dir):
            return []
        return sorted(int(f[5:-3]) for f in os.listdir(self.dir) if f.startswith("layer") and f.endswith(".pt"))

    def weights(self, layer: int) -> dict:
        """One layer's SAE on the GPU. Only one is kept there at a time: each takes about 256 MB."""
        if layer not in self.layers():
            raise ValueError(f"No SAE for layer {layer}. Run scripts/prepare_sae.py first.")
        with self._lock:
            if layer not in self._weights:
                self._weights.clear()
                torch.cuda.empty_cache()
                w = torch.load(os.path.join(self.dir, f"layer{layer}.pt"))
                self._weights[layer] = {k: w[k].to(self.lm.device) for k in ("W_enc", "b_enc", "W_dec", "b_dec")} | {"k": w["k"]}
            return self._weights[layer]

    def features(self, layer: int) -> dict:
        with self._lock:
            if layer not in self._features:
                with gzip.open(os.path.join(self.dir, f"layer{layer}.features.json.gz"), "rt", encoding="utf-8") as f:
                    self._features[layer] = json.load(f)
            return self._features[layer]

    def describe(self, layer: int, feature: int) -> dict:
        f = self.features(layer)
        return {"id": feature, "freq": f["freq"][feature], "tokens": f["tokens"][feature], "examples": f["examples"][feature]}

    @torch.no_grad()
    def encode(self, text: str, layer: int) -> dict:
        """The active features at every token of `text`, strongest first, with a short description of each."""
        lm = self.lm
        lm.require()
        w = self.weights(layer)
        ids = lm._ids(text)
        got = {}
        hook = lm.layers[layer].register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            with lm.lock:
                lm.model(input_ids=ids, use_cache=False)
        finally:
            hook.remove()
        pre = got["h"][0].to(w["W_enc"].dtype) @ w["W_enc"].T + w["b_enc"]  # (n, features)
        vals, idx = pre.float().topk(w["k"], dim=-1)
        acts = [[[int(i), round(float(v), 2)] for v, i in zip(vr, ir) if v > 0] for vr, ir in zip(vals.tolist(), idx.tolist())]
        f = self.features(layer)
        seen = {i for row in acts for i, _ in row}
        return {
            "tokens": [lm.piece(t) for t in ids[0].tolist()],
            "acts": acts,
            "labels": {i: {"tokens": f["tokens"][i], "freq": f["freq"][i]} for i in seen},
        }

    def direction(self, layer: int, feature: int) -> torch.Tensor:
        """The feature's decoder direction: what it adds to the residual stream when it is active."""
        return self.weights(layer)["W_dec"][:, feature].float()
