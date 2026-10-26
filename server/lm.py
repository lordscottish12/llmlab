"""A thin wrapper around a Hugging Face causal LM that exposes its internals for the demos.

Everything here works on raw text. Chat templates are applied by the caller (see app.py), so the
demos can show that a "chat" is just a text document with a particular format.
"""

import hashlib
import json
import math
import os
import random
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import DATA_DIR, MAX_CONTEXT, USE_GRAPHS, ModelSpec
from .decode import Decoder


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pca3(x: np.ndarray) -> np.ndarray:
    """Project rows of x onto their first three principal components, scaled to roughly [-1, 1]."""
    x = x - x.mean(axis=0, keepdims=True)
    if len(x) < 2:
        return np.zeros((len(x), 3))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:3]
    if comps.shape[0] < 3:
        comps = np.vstack([comps, np.zeros((3 - comps.shape[0], x.shape[1]))])
    y = x @ comps.T
    scale = np.abs(y).max() or 1.0
    return y / scale


def _byte_decoder() -> dict[str, int]:
    """Byte-level BPE vocabularies (GPT-2's, Qwen's) write each byte as a printable character. This maps them back."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


class LM:
    def __init__(self, role: str, spec: ModelSpec):
        self.role = role
        self.spec = spec
        self.tok = None
        self.model = None
        self.jlens = None  # the fitted Jacobian lens, if there is one
        self.neurons = None  # the super weight and language neurons (scripts/find_neurons.py), if found
        self.status = "not loaded"
        self.error: str | None = None
        self.lock = threading.RLock()  # one forward pass at a time per model
        self._load_lock = threading.Lock()
        self._steer_cache: dict[str, torch.Tensor] = {}
        self._attn_cache: dict[str, dict] = {}

    # ---------------------------------------------------------------- loading

    def load_tokenizer(self):
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(self.spec.repo)

    def load(self):
        with self._load_lock:  # the start-up preload and a first request may both ask
            if self.model is None:
                self._load()

    def _load(self):
        self.status = "loading"
        t0 = time.time()
        try:
            self.load_tokenizer()
            kwargs = dict(dtype=torch.bfloat16 if _device() == "cuda" else torch.float32)
            if self.spec.quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
                kwargs["device_map"] = {"": 0}
            # The base model serves the attention demo, which needs the explicit ("eager") attention weights.
            kwargs["attn_implementation"] = "eager" if self.role == "base" else "sdpa"
            model = AutoModelForCausalLM.from_pretrained(self.spec.repo, **kwargs)
            if self.spec.quant is None:
                model.to(_device())
            model.eval()
            self.model = model
            self._install_steering()
            # The base model only continues short documents, so it gets a smaller cache.
            self.decoder = Decoder(model, MAX_CONTEXT if self.role == "chat" else min(MAX_CONTEXT, 2048), USE_GRAPHS)
            self.jlens = self._load_jlens()
            self.neurons = self._load_neurons()
            if self.neurons:
                self._install_neurons()
            self.status = f"ready ({time.time() - t0:.0f}s)"
        except Exception as e:  # surfaced in /api/info so the UI can say what went wrong
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            raise

    def _load_jlens(self):
        """The fitted Jacobian lens for this model (scripts/fit_jlens.py), if there is one: one d x d matrix per
        layer, which carries that layer's vector into the last layer's basis."""
        path = os.path.join(DATA_DIR, "jlens", self.spec.repo.split("/")[-1] + ".pt")
        if not os.path.exists(path):
            return None
        return torch.load(path)["J"].to(self.device, self.model.model.norm.weight.dtype)

    def _load_neurons(self):
        if self.spec.quant:  # a 4-bit model's weights cannot be read one by one
            return None
        path = os.path.join(DATA_DIR, "neurons", self.spec.repo.split("/")[-1] + ".json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def require(self):
        if self.model is None:
            if self.status == "error":
                raise RuntimeError(f"{self.spec.repo} failed to load: {self.error}")
            self.load()

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def device(self):
        return self.model.model.embed_tokens.weight.device

    def info(self) -> dict:
        d = {"role": self.role, "repo": self.spec.repo, "label": self.spec.label, "status": self.status}
        if self.error:
            d["error"] = self.error
        if self.model is not None:
            cfg = self.model.config
            d.update(
                layers=cfg.num_hidden_layers,
                heads=cfg.num_attention_heads,
                hidden=cfg.hidden_size,
                vocab=cfg.vocab_size,
                params=sum(p.numel() for p in self.model.parameters()),
                jlens=self.jlens is not None,
            )
            if self.spec.quant == "4bit":  # 4-bit weights are packed two per byte, so count from the config
                d["params"] = int(d["params"] * 2 - cfg.vocab_size * cfg.hidden_size)
        return d

    # ---------------------------------------------------------------- tokens

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def piece(self, tid: int) -> str:
        """The text of one token. Tokens that hold part of a multi-byte character show as their raw byte form."""
        s = self.tok.decode([tid])
        if "�" in s:
            s = self.tok.convert_ids_to_tokens(tid)
        return s

    def raw_bytes(self, tid: int) -> list[int] | None:
        """The bytes of a token that holds only part of a character (so has no text of its own), else None."""
        if tid in self.tok.all_special_ids:
            return None
        form = self.tok.convert_ids_to_tokens(tid)
        if not all(c in BYTE_DECODER for c in form):
            return None
        raw = bytes(BYTE_DECODER[c] for c in form)
        try:
            raw.decode("utf-8")
            return None
        except UnicodeDecodeError:
            return list(raw)

    def tokenize(self, text: str) -> list[dict]:
        ids = self.encode(text)
        return [{"id": i, "text": self.piece(i), "bytes": self.raw_bytes(i), "special": i in self.tok.all_special_ids} for i in ids]

    # ---------------------------------------------------------------- forward passes

    def _ids(self, text: str) -> torch.Tensor:
        ids = self.encode(text)
        if not ids:
            raise ValueError("Type some text first.")
        return torch.tensor([ids], device=self.device)

    @contextmanager
    def _capture(self):
        """Record the residual stream: the embedding output and the output of every layer."""
        states: list[torch.Tensor] = []
        hooks = [self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: states.append(o[0]))]
        for layer in self.layers:
            hooks.append(
                layer.register_forward_hook(lambda m, i, o: states.append((o[0] if isinstance(o, tuple) else o)[0]))
            )
        try:
            yield states
        finally:
            for h in hooks:
                h.remove()

    def lens(self, h: torch.Tensor) -> torch.Tensor:
        """Logit lens: read a residual state with the model's own final norm and output layer."""
        h = h.to(self.model.model.norm.weight.dtype)
        return self.model.lm_head(self.model.model.norm(h)).float()

    @torch.no_grad()
    def forward_pass(self, text: str, top: int = 10) -> dict:
        self.require()
        with self.lock:
            ids = self._ids(text)
            with self._capture() as states:
                self.model(input_ids=ids, use_cache=False)
            hs = torch.stack([s.float() for s in states])  # (L+1, n, d)
            L1, n, d = hs.shape
            # The logit lens reads every vector as it is. The Jacobian lens first carries each layer's vector
            # into the last layer's basis (J @ h); the last layer needs no carrying.
            logit = self._readout(hs, top)
            jl = None
            if self.jlens is not None:
                carried = torch.stack([hs[l] @ self.jlens[l].float().T if l < len(self.jlens) else hs[l] for l in range(L1)])
                jl = self._readout(carried, top, axes_from=logit)
                del carried
            final = torch.softmax(self.lens(hs[-1][-1]), -1)
            entropy = float(-(final * final.clamp_min(1e-12).log2()).sum())
            fp, fi = final.topk(30)
            norms = hs.norm(dim=-1).cpu().numpy()
            del hs, final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the logit lens briefly needs a lot of memory; give it back

        tokens = [self.piece(i) for i in ids[0].tolist()]
        result = {
            "tokens": tokens,
            "layers": L1 - 1,
            "hidden": d,
            "cells": [
                [
                    {
                        "norm": round(float(norms[l, pos]), 1),
                        **self._cell(logit, l, pos),
                        **({"j": self._cell(jl, l, pos)} if jl else {}),
                    }
                    for pos in range(n)
                ]
                for l in range(L1)
            ],
            "final": [[self.piece(int(t)), float(pr)] for pr, t in zip(fp.tolist(), fi.tolist())],
            "axes": [[self.piece(a), self.piece(b)] for a, b in logit["axes"]],  # per column: the tokens its arrows are projected onto
            "entropy": entropy,
        }
        return result

    def _readout(self, hs: torch.Tensor, top: int, axes_from: dict | None = None) -> dict:
        """Read every vector in hs (L+1, n, d) as a prediction, with the model's own final norm and unembedding.

        Also the arrows: each vector as read here (after the final norm), projected onto the output directions
        of its own column's top two final predictions. The second direction is made perpendicular to the
        first, so the two axes span a proper plane.
        """
        rows, entropies = [], []
        for l in range(hs.shape[0]):
            probs = torch.softmax(self.lens(hs[l]), -1)  # (n, V)
            p, i = probs.topk(top, -1)
            rows.append((p.cpu().numpy(), i.cpu().numpy()))
            entropies.append((-(probs * probs.clamp_min(1e-12).log2()).sum(-1)).cpu().numpy())
        if axes_from is not None:
            top2 = torch.as_tensor(axes_from["axes"], device=hs.device)
        else:
            top2 = torch.as_tensor(rows[-1][1][:, :2], device=hs.device)  # (n, 2)
        wa, wb = self.model.lm_head.weight[top2[:, 0]].float(), self.model.lm_head.weight[top2[:, 1]].float()
        e1 = wa / wa.norm(dim=-1, keepdim=True)
        e2 = wb - (wb * e1).sum(-1, keepdim=True) * e1
        e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        nh = self.model.model.norm(hs.to(self.model.model.norm.weight.dtype)).float()  # (L+1, n, d)
        arrows = torch.stack([(nh * e1).sum(-1), (nh * e2).sum(-1)], -1).cpu().numpy()  # (L+1, n, 2)
        return {"rows": rows, "entropies": entropies, "arrows": arrows, "axes": top2.cpu().tolist()}

    def _cell(self, r: dict, l: int, pos: int) -> dict:
        return {
            "lens": [[self.piece(int(t)), round(float(pr), 4)] for pr, t in zip(*[a[pos] for a in r["rows"][l]])],
            "arrow": [round(float(v), 2) for v in r["arrows"][l, pos]],
            "entropy": round(float(r["entropies"][l][pos]), 3),
        }

    @torch.no_grad()
    def next_token(self, text: str, temperature: float = 1.0, top: int = 12) -> dict:
        self.require()
        with self.lock:
            ids = self._ids(text)
            logits = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
        t = max(temperature, 1e-4)
        probs = torch.softmax(logits / t, -1)
        p, i = probs.topk(top)
        raw = torch.softmax(logits, -1)
        entropy = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
        return {
            "candidates": [
                {"id": int(t_), "text": self.piece(int(t_)), "p": float(pr), "p1": float(raw[t_]), "logit": float(logits[t_])}
                for pr, t_ in zip(p.tolist(), i.tolist())
            ],
            "entropy": entropy,
            "rest": max(0.0, 1.0 - float(p.sum())),
            # One token drawn with no weighting at all, the way Llull's disks combine: every ordinary token in
            # the vocabulary equally likely. Its probability and rank show what the model makes of it.
            "uniform": self._uniform(raw),
        }

    def _uniform(self, raw: torch.Tensor) -> dict:
        # Only tokens that are whole pieces of text: the ~1% that hold part of a character cannot be
        # appended to the text on their own. The list is built once, on first use.
        if not hasattr(self, "_whole"):
            self._whole = [i for i in range(self.tok.vocab_size) if "�" not in self.tok.decode([i])]
        u = random.choice(self._whole)
        return {"id": u, "text": self.piece(u), "p": float(raw[u]), "rank": int((raw > raw[u]).sum()) + 1,
                "of": self.tok.vocab_size, "pool": len(self._whole)}

    @torch.no_grad()
    def continuation_logprob(self, prefix: str, continuation: str) -> dict:
        """How likely the model finds `continuation` right after `prefix`, token by token."""
        self.require()
        pre = self.encode(prefix)
        full = self.encode(prefix + continuation)
        # Tokenization can merge across the boundary; find where the prefix tokens stop matching.
        k = 0
        while k < min(len(pre), len(full)) and pre[k] == full[k]:
            k += 1
        k = max(1, k)
        with self.lock:
            logits = self.model(input_ids=torch.tensor([full], device=self.device), use_cache=False).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        parts = []
        total = 0.0
        for pos in range(k, len(full)):
            lp = float(logp[pos - 1, full[pos]])
            rank = int((logp[pos - 1] > lp).sum()) + 1
            parts.append({"text": self.piece(full[pos]), "logprob": lp, "rank": rank})
            total += lp
        return {"logprob": total, "p": math.exp(total), "bits": -total / math.log(2), "tokens": parts}

    @torch.no_grad()
    def attention(self, text: str) -> dict:
        """All attention weights for a text, cached so the UI can page through layers cheaply."""
        self.require()
        key = hashlib.sha1(text.encode()).hexdigest()
        if key not in self._attn_cache:
            with self.lock:
                ids = self._ids(text)
                if ids.shape[1] > 64:
                    raise ValueError("Keep attention examples under 64 tokens so the picture stays readable.")
                out = self.model(input_ids=ids, output_attentions=True, use_cache=False)
            att = torch.stack([a[0].float() for a in out.attentions]).cpu().numpy()  # (L, H, n, n)
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._attn_cache = {key: {"tokens": [self.piece(i) for i in ids[0].tolist()], "att": att}}
        return self._attn_cache[key]

    @torch.no_grad()
    def contextual(self, sentences: list[str], word: str, layer: int | None) -> dict:
        """The vector the model builds for `word` inside each sentence, at one layer."""
        self.require()
        layer = self.model.config.num_hidden_layers // 2 if layer is None else layer
        vecs, found = [], []
        for s in sentences:
            enc = self.tok(s, add_special_tokens=False, return_offsets_mapping=True)
            start = s.lower().find(word.lower())
            if start < 0:
                found.append(False)
                continue
            end = start + len(word)
            idx = [j for j, (a, b) in enumerate(enc.offset_mapping) if a < end and b > start]
            with self.lock:
                with self._capture() as states:
                    self.model(input_ids=torch.tensor([enc.input_ids], device=self.device), use_cache=False)
            vecs.append(states[layer][idx].float().mean(0).cpu().numpy())
            found.append(True)
        if not vecs:
            raise ValueError(f'None of the sentences contains "{word}".')
        v = np.stack(vecs)
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        return {"layer": layer, "found": found, "sim": np.round(u @ u.T, 3).tolist(), "xyz": np.round(pca3(u), 3).tolist()}

    # ---------------------------------------------------------------- steering

    def mean_state(self, texts: list[str], layer: int) -> torch.Tensor:
        """The residual state after `layer`, averaged over each text's tokens and then over the texts."""
        return self.text_states(texts, layer).mean(0)

    @torch.no_grad()
    def text_states(self, texts: list[str], layer: int, batch: int = 16) -> torch.Tensor:
        """For each text, its residual state after `layer` averaged over its tokens: (texts, hidden).

        Texts go through the model in batches, padded on the right: a token only attends to earlier tokens,
        so the padding leaves the real tokens' states unchanged, and it is left out of the average.
        """
        texts = [t for t in texts if t.strip()]
        if not texts:
            raise ValueError("Type some text first.")
        got: list[torch.Tensor] = []
        hook = self.layers[layer].register_forward_hook(lambda m, i, o: got.append(o[0] if isinstance(o, tuple) else o))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        acc = []
        try:
            for k in range(0, len(texts), batch):
                ids = [self.encode(t)[:MAX_CONTEXT] for t in texts[k : k + batch]]
                n = max(len(i) for i in ids)
                x = torch.tensor([i + [pad] * (n - len(i)) for i in ids], device=self.device)
                mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device=self.device)
                got.clear()
                with self.lock:
                    self.model(input_ids=x, attention_mask=mask, use_cache=False)
                states = got[0].float()  # (batch, n, d)
                for j, i in enumerate(ids):
                    s = states[j, : len(i)]
                    # The first token carries a few huge values that have nothing to do with the text; skip it.
                    acc.append((s[1:] if len(s) > 1 else s).mean(0))
        finally:
            hook.remove()
        return torch.stack(acc)

    def steer_vector(self, positive: list[str], negative: list[str], layer: int) -> torch.Tensor:
        """Difference of mean residual states: "what changes inside the model" between two kinds of text."""
        self.require()
        key = hashlib.sha1(repr((positive, negative, layer)).encode()).hexdigest()
        if key not in self._steer_cache:
            v = self.mean_state(positive, layer) - self.mean_state(negative, layer)
            self._steer_cache[key] = v
        return self._steer_cache[key]

    @torch.no_grad()
    def typical_norm(self, layer: int) -> float:
        key = f"norm{layer}"
        if key not in self._steer_cache:
            with self.lock:
                with self._capture() as states:
                    self.model(input_ids=self._ids("The weather was mild, and people walked along the river."), use_cache=False)
            self._steer_cache[key] = states[layer + 1][1:].float().norm(dim=-1).mean()
        return float(self._steer_cache[key])

    @torch.no_grad()
    def project(self, texts: list[str], vec: torch.Tensor, layer: int) -> list[float]:
        unit = vec / vec.norm()
        return (self.text_states(texts, layer) @ unit).tolist()

    def _install_steering(self):
        """Every layer adds a row of a buffer to its output while a generation runs.

        The buffer is all zeros unless a steering vector is set. The hooks stay installed, so the
        addition is part of the recorded CUDA graph and steering costs nothing extra to switch on.
        """
        cfg = self.model.config
        self._steer_buf = torch.zeros(cfg.num_hidden_layers, cfg.hidden_size, device=self.device, dtype=self.model.model.norm.weight.dtype)
        # Fixed storage: a captured decoding graph must see later changes to the direction.
        self._ablate_buf = torch.zeros(cfg.hidden_size, device=self.device, dtype=torch.float32)
        self._steer_on = False

        def ablate(module, inputs, output):
            if not self._steer_on:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            x = hidden.float()
            projected = (x - (x @ self._ablate_buf).unsqueeze(-1) * self._ablate_buf).to(hidden.dtype)
            return (projected,) + tuple(output[1:]) if isinstance(output, tuple) else projected

        # Project each write into the residual stream: embeddings, attention and MLP outputs.
        # By linearity this keeps the stream perpendicular to r after every sublayer.
        # Hooks run even for the zero direction during generation, so CUDA graphs can switch modes.
        self.model.model.embed_tokens.register_forward_hook(ablate)
        for layer in self.layers:
            layer.self_attn.register_forward_hook(ablate)
            layer.mlp.register_forward_hook(ablate)

        def make(l):
            def hook(module, inputs, output):
                if not self._steer_on:
                    return None
                if isinstance(output, tuple):
                    return (output[0] + self._steer_buf[l],) + tuple(output[1:])
                return output + self._steer_buf[l]
            return hook

        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(make(l))

    def _install_neurons(self):
        """Hooks for the "One number" chapter, on every layer's MLP output projection (down_proj).

        Before it: chosen neurons are clamped to set values (x <- x * (1 - mask) + value), for the language
        neurons. After it, in the super weight's layer: that one weight's contribution is taken away again,
        which is exactly what setting it to zero would do, without touching the model. All buffers are zero
        unless a generation asks for them, and the hooks stay installed, so they live inside the CUDA graph.
        """
        cfg, dt = self.model.config, self.model.model.norm.weight.dtype
        self._clamp_mask = torch.zeros(cfg.num_hidden_layers, cfg.intermediate_size, device=self.device, dtype=dt)
        self._clamp_val = torch.zeros_like(self._clamp_mask)
        self._sw_on = torch.zeros((), device=self.device, dtype=dt)
        sw = self.neurons["super_weight"]

        def clamp(l):
            def hook(module, inputs):
                if not self._steer_on:
                    return None
                x = inputs[0]
                return (x * (1 - self._clamp_mask[l]) + self._clamp_val[l],)
            return hook

        def without_super_weight(module, inputs, output):
            if not self._steer_on:
                return None
            w = module.weight[sw["row"], sw["col"]]
            output[..., sw["row"]] -= self._sw_on * w * inputs[0][..., sw["col"]]
            return output

        for l, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(clamp(l))
        self.layers[sw["layer"]].mlp.down_proj.register_forward_hook(without_super_weight)

    def neuron_plan(self, zero_super: bool = False, language: str | None = None, strength: float = 0.0, mute: str | None = "en",
                    exclude: str | None = None):
        """Buffers for one generation: (mask, values, super weight off). Language neurons are pushed to
        `strength` times their typical level, leaving out any shared with the `exclude` language (Swiss German
        without German: only what sets it apart); the `mute` language's own neurons are switched off."""
        if not self.neurons:
            raise ValueError("No neurons found for this model yet. Run scripts/find_neurons.py.")
        mask, val = torch.zeros_like(self._clamp_mask), torch.zeros_like(self._clamp_val)
        if language and strength:
            langs = self.neurons["languages"]
            if language not in langs:
                raise ValueError(f"Unknown language {language!r}")
            ours = set()
            shared = {(l, j) for l, j, _ in langs[exclude]["neurons"]} if exclude in langs else set()
            for l, j, v in langs[language]["neurons"]:
                if (l, j) in shared:
                    continue
                mask[l, j], val[l, j] = 1, v * strength
                ours.add((l, j))
            if mute and mute != language:
                for l, j, _ in langs[mute]["neurons"]:
                    if (l, j) not in ours:
                        mask[l, j], val[l, j] = 1, 0
        return mask, val, torch.tensor(1.0 if zero_super else 0.0, device=self.device, dtype=mask.dtype)

    def _set_neurons(self, plan):
        if not self.neurons:
            return
        if plan is None:
            self._clamp_mask.zero_(), self._clamp_val.zero_(), self._sw_on.zero_()
        else:
            self._clamp_mask.copy_(plan[0]), self._clamp_val.copy_(plan[1]), self._sw_on.copy_(plan[2])

    def _set_steer(self, vec: torch.Tensor | None, layer: int):
        self._steer_buf.zero_()
        if vec is not None:
            self._steer_buf[layer] = vec.to(self._steer_buf.dtype)

    def _set_ablation(self, direction: torch.Tensor | None):
        self._ablate_buf.zero_()
        if direction is not None:
            self._ablate_buf.copy_(direction)

    # ---------------------------------------------------------------- generation

    def stop_ids(self) -> set[int]:
        ids = set()
        for t in ("<|endoftext|>", "<|im_end|>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        if self.tok.eos_token_id is not None:
            ids.add(self.tok.eos_token_id)
        return ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        steer: tuple[torch.Tensor, int] | None = None,
        stop_strings: list[str] | None = None,
        seed: int | None = None,
        ablate: torch.Tensor | None = None,
        neurons: tuple | None = None,
    ):
        """Yield the continuation of `prompt` piece by piece. One forward pass per new token."""
        self.require()
        if ablate is not None:
            if steer is not None or neurons is not None:
                raise ValueError("Run ablation separately from other interventions.")
            ablate = ablate.to(device=self.device, dtype=torch.float32)
            if ablate.shape != self._ablate_buf.shape or not torch.isfinite(ablate).all() or ablate.norm() < 1e-6:
                raise ValueError("Ablation needs a finite, non-zero direction matching the model width.")
            ablate = ablate / ablate.norm()
        stops = self.stop_ids()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % 2**31)
        vec, layer = steer if steer else (None, 0)
        ids = self.encode(prompt)
        if not ids:
            raise ValueError("Type some text first.")
        max_new_tokens = min(max_new_tokens, self.decoder.max_len - len(ids))
        if max_new_tokens <= 0:
            raise ValueError(f"The text is longer than the {self.decoder.max_len}-token limit.")
        me = object()
        self.decoder.claim(me)
        out_ids: list[int] = []
        emitted = ""
        try:
            with self.lock:
                self._set_steer(vec, layer)
                self._set_ablation(ablate)
                self._set_neurons(neurons)
                self._steer_on = True
                try:
                    logits = self.decoder.prefill(ids).float()
                finally:
                    self._steer_on = False
            for _ in range(max_new_tokens):
                if temperature <= 0:
                    nxt = int(logits.argmax())
                else:
                    probs = torch.softmax(logits / temperature, -1)
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(0) - sp < top_p
                    sp = sp * keep
                    nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])
                if nxt in stops:
                    yield {"done": "stop", "token": self.tok.decode([nxt])}
                    return
                out_ids.append(nxt)
                text = self.tok.decode(out_ids)
                if not text.endswith("�"):
                    delta = text[len(emitted):]
                    emitted = text
                    if delta:
                        yield {"text": delta}
                if stop_strings and any(s in emitted for s in stop_strings):
                    yield {"done": "stop_string"}
                    return
                with self.lock:
                    preempted = not self.decoder.owns(me)
                    if not preempted:
                        self._set_steer(vec, layer)
                        self._set_ablation(ablate)
                        self._set_neurons(neurons)
                        self._steer_on = True
                        try:
                            logits = self.decoder.step(nxt).float()
                        finally:
                            self._steer_on = False
                if preempted:  # never yield while holding the lock: an abandoned generator may never resume
                    yield {"done": "preempted"}
                    return
            yield {"done": "length"}
        finally:
            # No lock here: an abandoned generator is closed by the garbage collector, possibly in a
            # thread that already holds self.lock. Each owner sets the steering buffer before every pass.
            self.decoder.release(me)
