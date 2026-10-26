"""A thin wrapper around a Hugging Face causal LM that exposes its internals for the demos.

Everything here works on raw text. Chat templates are applied by the caller (see app.py), so the
demos can show that a "chat" is just a text document with a particular format.
"""

import hashlib
import math
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import MAX_CONTEXT, USE_GRAPHS, ModelSpec
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
            self.status = f"ready ({time.time() - t0:.0f}s)"
        except Exception as e:  # surfaced in /api/info so the UI can say what went wrong
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            raise

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
    def forward_pass(self, text: str, top: int = 10, slice_dims: int = 64) -> dict:
        self.require()
        with self.lock:
            ids = self._ids(text)
            with self._capture() as states:
                self.model(input_ids=ids, use_cache=False)
            hs = torch.stack([s.float() for s in states])  # (L+1, n, d)
            L1, n, d = hs.shape
            # Scale each vector by its typical entry (median size), so that the few huge entries some
            # models carry do not wash out all the others in the picture.
            normed = hs / (hs.abs().median(-1, keepdim=True).values * 1.4826).clamp_min(1e-6)
            # A fixed spread of dimensions, so the same tile always shows the same coordinates.
            dims = torch.linspace(0, d - 1, slice_dims).long().to(hs.device)
            lens_rows, entropies = [], []
            for l in range(L1):
                probs = torch.softmax(self.lens(hs[l]), -1)  # (n, V)
                p, i = probs.topk(top, -1)
                lens_rows.append((p.cpu().numpy(), i.cpu().numpy()))
                entropies.append((-(probs * probs.clamp_min(1e-12).log2()).sum(-1)).cpu().numpy())
            # Arrows: each vector as the logit lens reads it (after the final norm), projected onto the
            # output directions of its own column's top two final predictions. The second direction is made
            # perpendicular to the first, so the two axes span a proper plane.
            top2 = torch.as_tensor(lens_rows[-1][1][:, :2], device=hs.device)  # (n, 2)
            wa, wb = self.model.lm_head.weight[top2[:, 0]].float(), self.model.lm_head.weight[top2[:, 1]].float()
            e1 = wa / wa.norm(dim=-1, keepdim=True)
            e2 = wb - (wb * e1).sum(-1, keepdim=True) * e1
            e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            nh = self.model.model.norm(hs.to(self.model.model.norm.weight.dtype)).float()  # (L+1, n, d)
            arrows = torch.stack([(nh * e1).sum(-1), (nh * e2).sum(-1)], -1).cpu().numpy()  # (L+1, n, 2)
            axes = top2.cpu().tolist()
            del nh
            final = torch.softmax(self.lens(hs[-1][-1]), -1)
            entropy = float(-(final * final.clamp_min(1e-12).log2()).sum())
            fp, fi = final.topk(30)
            traj = pca3(normed.reshape(L1 * n, d).cpu().numpy()).reshape(L1, n, 3)
            vals = np.round(normed[:, :, dims].cpu().numpy(), 2)  # copy to the CPU once, not per tile
            norms = hs.norm(dim=-1).cpu().numpy()
            del hs, normed, final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the logit lens briefly needs a lot of memory; give it back

        tokens = [self.piece(i) for i in ids[0].tolist()]
        result = {
            "tokens": tokens,
            "layers": L1 - 1,
            "hidden": d,
            "slice_dims": dims.tolist(),
            "cells": [
                [
                    {
                        "v": vals[l, pos].tolist(),
                        "norm": round(float(norms[l, pos]), 1),
                        "xyz": np.round(traj[l, pos], 3).tolist(),
                        "lens": [[self.piece(int(t)), round(float(pr), 4)] for pr, t in zip(*[a[pos] for a in lens_rows[l]])],
                        "arrow": [round(float(v), 2) for v in arrows[l, pos]],
                        "entropy": round(float(entropies[l][pos]), 3),
                    }
                    for pos in range(n)
                ]
                for l in range(L1)
            ],
            "final": [[self.piece(int(t)), float(pr)] for pr, t in zip(fp.tolist(), fi.tolist())],
            "axes": [[self.piece(a), self.piece(b)] for a, b in axes],  # per column: the tokens its arrows are projected onto
            "entropy": entropy,
        }
        return result

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
        }

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

    @torch.no_grad()
    def mean_state(self, texts: list[str], layer: int) -> torch.Tensor:
        acc = []
        for t in texts:
            with self.lock:
                with self._capture() as states:
                    self.model(input_ids=self._ids(t), use_cache=False)
            s = states[layer + 1].float()
            # The first token carries a few huge values that have nothing to do with the text; skip it.
            acc.append((s[1:] if len(s) > 1 else s).mean(0))
        return torch.stack(acc).mean(0)

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
        out = []
        for t in texts:
            out.append(float(self.mean_state([t], layer) @ unit))
        return out

    def _install_steering(self):
        """Every layer adds a row of a buffer to its output while a generation runs.

        The buffer is all zeros unless a steering vector is set. The hooks stay installed, so the
        addition is part of the recorded CUDA graph and steering costs nothing extra to switch on.
        """
        cfg = self.model.config
        self._steer_buf = torch.zeros(cfg.num_hidden_layers, cfg.hidden_size, device=self.device, dtype=self.model.model.norm.weight.dtype)
        self._steer_on = False

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

    def _set_steer(self, vec: torch.Tensor | None, layer: int):
        self._steer_buf.zero_()
        if vec is not None:
            self._steer_buf[layer] = vec.to(self._steer_buf.dtype)

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
    ):
        """Yield the continuation of `prompt` piece by piece. One forward pass per new token."""
        self.require()
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
