"""Static word vectors (GloVe, 300 dimensions) for the embedding demo.

GloVe gives each word one fixed vector learned from co-occurrence counts. That is not what a
modern LLM uses internally, but it is the cleanest way to show that geometry can carry meaning.
The contextual demo in lm.py shows the LLM's own, context-dependent vectors.
"""

import json
import os
import re

import numpy as np

from .config import DATA_DIR
from .lm import pca3


class WordVectors:
    def __init__(self):
        self.words: list[str] = []
        self.index: dict[str, int] = {}
        self.unit: np.ndarray | None = None

    def load(self):
        path = os.path.join(DATA_DIR, "glove.npy")
        if not os.path.exists(path):
            raise RuntimeError("Word vectors missing: run `uv run python scripts/prepare_glove.py` first.")
        self.unit = np.load(path).astype(np.float32)
        with open(os.path.join(DATA_DIR, "glove_vocab.json"), encoding="utf-8") as f:
            self.words = json.load(f)
        self.index = {w: i for i, w in enumerate(self.words)}

    @property
    def ready(self) -> bool:
        return self.unit is not None

    def vec(self, word: str) -> np.ndarray:
        w = word.strip().lower()
        if w not in self.index:
            raise KeyError(f'"{word}" is not among the {len(self.words):,} words in the vocabulary.')
        return self.unit[self.index[w]]

    def parse(self, expr: str) -> list[tuple[int, str]]:
        """'king - man + woman' -> [(+1, 'king'), (-1, 'man'), (+1, 'woman')]"""
        terms = []
        for sign, word in re.findall(r"([+-]?)\s*([^\s+-]+)", expr):
            terms.append((-1 if sign == "-" else 1, word.lower()))
        if not terms:
            raise ValueError("Type words joined by + and -, for example: king - man + woman")
        return terms

    def evaluate(self, expr: str) -> tuple[np.ndarray, list[tuple[int, str]]]:
        terms = self.parse(expr)
        v = sum(s * self.vec(w) for s, w in terms)
        return v, terms

    def neighbors(self, v: np.ndarray, k: int = 10, exclude: set[str] = frozenset()) -> list[list]:
        sims = self.unit @ (v / (np.linalg.norm(v) or 1))
        order = np.argpartition(-sims, k + len(exclude) + 1)[: k + len(exclude) + 1]
        order = order[np.argsort(-sims[order])]
        out = [[self.words[i], round(float(sims[i]), 3)] for i in order if self.words[i] not in exclude]
        return out[:k]

    def similarity(self, a: str, b: str) -> float:
        return float(self.vec(a) @ self.vec(b))

    def project(self, items: list[dict], method: str = "pca", axes: list[list[str]] | None = None) -> list:
        """items: [{"label": ..., "expr": ...}]. Returns a 3D point per item, in the same order."""
        vs = []
        for it in items:
            v, _ = self.evaluate(it.get("expr") or it["label"])
            vs.append(v / (np.linalg.norm(v) or 1))
        x = np.stack(vs)
        if method == "axes" and axes:
            cols = []
            for a, b in axes[:3]:
                d = self.vec(b) - self.vec(a)
                cols.append(x @ (d / np.linalg.norm(d)))
            while len(cols) < 3:
                cols.append(np.zeros(len(x)))
            y = np.stack(cols, axis=1)
            y = y - y.mean(axis=0, keepdims=True)
            y = y / (np.abs(y).max() or 1)
            return np.round(y, 4).tolist()
        return np.round(pca3(x), 4).tolist()
