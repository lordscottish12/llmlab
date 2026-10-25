"""Download GloVe (6B tokens, 300 dimensions) and keep the 50,000 most frequent plain words.

Writes data/glove.npy (unit-length vectors, float16) and data/glove_vocab.json.
"""

import json
import os
import re

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"
KEEP = 50_000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    weights = load_file(hf_hub_download(REPO, "0_WordEmbeddings/model.safetensors"))
    matrix = next(iter(weights.values()))
    with open(hf_hub_download(REPO, "0_WordEmbeddings/whitespacetokenizer_config.json"), encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]
    offset = matrix.shape[0] - len(vocab)  # a padding row, if there is one, comes first
    print(f"{len(vocab):,} words, matrix {matrix.shape}, offset {offset}")

    words, rows = [], []
    for i, w in enumerate(vocab):  # GloVe's vocabulary is ordered by frequency
        if re.fullmatch(r"[a-z][a-z'\-]*", w):
            words.append(w)
            rows.append(i + offset)
            if len(words) == KEEP:
                break
    m = matrix[rows].astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "glove.npy"), m.astype(np.float16))
    with open(os.path.join(OUT, "glove_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(words, f)

    idx = {w: i for i, w in enumerate(words)}
    v = m[idx["king"]] - m[idx["man"]] + m[idx["woman"]]
    sims = m @ (v / np.linalg.norm(v))
    print("king - man + woman ->", [words[i] for i in np.argsort(-sims)[:5]])


if __name__ == "__main__":
    main()
