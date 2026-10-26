# LLM Lab

Interactive classroom demos of how a large language model works, run on a real model. Written
for an audience without a technical background (it was made for a philosophy class): each demo
shows the mechanism, explains the math in a few plain sentences, and ends with questions
for discussion.

| # | Chapter | What it shows |
|---|------|---------------|
| 1 | Tokens | Text is cut into tokens and replaced by numbers |
| 2 | Embeddings | Words as vectors (GloVe); `king − man + woman`; meaning axes; the same token in different sentences inside the LLM |
| 3 | One prediction | A full forward pass in 3D: the residual stream rising through every layer, the logit lens, and the output distribution |
| 4 | Writing text | The generation loop, one click per token; greedy vs. random; temperature; forcing a token and branching |
| 5 | Attention | A draggable query/key toy, then real attention arcs for every head, and a search for heads that show a pattern |
| 6 | Chat vs. document | Base model vs. chat model; the raw chat template; system prompts; prefilling |
| 7 | Reasoning | A thinking model's scratchpad, streamed; pause, edit, inject a thought, and let it continue or answer |
| 8 | Steering | A concept direction from contrasting examples, a "concept meter", and adding the direction while the model writes |

## Running it

Needs [uv](https://docs.astral.sh/uv/) and, for reasonable speed, an NVIDIA GPU.

```sh
uv sync                                   # Python 3.12, PyTorch with CUDA, transformers
uv run python scripts/prepare_glove.py    # word vectors for chapter 2 (one-off, ~1 GB download)
uv run python -m server                   # then open http://127.0.0.1:8000
```

The first start downloads the models from Hugging Face. Both models load in the background;
the status indicator at the top right turns green when they are ready.

### Choosing models

The demos use two models of the same family: a **base** model (pre-trained only) and its
**chat** sibling (post-trained, with a thinking mode). Qwen3 is used because it is a standard
transformer, so attention can be shown in every layer. The newer Qwen3.5 small models replace
three of every four attention layers with linear attention, which would break chapter 5.

Pick a profile with `LLMLAB_PROFILE`:

| profile | for | base | chat | GPU memory |
|---|---|---|---|---|
| `laptop` (default) | 8 GB GPU, e.g. RTX 3070 laptop | Qwen3-1.7B-Base | Qwen3-4B, 4-bit | ~6 GB |
| `desktop` | 24 GB GPU, e.g. RTX 3090 | Qwen3-4B-Base | Qwen3-4B | ~17 GB |
| `server` | 48 GB+ cloud GPU | Qwen3-8B-Base | Qwen3-14B | ~46 GB |
| `cpu` | no GPU | Qwen3-0.6B-Base | Qwen3-0.6B | — |

Or override a single role: `LLMLAB_CHAT=Qwen/Qwen3-8B:4bit uv run python -m server`.

PowerShell: `$env:LLMLAB_PROFILE = "desktop"; uv run python -m server`

### Running the models on a cloud GPU (RunPod)

The same server runs on a GPU pod. On the pod:

```sh
git clone https://github.com/lordscottish12/llmlab.git && cd llmlab
uv sync && uv run python scripts/prepare_glove.py
LLMLAB_PROFILE=server LLMLAB_HOST=0.0.0.0 uv run python -m server
```

Expose HTTP port 8000 on the pod and open `https://<pod-id>-8000.proxy.runpod.net`. To keep
the page on your laptop but use the pod's models, click the status indicator and enter that
address, or open `http://127.0.0.1:8000/?server=https://<pod-id>-8000.proxy.runpod.net`.
Clearing the address switches back to the local models, which makes the laptop the fallback.

## Before class

With the server running, `uv run python scripts/check.py` calls every demo once and reports
speed and GPU memory. On an RTX 3090 with the `laptop` profile: base model ~60 tokens/s, chat
model ~40 tokens/s, 6.9 GB of GPU memory at peak. If an 8 GB card runs out of memory, use the
smaller chat model (`LLMLAB_CHAT=Qwen/Qwen3-1.7B`) or a shorter context (`LLMLAB_MAX_CONTEXT=2048`).

Generation records each decoding step as a CUDA graph and replays it, which is 5-6x faster
than plain PyTorch on Windows. If text ever comes out garbled, set `LLMLAB_GRAPHS=0`.

## In class

- **Projector** (top right) enlarges all text. **Day / Night** switches the theme.
- Every demo has preset examples; they are chosen to show the effect reliably on the small models.
- All numbers are computed live by the model. Sampling is random, so results vary from run to run.

## Layout

```
server/      FastAPI app (app.py), model wrapper with hooks (lm.py), word vectors, profiles (config.py)
web/         the page: plain ES modules, no build step; three.js vendored in web/vendor
scripts/     one-off data preparation
data/        generated word vectors (not committed)
```
