# LLM Lab

Interactive classroom demos of how a large language model works, run on a real model. Written
for an audience without a technical background (it was made for a philosophy class): each demo
shows the mechanism, explains the math in a few plain sentences, and ends with questions
for discussion.

| # | Chapter | What it shows |
|---|------|---------------|
| Intro | Llull's machine | Borges on Ramon Llull's thinking machine: three disks to turn by hand, loaded three ways |
| 1 | Tokens | Text is cut into tokens and replaced by numbers |
| 2 | Embeddings | Words as vectors (GloVe); `king − man + woman`; meaning axes; the same token in different sentences inside the LLM |
| 3 | One prediction | A full forward pass in 3D: the residual stream rising through every layer, the logit lens, and the output distribution |
| 4 | Writing text | The generation loop, one click per token; greedy vs. random; temperature; forcing a token and branching |
| 5 | Attention | A draggable query/key toy, then real attention arcs for every head, and a search for heads that show a pattern |
| 6 | Chat vs. document | Base model vs. chat model; the raw chat template; system prompts; prefilling |
| 7 | Reasoning | A thinking model's scratchpad, streamed; pause, edit, inject a thought, and let it continue or answer |
| 8 | Steering | A concept direction from contrasting examples, a "concept meter", and adding the direction while the model writes |
| 9 | Ablation | Fit a candidate refusal direction, remove it during generation, and compare normal, ablated, and random-control replies |
| 10 | Features | A sparse autoencoder takes a vector apart into a few dozen features; each feature with the web text that switches it on, and turned up while the model writes |
| 11 | One number | The super weight: one weight out of 1.7 billion that the model cannot do without; and neurons for a language, pushed until the model switches to it |
| → | Reading list | Papers, essays and videos behind each chapter, with a philosophy section and “start here” picks |

## Running it

Needs [uv](https://docs.astral.sh/uv/) and, for reasonable speed, an NVIDIA GPU.

```sh
uv sync                                   # Python 3.12, PyTorch with CUDA, transformers
uv run python scripts/prepare_glove.py    # word vectors for chapter 2 (one-off, ~1 GB download)
uv run python -m server                   # then open http://127.0.0.1:8000
```

The first start downloads the models from Hugging Face. Both models load in the background;
the status indicator at the top right turns green when they are ready.

Three optional extras, each prepared once per machine (they go into `data/`, which is not in git):

```sh
uv run python scripts/fit_jlens.py        # Jacobian lens for chapter 3 (~27 s per passage on an RTX 3090; 150 passages ≈ 70 min)
uv run python scripts/prepare_sae.py      # Qwen-Scope SAEs + feature descriptions for chapter 10 (~1.5 GB download)
uv run python scripts/find_neurons.py     # super weight + language neurons for chapter 11 (~270 MB of text)
```

The **Jacobian lens** (Anthropic, 2026) is a refinement of the logit lens: it carries each layer's
vector into the last layer's terms before reading it. Once fitted, chapter 3 offers a
"Logit lens / J-lens" switch. The fit for one model can be copied to another machine
(`data/jlens/<model>.pt`). The **SAEs** are Qwen's own (Qwen-Scope) and exist for
Qwen3-1.7B-Base, the laptop profile's base model.

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

### Refusal-direction ablation (chapter 9)

Open **Ablation**, then **Compare replies**. Qwen3-4B defaults to **layer 20**, the classroom
preset chosen after trying the demo; no layer search is needed. The first run fits a direction from 16 pairs of
editable requests; subsequent runs reuse the recorded activations. No extra download or
preparation is required. The current chat model supplies its own non-thinking template.
The layer slider selects where to *read* the direction; the intervention removes it from
the embeddings and every attention/MLP contribution throughout the model while it generates.
Weights are never changed. Both replies use greedy decoding; the optional random direction
provides a control for nonspecific disruption.

Start with a prompt the normal model actually refuses, then try the philosophy and arithmetic
controls. Other checkpoints start with an estimated layer, and the small fitting set also contains differences
in topic and wording: separation of the training examples is not proof of a refusal mechanism.
Effects vary with the checkpoint, quantisation, examples, and extraction layer. This adapts
[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) to a classroom experiment on Qwen3;
it does not reproduce the paper's full selection procedure or evaluation.

Use the **Impersonation** preset to begin, or choose another request. Read the entire reply,
since a changed opening phrase can still lead to a refusal or a benign redirection.
The separate philosophy and arithmetic prompts provide quick checks for collateral changes.

**Search layers** is an optional experiment for comparing the preset with other directions. Edit the two
validation lists to match the task; the API rejects overlap with fitting examples. The search
generates baseline replies, scans every layer using an English refusal-opening token proxy,
and measures next-token KL divergence across the first eight positions of the normal harmless
replies, using the same continuation for each candidate. It verifies the best three eligible
layers with generated replies by default. Under **Search settings**, choose eight finalists or
**All eligible layers** for a slower, broader search, change the reply length, or adjust the
harmless-change limit (default 0.5 nats). Replies get up to 512 tokens by default so long
warnings do not use up the budget before the substantive answer. The final fifth of layers is excluded from automatic
selection to reduce direct interference with output-token directions.

The unmodified local chat model judges each generated reply as an answer, refusal/redirection,
or unclear, using the request and the full available reply. This catches some polite redirections
that a string-only detector misses, but remains a fallible self-evaluation. The selected candidate
must improve the judged answer count on requests the baseline refused, without new judged
refusals, unclear answers, or incomplete replies on the harmless controls. Truncated replies
cannot count as successful reductions; ties favour lower harmless KL. If nothing improves, the
current layer is kept. Progress, scores, judge decisions, matched wording, and baseline/finalist replies are shown;
**Stop search** cancels between model passes. Inspect the replies, then try new test prompts:
neither token probabilities, string matches, nor self-judgements establish meaningful compliance, and next-token
KL is only a limited check on preserved abilities. The three-finalist mode can miss a better
layer; the selected layer is best only under the chosen heuristic and validation set.

`uv run python -m unittest discover -s tests -v` checks the projection, fitting position,
validation, search ranking, cancellation, restoration, and CUDA graph switching on a tiny random Qwen3,
without downloading weights. Restart an already running server after updating the code.

## In class

- **Projector** (top right) enlarges all text. **Day / Night** switches the theme.
- Every demo has preset examples; they are chosen to show the effect reliably on the small models.
- All numbers are computed live by the model. Most generation demos sample randomly; ablation uses greedy decoding for repeatable comparisons.

## Layout

```
server/      FastAPI app (app.py), model wrapper with hooks (lm.py), word vectors, profiles (config.py)
web/         the page: plain ES modules, no build step; three.js vendored in web/vendor
scripts/     one-off data preparation
data/        generated word vectors (not committed)
```
