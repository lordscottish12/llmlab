"""HTTP API for the LLM Lab. Run with:  uv run python -m server  (see README)."""

import asyncio
import json
import os
import queue
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .embeddings import WordVectors
from .lm import LM
from .sae import SAEs
from .refusal import RefusalDirections, SearchCancelled

MODELS = {role: LM(role, spec) for role, spec in config.ROLES.items()}
WORDS = WordVectors()
SAES = SAEs(MODELS["base"])  # sparse autoencoders for the base model, if prepared
REFUSAL = RefusalDirections(MODELS["chat"])


@asynccontextmanager
async def lifespan(app):
    try:
        WORDS.load()
    except Exception as e:
        print(f"[llmlab] word vectors: {e}")
    if config.PRELOAD:
        threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="LLM Lab", lifespan=lifespan)
# The page may be served from somewhere else (a laptop) while the models run on a cloud GPU.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    # Make browsers revalidate the page's scripts, so an edited demo shows up on reload.
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _preload():
    for lm in MODELS.values():
        try:
            lm.load()
            print(f"[llmlab] {lm.role}: {lm.spec.repo} {lm.status}")
        except Exception as e:
            print(f"[llmlab] {lm.role}: failed to load {lm.spec.repo}: {e}")


@app.exception_handler(KeyError)
@app.exception_handler(ValueError)
@app.exception_handler(RuntimeError)
async def user_error(request: Request, exc: Exception):
    msg = exc.args[0] if exc.args else str(exc)
    return JSONResponse({"error": str(msg)}, status_code=400)


def lm(role: str) -> LM:
    if role not in MODELS:
        raise HTTPException(404, f"No model role {role!r}")
    return MODELS[role]


# ------------------------------------------------------------------ info


@app.get("/api/info")
def info():
    import torch

    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "used_gb": round((total - free) / 2**30, 1),  # by every program on the card
            "lab_gb": round(torch.cuda.memory_reserved() / 2**30, 1),  # by this server
            "lab_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
            "total_gb": round(total / 2**30, 1),
        }
    return {
        "profile": config.PROFILE,
        "gpu": gpu,
        "models": {r: m.info() for r, m in MODELS.items()},
        "words": len(WORDS.words) if WORDS.ready else 0,
    }


# ------------------------------------------------------------------ tokens


class TextReq(BaseModel):
    text: str
    role: str = "base"


@app.post("/api/tokenize")
def tokenize(r: TextReq):
    m = lm(r.role)
    m.load_tokenizer()  # no need to wait for the model itself
    return {"tokens": m.tokenize(r.text)}


# ------------------------------------------------------------------ static word vectors


class ExprReq(BaseModel):
    expr: str
    k: int = 10
    exclude_inputs: bool = True


@app.post("/api/words/expr")
def words_expr(r: ExprReq):
    v, terms = WORDS.evaluate(r.expr)
    inputs = {w for _, w in terms}
    return {
        "terms": terms,
        "neighbors": WORDS.neighbors(v, r.k, inputs if r.exclude_inputs else set()),
        "with_inputs": WORDS.neighbors(v, 5),
    }


class ProjectReq(BaseModel):
    items: list[dict]
    method: str = "pca"
    axes: list[list[str]] | None = None


@app.post("/api/words/project")
def words_project(r: ProjectReq):
    missing = []
    for it in r.items:
        for _, w in WORDS.parse(it.get("expr") or it["label"]):
            if w not in WORDS.index:
                missing.append(w)
    if missing:
        raise ValueError(f"Not in the vocabulary: {', '.join(sorted(set(missing)))}")
    return {"xyz": WORDS.project(r.items, r.method, r.axes)}


class SimReq(BaseModel):
    words: list[str]


@app.post("/api/words/similarity")
def words_similarity(r: SimReq):
    vs = np.stack([WORDS.vec(w) for w in r.words])
    return {"sim": np.round(vs @ vs.T, 3).tolist(), "values": [np.round(WORDS.vec(w)[:24], 2).tolist() for w in r.words]}


class ContextReq(BaseModel):
    sentences: list[str]
    word: str
    layer: int | None = None
    role: str = "base"


@app.post("/api/contextual")
def contextual(r: ContextReq):
    return lm(r.role).contextual(r.sentences, r.word, r.layer)


# ------------------------------------------------------------------ prediction


@app.post("/api/forward")
def forward(r: TextReq):
    return lm(r.role).forward_pass(r.text)


class NextReq(BaseModel):
    text: str
    temperature: float = 1.0
    top: int = 12
    role: str = "base"


@app.post("/api/next")
def next_token(r: NextReq):
    return lm(r.role).next_token(r.text, r.temperature, min(r.top, 50))


class GuessReq(BaseModel):
    prefix: str
    guesses: list[str]
    role: str = "base"


@app.post("/api/guess")
def guess(r: GuessReq):
    m = lm(r.role)
    out = []
    for g in r.guesses:
        g = g.strip()
        if not g:
            continue
        sep = "" if r.prefix.endswith((" ", "\n")) else " "
        out.append({"guess": g, **m.continuation_logprob(r.prefix, sep + g)})
    return {"guesses": out}


# ------------------------------------------------------------------ attention


class AttnReq(BaseModel):
    text: str
    layer: int = 0
    role: str = "base"


@app.post("/api/attention")
def attention(r: AttnReq):
    a = lm(r.role).attention(r.text)
    att = a["att"]
    layer = max(0, min(r.layer, att.shape[0] - 1))
    return {
        "tokens": a["tokens"],
        "layers": att.shape[0],
        "heads": att.shape[1],
        "layer": layer,
        "weights": np.round(att[layer], 3).tolist(),
    }


class AttnSearchReq(BaseModel):
    text: str
    kind: str = "pair"  # pair | previous | induction | first
    query: int | None = None
    key: int | None = None
    role: str = "base"
    top: int = 12


@app.post("/api/attention/search")
def attention_search(r: AttnSearchReq):
    """Rank every head in the model by how strongly it shows one pattern."""
    a = lm(r.role).attention(r.text)
    att, toks = a["att"], a["tokens"]
    L, H, n, _ = att.shape
    if r.kind == "pair":
        if r.query is None or r.key is None:
            raise ValueError("Pick a word that looks and a word it should look at.")
        score = att[:, :, r.query, r.key]
        if r.key != 0:  # measure against the attention not parked on the first token, as the UI shows it
            score = score / np.clip(1 - att[:, :, r.query, 0], 1e-3, None)
    elif r.kind == "previous":
        score = np.stack([att[:, :, i, i - 1] for i in range(1, n)], -1).mean(-1)
    elif r.kind == "first":
        score = att[:, :, 1:, 0].mean(-1)
    elif r.kind == "induction":
        # For a repeated token at position i, look at the token that followed its earlier occurrence.
        pairs = [(i, j + 1) for i in range(n) for j in range(i - 1) if toks[j] == toks[i] and j + 1 < i]
        if not pairs:
            raise ValueError("Induction needs repeated text: use the repeated-sequence example.")
        score = np.stack([att[:, :, i, j] for i, j in pairs], -1).mean(-1)
    else:
        raise ValueError(f"Unknown pattern {r.kind!r}")
    flat = [(float(score[l, h]), l, h) for l in range(L) for h in range(H)]
    flat.sort(reverse=True)
    return {"heads": [{"layer": l, "head": h, "score": round(s, 3)} for s, l, h in flat[: r.top]]}


# ------------------------------------------------------------------ chat templates and generation


class TemplateReq(BaseModel):
    messages: list[dict]
    add_generation_prompt: bool = True
    enable_thinking: bool = False


@app.post("/api/template")
def template(r: TemplateReq):
    m = lm("chat")
    m.load_tokenizer()
    text = m.tok.apply_chat_template(
        r.messages, tokenize=False, add_generation_prompt=r.add_generation_prompt, enable_thinking=r.enable_thinking
    )
    return {"text": text}


class Steer(BaseModel):
    positive: list[str]
    negative: list[str]
    layer: int
    strength: float  # as a fraction of the typical size of the residual state at that layer


class FeatureSteer(BaseModel):
    layer: int
    index: int
    strength: float  # as a fraction of the typical size of the residual state at that layer


class AblationFit(BaseModel):
    positive: list[str] = Field(min_length=2, max_length=64)
    negative: list[str] = Field(min_length=2, max_length=64)
    layer: int | None = Field(default=None, ge=0)


class Ablation(AblationFit):
    control: bool = False


@app.post("/api/ablation/fit")
def ablation_fit(r: AblationFit):
    _, metadata = REFUSAL.fit(r.positive, r.negative, r.layer)
    return metadata


class AblationSearch(AblationFit):
    validation: list[str] = Field(min_length=2, max_length=8)
    controls: list[str] = Field(min_length=2, max_length=8)
    layers: list[int] | None = Field(default=None, min_length=1, max_length=256)
    finalists: int = Field(default=3, ge=1, le=256)
    max_new_tokens: int = Field(default=512, ge=32, le=512)
    max_control_kl: float = Field(default=0.5, gt=0, le=10, allow_inf_nan=False)


@app.post("/api/ablation/search")
def ablation_search(r: AblationSearch):
    async def events():
        cancelled = threading.Event()
        messages = queue.Queue()

        def work():
            search = REFUSAL.search(**r.model_dump(exclude={"layer"}), cancel=cancelled)
            try:
                for event in search:
                    if cancelled.is_set():
                        break
                    messages.put(event)
            except SearchCancelled:
                pass
            except Exception as exc:
                messages.put({"error": str(exc)})
            finally:
                search.close()
                messages.put(None)

        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                try:
                    event = messages.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                if event is None:
                    return
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Stop also when the browser disconnects or navigates away, between forward passes.
            cancelled.set()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class NeuronReq(BaseModel):
    zero_super: bool = False
    language: str | None = None
    strength: float = 0.0
    mute: str | None = "en"
    exclude: str | None = None


class GenReq(BaseModel):
    prompt: str
    role: str = "chat"
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int | None = None
    stop: list[str] | None = None
    steer: Steer | None = None
    feature: FeatureSteer | None = None
    neurons: NeuronReq | None = None
    ablate: Ablation | None = None


def _steer_args(m: LM, s: Steer | None):
    if not s or s.strength == 0:
        return None
    v = m.steer_vector(s.positive, s.negative, s.layer)
    scale = s.strength * m.typical_norm(s.layer) / float(v.norm())
    return v * scale, s.layer


@app.post("/api/generate")
def generate(r: GenReq):
    if r.ablate and (r.role != "chat" or r.steer or r.feature or r.neurons):
        raise ValueError("Refusal ablation uses the chat model, without another intervention.")
    if r.neurons and r.role != "base":
        raise ValueError("The super weight and language neurons belong to the base model.")
    m = lm(r.role)
    m.require()
    steer = _steer_args(m, r.steer)
    if r.feature and r.feature.strength != 0:
        # Turn one SAE feature up: add its decoder direction where the SAE reads the residual stream.
        v = SAES.direction(r.feature.layer, r.feature.index)
        steer = (v * (r.feature.strength * m.typical_norm(r.feature.layer) / float(v.norm())), r.feature.layer)
    plan = m.neuron_plan(**r.neurons.model_dump()) if r.neurons else None
    limit = min(r.max_new_tokens, 4096)
    ablate = REFUSAL.direction(r.ablate.positive, r.ablate.negative, r.ablate.layer, r.ablate.control) if r.ablate else None

    def events():
        try:
            for ev in m.generate(r.prompt, limit, r.temperature, r.top_p, steer, r.stop, r.seed, ablate=ablate, neurons=plan):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class SteerProjectReq(BaseModel):
    positive: list[str]
    negative: list[str]
    layer: int
    texts: list[str]
    role: str = "chat"


@app.post("/api/steer/project")
def steer_project(r: SteerProjectReq):
    m = lm(r.role)
    v = m.steer_vector(r.positive, r.negative, r.layer)
    pos = m.project(r.positive, v, r.layer)
    neg = m.project(r.negative, v, r.layer)
    mid = (np.mean(pos) + np.mean(neg)) / 2
    half = (np.mean(pos) - np.mean(neg)) / 2 or 1
    scores = [(s - mid) / half for s in m.project(r.texts, v, r.layer)]
    return {"scores": [round(s, 3) for s in scores]}


# ------------------------------------------------------------------ one number (super weight, language neurons)


@app.get("/api/neurons")
def neurons_info():
    m = lm("base")
    m.require()
    if not m.neurons:
        return {"model": m.spec.repo, "available": False}
    cfg = m.model.config
    return {
        "model": m.spec.repo,
        "available": True,
        "super_weight": m.neurons["super_weight"],
        "params": sum(p.numel() for p in m.model.parameters()),
        "neurons_total": cfg.num_hidden_layers * cfg.intermediate_size,
        "languages": {c: {"name": d["name"], "count": len(d["neurons"])} for c, d in m.neurons["languages"].items()},
    }


# ------------------------------------------------------------------ features (sparse autoencoders)


@app.get("/api/sae")
def sae_info():
    return {"model": SAES.lm.spec.repo, "layers": SAES.layers()}


class SaeEncodeReq(BaseModel):
    text: str
    layer: int


@app.post("/api/sae/encode")
def sae_encode(r: SaeEncodeReq):
    return SAES.encode(r.text, r.layer)


class SaeFeatureReq(BaseModel):
    layer: int
    feature: int


@app.post("/api/sae/feature")
def sae_feature(r: SaeFeatureReq):
    return SAES.describe(r.layer, r.feature)


WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
