"""Model profiles: which checkpoints play which role, sized for the machine the server runs on.

Two roles:
  base  a pretrained-only model. It continues text and has never been taught to chat.
        All the "internals" demos (forward pass, attention, logit lens) use it.
  chat  the post-trained sibling of the base model. It follows the chat template,
        answers questions and can "think" before it answers.

Both roles use the same model family (Qwen3), so the base/chat comparison is a fair one.
Pick a profile with LLMLAB_PROFILE, or override one role with LLMLAB_BASE / LLMLAB_CHAT.
"""

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    quant: str | None = None  # None (bf16) or "4bit"
    label: str = ""


PROFILES: dict[str, dict[str, ModelSpec]] = {
    # RTX 3070 laptop, 8 GB: a 1.7B base model in bf16 next to a 4-bit 4B chat model, about 6 GB together.
    "laptop": {
        "base": ModelSpec("Qwen/Qwen3-1.7B-Base", label="Qwen3 1.7B base"),
        "chat": ModelSpec("Qwen/Qwen3-4B", quant="4bit", label="Qwen3 4B chat (4-bit)"),
    },
    # 24 GB desktop card: both 4B models in bf16, about 16 GB together.
    "desktop": {
        "base": ModelSpec("Qwen/Qwen3-4B-Base", label="Qwen3 4B base"),
        "chat": ModelSpec("Qwen/Qwen3-4B", label="Qwen3 4B chat"),
    },
    # A 48 GB+ cloud GPU (e.g. a RunPod L40S or A6000).
    "server": {
        "base": ModelSpec("Qwen/Qwen3-8B-Base", label="Qwen3 8B base"),
        "chat": ModelSpec("Qwen/Qwen3-14B", label="Qwen3 14B chat"),
    },
    # No GPU at all: small enough to run on a CPU, slowly.
    "cpu": {
        "base": ModelSpec("Qwen/Qwen3-0.6B-Base", label="Qwen3 0.6B base"),
        "chat": ModelSpec("Qwen/Qwen3-0.6B", label="Qwen3 0.6B chat"),
    },
}


def _spec_from_env(var: str, default: ModelSpec) -> ModelSpec:
    """LLMLAB_CHAT=Qwen/Qwen3-8B or LLMLAB_CHAT=Qwen/Qwen3-8B:4bit"""
    value = os.environ.get(var)
    if not value:
        return default
    repo, _, quant = value.partition(":")
    return ModelSpec(repo, quant=quant or None, label=repo.split("/")[-1])


PROFILE = os.environ.get("LLMLAB_PROFILE", "laptop")
if PROFILE not in PROFILES:
    raise SystemExit(f"Unknown LLMLAB_PROFILE {PROFILE!r}; choose one of {', '.join(PROFILES)}")

ROLES: dict[str, ModelSpec] = {
    "base": _spec_from_env("LLMLAB_BASE", PROFILES[PROFILE]["base"]),
    "chat": _spec_from_env("LLMLAB_CHAT", PROFILES[PROFILE]["chat"]),
}
ROLES = {role: replace(spec, label=spec.label or spec.repo) for role, spec in ROLES.items()}

# Load both models at start-up so the first demo in class does not wait for a load.
PRELOAD = os.environ.get("LLMLAB_PRELOAD", "1") != "0"

# Longest text (prompt plus reply) a generation can reach; the KV cache is allocated at this size.
MAX_CONTEXT = int(os.environ.get("LLMLAB_MAX_CONTEXT", "4096"))
# Record decoding steps as CUDA graphs (much faster on Windows). Set to 0 if decoding misbehaves.
USE_GRAPHS = os.environ.get("LLMLAB_GRAPHS", "1") != "0"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
