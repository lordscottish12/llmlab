"""Fast token-by-token decoding: a static KV cache plus a captured CUDA graph.

A plain PyTorch decoding step launches about two thousand small GPU kernels. On Windows each launch
costs ~20 µs of CPU time, so the CPU, not the GPU, sets the pace (around 10 tokens/s). Recording one
step as a CUDA graph and replaying it removes that cost (5-8x faster).

The graph writes into one fixed cache, so one generation per model runs at a time. A new request
waits for the running one; if that one has stopped asking for tokens (its browser tab went away),
the new request takes over after a few seconds.
"""

import threading
import time

import torch
from transformers import StaticCache

STALE_AFTER = 3.0  # seconds without a step before a generation counts as abandoned


class Decoder:
    def __init__(self, model, max_len: int, use_graph: bool):
        self.model = model
        self.max_len = max_len
        self.use_graph = use_graph and torch.cuda.is_available()
        self.cache = None
        self.graph = None
        self.cv = threading.Condition()
        self.owner = None
        self.owner_seen = 0.0

    # ---------------------------------------------------------- ownership of the cache

    def claim(self, me):
        with self.cv:
            while self.owner is not None and time.time() - self.owner_seen < STALE_AFTER:
                self.cv.wait(0.2)
            self.owner, self.owner_seen = me, time.time()

    def owns(self, me) -> bool:
        if self.owner is not me:
            return False
        self.owner_seen = time.time()
        return True

    def release(self, me):
        with self.cv:
            if self.owner is me:
                self.owner = None
                self.cv.notify_all()

    # ---------------------------------------------------------- the cache and the graph

    def _layers(self):
        return self.cache.layers

    def _set_length(self, n: int):
        # StaticCache keeps its write position in a counter on the GPU, advanced by every forward pass.
        for layer in self._layers():
            if getattr(layer, "is_initialized", True) and hasattr(layer, "cumulative_length"):
                layer.cumulative_length.fill_(n)

    def prefill(self, ids: list[int]) -> torch.Tensor:
        dev = self.model.model.embed_tokens.weight.device
        if self.cache is None:
            self.cache = StaticCache(config=self.model.config, max_cache_len=self.max_len)
        else:
            self._set_length(0)
        n = len(ids)
        out = self.model(
            input_ids=torch.tensor([ids], device=dev),
            past_key_values=self.cache,
            cache_position=torch.arange(n, device=dev),
            use_cache=True,
        )
        self.pos = n
        return out.logits[0, -1]

    def _step_eager(self, ids, pos):
        return self.model(input_ids=ids, past_key_values=self.cache, cache_position=pos, position_ids=pos.view(1, 1), use_cache=True).logits

    def _capture(self, token: int):
        dev = self.model.model.embed_tokens.weight.device
        self.s_ids = torch.full((1, 1), token, dtype=torch.long, device=dev)
        self.s_pos = torch.full((1,), self.pos, dtype=torch.long, device=dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up; these write the same entry twice
                self._set_length(self.pos)
                self._step_eager(self.s_ids, self.s_pos)
        torch.cuda.current_stream().wait_stream(side)
        self._set_length(self.pos)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.s_logits = self._step_eager(self.s_ids, self.s_pos)
        self._set_length(self.pos)
        self.graph = g

    def step(self, token: int) -> torch.Tensor:
        """Feed one token at the current position; return the logits for the next one."""
        if self.pos >= self.max_len:
            raise ValueError(f"The text is longer than the {self.max_len}-token limit.")
        if self.use_graph:
            try:
                if self.graph is None:
                    self._capture(token)
                self.s_ids.fill_(token)
                self.s_pos.fill_(self.pos)
                self.graph.replay()
                self.pos += 1
                return self.s_logits[0, -1]
            except Exception as e:  # some quantized kernels cannot be captured; fall back once and for all
                print(f"[llmlab] CUDA graph unavailable, decoding without it: {e}")
                self.use_graph = False
                self.graph = None
                self._set_length(self.pos)
        dev = self.model.model.embed_tokens.weight.device
        logits = self._step_eager(torch.tensor([[token]], device=dev), torch.tensor([self.pos], device=dev))
        self.pos += 1
        return logits[0, -1]
