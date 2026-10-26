"""Numerical and lifecycle checks on a tiny, randomly initialised Qwen3; no downloads.

Run: uv run python -m unittest discover -s tests -v
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from transformers import Qwen3Config, Qwen3ForCausalLM

from server.config import ModelSpec
from server.decode import Decoder
from server.lm import LM
from server.refusal import RefusalDirections


class Tokenizer:
    eos_token_id = None

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) % 61 + 1 for c in text])

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "user:" + messages[0]["content"] + "\nassistant:"

    def convert_tokens_to_ids(self, text):
        return None

    def decode(self, ids):
        return "".join(chr(i + 64) for i in ids)


def tiny_model(device="cpu", graph=False):
    torch.manual_seed(123)
    lm = LM("chat", ModelSpec("test/qwen3"))
    lm.model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        attn_implementation="eager",
    )).eval().to(device)
    lm.tok = Tokenizer()
    lm._install_steering()
    lm.decoder = Decoder(lm.model, 128, graph)
    return lm


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.lm = tiny_model()
        self.direction = torch.arange(1, 17, dtype=torch.float32)
        self.direction /= self.direction.norm()

    @torch.no_grad()
    def test_projection_all_layers_and_zero_identity(self):
        lm = self.lm
        ids = lm._ids("test")
        baseline = lm.model(input_ids=ids).logits
        lm._steer_on = True
        lm._set_ablation(None)
        torch.testing.assert_close(baseline, lm.model(input_ids=ids).logits, rtol=0, atol=0)
        lm._set_ablation(self.direction)
        with lm._capture() as states:
            changed = lm.model(input_ids=ids).logits
        for state in states:
            torch.testing.assert_close(state @ self.direction, torch.zeros(state.shape[:-1]), rtol=0, atol=2e-7)
        self.assertFalse(torch.equal(baseline, changed))
        lm._steer_on = False
        torch.testing.assert_close(baseline, lm.model(input_ids=ids).logits, rtol=0, atol=0)

    def test_generation_restores_and_cancel_releases(self):
        lm = self.lm
        before = list(lm.generate("test", 8, temperature=0))
        list(lm.generate("test", 8, temperature=0, ablate=self.direction))
        self.assertEqual(before, list(lm.generate("test", 8, temperature=0)))
        running = lm.generate("test", 8, ablate=self.direction)
        next(running)
        self.assertFalse(lm._steer_on)
        running.close()
        self.assertIsNone(lm.decoder.owner)
        self.assertEqual(before, list(lm.generate("test", 8, temperature=0)))

    def test_forward_exception_disables_intervention(self):
        with patch.object(self.lm.decoder, "prefill", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                list(self.lm.generate("test", ablate=self.direction))
        self.assertFalse(self.lm._steer_on)
        self.assertIsNone(self.lm.decoder.owner)

    def test_invalid_directions_and_mixed_interventions(self):
        for direction in [torch.zeros(16), torch.ones(3), torch.full((16,), float("nan"))]:
            with self.assertRaises(ValueError):
                list(self.lm.generate("test", ablate=direction))
        with self.assertRaises(ValueError):
            list(self.lm.generate("test", steer=(self.direction, 0), ablate=self.direction))

    def test_fit_final_token_cache_and_validation(self):
        fitter = RefusalDirections(self.lm)
        positive, negative = ["refuse", "decline"], ["answer", "help"]
        unit, meta = fitter.fit(positive, negative, 0)
        self.assertAlmostEqual(float(unit.norm()), 1, places=6)
        self.assertAlmostEqual(sum(meta["positive_scores"]) / 2, 1, places=5)
        self.assertAlmostEqual(sum(meta["negative_scores"]) / 2, -1, places=5)
        with patch.object(fitter, "_states", side_effect=AssertionError("cache missed")):
            fitter.fit(positive, negative, 1)
        prompts = fitter._prompts(positive + negative)
        with torch.no_grad(), self.lm._capture() as states:
            self.lm.model(input_ids=self.lm._ids(prompts[0]))
        cached = next(iter(fitter.cache.values()))
        torch.testing.assert_close(cached[0, 0], states[1][-1])
        random = fitter.direction(positive, negative, 0, control=True)
        self.assertAlmostEqual(float(random.norm()), 1, places=6)
        self.assertAlmostEqual(float(random @ unit), 0, places=6)
        for pos, neg, layer in [([], negative, 0), (["", "x"], negative, 0), (positive, positive, 0), (positive, negative, 5)]:
            with self.assertRaises(ValueError):
                fitter.fit(pos, neg, layer)

    def test_capture_hooks_removed_after_failure(self):
        fitter = RefusalDirections(self.lm)
        counts = [len(layer._forward_hooks) for layer in self.lm.layers]
        with patch.object(self.lm.model.model, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                fitter.fit(["a", "b"], ["c", "d"], 0)
        self.assertEqual(counts, [len(layer._forward_hooks) for layer in self.lm.layers])

    def test_api_errors_and_intervention_exclusion(self):
        from server.app import app
        with patch("server.app.REFUSAL", RefusalDirections(self.lm)), patch.dict("server.app.MODELS", {"chat": self.lm}):
            client = TestClient(app)
            data = {"positive": ["refuse", "decline"], "negative": ["answer", "help"], "layer": 0}
            self.assertEqual(client.post("/api/ablation/fit", json=data).status_code, 200)
            self.assertEqual(client.post("/api/ablation/fit", json={**data, "positive": []}).status_code, 422)
            self.assertEqual(client.post("/api/ablation/fit", json={**data, "layer": 99}).status_code, 400)
            self.assertEqual(client.post("/api/generate", json={"prompt": "test", "role": "base", "ablate": data}).status_code, 400)
            response = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": 4, "ablate": data})
            self.assertIn('"done":', response.text)
            self.assertNotIn('"error":', response.text)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_captured_graph_switches_direction_and_restores(self):
        lm = tiny_model("cuda", graph=True)
        baseline = list(lm.generate("test", 8, temperature=0))
        self.assertIsNotNone(lm.decoder.graph)
        direction = self.direction.cuda()
        graphed = list(lm.generate("test", 8, temperature=0, ablate=direction))
        self.assertEqual(baseline, list(lm.generate("test", 8, temperature=0)))
        lm.decoder.use_graph = False
        eager = list(lm.generate("test", 8, temperature=0, ablate=direction))
        self.assertEqual(graphed, eager)


if __name__ == "__main__":
    unittest.main()
