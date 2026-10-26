import asyncio
import json
import threading
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from test_ablation import tiny_model
from server.refusal import RefusalDirections, SearchCancelled, reply_summary, search_recommendation


REFUSAL = reply_summary("I'm unable to assist with that request.", "stop")
ANSWER = reply_summary("Here is an answer.", "stop")
INCOMPLETE = reply_summary("Certainly, here is", "length")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.lm = tiny_model()
        self.fit = RefusalDirections(self.lm)
        self.data = dict(positive=["refuse", "decline"], negative=["answer", "help"],
                         validation=["question a", "question b"], controls=["sum", "logic"])

    @staticmethod
    def judge(request, reply):
        return {"label": "refusal" if reply["flagged"] else "answer", "relative_confidence": 1.0}

    @torch.no_grad()
    def test_score_matches_normal_logits_and_cleans_up(self):
        prompt = self.fit._prompts(["hello", "bye"])[0]
        expected = self.lm.model(input_ids=self.lm._ids(prompt), use_cache=False).logits[0, -1].float().log_softmax(-1)
        torch.testing.assert_close(self.fit._next_logprobs(prompt), expected)
        full = torch.cat([self.lm._ids(prompt), torch.tensor([self.lm.encode("answer")[:7]])], dim=1)
        expected_continuation = self.lm.model(input_ids=full, use_cache=False).logits[0, len(self.lm.encode(prompt)) - 1:].float().log_softmax(-1)
        torch.testing.assert_close(self.fit._next_logprobs(prompt, continuation="answer"), expected_continuation)
        unit, _ = self.fit.fit(self.data["positive"], self.data["negative"], 0)
        cache = self.lm.decoder.cache
        self.fit._next_logprobs(prompt, unit)
        self.assertIs(self.lm.decoder.cache, cache)
        self.assertFalse(self.lm._steer_on)
        self.assertEqual(float(self.lm._ablate_buf.norm()), 0)
        torch.testing.assert_close(self.fit._next_logprobs(prompt), expected)
        with patch.object(self.lm.model.model, "forward", side_effect=RuntimeError("scoring failure")):
            with self.assertRaises(RuntimeError):
                self.fit._next_logprobs(prompt, unit)
        self.assertFalse(self.lm._steer_on)
        self.assertEqual(float(self.lm._ablate_buf.norm()), 0)

    def test_selection_uses_generated_replies_then_control_kl(self):
        baseline = dict(refusal_replies=[REFUSAL, REFUSAL], control_replies=[ANSWER, ANSWER])
        def row(layer, replies, kl, proxy=0.1, controls=None):
            return dict(layer=layer, eligible=True, refusal_replies=replies,
                        control_replies=controls or [ANSWER, ANSWER], control_kl=kl, proxy=proxy)
        rows = [row(0, [REFUSAL, ANSWER], .01, .001), row(1, [ANSWER, ANSWER], .2),
                row(2, [ANSWER, ANSWER], .1), row(3, [ANSWER, ANSWER], .001, controls=[REFUSAL, ANSWER])]
        self.assertEqual(search_recommendation(baseline, rows), 2)
        self.assertEqual(rows[3]["control_regressions"], 1)
        self.assertIsNone(search_recommendation(baseline, [row(0, [INCOMPLETE, INCOMPLETE], 0)]))
        self.assertIsNone(search_recommendation(baseline, [row(0, [REFUSAL, REFUSAL], 0)]))
        self.assertIsNone(search_recommendation(dict(refusal_replies=[ANSWER], control_replies=[]), []))
        redirected = {**ANSWER, "judgement": {"label": "refusal", "relative_confidence": .99}}
        self.assertIsNone(search_recommendation(baseline, [row(0, [redirected, redirected], 0)]))

    def test_judge_scores_labels_and_abstains_when_ambiguous(self):
        logp = torch.full((64,), -20.0)
        labels = [self.lm.encode(t)[0] for t in ["A", "B", "C"]]
        logp[labels[1]] = 0
        with patch.object(self.fit, "_next_logprobs", return_value=logp):
            self.assertEqual(self.fit._judge("request", ANSWER)["label"], "refusal")
        logp[labels] = 0
        with patch.object(self.fit, "_next_logprobs", return_value=logp):
            self.assertEqual(self.fit._judge("request", ANSWER)["label"], "unclear")

    def test_refusal_detection_checks_entire_reply_and_reports_matches(self):
        reply = reply_summary("Certainly! On reflection, I cannot help with that request.", "stop")
        self.assertTrue(reply["flagged"])
        self.assertIn("i cannot help", reply["matches"])
        self.assertFalse(reply_summary("Here is the result: 84.", "stop")["flagged"])
        self.assertTrue(reply_summary("", "stop")["incomplete"])

    def test_search_rejects_validation_leakage_and_bad_limits(self):
        for change in [dict(validation=[" REFUSE ", "new"]), dict(controls=["sum", " SUM "]),
                       dict(layers=[99]), dict(max_control_kl=float("nan")), dict(finalists=0)]:
            with self.assertRaises(ValueError):
                list(self.fit.search(**{**self.data, **change}))

    def test_cancel_and_generator_close_release_model(self):
        cancel = threading.Event()
        search = self.fit.search(**self.data, cancel=cancel)
        self.assertEqual(next(search)["stage"], "fit")
        cancel.set()
        with self.assertRaises(SearchCancelled):
            next(search)
        search = self.fit.search(**self.data, max_new_tokens=8)
        next(search)  # fit
        self.assertEqual(next(search)["stage"], "baseline")  # real generation suspended at a token
        search.close()
        self.assertIsNone(self.lm.decoder.owner)
        self.assertFalse(self.lm._steer_on)

    def test_search_pipeline_and_streaming_api(self):
        def replies(prompt, direction, length, cancel, progress):
            yield {"type": "progress", **progress}
            return dict(REFUSAL if direction is None and "question" in prompt else ANSWER)
        with patch.object(self.fit, "_reply", side_effect=replies), patch.object(self.fit, "_judge", side_effect=self.judge):
            events = list(self.fit.search(**self.data, finalists=2, max_control_kl=10))
        result = events[-1]
        self.assertEqual(result["type"], "done")
        self.assertIsNotNone(result["best_layer"])
        self.assertEqual(len([e for e in events if e["type"] == "candidate"]), 2)
        self.assertTrue(all("control_replies" in row for row in result["rows"]))
        from server.app import app
        with patch("server.app.REFUSAL", self.fit), patch.object(self.fit, "_reply", side_effect=replies), patch.object(self.fit, "_judge", side_effect=self.judge):
            client = TestClient(app)
            response = client.post("/api/ablation/search", json={**self.data, "finalists": 2, "max_control_kl": 10})
            self.assertEqual(response.status_code, 200)
            parsed = [json.loads(s[6:]) for s in response.text.split("\n\n") if s]
            self.assertEqual(parsed[-1]["type"], "done")
            self.assertEqual(client.post("/api/ablation/search", json={**self.data, "finalists": 257}).status_code, 422)

    def test_no_baseline_refusals_returns_no_recommendation(self):
        def replies(*args):
            yield {"type": "progress"}
            return dict(ANSWER)
        with patch.object(self.fit, "_reply", side_effect=replies), patch.object(self.fit, "_judge", side_effect=self.judge):
            events = list(self.fit.search(**self.data))
        self.assertIsNone(events[-1]["best_layer"])
        self.assertFalse(any(e["type"] == "candidate" for e in events))

    def test_no_usable_harmless_controls_returns_no_recommendation(self):
        def replies(*args):
            yield {"type": "progress"}
            return dict(REFUSAL)
        with patch.object(self.fit, "_reply", side_effect=replies), patch.object(self.fit, "_judge", side_effect=self.judge):
            events = list(self.fit.search(**self.data))
        self.assertIsNone(events[-1]["best_layer"])
        self.assertIn("No harmless control", events[-1]["message"])
        self.assertFalse(any(e["type"] == "candidate" for e in events))

    def test_stream_close_signals_worker_cancellation(self):
        from server.app import AblationSearch, ablation_search
        stopped = threading.Event()
        def search(**kwargs):
            yield {"type": "progress"}
            kwargs["cancel"].wait(2)
            if kwargs["cancel"].is_set():
                stopped.set()
        async def close_stream():
            response = ablation_search(AblationSearch(**self.data))
            await anext(response.body_iterator)
            await response.body_iterator.aclose()
        with patch("server.app.REFUSAL.search", side_effect=search):
            asyncio.run(close_stream())
            self.assertTrue(stopped.wait(1))


if __name__ == "__main__":
    unittest.main()
