import unittest
from unittest.mock import patch

import torch

from train import evaluate, validate_fingerprint


class FingerprintTests(unittest.TestCase):
    def test_accepts_legacy_and_explicit_modes(self):
        for mode, key, digest in (
            ("corpus", "corpus_sha256", "hash"),
            ("documents", "document_sha256", {"train/doc.txt": "hash"}),
        ):
            with self.subTest(mode=mode):
                validate_fingerprint({key: digest}, mode, digest)
                validate_fingerprint({"data_mode": mode, key: digest}, mode, digest)

    def test_rejects_wrong_or_ambiguous_mode(self):
        for saved, mode in (
            ({"corpus_sha256": "hash"}, "documents"),
            ({"document_sha256": "hash"}, "corpus"),
            ({"data_mode": "documents", "corpus_sha256": "hash"}, "corpus"),
            ({"corpus_sha256": "hash", "document_sha256": "hash"}, "corpus"),
            ({"data_mode": "corpus"}, "corpus"),
        ):
            with self.subTest(saved=saved, mode=mode):
                with self.assertRaisesRegex(ValueError, "modalidade"):
                    validate_fingerprint(saved, mode, "hash")

    def test_rejects_changed_hash_in_same_mode(self):
        with self.assertRaisesRegex(ValueError, "corpus difere"):
            validate_fingerprint({"corpus_sha256": "old"}, "corpus", "new")
        with self.assertRaisesRegex(ValueError, "corpus difere"):
            validate_fingerprint(
                {"document_sha256": {"validation/doc.txt": "old"}},
                "documents",
                {"validation/doc.txt": "new"},
            )


class EvaluateTests(unittest.TestCase):
    def test_uses_eval_mode_and_restores_original_mode(self):
        model = torch.nn.Dropout()

        def measure(model, data, batch_size, rng):
            self.assertFalse(model.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            return torch.rand((), generator=rng)

        with patch("train.loss_on", side_effect=measure):
            first = evaluate(model, None, 1, 42)
            self.assertTrue(model.training)
            model.eval()
            self.assertEqual(first, evaluate(model, None, 1, 42))
            self.assertFalse(model.training)

    def test_restores_original_mode_on_exception(self):
        model = torch.nn.Dropout()
        with patch("train.loss_on", side_effect=RuntimeError("falha de avaliação")):
            for training in (True, False):
                with self.subTest(training=training):
                    model.train(training)
                    with self.assertRaisesRegex(RuntimeError, "falha de avaliação"):
                        evaluate(model, None, 1, 42)
                    self.assertIs(model.training, training)
