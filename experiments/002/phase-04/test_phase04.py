import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from corpus import Partition, read_documents
from generate import Vocabulary, load_checkpoint
from run_training import SOURCE, StopState, evaluate_fixed, restore, save_state, update_weights, verify_hashes


def compare(test, expected, actual):
    if torch.is_tensor(expected):
        test.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, dict):
        test.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            compare(test, expected[key], actual[key])
    else:
        test.assertEqual(expected, actual)


class PhaseFourTests(unittest.TestCase):
    def test_overfit_increments_resets_and_stops(self):
        state = StopState(5000, 1.9, 5000, 1.9)
        for step in (5500, 6000):
            self.assertIsNone(state.update(step, 1.92)[1])
        self.assertEqual(state.overfit_count, 2)
        self.assertIsNone(state.update(6500, 1.91)[1])
        self.assertEqual(state.overfit_count, 0)
        self.assertEqual(state.update(7000, 1.85), (True, None))
        for step in (7500, 8000, 8500):
            self.assertIsNone(state.update(step, 1.88)[1])
        self.assertEqual(state.update(9000, 1.88), (False, "overfitting_sustentado"))

    def test_significant_reference_accumulates_and_times_out(self):
        state = StopState(5000, 2.0, 5000, 2.0)
        self.assertEqual(state.update(5500, 1.997), (True, None))
        self.assertEqual(state.significant_step, 5000)
        self.assertEqual(state.update(6000, 1.994), (True, None))
        self.assertEqual((state.significant_step, state.significant_loss), (6000, 1.994))
        for step in range(6500, 9000, 500):
            self.assertIsNone(state.update(step, 1.992)[1])
        self.assertEqual(state.update(9000, 1.991), (True, "saturacao"))

    def test_best_remains_when_latest_is_replaced(self):
        state = StopState(5000, 2.0, 5000, 2.0)
        with TemporaryDirectory() as folder:
            path = Path(folder)
            save_state(path, {"step": 5000}, True)
            save_state(path, {"step": 7500}, state.update(7500, 1.9)[0], snapshot=True)
            save_state(path, {"step": 8000}, state.update(8000, 1.91)[0])
            self.assertEqual(load_checkpoint(path / "latest.pt")["step"], 8000)
            self.assertEqual(load_checkpoint(path / "best.pt")["step"], 7500)
            self.assertEqual(load_checkpoint(path / "model-07500.pt")["step"], 7500)

    def test_non_finite_loss_does_not_update_weights(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        original = {key: value.clone() for key, value in model.state_dict().items()}
        with patch("run_training.loss_on", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                update_weights(model, optimizer, None, 8, torch.Generator())
        compare(self, original, model.state_dict())
        self.assertFalse(optimizer.state)
        with self.assertRaises(FloatingPointError):
            StopState(5000, 2.0, 5000, 2.0).update(5500, float("inf"))

    def test_evaluation_restores_mode_and_rng_in_success_and_failure(self):
        model = torch.nn.Dropout()
        train_rng = torch.Generator().manual_seed(42)
        initial = torch.get_rng_state(), train_rng.get_state()

        def loss_eval(model, data, batch_size, generator):
            self.assertFalse(model.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            return torch.rand((), generator=generator)

        for mode in (True, False):
            model.train(mode)
            with patch("run_training.loss_on", side_effect=loss_eval):
                self.assertEqual(evaluate_fixed(model, None, 8, 43, 64), evaluate_fixed(model, None, 8, 43, 64))
            self.assertIs(model.training, mode)
            with patch("run_training.loss_on", return_value=torch.tensor(float("inf"))):
                with self.assertRaises(FloatingPointError):
                    evaluate_fixed(model, None, 8, 43, 4)
            self.assertIs(model.training, mode)
            compare(self, initial[0], torch.get_rng_state())
            compare(self, initial[1], train_rng.get_state())

    def test_historical_resume_two_updates_equal_to_original(self):
        verify_hashes()
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        with TemporaryDirectory() as folder:
            output = Path(folder) / "reference.pt"
            subprocess.run(
                [sys.executable, "train.py", "--documents", "data/experiment-002", "--resume", str(SOURCE),
                 "--steps", "5002", "--eval-interval", "1", "--checkpoint", str(output)],
                check=True, capture_output=True, text=True,
            )
            reference = load_checkpoint(output)
        saved = load_checkpoint(SOURCE)
        vocab = Vocabulary(saved["vocabulary"])
        train_docs, val_docs, _ = read_documents(Path("data/experiment-002"), 128)
        train_data, val_data = (Partition(docs, vocab, 128) for docs in (train_docs, val_docs))
        model, optimizer, train_rng = restore(saved)
        update_weights(model, optimizer, train_data, 8, train_rng)
        evaluate_fixed(model, val_data, 8, 43, 64)
        update_weights(model, optimizer, train_data, 8, train_rng)
        compare(self, reference["model"], model.state_dict())
        expected_optimizer = reference["optimizer"]
        actual_optimizer = optimizer.state_dict()
        compare(self, expected_optimizer["state"], actual_optimizer["state"])
        self.assertEqual(expected_optimizer["param_groups"], actual_optimizer["param_groups"])
        compare(self, reference["torch_rng"], torch.get_rng_state())
        compare(self, reference["train_rng"], train_rng.get_state())
