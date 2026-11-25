"""Synthetic tests only; never train on the actual biometric datasets."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.tests.helpers import multisample_data
from scripts.s11_threshold_v5.model import ThresholdingModelV5, V5Config
from scripts.s11_threshold_v5.losses import compute_loss as v5_loss
from scripts.s11_threshold_v5.protocol import batch_forward, make_plan, prepare_bank, validate_population
from .model import Config, NeuralThreshold, load_checkpoint, save_checkpoint, state_digest
from .losses import compute_loss, recovery_terms
from . import train


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data, self.users = multisample_data()
        self.args = train.parser().parse_args(["--device", "cpu", "--threads", "1", "--seeds", "17",
            "--epochs", "2", "--warmup-epochs", "1", "--batch-size", "4", "--train-rounds", "1",
            "--attack-rounds", "1", "--attack-epochs", "1", "--attack-batch-size", "4"])
        self.plan = make_plan(self.data, self.users, "train", 17)
        self.bank = prepare_bank(self.data, self.plan, "cpu")
        self.positions = torch.arange(len(self.bank["rows"]))

    def test_coordinate_model_exactly_matches_v5(self):
        torch.manual_seed(19)
        original = ThresholdingModelV5(V5Config(coordinate_hidden=128))
        torch.manual_seed(19)
        current = NeuralThreshold()
        self.assertEqual(state_digest(current), state_digest(original))
        # Exercise nonzero thresholds, not only the zero-initialized output.
        with torch.no_grad():
            for name in ("face_mlp", "finger_mlp"):
                getattr(original, name)[-1].weight.normal_(std=.02)
        current.load_state_dict(original.state_dict())
        a = batch_forward(original, self.bank, self.positions)
        b = batch_forward(current, self.bank, self.positions)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_set_invariance_modality_independence_and_gradients(self):
        model = NeuralThreshold(Config(architecture="set"))
        rows = self.bank["rows"]
        face = self.bank["face"]["enrollment"][rows[:, 0]]
        finger = self.bank["fingerprint"]["enrollment"][rows[:, 1]]
        self.assertEqual(float(model(face, finger)["face_tau"].abs().sum()), 0.)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        for _ in range(3):
            optimizer.zero_grad()
            out = model(face, finger)
            loss = (out["face_tau"]-.01).square().mean()+(out["finger_tau"]+.01).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(float(model.face_mlp.encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.face_mlp.attention.weight.grad.abs().sum()), 0)
        out = model(face, finger)
        permuted = model(face[:, [4, 0, 2, 1, 3]], finger[:, [2, 1, 3, 4, 0]])
        for name, x in (("face", face), ("finger", finger)):
            torch.testing.assert_close(out[name+"_tau"], permuted[name+"_tau"])
            self.assertTrue(torch.equal(out[name+"_target_bits"], (x.mean(1) >= 0).float()))
            self.assertLessEqual(float(out[name+"_tau"].abs().max()), .02000001)
            torch.testing.assert_close(out[name+"_attention"].sum(1), torch.ones(len(face)))
        torch.testing.assert_close(out["face_tau"], model(face, -finger)["face_tau"], rtol=0, atol=0)

    def test_enrollment_only_and_query_padding(self):
        model = NeuralThreshold(Config(architecture="set"))
        before = batch_forward(model, self.bank, self.positions)
        changed = deepcopy(self.bank)
        for m in ("face", "fingerprint"):
            q = changed[m]["queries"]
            changed[m]["queries"] = torch.cat((q, torch.full((len(q), 2, 512), 99.)), dim=1)
            changed[m]["query_mask"] = torch.cat((changed[m]["query_mask"], torch.zeros(len(q), 2, dtype=torch.bool)), dim=1)
        after = batch_forward(model, changed, self.positions)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        reference = {key: value.detach() for key, value in before.items()}
        for improved in (False, True):
            a = compute_loss(before, self.bank, self.positions, self.args, torch.Generator().manual_seed(9),
                             improved=improved, reference=reference)
            b = compute_loss(after, changed, self.positions, self.args, torch.Generator().manual_seed(9),
                             improved=improved, reference=reference)
            torch.testing.assert_close(a["total"], b["total"])
        for m in ("face", "fingerprint"):
            changed[m]["queries"].neg_()
        out = batch_forward(model, changed, self.positions)
        torch.testing.assert_close(before["face_tau"], out["face_tau"], rtol=0, atol=0)

    def test_old_loss_exact_and_new_loss_differentiable(self):
        model = NeuralThreshold()
        outputs = batch_forward(model, self.bank, self.positions)
        original = v5_loss(outputs, self.bank, self.positions, "D", self.args, torch.Generator().manual_seed(7))
        actual = compute_loss(outputs, self.bank, self.positions, self.args, torch.Generator().manual_seed(7))
        torch.testing.assert_close(actual["total"], original["total"], rtol=0, atol=0)
        ga = torch.autograd.grad(actual["total"], tuple(model.parameters()), retain_graph=True)
        gb = torch.autograd.grad(original["total"], tuple(model.parameters()), retain_graph=True)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        ref = {key: value.detach() for key, value in outputs.items()}
        improved = compute_loss(outputs, self.bank, self.positions, self.args, torch.Generator().manual_seed(7),
                                improved=True, reference=ref)
        improved["total"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
        with self.assertRaises(ValueError):
            compute_loss(outputs, self.bank, self.positions, self.args, torch.Generator(), improved=True)

    def test_keep_penalizes_only_lost_reference_successes(self):
        distance = torch.tensor([[31., 33., 80., 100.]], requires_grad=True)
        reference = torch.tensor([[31., 31., 80., 0.]])
        mask = torch.tensor([[True, True, True, False]])
        boundary, keep, protected = recovery_terms(distance, reference, mask, 32, 4.)
        torch.testing.assert_close(keep, torch.tensor([1/32**2/3]))
        self.assertEqual(protected.tolist(), [[True, True, False, False]])
        keep.sum().backward(retain_graph=True)
        self.assertEqual(distance.grad[0, 0], 0)
        self.assertGreater(distance.grad[0, 1], 0)
        self.assertEqual(distance.grad[0, 2], 0)
        self.assertEqual(distance.grad[0, 3], 0)
        distance.grad.zero_()
        boundary.sum().backward()
        self.assertGreater(distance.grad[0, 1], distance.grad[0, 2])
        _, empty, _ = recovery_terms(distance, reference+100, mask, 32, 4.)
        self.assertEqual(float(empty), 0.)

    def test_protocol_and_checkpoint_round_trip(self):
        validate_population(self.data, self.users)
        for records in self.plan["modalities"].values():
            for row in records:
                self.assertEqual(len(row["enrollment"]), 5)
                self.assertFalse(set(row["enrollment"]) & set(row["queries"]))
        model = NeuralThreshold(Config(architecture="set"))
        with tempfile.TemporaryDirectory(prefix="neural-checkpoint-") as tmp:
            path = Path(tmp)/"model.pt"
            save_checkpoint(model, path, {"purpose": "synthetic"})
            restored, meta = load_checkpoint(path)
            self.assertEqual(state_digest(restored), state_digest(model))
            self.assertEqual(meta["purpose"], "synthetic")

    @unittest.skipUnless(shutil.which("g++"), "Existing g++ required for actual BCH smoke test")
    def test_all_four_arms_end_to_end_and_test_barrier(self):
        with tempfile.TemporaryDirectory(prefix="neural-abcd-") as tmp:
            self.args.output_dir = Path(tmp)/"run"
            original_prepare = train.prepare_bank
            def checked_prepare(data, plan, device):
                if plan["split"] == "test":
                    self.assertTrue((self.args.output_dir/"selection_complete.json").is_file())
                return original_prepare(data, plan, device)
            with patch.object(train, "prepare_bank", checked_prepare), redirect_stdout(io.StringIO()):
                output = train.run_experiments(self.data, self.users, self.args, {})
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["neural_trials"], 4)
            self.assertEqual(status["endpoints"], 10)
            self.assertEqual(status["fresh_attackers"], 40)
            paired = json.loads((output/"matched_warmup.json").read_text())
            self.assertTrue(paired["passed"])
            self.assertEqual(len(paired["comparisons"]), 2)
            barrier = json.loads((output/"selection_complete.json").read_text())
            self.assertEqual(len(barrier["model_checkpoints"]), 8)
            self.assertEqual(len(barrier["reference_checkpoints"]), 4)
            self.assertEqual(len(barrier["attack_checkpoints"]), 40)
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(report["format"], train.FORMAT)
            self.assertEqual({r["condition"] for r in report["results"]}, {"A", "B", "C", "D", "tau_zero", "leak_control"})
            for row in report["results"]:
                self.assertIn("bch_genuine", row["metrics"]["test"]["face"])
                payload = np.load(train.row_directory(output, row)/"test/public_payload.npz")
                if row["kind"] == "checkpoint":
                    bits = np.load(train.row_directory(output, row)/"test/bits_tau.npz")
                    np.testing.assert_array_equal(payload["face"], bits["face_tau"])
                    self.assertEqual(payload["face"].dtype, np.float32)
                    self.assertEqual(payload["face"].shape[1]*payload["face"].itemsize*8, 16384)
            self.assertTrue((output/"bch/bch.so").is_file())
            with self.assertRaises(FileExistsError):
                train.run_experiments(self.data, self.users, self.args, {})


if __name__ == "__main__":
    unittest.main()
