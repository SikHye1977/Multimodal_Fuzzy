"""Synthetic-only training and numerical checks for the BCH-aware ablation."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import textwrap

import numpy as np
import torch

from scripts.s06_bch_loss_abc import train as ex
from scripts.tests.helpers import synthetic_data


class FusionFreeImportTests(unittest.TestCase):
    def test_all_five_experiments_import_without_fusion(self):
        # A fresh interpreter catches accidental imports even when another test
        # has already loaded a model into sys.modules.
        code = textwrap.dedent("""
            import importlib
            import importlib.abc
            import sys

            forbidden = {"Fusion_Encoder", "Fusion_Decoder", "scripts.train_fusion",
                         "scripts.train_thresholding", "scripts.train_reliability_distance",
                         "scripts.train_thresholding_abcd", "scripts.train_thresholding_architectures",
                         "scripts.train_thresholding_ablation", "scripts.train_thresholding_reliability_abc"}

            class BlockFusion(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in forbidden:
                        raise AssertionError("Unexpected Fusion dependency: " + fullname)
                    return None

            sys.meta_path.insert(0, BlockFusion())
            for name in (
                "s05_threshold512.train", "s06_bch_loss_abc.train",
                "s07_fixed_dynamic.train", "s08_threshold_v3.train",
                "s09_multisample.train",
            ):
                importlib.import_module("scripts." + name)
            from scripts.common.bch_evaluator import BCHEvaluator
            from scripts.s06_bch_loss_abc.train import BCHEvaluator as used
            assert used is BCHEvaluator
            assert not forbidden.intersection(sys.modules)
        """)
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class BCHABCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.build = tempfile.TemporaryDirectory()
        cls.bch = ex.BCHEvaluator(Path(cls.build.name), 32)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args(["--device", "cpu", "--seeds", "17", "--epochs", "1",
            "--batch-size", "4", "--patience", "1", "--log-interval", "100",
            "--output-dir", str(Path(self.tmp.name) / "result")])
        self.args.temperature, self.args.data_seed = .02, 17
        self.data, self.users, self.pairs = synthetic_data()
        self.cache = {m: torch.from_numpy(x.copy()) for m, x in self.data.arrays.items()}

    def test_a_is_exact_original_loss(self):
        model = ex.train_ops.make_coordinate_model(self.args, "dynamic", 17).set_stage("threshold")
        rows = self.pairs["train"]
        e, q = ex.quantize_pair(model, self.cache, rows)
        actual = ex.threshold_loss(model, e, q, rows, self.users, self.args, "A")
        expected = ex.loss_ops.coordinate_threshold_loss(model, e, q, rows, self.users, self.args)
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_ste_exact_forward_and_shared_threshold_gradient(self):
        tau = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        x = torch.tensor([[.015], [-.005]], dtype=torch.float64)
        soft = torch.sigmoid((x - tau) / .02)
        hard = x >= tau
        bits = ex.ste_bits(soft, hard)
        torch.testing.assert_close(bits, hard.double(), rtol=0, atol=0)
        d = ex.xor_distance(bits[:1], bits[1:]).sum()
        d.backward()
        self.assertGreater(abs(float(tau.grad)), 1.)
        # Identity straight-through would cancel for a shared fixed tau.
        tau2 = torch.tensor(0., requires_grad=True)
        margin = x - tau2
        identity_ste = hard.double() + (margin - margin.detach())
        (identity_ste[:1] - identity_ste[1:]).abs().sum().backward()
        self.assertEqual(float(tau2.grad), 0.)

    def test_hamming_matches_xor_and_matching_negative_has_gradient(self):
        generator = torch.Generator().manual_seed(3)
        e, q = (torch.randint(0, 2, (7, 512), generator=generator).float() for _ in range(2))
        torch.testing.assert_close(ex.xor_distance(e, q), (e != q).sum(1).float(), rtol=0, atol=0)
        torch.testing.assert_close(ex.xor_distance(e, q, matrix=True),
                                   (e[:, None] != q[None]).sum(2).float(), rtol=0, atol=0)
        soft = torch.tensor([[.1], [.2]], requires_grad=True)
        b = ex.ste_bits(soft, torch.zeros_like(soft))
        negative = (1 - ex.xor_distance(b[:1], b[1:])).square().sum()
        negative.backward()
        self.assertTrue(bool((soft.grad != 0).all()))

    def test_radius_boundary_and_gradient(self):
        d = torch.tensor([0., 27., 28., 29., 32., 80.], requires_grad=True)
        loss = ex.bch_positive(d, 32, 4)
        expected = (1 + 16 + 52**2) / (32**2 * 6)
        self.assertAlmostEqual(float(loss), expected, places=6)
        loss.backward()
        self.assertTrue(bool((d.grad[:3] == 0).all()))
        self.assertTrue(bool((d.grad[3:] > 0).all()))

    def test_negative_mining_matches_original_and_excludes_identity(self):
        generator = torch.Generator().manual_seed(7)
        e, q = (torch.rand((5, 512), generator=generator) for _ in range(2))
        labels = torch.tensor([0, 0, 1, 2, 2])
        distances = torch.cdist(e, q, p=1) / 512
        expected = ex.loss_ops.mined_negative(e, q, labels, .4, 4, .5)
        actual = ex.mined_negative(distances, labels, .4, 4, .5)
        altered = distances.masked_fill(labels[:, None] == labels[None, :], 0)
        other = ex.mined_negative(altered, labels, .4, 4, .5)
        for a, b, c in zip(actual, expected, other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_both_hard_arms_backprop_only_thresholds(self):
        for mode in ("fixed", "dynamic"):
            for arm in ("B", "C"):
                model = ex.train_ops.make_coordinate_model(self.args, mode, 17).set_stage("threshold")
                e, q = ex.quantize_pair(model, self.cache, self.pairs["train"])
                losses = ex.threshold_loss(model, e, q, self.pairs["train"], self.users, self.args, arm)
                losses["total"].backward()
                self.assertTrue(all(p.grad is None for p in model.reliability.parameters()))
                for m in ex.MODALITIES:
                    grad = model.thresholds[m].base_logit.grad
                    self.assertTrue(bool(torch.isfinite(grad).all()))
                    self.assertGreater(float(grad.abs().sum()), 0.)

    def test_actual_bch_every_error_count_through_t(self):
        rng = np.random.default_rng(18)
        enrolled = rng.integers(0, 2, (33, 512), dtype=np.uint8)
        query = enrolled.copy()
        for errors in range(33):
            query[errors, rng.choice(512, errors, replace=False)] ^= 1
        np.testing.assert_array_equal(self.bch.recover(enrolled, query), np.ones(33, dtype=np.uint8))

    def test_dedup_and_fingerprint_participant_scopes(self):
        # Multiple fingers of ONE participant; no different-person FAR exists.
        users = [replace(u, fingerprint_person="one", finger_id=u.face_person) for u in self.users]
        rows = np.concatenate([self.pairs["validation"], self.pairs["validation"][:2]])
        model = ex.train_ops.make_coordinate_model(self.args, "fixed", 17)
        e, q = ex.train_ops.collect(model, self.cache, rows, 4)
        result = ex.negative_diagnostics(e, q, rows, users, "fingerprint", 32)
        self.assertEqual(result["different_people"]["pair_count"], 0)
        self.assertIsNone(result["different_people"]["FAR_h"])
        self.assertGreater(result["same_person_different_fingers"]["pair_count"], 0)
        original = ex.negative_diagnostics({k: v[:-2] for k, v in e.items()},
            {k: v[:-2] for k, v in q.items()}, rows[:-2], users, "fingerprint", 32)
        self.assertEqual(result, original)

    def test_wrong_candidate_never_counts_as_exact_recovery(self):
        model = ex.train_ops.make_coordinate_model(self.args, "fixed", 17)
        rows = self.pairs["validation"]
        plan = ex.metric_ops.evaluation_plan(rows, self.users, "cpu")
        class FakeBCH:
            def recover(self, e, q):
                return np.where(np.count_nonzero(e != q, axis=1) <= 32, 1, 2).astype(np.uint8)
        result, _, _, _ = ex.evaluate(model, self.cache, rows, self.users, plan, self.args, FakeBCH())
        for m in ex.MODALITIES:
            values = result["unique_pairs"][m]
            self.assertEqual(values["bch_exact_recovery_fraction"], values["within_radius_fraction"])
            self.assertAlmostEqual(values["bch_wrong_candidate_fraction"], 1 - values["within_radius_fraction"])

    def test_invalid_arguments_and_existing_output_rejected(self):
        cases = [["--safety-margin", "32"], ["--bch-t", "0"], ["--arms", "A", "A"],
                 ["--seeds", "17", "17"], ["--threshold-lr", "nan"], ["--batch-size", "1"],
                 ["--output-dir", self.tmp.name]]
        for values in cases:
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ex.parse_args(values)

    def test_end_to_end_synthetic_abc_both_modes(self):
        self.args.modes = ["fixed", "dynamic"]
        with redirect_stdout(io.StringIO()):
            report = ex.run(self.args, self.data, self.users, self.pairs,
                            {"temperature": .02, "seed": 17}, {}, synthetic=True)
        self.assertEqual(len(report["trials"]), 6)
        self.assertEqual(len(report["shared_schedule_sha256"]), 1)
        self.assertEqual(len(report["initial_state_sha256"]), 2)
        self.assertTrue(report["input_and_source_unchanged"])
        self.assertEqual(json.loads((self.args.output_dir / "status.json").read_text())["state"], "completed")
        recorded = json.loads((self.args.output_dir / "config.json").read_text())["code_sha256"]
        source_names = {Path(path).name for path in recorded}
        self.assertIn("bch_evaluator.py", source_names)
        self.assertTrue({"train_fusion.py", "Fusion_Encoder.py", "Fusion_Decoder.py"}.isdisjoint(source_names))
        for trial in report["trials"]:
            model, metadata = ex.ThresholdingModel512.load_checkpoint(trial["checkpoint"])
            self.assertFalse(bool(model.reliability_fitted))
            self.assertEqual(metadata["stage"], "threshold_only")
            self.assertEqual(trial["threshold"]["epochs_run"], 1)
            path = Path(trial["checkpoint"]).parent
            with np.load(path / "test_bits_tau.npz") as arrays:
                self.assertEqual(arrays["enrollment_face_tau"].shape[1], 512)
                self.assertTrue(np.isin(arrays["query_fingerprint_bits"], [0, 1]).all())
        for name in ("comparison.csv", "aggregate.csv", "paired_deltas.csv", "zero_test.json"):
            self.assertTrue((self.args.output_dir / name).is_file())
        self.data.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
