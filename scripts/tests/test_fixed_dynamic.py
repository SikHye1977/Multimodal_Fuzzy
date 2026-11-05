"""Synthetic-only checks; never optimize real biometric data."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.s07_fixed_dynamic import train as ex
from scripts.tests.helpers import synthetic_data


class CFixedDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args(["--device", "cpu", "--seeds", "17", "--epochs", "2",
            "--batch-size", "4", "--patience", "2", "--log-interval", "100",
            "--output-dir", str(Path(self.tmp.name)/"result")])
        self.args.temperature, self.args.data_seed = .02, 17
        self.data, self.users, self.pairs = synthetic_data()
        self.cache = {m: torch.from_numpy(x.copy()) for m, x in self.data.arrays.items()}

    def test_initial_outputs_and_training_schedules_match(self):
        digests, schedules, batches = [], {}, []
        for mode in ex.MODES:
            model = ex.train_ops.make_coordinate_model(self.args, mode, 17)
            digests.append(ex.initial_output_digest(model, self.cache, self.pairs["train"]))
            rows, _ = ex.train_ops.make_schedule(self.data, self.users, self.args, 17, 1, "threshold", schedules)
            batches.append(rows)
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(len(schedules), 1)
        for a, b in zip(*batches):
            np.testing.assert_array_equal(a[0], b[0])

    def test_dynamic_uses_own_input_and_c_loss_updates_only_thresholds(self):
        for mode in ex.MODES:
            model = ex.train_ops.make_coordinate_model(self.args, mode, 17).set_stage("threshold")
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
            rows = self.pairs["train"]
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                e, q = ex.train_ops.quantize_pair(model, self.cache, rows)
                loss = ex.loss_ops.bch_threshold_loss(model, e, q, rows, self.users, self.args, "C")["total"]
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.reliability.parameters()))
                for m in ex.MODALITIES:
                    self.assertGreater(float(model.thresholds[m].base_logit.grad.abs().sum()), 0.)
                optimizer.step()
            for m in ex.MODALITIES:
                out = model.quantize(m, self.cache[m][:4])
                tau = out["tau"]
                self.assertEqual(tau.shape, (4, 512))
                self.assertLessEqual(float(tau.abs().max()), .02)
                std = float(tau.std(dim=0, unbiased=False).mean())
                self.assertGreater(std, 0.) if mode == "dynamic" else self.assertEqual(std, 0.)
                # A different batch companion must not change this image's tau.
                single = model.quantize(m, self.cache[m][:1])["tau"]
                torch.testing.assert_close(single, tau[:1], atol=1e-8, rtol=1e-5)

    def test_success_includes_32_but_not_33_and_deduplicates(self):
        rows = self.pairs["validation"]
        distances = np.array([0, 28, 32, 33, 64, 80, 31, 32, 100])
        self.assertEqual(len(rows), len(distances))
        e, q = {}, {}
        for m in ex.MODALITIES:
            bits = torch.zeros((len(rows), 512), dtype=torch.uint8)
            for i, n in enumerate(distances):
                bits[i, :n] = 1
            e[m+"_bits"], q[m+"_bits"] = torch.zeros_like(bits), bits
            e[m+"_tau"], q[m+"_tau"] = torch.zeros_like(bits, dtype=torch.float32), torch.zeros_like(bits, dtype=torch.float32)
        # Repeat a successful pair; its contribution must not be counted twice.
        rows = np.concatenate((rows, rows[:1]))
        e, q = ({k: torch.cat((v, v[:1])) for k, v in out.items()} for out in (e, q))
        plan = ex.metric_ops.evaluation_plan(rows, self.users, "cpu")
        model = ex.train_ops.make_coordinate_model(self.args, "fixed", 17)
        with patch.object(ex.train_ops, "collect", return_value=(e, q)):
            result, _, _ = ex.evaluate(model, self.cache, rows, self.users, plan, self.args)
        for m in ex.MODALITIES:
            metric = result["unique_pairs"][m]
            self.assertEqual(metric["genuine_pair_count"], 9)
            self.assertEqual(metric["within_radius_count"], 5)
            self.assertAlmostEqual(metric["within_radius_fraction"], 5/9)
            self.assertAlmostEqual(metric["zero_error_fraction"], 1/9)
            self.assertNotIn("bch_exact_recovery_fraction", metric)

    def test_selection_prioritizes_within_radius_not_exact_recovery(self):
        best = {ex.PRIMARY: .5, "mean_unique_genuine_hamming_bits": 50., "mean_unique_bch_exact_recovery_fraction": 1.}
        candidate = {ex.PRIMARY: .6, "mean_unique_genuine_hamming_bits": 60., "mean_unique_bch_exact_recovery_fraction": 0.}
        self.assertTrue(ex.better(candidate, best, self.args))
        candidate[ex.PRIMARY] = .4
        candidate["mean_unique_genuine_hamming_bits"] = 10.
        self.assertFalse(ex.better(candidate, best, self.args))

    def test_invalid_arguments_and_output_reuse_rejected(self):
        for values in (["--seeds", "17", "17"], ["--safety-margin", "32"], ["--threshold-lr", "0"],
                       ["--output-dir", self.tmp.name], ["--arms", "A"], ["--batch-size", "1"]):
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ex.parse_args(values)

    def test_synthetic_pipeline_exports_paired_comparison_without_bch(self):
        with patch.object(ex, "BCHEvaluator", side_effect=AssertionError("No decoder is required")), redirect_stdout(io.StringIO()):
            report = ex.run(self.args, self.data, self.users, self.pairs, {"temperature": .02, "seed": 17}, {}, synthetic=True)
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(len(report["paired_deltas"]), 4)
        self.assertEqual(len(report["shared_schedule_sha256"]), 2)
        self.assertTrue(report["input_and_source_unchanged"])
        config = json.loads((self.args.output_dir/"config.json").read_text())
        self.assertFalse(config["BCH_decoding"])
        self.assertFalse(config["test_used_for_selection"])
        self.assertFalse((self.args.output_dir/"bch_evaluator").exists())
        for trial in report["trials"]:
            folder = Path(trial["checkpoint"]).parent
            model, meta = ex.ThresholdingModel512.load_checkpoint(trial["checkpoint"])
            self.assertEqual(meta["selection_metric"], ex.PRIMARY)
            self.assertFalse(bool(model.reliability_fitted))
            self.assertEqual(trial["threshold"]["epochs_run"], 2)
            for split in ("validation", "test"):
                with np.load(folder/f"{split}_bits_tau.npz") as a:
                    for m in ex.MODALITIES:
                        self.assertEqual(a[f"enrollment_{m}_tau"].shape[1], 512)
                        d=np.count_nonzero(a[f"enrollment_{m}_bits"]!=a[f"query_{m}_bits"], axis=1)
                        np.testing.assert_array_equal(a[f"{m}_within_radius"], d <= 32)
            if trial["mode"] == "fixed":
                self.assertEqual(trial["test"]["unique_pairs"]["face"]["tau_across_unique_images_std_mean"], 0.)
        with (self.args.output_dir/"comparison.csv").open() as f:
            rows=list(csv.DictReader(f))
        lookup={(r['mode'],r['split'],r['modality']):r for r in rows}
        for delta in report['paired_deltas']:
            key=delta['split'],delta['modality']
            expected=float(lookup[('dynamic',*key)]['within_radius_percent'])-float(lookup[('fixed',*key)]['within_radius_percent'])
            self.assertAlmostEqual(delta['within_radius_percent_delta'],expected)
        self.assertEqual(json.loads((self.args.output_dir/"status.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main()
