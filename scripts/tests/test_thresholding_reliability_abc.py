"""Synthetic checks only: no optimizer ever receives actual biometric data."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts import train_thresholding_reliability_abc as ex


def synthetic_data():
    rng = np.random.default_rng(91)
    users, groups, files, arrays, cameras = [], {}, {}, {}, []
    for m in ex.MODALITIES:
        groups[m], files[m], arrays[m] = {}, [], []
        for split_index, split in enumerate(ex.common.SPLITS):
            for person in range(3):
                name = f"{split_index}{person}"
                identity = name if m == "face" else (name, "1")
                center = rng.normal(size=512)
                indices = []
                for capture in range(4):
                    indices.append(len(arrays[m]))
                    x = center + rng.normal(scale=.45, size=512)
                    arrays[m].append((x / np.linalg.norm(x)).astype(np.float32))
                    files[m].append(f"synthetic_{m}_{name}_{capture}")
                    if m == "face":
                        cameras.append("a" if capture == 0 else "b")
                groups[m][identity] = indices
                if m == "face":
                    users.append(ex.common.VirtualUser(split, name, name, "1"))
        arrays[m] = np.stack(arrays[m])
    original = {m: a.copy() for m, a in arrays.items()}

    def assert_unchanged():
        for m in ex.MODALITIES:
            np.testing.assert_array_equal(original[m], arrays[m])

    data = SimpleNamespace(groups=groups, files=files, arrays=arrays, cameras=cameras,
                           hashes={}, assert_unchanged=assert_unchanged)
    pairs = {s: ex.common.make_pairs(data, users, s, 17, epoch=1 if s == "train" else 0)
             for s in ex.common.SPLITS}
    return data, users, pairs


class ReliabilityABCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args([
            "--device", "cpu", "--seeds", "17", "--threshold-epochs", "1",
            "--reliability-epochs", "1", "--batch-size", "4", "--patience", "1",
            "--log-interval", "100", "--output-dir", str(Path(self.tmp.name) / "result")])
        self.args.temperature, self.args.data_seed = .02, 17
        self.data, self.users, self.pairs = synthetic_data()
        self.cache = {m: torch.from_numpy(x.copy()) for m, x in self.data.arrays.items()}

    def model(self, arm):
        return ex.make_model(self.args, arm, 17)

    def statistics(self, model):
        return ex.fit_statistics(model, self.cache, self.data, self.users, self.pairs["train"], self.args)

    def test_initial_bits_and_reliability_identical_across_arms(self):
        stats = self.statistics(self.model("A"))
        expected = None
        for arm in ex.ARMS:
            model = self.model(arm)
            for m in ex.MODALITIES:
                model.reliability[m].set_statistics(stats[m])
            out = model(self.cache["face"][:4], self.cache["fingerprint"][:4])
            for m in ex.MODALITIES:
                self.assertEqual(out[f"{m}_tau"].shape, (4, 1))
                self.assertEqual(out[f"{m}_reliability"].shape, (4, 512))
                self.assertTrue(torch.equal(out[f"{m}_bits"], (self.cache[m][:4] >= 0).to(torch.uint8)))
            if expected is not None:
                for key in out:
                    torch.testing.assert_close(out[key], expected[key], atol=0, rtol=0)
            expected = out
        b, c = self.model("B"), self.model("C")
        for name, param in b.reliability.state_dict().items():
            torch.testing.assert_close(param, c.reliability.state_dict()[name], atol=0, rtol=0)

    def test_tau_scalars_independent_and_bounded(self):
        model = self.model("C")
        with torch.no_grad():
            model.thresholds["face"].base_logit.fill_(.4)
            model.thresholds["fingerprint"].base_logit.fill_(-.4)
            for head in model.thresholds.values():
                head.delta.weight.fill_(.5)
        f, p = self.cache["face"][:4], self.cache["fingerprint"][:4]
        original, changed_p, changed_f = model(f, p), model(f, -p), model(-f, p)
        for suffix in ("tau", "bits", "reliability"):
            torch.testing.assert_close(original[f"face_{suffix}"], changed_p[f"face_{suffix}"], atol=0, rtol=0)
            torch.testing.assert_close(original[f"fingerprint_{suffix}"], changed_f[f"fingerprint_{suffix}"], atol=0, rtol=0)
        for m in ex.MODALITIES:
            self.assertLess(float(original[f"{m}_tau"].abs().max()), model.config.tau_bound)
        self.assertFalse(torch.equal(original["face_tau"], changed_f["face_tau"]))

    def test_threshold_backward_and_reliability_detachment(self):
        model = self.model("C").set_stage("threshold")
        rows = self.pairs["train"][[0, 4, 8]]
        e, q = ex.common.forward_pair(model, self.cache, rows)
        ex.tau_losses(model, e, q, rows, self.users, self.args)["total"].backward()
        for m in ex.MODALITIES:
            self.assertGreater(float(model.thresholds[m].delta.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.thresholds[m].parameters()))
        self.assertTrue(all(p.grad is None for p in model.reliability.parameters()))
        # Even accidentally unfreezing tau must not allow BCE to move it.
        model.set_stage("reliability")
        for p in model.thresholds.parameters():
            p.requires_grad_(True)
        e, q = ex.common.forward_pair(model, self.cache, rows)
        ex.reliability_loss(e, q).backward()
        self.assertTrue(all(p.grad is None for p in model.thresholds.parameters()))

    def test_reliability_step_cannot_change_bits_or_fitted_statistics(self):
        for arm in ex.ARMS:
            model = self.model(arm)
            for m, stats in self.statistics(model).items():
                model.reliability[m].set_statistics(stats)
            model.set_stage("reliability")
            frozen = model.fixed_reliability_state()
            rows = self.pairs["train"][:4]
            before, _ = ex.collect(model, self.cache, rows, 4)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
            e, q = ex.common.forward_pair(model, self.cache, rows)
            ex.reliability_loss(e, q).backward()
            optimizer.step()
            ex.assert_fixed(model, frozen)
            after, _ = ex.collect(model, self.cache, rows, 4)
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)

    def test_statistics_ignore_heldout_and_duplicate_virtual_mappings(self):
        model = self.model("B")
        original = self.statistics(model)
        for m in ex.MODALITIES:
            self.cache[m][12:] = 10
        self.assertEqual(original, self.statistics(model))
        duplicated = ex.fit_statistics(model, self.cache, self.data, self.users + [self.users[0]],
                                       self.pairs["train"], self.args)
        self.assertEqual(original, duplicated)
        with self.assertRaisesRegex(ValueError, "training participants"):
            ex.fit_statistics(model, self.cache, self.data, self.users, self.pairs["test"], self.args)
        for m in ex.MODALITIES:
            self.assertEqual(original[m]["unique_capture_count"], 12)
            self.assertEqual(original[m]["variance_person_count"], 3)

    def test_fingerprint_variance_weights_people_equally(self):
        # Add another finger of participant 00 to expose finger-vs-person weighting.
        identity = ("00", "2")
        indices = [12, 13, 14, 15]
        self.data.groups["fingerprint"][identity] = indices
        users = self.users + [ex.common.VirtualUser("train", "00", *identity)]
        model = self.model("B")
        actual = ex.fit_statistics(model, self.cache, self.data, users, self.pairs["train"], self.args)
        variances = [self.cache["fingerprint"][i:i+4].double().var(0) for i in (0, 4, 8, 12)]
        expected = ((variances[0] + variances[3])/2 + variances[1] + variances[2])/3
        expected = ((1-self.args.variance_shrinkage)*expected + self.args.variance_shrinkage*expected.mean())
        expected = expected.clamp_min(self.args.variance_floor**2).sqrt()
        np.testing.assert_allclose(actual["fingerprint"]["variability_scale"], expected.numpy())

    def test_scalar_oracle_matches_brute_force_with_ties_and_bounds(self):
        rng = np.random.default_rng(98)
        for _ in range(30):
            x = (rng.integers(-5, 6, size=32) / 100).astype(np.float32)
            for target in (rng.integers(0, 2, size=32), np.zeros(32), np.ones(32)):
                for bound in (None, .02, .0001):
                    candidates = np.r_[np.unique(x), np.nextafter(x.max(), np.float32(np.inf))]
                    if bound is not None:
                        candidates = np.r_[candidates[(candidates >= -bound) & (candidates <= bound)],
                                           np.float32(-bound), np.float32(bound)]
                    expected = min(np.count_nonzero((x >= t) != target) for t in candidates)
                    count, tau = ex.scalar_oracle(x, target, bound)
                    self.assertEqual(count, expected)
                    self.assertEqual(count, np.count_nonzero((x >= np.float32(tau)) != target))

    def test_probability_metrics_and_fractional_ties(self):
        labels = np.array([1, 0, 1, 1], bool)
        logits = np.repeat(np.log(3), 4)
        metrics = ex.reliability_metrics(labels, logits, .75, np.ones(4))
        self.assertAlmostEqual(metrics["agreement_auroc"], .5)
        self.assertAlmostEqual(metrics["error_average_precision"], .25)
        self.assertAlmostEqual(metrics["brier"], .1875)
        self.assertAlmostEqual(metrics["ece_10"], 0.)
        self.assertAlmostEqual(metrics["lowest10_error_precision"], .25)
        self.assertAlmostEqual(metrics["bce"], metrics["constant_bce"])
        perfect = ex.reliability_metrics(labels, np.where(labels, 8., -8.), .75)
        self.assertEqual(perfect["agreement_auroc"], 1)
        self.assertEqual(perfect["error_average_precision"], 1)
        self.assertEqual(perfect["lowest10_error_precision"], 1)

    def test_checkpoint_round_trip_and_training_schedules(self):
        schedules = {}
        for arm in ex.ARMS:
            model = self.model(arm)
            path = Path(self.tmp.name) / f"{arm}.pt"
            model.save_checkpoint(path, {"epoch": 2})
            loaded, meta = ex.VariabilityThresholdModel.load_checkpoint(path)
            self.assertEqual(meta, {"epoch": 2})
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
            ex.make_schedule(self.data, self.users, self.args, 17, 1, "reliability", schedules)
        self.assertEqual(len(schedules), 1)

    def test_cli_rejects_existing_output_and_nonfinite_values(self):
        for extra in (("--output-dir", self.tmp.name), ("--tau-base-bound", "nan"),
                      ("--variance-shrinkage", "1.1"), ("--max-auroc-drop", "nan")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ex.parse_args(list(extra))

    def test_all_three_arms_end_to_end_on_synthetic_data(self):
        with redirect_stdout(io.StringIO()):
            report = ex.run_experiments(self.args, self.data, self.users, self.pairs,
                                        {"temperature": .02, "seed": 17}, {}, synthetic=True)
        self.assertEqual(len(report["trials"]), 3)
        self.assertTrue(report["synthetic_test_only"])
        self.assertTrue(report["input_and_source_unchanged"])
        self.assertEqual(json.loads((self.args.output_dir/"status.json").read_text())["state"], "completed")
        for name in ("comparison.csv", "aggregate.csv", "paired_deltas.csv", "oracle_diagnostics.json"):
            self.assertTrue((self.args.output_dir/name).is_file())
        a, b, c = report["trials"]
        for split in ("validation", "test"):
            self.assertEqual(a[split]["bits"], b[split]["bits"])
        for trial in (a, b, c):
            folder = Path(trial["checkpoint"]).parent
            model, _ = ex.VariabilityThresholdModel.load_checkpoint(trial["checkpoint"])
            self.assertEqual(model.config.arm, trial["arm"])
            with np.load(folder/"test_outputs.npz") as archive:
                self.assertEqual(archive["query_face_bits"].shape, (len(self.pairs["test"]), 512))
                self.assertTrue(np.isfinite(archive["query_fingerprint_reliability_logits"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_backward_both_training_stages(self):
        self.args.device = "cuda"
        cache = {m: v.cuda() for m, v in self.cache.items()}
        rows = self.pairs["train"][[0, 4, 8]]
        for stage in ("threshold", "reliability"):
            model = self.model("C").set_stage(stage)
            e, q = ex.common.forward_pair(model, cache, rows)
            loss = (ex.tau_losses(model, e, q, rows, self.users, self.args)["total"]
                    if stage == "threshold" else ex.reliability_loss(e, q))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in model.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
