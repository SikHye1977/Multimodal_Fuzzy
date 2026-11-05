"""CPU synthetic tests only; actual biometric data never reaches an optimizer."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.s08_threshold_v3 import train as ex
from scripts.tests.helpers import synthetic_data


class ThresholdingV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args(["--device", "cpu", "--seeds", "17",
            "--epochs", "2", "--patience", "2", "--batch-size", "4", "--log-interval", "100",
            "--output-dir", str(Path(self.tmp.name)/"result")])
        self.args.data_seed = 17
        self.data, self.users, self.pairs = synthetic_data()
        self.cache = {m: torch.from_numpy(x.copy()) for m, x in self.data.arrays.items()}

    def model(self, mode="separate_512"):
        return ex.make_model(self.args, mode, 17)

    def train_steps(self, model, steps=3):
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
        rows = self.pairs["train"][[0, 4, 8]]
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            e, q = ex.forward_pair(model, self.cache, rows)
            loss = ex.loss_function(model, e, q, rows, self.users, self.args)
            loss["total"].backward()
            optimizer.step()
        return loss

    def test_four_shapes_bounds_zero_baseline_and_shared_contract(self):
        digests = []
        for mode in ex.MODES:
            with self.subTest(mode=mode):
                model = self.model(mode)
                digests.append(ex.initial_digest(model, self.cache, self.pairs["train"]))
                f, p = self.cache["face"][:3], self.cache["fingerprint"][:3]
                out = ex.forward_model(model, f, p)
                dim = 1 if mode.endswith("scalar") else 512
                for m, x in zip(ex.MODALITIES, (f, p)):
                    self.assertEqual(out[m+"_tau"].shape, (3, dim))
                    self.assertEqual(out[m+"_bits"].shape, (3, 512))
                    self.assertTrue(torch.equal(out[m+"_bits"], (x >= 0).to(torch.uint8)))
                if mode.startswith("shared"):
                    self.assertTrue(torch.equal(out["face_tau"], out["fingerprint_tau"]))
                self.train_steps(model)
                out = ex.forward_model(model, f, p)
                for m in ex.MODALITIES:
                    self.assertLessEqual(float(out[m+"_tau"].abs().max().detach()), model.config.tau_bound)
                    self.assertGreater(float(out[m+"_tau"].abs().sum().detach()), 0)
                if mode.startswith("shared"):
                    self.assertTrue(torch.equal(out["face_tau"], out["fingerprint_tau"]))
                else:
                    self.assertFalse(torch.equal(out["face_tau"], out["fingerprint_tau"]))
        self.assertEqual(len(set(digests)), 1)

    def test_gradients_reach_both_inputs_fusion_and_heads(self):
        for mode in ex.model_module.MODES:
            with self.subTest(mode=mode):
                model = self.model(mode)
                losses = self.train_steps(model)
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                for component in (*model.branches.values(), model.fusion, *model.heads.values()):
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None), 0)
                self.assertFalse(any("agreement" in k or "stability" in k for k in losses))
                self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_companion_input_matters_but_unrelated_batch_items_do_not(self):
        for mode in ex.model_module.MODES:
            with self.subTest(mode=mode):
                model = self.model(mode)
                self.train_steps(model)
                model.eval()
                f, p = self.cache["face"][:3], self.cache["fingerprint"][:3]
                with torch.no_grad():
                    normal, singleton = model(f, p), model(f[:1], p[:1])
                    wrong_p, wrong_f = model(f, -p), model(-f, p)
                for m in ex.MODALITIES:
                    torch.testing.assert_close(normal[m+"_tau"][:1], singleton[m+"_tau"], atol=1e-7, rtol=1e-5)
                    self.assertTrue(torch.equal(normal[m+"_bits"][:1], singleton[m+"_bits"]))
                self.assertFalse(torch.equal(normal["face_tau"], wrong_p["face_tau"]))
                self.assertFalse(torch.equal(normal["fingerprint_tau"], wrong_f["fingerprint_tau"]))

    def test_checkpoint_reload_and_input_validation(self):
        model = self.model()
        self.train_steps(model)
        path = Path(self.tmp.name)/"model.pt"
        model.save_checkpoint(path, {"test_only": True, "epoch": 3})
        loaded, meta = ex.load_checkpoint(path, "separate_512", "cpu")
        self.assertEqual(meta, {"test_only": True, "epoch": 3})
        inputs = [self.cache[m][:3] for m in ex.MODALITIES]
        with torch.no_grad():
            a, b = model(*inputs), loaded(*inputs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        for bad in (torch.zeros(3, 256), torch.zeros(0, 512), torch.full((3, 512), float("nan")),
                    torch.zeros(3, 512, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                model(bad, inputs[1])
        with self.assertRaises(ValueError):
            model(inputs[0], inputs[1][:2])

    def test_hard_forward_with_surrogate_backward_and_radius_boundary(self):
        logits = torch.tensor([-.2, .4], requires_grad=True)
        soft = logits.sigmoid()
        hard = logits >= 0
        bits = ex.ste(soft, hard)
        self.assertTrue(torch.equal(bits, hard.float()))
        bits.sum().backward()
        torch.testing.assert_close(logits.grad, soft.detach()*(1-soft.detach()))
        a = torch.zeros(2, 512)
        b = a.clone()
        b[0, :32], b[1, :33] = 1, 1
        d = ex.hamming(a, b)
        self.assertEqual(d.tolist(), [32, 33])
        self.assertEqual((d <= self.args.bch_t).tolist(), [True, False])
        stats = ex.distance_stats(d.numpy(), self.args.bch_t)
        self.assertEqual(stats["within_radius_count"], 1)
        self.assertEqual(stats["FAR_h"], .5)

    def test_negative_labels_mask_same_identity_and_balance_duplicates(self):
        labels = torch.tensor([0, 0, 1])
        d = torch.tensor([[0., 0., 100.], [0., 0., 100.], [100., 100., 0.]], requires_grad=True)
        loss = ex.negative_loss(d, labels, 160., 4, .5)
        self.assertAlmostEqual(float(loss.detach()), ((160-100)/160)**2)
        loss.backward()
        self.assertEqual(float(d.grad[0, 1]), 0)
        self.assertEqual(float(d.grad[0, 0]), 0)
        self.assertLess(float(d.grad[0, 2]), 0)
        duplicated = torch.tensor([0, 0, 0, 1])
        d2 = torch.where(duplicated[:, None] == duplicated[None, :], 0., 100.)
        torch.testing.assert_close(ex.negative_loss(d2, duplicated, 160., 4, .5), loss.detach())

    def test_evaluation_preserves_joint_context_and_separates_finger_scopes(self):
        users = [
            ex.common.VirtualUser("validation", "face0", "person0", "1"),
            ex.common.VirtualUser("validation", "face1", "person0", "1"),
            ex.common.VirtualUser("validation", "face2", "person0", "2"),
            ex.common.VirtualUser("validation", "face3", "person1", "1")]
        rows = np.array([[0, 1, 0, 1, 0], [4, 5, 0, 1, 1],
                         [8, 9, 4, 5, 2], [12, 13, 8, 9, 3]], dtype=np.int64)
        plan = ex.evaluation_plan(np.concatenate((rows, rows[:1])), users)
        self.assertEqual(len(plan["rows"]), 4)
        self.assertEqual(len(plan["anchors"]), 4)
        self.assertEqual(plan["labels"]["fingerprint"][0], plan["labels"]["fingerprint"][1])
        self.assertNotEqual(plan["labels"]["fingerprint"][0], plan["labels"]["fingerprint"][2])
        model = self.model()
        e, q = ex.collect(model, self.cache, plan["rows"], self.args)
        stats = ex.summarize(e, q, plan, self.args, model.config.tau_bound)
        fp = stats["modalities"]["fingerprint"]
        self.assertEqual(fp["context_pair_count"], 4)
        self.assertEqual(fp["image_pair_count"], 3)
        self.assertEqual(fp["negative_scopes"]["same_person_different_fingers"]["pair_count"], 4)
        self.assertEqual(fp["negative_scopes"]["different_people"]["pair_count"], 6)
        self.assertEqual(fp["negative_scopes"]["all_different_identities"]["pair_count"], 10)
        for m in ex.MODALITIES:
            self.assertTrue(np.all(plan["labels"][m] != plan["labels"][m][plan["swaps"][m]]))

    def test_identity_macro_primary_and_image_context_averages(self):
        users = [ex.common.VirtualUser("validation", "A", "P", "1"),
                 ex.common.VirtualUser("validation", "B", "Q", "1")]
        rows = np.array([[0, 1, 0, 1, 0], [0, 2, 0, 2, 0], [4, 5, 4, 5, 1]])
        plan = ex.evaluation_plan(rows, users)
        e, q = {}, {}
        for m in ex.MODALITIES:
            eb = torch.zeros(3, 512, dtype=torch.uint8)
            qb = eb.clone()
            qb[2, :33] = 1
            e[m+"_bits"], q[m+"_bits"] = eb, qb
            e[m+"_tau"] = q[m+"_tau"] = torch.zeros(3, 1)
        report = ex.summarize(e, q, plan, self.args, .02)
        self.assertAlmostEqual(report[ex.PRIMARY], .5)
        self.assertAlmostEqual(report["modalities"]["face"]["within_radius_fraction"], 2/3)

    def test_selection_targets_radius_and_enforces_far_and_collapse_guards(self):
        plan = ex.evaluation_plan(self.pairs["validation"], self.users)
        base, _, _ = ex.evaluate(self.model(), self.cache, plan, self.args)
        candidate = deepcopy(base)
        for m in ex.MODALITIES:
            candidate["modalities"][m]["negative_scopes"]["all_different_identities"]["mean_hamming_bits"] -= 15
        self.assertTrue(ex.selection_guard(candidate, base, self.args)[0])
        candidate[ex.PRIMARY] += .1
        candidate["mean_identity_macro_hamming_bits"] += 2
        self.assertTrue(ex.better(candidate, base, self.args))
        tests = [("far", lambda c: c["modalities"]["face"]["negative_scopes"]["all_different_identities"].update(FAR_h=1)),
                 ("swap", lambda c: c["swaps"]["wrong_face"].update(changed_identity_macro_FAR_h=1)),
                 ("balance", lambda c: c["modalities"]["face"].update(coordinate_balance_error=1)),
                 ("constant", lambda c: c["modalities"]["face"].update(constant_coordinate_fraction=1)),
                 ("occupancy", lambda c: c["modalities"]["face"].update(outside_row_band_fraction=1))]
        for name, mutate in tests:
            with self.subTest(name=name):
                changed = deepcopy(base)
                mutate(changed)
                self.assertFalse(ex.selection_guard(changed, base, self.args)[0])

    def test_full_synthetic_pipeline_and_delayed_test_evaluation(self):
        self.args.modes = list(ex.MODES)
        events = []
        original_train, original_eval = ex.train_trial, ex.evaluate
        def train(*args, **kwargs):
            events.append("train:"+args[1])
            return original_train(*args, **kwargs)
        def evaluate(model, cache, plan, args, *rest, **kwargs):
            split = self.users[int(plan["rows"][0, ex.common.USER])].split
            events.append("evaluate:"+split)
            return original_eval(model, cache, plan, args, *rest, **kwargs)
        with redirect_stdout(io.StringIO()), patch.object(ex, "train_trial", train), patch.object(ex, "evaluate", evaluate):
            report = ex.run(self.args, self.data, self.users, self.pairs, {"seed": 17}, {}, synthetic=True)
        first_test = events.index("evaluate:test")
        self.assertEqual(sum(v.startswith("train:") for v in events[:first_test]), len(ex.MODES))
        self.assertEqual(len(report["trials"]), 5)
        self.assertTrue(report["input_and_source_unchanged"])
        for trial in report["trials"]:
            folder = Path(trial["checkpoint"]).parent
            self.assertTrue((folder/"last_threshold.pt").exists())
            self.assertTrue((folder/"test_context_pairs.csv").exists())
            with np.load(folder/"test_bits_tau.npz", allow_pickle=False) as out:
                for m in ex.MODALITIES:
                    d = np.count_nonzero(out["enrollment_"+m+"_bits"] != out["query_"+m+"_bits"], axis=1)
                    np.testing.assert_array_equal(d <= 32, out[m+"_within_radius"])
            history = [json.loads(line) for line in (folder/"history.jsonl").read_text().splitlines()]
            self.assertEqual(history[1]["epoch"], 1)
            self.assertGreater(history[1]["train_loss"]["gradient_norm"], 0)
        self.assertEqual(json.loads((self.args.output_dir/"status.json").read_text())["state"], "completed")
        self.assertTrue((self.args.output_dir/"aggregate.csv").exists())

    def test_check_only_never_starts_optimizer_or_creates_output(self):
        self.data.directory = Path(self.tmp.name)/"inputs"
        with patch.object(ex.provenance, "prepare_source", return_value=(self.data, self.users, self.pairs, {"seed": 17}, {})), \
             patch.object(ex, "run", side_effect=AssertionError("must not run")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("must not optimize")), redirect_stdout(io.StringIO()):
            ex.main(["--check-only", "--output-dir", str(self.args.output_dir)])
        self.assertFalse(self.args.output_dir.exists())

    def test_argument_rejections(self):
        for options in (["--temperature", "0"], ["--seeds", "17", "17"], ["--bch-t", "0"],
                        ["--safety-margin", "32"], ["--row-lower", ".5"], ["--hard-mix", "nan"],
                        ["--face-margin-bits", "32"], ["--output-dir", str(ex.ROOT/"scripts/new_run")],
                        ["--output-dir", str(ex.ROOT/"thresholding/results/train_20261029_162319_486882/nested")]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ex.parse_args(options)


if __name__ == "__main__":
    unittest.main()

