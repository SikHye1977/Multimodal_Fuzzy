"""Synthetic CPU tests; real biometric caches never reach an optimizer."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.s09_multisample import train as ex


from scripts.tests.helpers import multisample_data as synthetic_data


class MultiSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = ex.parse_args(["--device", "cpu", "--seeds", "17", "29", "--epochs", "1",
            "--patience", "1", "--batch-size", "16", "--hidden-dim", "16", "--context-dim", "8",
            "--evaluation-rounds", "1", "--log-interval", "100",
            "--output-dir", str(Path(self.tmp.name)/"result")])
        self.args.data_seed = 17
        self.data, self.users = synthetic_data()
        self.plan = ex.make_protocol(self.data, self.users, self.args, "train", 1)
        self.bank = ex.build_bank(self.data, self.plan, "cpu")

    def train_steps(self, model, steps=3):
        rows = self.plan["rows"][[0, 10, 20, 30]]
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            e, q = ex.forward_pair(model, "C", self.bank, rows)
            losses = ex.loss_ops.joint_threshold_loss(model, e, q, rows, self.users, self.args)
            losses["total"].backward()
            optimizer.step()
        return losses

    def test_sets_are_disjoint_identity_correct_cover_all_and_reproducible(self):
        ex.validate_population(self.data, self.users, self.args)
        again = ex.make_protocol(self.data, self.users, self.args, "train", 1)
        changed = ex.make_protocol(self.data, self.users, self.args, "train", 2)
        self.assertEqual(self.plan["sha256"], again["sha256"])
        self.assertNotEqual(self.plan["sha256"], changed["sha256"])
        for row in self.plan["rows"]:
            user = self.users[row[ex.common.USER]]
            self.assertEqual(user.split, "train")
            for m, ec, qc in (("face", ex.common.EF, ex.common.QF), ("fingerprint", ex.common.EP, ex.common.QP)):
                e, q = (self.plan["members"][m][row[c]] for c in (ec, qc))
                self.assertEqual((len(e), len(q)), (5, 1))
                self.assertFalse(set(e) & set(q))
                identity = user.face_person if m == "face" else user.finger
                self.assertTrue(set(e+q) <= set(self.data.groups[m][identity]))
        self.assertEqual(self.plan["capture_coverage"], {"face": 40, "fingerprint": 32})
        self.args.enroll_captures = self.args.query_captures = 3
        plan = ex.make_protocol(self.data, self.users, self.args, "validation")
        for row in plan["rows"]:
            for m, ec, qc in (("face", 0, 1), ("fingerprint", 2, 3)):
                e, q = plan["members"][m][row[ec]], plan["members"][m][row[qc]]
                self.assertEqual((len(e), len(q)), (3, 3))
                self.assertFalse(set(e) & set(q))

    def test_statistics_are_order_invariant_and_singleton_variance_is_missing(self):
        x = torch.from_numpy(self.data.arrays["face"][:5])[None]
        normal = ex.set_statistics(x)
        torch.testing.assert_close(normal, ex.set_statistics(x[:, [4, 1, 3, 0, 2]]))
        expected = torch.nn.functional.normalize(x.mean(1), dim=1)
        torch.testing.assert_close(normal[:, :512], expected)
        torch.testing.assert_close(normal[:, 512:1024], x.std(1, unbiased=True))
        self.assertEqual(float(normal[0, -1]), 1)
        one = ex.set_statistics(x[:, :1])
        self.assertTrue(torch.equal(one[:, 512:1024], torch.zeros(1, 512)))
        self.assertEqual(float(one[0, -1]), 0)
        self.assertLess(float(one[0, -2]), float(normal[0, -2]))
        with self.assertRaises(ValueError):
            ex.set_statistics(torch.cat((x[:, :1], -x[:, :1]), dim=1))

    def test_matched_arms_and_gradients_checkpoint(self):
        model = ex.make_model(self.args, 17)
        rows = self.plan["rows"][:6]
        a, b, c = (ex.forward_pair(model if arm == "C" else None, arm, self.bank, rows) for arm in "ABC")
        for role in range(2):
            for m, col in (("face", role), ("fingerprint", role+2)):
                expected = self.bank[m]["single"][rows[:, col]] >= 0
                self.assertTrue(torch.equal(a[role][m+"_bits"], expected.to(torch.uint8)))
                self.assertTrue(torch.equal(b[role][m+"_bits"], c[role][m+"_bits"]))
                self.assertTrue(torch.equal(c[role][m+"_tau"], torch.zeros(6, 512)))
        losses = self.train_steps(model)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        for component in (*model.branches.values(), model.fusion, *model.heads.values()):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None), 0)
        e, q = ex.forward_pair(model, "C", self.bank, rows)
        for m in ex.MODALITIES:
            self.assertGreater(float(e[m+"_tau"].detach().abs().sum()), 0)
            self.assertLessEqual(float(e[m+"_tau"].detach().abs().max()), model.config.tau_bound)
        path = Path(self.tmp.name)/"model.pt"
        model.save_checkpoint(path, {"synthetic": True})
        loaded, meta = ex.SetThresholdingModel.load_checkpoint(path)
        self.assertEqual(meta, {"synthetic": True})
        re, rq = ex.forward_pair(loaded, "C", self.bank, rows)
        for expected, actual in ((e, re), (q, rq)):
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)

    def test_no_query_leakage_and_no_batch_dependence(self):
        model = ex.make_model(self.args, 17)
        self.train_steps(model)
        rows = self.plan["rows"][:4].copy()
        e, q = ex.forward_pair(model, "C", self.bank, rows)
        changed = rows.copy()
        changed[:, [ex.common.QF, ex.common.QP]] = self.plan["rows"][10:14, [ex.common.QF, ex.common.QP]]
        changed_e, _ = ex.forward_pair(model, "C", self.bank, changed)
        solo_e, solo_q = ex.forward_pair(model, "C", self.bank, rows[:1])
        for m in ex.MODALITIES:
            torch.testing.assert_close(e[m+"_tau"], changed_e[m+"_tau"], atol=0, rtol=0)
            torch.testing.assert_close(e[m+"_tau"][:1], solo_e[m+"_tau"], atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(q[m+"_tau"][:1], solo_q[m+"_tau"], atol=1e-7, rtol=1e-5)
        # Raw-set inference and cached-statistic inference must agree.
        inputs = [torch.from_numpy(self.data.arrays[m])[torch.tensor([self.plan["members"][m][row[c]] for row in rows])]
                  for m, c in (("face", ex.common.EF), ("fingerprint", ex.common.EP))]
        direct = model(*inputs)
        for m in ex.MODALITIES:
            torch.testing.assert_close(direct[m+"_tau"], e[m+"_tau"])

    def test_full_pipeline_delays_test_and_exports_recomputable_results(self):
        events = []
        original_train, original_evaluate = ex.train_trial, ex.evaluate
        def train(*args, **kwargs):
            events.append("train")
            return original_train(*args, **kwargs)
        def evaluate(model, arm, bank, plan, *args, **kwargs):
            events.append("eval:"+plan["split"])
            return original_evaluate(model, arm, bank, plan, *args, **kwargs)
        with patch.object(ex, "train_trial", train), patch.object(ex, "evaluate", evaluate), redirect_stdout(io.StringIO()):
            report = ex.run(self.args, self.data, self.users, {}, synthetic=True)
        self.assertEqual(events[:events.index("eval:test")].count("train"), 2)
        self.assertEqual(len(report["trials"]), 4)
        for trial in report["trials"]:
            folder = Path(trial["folder"])
            with np.load(folder/"test_bits_tau.npz", allow_pickle=False) as output:
                for m in ex.MODALITIES:
                    d = np.count_nonzero(output["enrollment_"+m+"_bits"] != output["query_"+m+"_bits"], axis=1)
                    np.testing.assert_array_equal(d <= 32, output[m+"_within_radius"])
            if trial["arm"] == "C":
                history = [json.loads(line) for line in (folder/"history.jsonl").read_text().splitlines()]
                self.assertGreater(history[1]["train_loss"]["gradient_norm"], 0)
                self.assertEqual(trial["validation"], trial["training"]["validation"])
        self.assertTrue(report["input_and_source_unchanged"])
        self.assertEqual(json.loads((self.args.output_dir/"status.json").read_text())["state"], "completed")
        for name in ("comparison.csv", "aggregate.csv", "paired_deltas.csv", "validation_sets.json", "test_sets.json"):
            self.assertTrue((self.args.output_dir/name).exists())

    def test_check_only_and_invalid_protocols(self):
        self.data.directory = Path(self.tmp.name)/"input"
        with patch.object(ex.provenance, "prepare_source", return_value=(self.data, self.users, {}, {"seed": 17}, {})), \
             patch.object(ex, "run", side_effect=AssertionError("must not run")), \
             patch.object(torch.optim, "Adam", side_effect=AssertionError("must not optimize")), redirect_stdout(io.StringIO()):
            ex.main(["--check-only", "--output-dir", str(self.args.output_dir)])
        self.assertFalse(self.args.output_dir.exists())
        for opts in (["--query-captures", "5"], ["--seeds", "17", "17"], ["--temperature", "0"],
                     ["--safety-margin", "32"], ["--face-margin-bits", "32"], ["--evaluation-rounds", "0"],
                     ["--output-dir", str(ex.ROOT/"scripts/results")]):
            with self.subTest(opts=opts), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ex.parse_args(opts)
        bad = deepcopy(self.users)
        bad[4] = ex.common.VirtualUser("validation", bad[0].face_person, bad[4].fingerprint_person, bad[4].finger_id)
        with self.assertRaises(ValueError):
            ex.validate_population(self.data, bad, self.args)


if __name__ == "__main__":
    unittest.main()
