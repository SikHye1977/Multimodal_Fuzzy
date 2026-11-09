"""Independent C1 publication tests using synthetic embeddings only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.common.paths import ROOT
from scripts.tests.helpers import multisample_data
from . import attacks
from .c1 import parser, restore_checkpoint, run_experiments
from .diagnostics import AnchorShiftControl
from .model import QuantileConfig, QuantileThreshold, pack_codes, unpack_codes
from .protocol import batch_forward, make_plan, prepare_bank, validate_population
from .sketch import public_sketch


class C1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data, self.users = multisample_data()
        self.args = parser().parse_args(["--device", "cpu", "--threads", "1", "--batch-size", "4",
            "--fit-rounds", "1", "--attack-rounds", "1", "--attack-epochs", "1", "--attack-batch-size", "4"])
        self.plan = make_plan(self.data, self.users, "train", 17)
        self.bank = attacks.add_sketches(prepare_bank(self.data, self.plan, "cpu"), 32)

    def test_person_split_registration_and_query_separation(self):
        validate_population(self.data, self.users)
        for split in ("train", "validation", "test"):
            plan = make_plan(self.data, self.users, split, 17)
            self.assertEqual(plan, make_plan(self.data, self.users, split, 17))
            for records in plan["modalities"].values():
                for row in records:
                    self.assertEqual(len(row["enrollment"]), 5)
                    self.assertFalse(set(row["enrollment"]) & set(row["queries"]))

    def test_train_only_fit_packed_payload_and_enrollment_only_inference(self):
        model = QuantileThreshold(QuantileConfig(helper_bits=4))
        with self.assertRaises(ValueError):
            model.fit(self.bank, {**self.plan, "split": "test"})
        model.fit(self.bank, self.plan)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
        positions = torch.arange(len(self.bank["rows"]))
        out = batch_forward(model, self.bank, positions)
        for column, (m, name) in enumerate((("face", "face"), ("fingerprint", "finger"))):
            payload = out[name+"_payload"]
            self.assertEqual(tuple(payload.shape), (len(positions), 256))
            self.assertEqual(payload.dtype, torch.uint8)
            self.assertTrue(torch.equal(pack_codes(unpack_codes(payload, 4), 4), payload))
            self.assertTrue(torch.equal(model.tau_from_payload(payload, name), out[name+"_tau"]))
            self.assertTrue(torch.equal(out[name+"_target_bits"], self.bank[m]["target"][self.bank["rows"][:, column]]))
            self.assertLessEqual(float(out[name+"_tau"].abs().max()), .020000001)
            self.bank[m]["queries"].fill_(999)
        after = batch_forward(model, self.bank, positions)
        for key in out:
            self.assertTrue(torch.equal(out[key], after[key]))
        self.data.assert_unchanged()

    def test_public_attack_inputs_and_positive_control(self):
        dataset = attacks.freeze_dataset(AnchorShiftControl(.02), self.bank, self.args)
        self.assertTrue(attacks.qualification(dataset, self.args)["passed"])
        prior = attacks.estimate_prior(dataset)
        for kind, view in attacks.SPECS:
            net = attacks.AttackNet(kind, view, dataset["helper"].shape[1], prior).initialize_joint(dataset["sketch"].shape[1])
            before = attacks.predict(net, dataset)
            after = attacks.predict(net, {**dataset, "target": 1-dataset["target"],
                                           "identities": dataset["identities"]+999})
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        x = torch.randint(0, 2, (3, 512)).float()
        y = torch.randint(0, 2, (3, 512)).float()
        torch.testing.assert_close(public_sketch((x+y)%2, 32), (public_sketch(x, 32)+public_sketch(y, 32))%2)

    def test_import_and_cli_do_not_require_other_experiments(self):
        code = '''
import importlib.abc, runpy, sys
class BlockOtherExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("scripts.s") and not fullname.startswith("scripts.s16_quantile_thresholding")) or "Fusion" in fullname:
            raise AssertionError("Unexpected experiment import: " + fullname)
sys.meta_path.insert(0, BlockOtherExperiments())
sys.argv = ["c1", "--help"]
runpy.run_module("scripts.s16_quantile_thresholding.c1", run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    @unittest.skipUnless(shutil.which("g++"), "Actual BCH requires existing g++")
    def test_synthetic_end_to_end_checkpoint_barrier_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="c1-test-") as tmp:
            self.args.output_dir = Path(tmp)/"run"
            with redirect_stdout(io.StringIO()):
                output = run_experiments(self.data, self.users, self.args, {})
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["endpoints"], 3)
            self.assertEqual(status["fresh_attackers"], 12)
            barrier = json.loads((output/"selection_complete.json").read_text())
            self.assertEqual(len(barrier["model_checkpoints"]), 1)
            self.assertEqual(len(barrier["attack_checkpoints"]), 12)
            report = json.loads((output/"report.json").read_text())
            self.assertEqual({r["condition"] for r in report["results"]}, {"C1", "tau_zero", "leak_control"})
            self.assertTrue((output/"bch/bch.so").is_file())
            for row in report["results"]:
                for m in ("face", "fingerprint"):
                    self.assertIn("bch_genuine", row["metrics"]["test"][m])
            model, _ = restore_checkpoint(output/"models/C1/fitted.pt", "cpu")
            self.assertEqual(model.config.helper_bits, 4)
            with self.assertRaises(FileExistsError):
                run_experiments(self.data, self.users, self.args, {})


if __name__ == "__main__":
    unittest.main()
