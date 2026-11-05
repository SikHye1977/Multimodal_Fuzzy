"""Regression checks for independent entry points and source-run provenance."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from scripts.common import provenance, sha256


EXPERIMENTS = (
    "s05_threshold512", "s06_bch_loss_abc", "s07_fixed_dynamic",
    "s08_threshold_v3", "s09_multisample",
)


class LayoutTests(unittest.TestCase):
    def historical_config(self):
        return {
            "model_sha256": sha256(provenance.ROOT / "Thresholding Model.py"),
            "trainer_sha256": provenance.HISTORICAL_TRAINER_SHA256,
        }

    def current_config(self):
        trainer = provenance.ROOT / "scripts/bootstrap/train.py"
        return {
            "model_sha256": sha256(provenance.ROOT / "Thresholding Model.py"),
            "trainer_sha256": sha256(trainer),
            "source_format": provenance.SOURCE_FORMAT,
            "code_sha256": provenance.code_hashes(trainer, relative=True),
        }

    def test_historical_source_is_validated_against_the_pinned_snapshot(self):
        self.assertEqual(sha256(provenance.HISTORICAL_TRAINER),
                         provenance.HISTORICAL_TRAINER_SHA256)
        provenance.validate_source_code(self.historical_config())
        for field in ("model_sha256", "trainer_sha256"):
            config = self.historical_config()
            config[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                provenance.validate_source_code(config)

    def test_tampered_snapshot_is_not_accepted(self):
        config = self.historical_config()

        def changed(path):
            return "0" * 64 if Path(path) == provenance.HISTORICAL_TRAINER else sha256(path)

        with patch.object(provenance.common, "sha256", side_effect=changed), self.assertRaises(ValueError):
            provenance.validate_source_code(config)

    def test_new_source_hashes_include_shared_dependencies(self):
        config = self.current_config()
        provenance.validate_source_code(config)
        hashes = config["code_sha256"]
        for name in ("scripts/common/data.py", "scripts/common/losses.py",
                     "scripts/common/metrics.py", "scripts/common/provenance.py",
                     "scripts/common/bch_bridge.cpp", "BCH/BCH.cpp"):
            self.assertIn(name, hashes)
        self.assertTrue(all(not Path(name).is_absolute() for name in hashes))
        for change in ("missing", "modified", "unknown_format"):
            changed = deepcopy(config)
            if change == "missing":
                del changed["code_sha256"]["scripts/common/losses.py"]
            elif change == "modified":
                changed["code_sha256"]["scripts/common/losses.py"] = "0" * 64
            else:
                changed["source_format"] = "unknown"
            with self.subTest(change=change), self.assertRaises(ValueError):
                provenance.validate_source_code(changed)

    def test_each_experiment_imports_without_other_experiments_or_fusion(self):
        for folder in EXPERIMENTS:
            code = textwrap.dedent("""
                import importlib
                import importlib.abc
                import sys
                package = "scripts." + sys.argv[1]

                class Independent(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        other = fullname.startswith("scripts.s0") and not (
                            fullname == package or fullname.startswith(package + "."))
                        old = fullname.startswith("scripts.train_")
                        fusion = fullname in {"Fusion_Encoder", "Fusion_Decoder"}
                        if other or old or fusion:
                            raise AssertionError("Unexpected dependency: " + fullname)
                        return None

                sys.meta_path.insert(0, Independent())
                importlib.import_module(package + ".train")
            """)
            result = subprocess.run([sys.executable, "-B", "-c", code, folder],
                                    cwd=provenance.ROOT, capture_output=True, text=True)
            with self.subTest(experiment=folder):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
