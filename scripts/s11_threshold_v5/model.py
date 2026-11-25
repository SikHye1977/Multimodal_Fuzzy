"""Stable import path for the user-requested model filename containing spaces."""
import importlib.util
import sys

from scripts.common.paths import ROOT

MODEL_PATH = ROOT / "Thresholding Model V5.py"
MODULE_NAME = "thresholding_model_v5"
if MODULE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
module = sys.modules[MODULE_NAME]
ThresholdingModelV5 = module.ThresholdingModelV5
V5Config = module.V5Config
MODES = module.MODES
