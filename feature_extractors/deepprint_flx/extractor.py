"""Frozen flx DeepPrint_TexMinu_512; texture 256 + minutiae 256."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

_VENDOR = Path(__file__).parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
from flx.models.deep_print_arch import DeepPrint_TexMinu
from flx.image_processing.binarization import LazilyAllocatedBinarizer


class DeepPrintFlx:
    output_dim = 512

    def __init__(self, checkpoint=None, device="auto", ridge_width=3.8):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not np.isfinite(ridge_width) or ridge_width <= 0:
            raise ValueError("ridge_width must be positive and finite")
        self.ridge_width = float(ridge_width)
        self.checkpoint = Path(checkpoint or Path(__file__).parent / "weights/best_model.pyt")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Missing pretrained weights: {self.checkpoint}")
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = saved["model_state_dict"]
        texture_dim = state["texture_branch._6_linear.weight"].shape[0]
        minutia_dim = state["minutia_embedding._4_linear.weight"].shape[0]
        classes = state["texture_logits.0.weight"].shape[0]
        if (texture_dim, minutia_dim) != (256, 256):
            raise ValueError(f"Expected 256 + 256 pretrained outputs, got {texture_dim} + {minutia_dim}")
        self.model = DeepPrint_TexMinu(classes, texture_dim, minutia_dim)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.binarizer = LazilyAllocatedBinarizer(self.ridge_width)

    def preprocess(self, image_path):
        """Palette-aware grayscale -> white square padding -> 299px -> flx Gabor.

        U.are.U has no upstream loader. We adapt the optical-sensor pipeline,
        retaining every source pixel and handling its odd width/height difference.
        Ridge width 3.8 is the upstream MCYT optical default, not fitted to U.are.U.
        This is image preprocessing, unrelated to the later embedding threshold tau.
        """
        with Image.open(image_path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame fingerprint image")
            gray = np.array(image.convert("L"), dtype=np.uint8, copy=True)
        tensor = TF.to_tensor(gray)
        height, width = gray.shape
        side = max(height, width)
        left, top = (side - width) // 2, (side - height) // 2
        right, bottom = side - width - left, side - height - top
        tensor = TF.pad(tensor, [left, top, right, bottom], fill=1.0)
        tensor = TF.resize(tensor, [299, 299], antialias=True)
        tensor = self.binarizer(tensor)
        if tensor.shape != (1, 299, 299) or not torch.isfinite(tensor).all():
            raise RuntimeError("Invalid preprocessed fingerprint")
        if tensor.max() == tensor.min():
            raise ValueError(f"Fingerprint preprocessing produced a constant image: {image_path}")
        return tensor

    @torch.inference_mode()
    def embed_tensor(self, batch):
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (1, 299, 299):
            raise ValueError("Expected batch shape (N, 1, 299, 299)")
        if not torch.isfinite(batch).all():
            raise ValueError("Fingerprint batch contains nonfinite pixels")
        # Keep batch/single-image inference numerically close for downstream binarization.
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            result = self.model(batch.to(device=self.device, dtype=torch.float32))
        combined = torch.cat([result.texture_embeddings, result.minutia_embeddings], dim=1)
        combined = torch.nn.functional.normalize(combined, dim=1)
        output = combined.cpu().numpy().astype(np.float32, copy=False)
        if output.shape != (len(batch), self.output_dim) or not np.isfinite(output).all():
            raise RuntimeError("Invalid DeepPrint embedding")
        if not np.allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-5):
            raise RuntimeError("DeepPrint returned an invalid embedding norm")
        return output

    def extract_batch(self, image_paths):
        paths = list(image_paths)
        if not paths:
            raise ValueError("Empty fingerprint batch")
        return self.embed_tensor(torch.stack([self.preprocess(p) for p in paths]))

    def extract(self, image_path):
        """Return a finite, L2-normalized float32 vector of shape (512,)."""
        return self.extract_batch([image_path])[0]
