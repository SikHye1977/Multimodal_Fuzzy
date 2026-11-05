"""Frozen InsightFace w600k_r50 (IResNet50), using automatic SCRFD alignment."""
from pathlib import Path

import cv2
import numpy as np
import torch  # Load the existing CUDA runtime before creating ORT sessions.
import onnxruntime as ort

from .vendor.face_align import norm_crop
from .vendor.scrfd import SCRFD


class FaceNotFoundError(ValueError):
    """No face passed the configured detector threshold."""


class ArcFaceIResNet50:
    output_dim = 512

    def __init__(self, weights_dir=None, device="auto", detection_threshold=0.5):
        self.weights_dir = Path(weights_dir or Path(__file__).parent / "weights")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use device='cpu' or a GPU-enabled process")
        if not 0 < detection_threshold <= 1:
            raise ValueError("detection_threshold must be in (0, 1]")
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {
                "device_id": 0, "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC", "use_tf32": 0,
            }))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        sessions = []
        for filename in ("w600k_r50.onnx", "det_10g.onnx"):
            path = self.weights_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing pretrained weights: {path}")
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            if self.device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("ORT CUDA failed to load. Use feature_extractors/run.sh to set local CUDA library paths.")
            sessions.append(session)
        self.session, detector_session = sessions
        self.detector = SCRFD(session=detector_session)
        self.detector.det_thresh = detection_threshold
        self.detector.input_size = (640, 640)
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[1:] != [3, 112, 112]:
            raise ValueError("Unexpected ArcFace input shape")
        if self.session.get_outputs()[0].shape[-1] != self.output_dim:
            raise ValueError("Expected a 512-dimensional recognition model")

    def extract_with_metadata(self, image_path):
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        boxes, landmarks = self.detector.detect(image)
        if len(boxes) == 0:
            raise FaceNotFoundError(f"No face found: {image_path.name}")
        # MUCT contains one intended face. Record multiple detections explicitly.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected = int(np.argmax(areas))
        aligned = norm_crop(image, landmarks[selected], image_size=112)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        raw = self.session.run(None, {self.input_name: blob})[0]
        if raw.shape != (1, self.output_dim) or not np.isfinite(raw).all():
            raise RuntimeError(f"Invalid ArcFace output: {raw.shape}")
        raw_norm = float(np.linalg.norm(raw[0]))
        if raw_norm <= 1e-12:
            raise RuntimeError("ArcFace returned a zero vector")
        embedding = np.asarray(raw[0] / raw_norm, dtype=np.float32)
        return embedding, {
            "alignment": "automatic_scrfd_5point", "faces_detected": len(boxes),
            "selected_face": selected, "selection": "largest_bounding_box",
            "detection_score": float(boxes[selected, 4]), "raw_embedding_norm": raw_norm,
        }

    def extract(self, image_path):
        """Return a finite, L2-normalized float32 vector of shape (512,)."""
        return self.extract_with_metadata(image_path)[0]
