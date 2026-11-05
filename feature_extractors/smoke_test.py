"""Run frozen inference on local datasets, recording every attempted file.

This checks the input/output pipeline, not identity verification or FE accuracy.
Original files are read only; all derived artifacts go into the output directory.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(paths, modality):
    rows = []
    for path in paths:
        row = {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
               "person_id": "", "finger_id": "", "capture": "", "lighting": "",
               "camera": "", "annotation_suffix": "", "status": "pending", "error": "",
               "embedding_row": "", "sha256": sha256(path)}
        if modality == "face":
            match = re.fullmatch(r"i(\d+)([a-z])([a-e])-(.+)", path.stem)
            if match:
                row.update(zip(("person_id", "lighting", "camera", "annotation_suffix"), match.groups()))
        else:
            match = re.fullmatch(r"(\d+)_(\d+)_(\d+)", path.stem)
            if match:
                row.update(zip(("person_id", "finger_id", "capture"), match.groups()))
        # Unrecognized names remain included; no guessed identity or silent exclusion.
        row["filename_parsed"] = match is not None
        with Image.open(path) as image:
            row.update(mode=image.mode, width=image.width, height=image.height,
                       frames=getattr(image, "n_frames", 1))
        rows.append(row)
    return rows


def save_result(out_dir, modality, paths, rows, embeddings, model, seconds, config):
    success_rows = [r for r in rows if r["status"] == "ok"]
    vectors = np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    np.savez_compressed(out_dir / f"{modality}_embeddings.npz", embeddings=vectors,
                        files=np.array([r["file"] for r in success_rows], dtype=str))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out_dir / f"{modality}_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    unchanged = all(sha256(path) == row["sha256"] for path, row in zip(paths, rows))
    checks = {"source_files_unchanged": unchanged, "all_outputs_finite": bool(np.isfinite(vectors).all()),
              "output_shape_valid": vectors.shape == (len(success_rows), 512),
              "all_attempted_files_succeeded": len(success_rows) == len(paths)}
    extra = {}
    if success_rows:
        first_idx = next(i for i, row in enumerate(rows) if row["status"] == "ok")
        repeated = model.extract(paths[first_idx])
        repeat_error = float(np.max(np.abs(repeated - vectors[0])))
        extra["repeat_max_abs_difference"] = repeat_error
        checks["repeat_consistent"] = repeat_error <= 1e-5
        norms = np.linalg.norm(vectors, axis=1)
        extra["l2_norm_min_max"] = [float(norms.min()), float(norms.max())]
        checks["l2_normalized"] = bool(np.allclose(norms, 1, atol=1e-5))
        if len(vectors) > 1:
            max_std = float(np.max(np.std(vectors, axis=0)))
            extra["max_dimension_std"] = max_std
            checks["outputs_not_constant_across_images"] = max_std > 1e-6
    people = {r["person_id"] for r in rows if r["person_id"]}
    summary = {"modality": modality, "attempted": len(paths), "succeeded": len(success_rows),
               "failed": len(paths) - len(success_rows), "people": len(people),
               "embedding_shape": list(vectors.shape), "dtype": str(vectors.dtype),
               "source_image_modes": dict(Counter(r["mode"] for r in rows)),
               "source_image_sizes": dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
               "unparsed_filenames": sum(not r["filename_parsed"] for r in rows),
               "inference_and_preprocessing_seconds": seconds, "config": config,
               "checks": checks, **extra}
    if modality == "fingerprint":
        summary["fingers"] = len({(r["person_id"], r["finger_id"]) for r in rows})
        summary["fingers_per_person"] = {p: len({r["finger_id"] for r in rows if r["person_id"] == p}) for p in sorted(people)}
    else:
        summary["cameras"] = dict(Counter(r["camera"] for r in rows))
        summary["multiple_face_detections"] = sum(r.get("faces_detected", 0) > 1 for r in rows)
    summary["passed"] = bool(success_rows) and all(checks.values())
    return summary


def run_face(paths, out_dir, device):
    from .arcface_iresnet50 import ArcFaceIResNet50, FaceNotFoundError
    rows = inventory(paths, "face")
    model = ArcFaceIResNet50(device=device)
    embeddings = []
    start = time.perf_counter()
    for i, (path, row) in enumerate(zip(paths, rows)):
        try:
            vector, metadata = model.extract_with_metadata(path)
        except (FaceNotFoundError, ValueError) as error:
            row.update(status="failed", error=str(error))
        else:
            row.update(status="ok", embedding_row=len(embeddings), **metadata)
            embeddings.append(vector)
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            print(f"face {i + 1}/{len(paths)}; succeeded={len(embeddings)}", flush=True)
    summary = save_result(out_dir, "face", paths, rows, embeddings, model, time.perf_counter() - start, {
        "model": "InsightFace ArcFace IResNet50 / w600k_r50",
        "device": model.device, "providers": model.session.get_providers(),
        "alignment": "automatic SCRFD 5-point; no MUCT manual landmarks",
        "detector_size": [640, 640], "detector_threshold": 0.5,
        "recognition_input": [3, 112, 112], "normalization": "RGB: (pixel-127.5)/127.5; embedding L2",
        "weights_sha256": {p.name: sha256(p) for p in model.weights_dir.glob("*.onnx")},
    })
    return summary


def run_fingerprint(paths, out_dir, device, batch_size):
    from .deepprint_flx import DeepPrintFlx
    rows = inventory(paths, "fingerprint")
    model = DeepPrintFlx(device=device)
    embeddings = []
    start = time.perf_counter()
    for start_idx in range(0, len(paths), batch_size):
        tensors, batch_rows = [], []
        for path, row in zip(paths[start_idx:start_idx + batch_size], rows[start_idx:start_idx + batch_size]):
            try:
                tensors.append(model.preprocess(path))
                batch_rows.append(row)
            except ValueError as error:
                row.update(status="failed", error=str(error))
        if tensors:
            vectors = model.embed_tensor(torch.stack(tensors))
            for row, vector in zip(batch_rows, vectors):
                row.update(status="ok", embedding_row=len(embeddings))
                embeddings.append(vector)
        end = min(start_idx + batch_size, len(paths))
        if end % 40 == 0 or end == len(paths):
            print(f"fingerprint {end}/{len(paths)}; succeeded={len(embeddings)}", flush=True)
    summary = save_result(out_dir, "fingerprint", paths, rows, embeddings, model, time.perf_counter() - start, {
        "model": "flx DeepPrint_TexMinu_512", "device": model.device,
        "branches": {"texture": 256, "minutiae": 256}, "concatenation_order": ["texture", "minutiae"],
        "weights_sha256": sha256(model.checkpoint), "input": [1, 299, 299],
        "preprocessing": "palette-aware grayscale; white square pad; antialiased resize; flx Gabor binarizer",
        "ridge_width": model.ridge_width, "ridge_width_source": "upstream MCYT optical default; not calibrated on U.are.U",
        "embedding_normalization": "each branch L2, concatenate, joint L2",
        "frozen": not model.model.training and all(not p.requires_grad for p in model.model.parameters()),
        "batch_size": batch_size,
    })
    if summary["succeeded"]:
        valid_paths = [p for p, r in zip(paths, rows) if r["status"] == "ok"][:2]
        batch = model.extract_batch(valid_paths)
        singles = np.stack([model.extract(p) for p in valid_paths])
        difference = float(np.max(np.abs(batch - singles)))
        summary["batch_vs_singleton_max_abs_difference"] = difference
        summary["checks"]["batch_and_singleton_consistent"] = difference <= 1e-5
        summary["checks"]["model_frozen"] = summary["config"]["frozen"]
        summary["passed"] = all(summary["checks"].values())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["both", "face", "fingerprint"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 = every original image; otherwise first N sorted files per modality")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--face-root", type=Path, default=ROOT / "muct-master/extracted/jpg")
    parser.add_argument("--fingerprint-root", type=Path, default=ROOT / "UareU")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("limit must be nonnegative and batch-size must be positive")
    out_dir = args.output_dir or ROOT / "feature_extractors/results" / datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
    # Never write derived data inside either source dataset.
    for source in (ROOT / "muct-master", args.face_root, args.fingerprint_root):
        if out_dir.resolve().is_relative_to(source.resolve()):
            parser.error("output-dir must be outside the original datasets")
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "frozen inference smoke test, not recognition accuracy evaluation",
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                              "cuda_available": torch.cuda.is_available(),
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "limit": args.limit, "modalities": {}}
    for modality, source, pattern in [("face", args.face_root, "*.jpg"), ("fingerprint", args.fingerprint_root, "*.tif")]:
        if args.modality not in {"both", modality}:
            continue
        paths = sorted(source.resolve().glob(pattern))
        if modality == "face":
            mirrored = [p for p in paths if p.stem.startswith("ir")]
            report["muct_mirrored_files_excluded"] = len(mirrored)
            paths = [p for p in paths if not p.stem.startswith("ir")]
        report[f"{modality}_available_originals"] = len(paths)
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            raise FileNotFoundError(f"No {pattern} images found in {source}")
        summary = run_face(paths, out_dir, args.device) if modality == "face" else run_fingerprint(paths, out_dir, args.device, args.batch_size)
        report["modalities"][modality] = summary
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(m["passed"] for m in report["modalities"].values())
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(out_dir / "report.json"),
                      "counts": {k: [v["succeeded"], v["attempted"]] for k, v in report["modalities"].items()}}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
