#!/usr/bin/env python3
"""Train Thresholding Model.py from the complete frozen embedding archives.

No feature extraction, image modification or FE/key-recovery experiment occurs.
Default: face people 220/28/28, fingerprint people 5/1/1, batch 64,
100 threshold epochs then 50 reliability epochs, with validation early stopping.
Run with the project's existing feature_extractors/run.sh environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common import Embeddings, Log, Progress, checkpoint, duration, forward_pair, load_model_module, make_pairs, make_users, model_inputs, negative_permutation, sha256, training_batches, write_csv, write_json
from scripts.common.metrics import bit_statistics
from scripts.common import provenance

SPLITS = ("train", "validation", "test")
MODALITIES = ("face", "fingerprint")
# Pair columns; user_index is metadata and NEVER passed into the model.
EF, QF, EP, QP, USER = range(5)


def loss_kwargs(args):
    return {"margin": args.margin, "different_weight": args.different_weight,
            "balance_weight": args.balance_weight}


@torch.no_grad()
def evaluate(module, model, cache, rows, users, args, on_batch=None):
    model.eval()
    collected = ({}, {})
    needed = ("tau",) + tuple(f"{m}_{field}" for m in MODALITIES for field in
                              ("soft_bits", "bits", "reliability_logits", "reliability"))
    for step, offset in enumerate(range(0, len(rows), args.batch_size), start=1):
        outputs = forward_pair(model, cache, rows[offset:offset + args.batch_size])
        for destination, output in zip(collected, outputs):
            for key in needed:
                destination.setdefault(key, []).append(output[key])
        if on_batch:
            on_batch(step)
    e, q = ({key: torch.cat(value) for key, value in role.items()} for role in collected)
    if not all(bool(torch.isfinite(value).all()) for role in (e, q) for value in role.values()):
        raise RuntimeError("Nonfinite validation/test output")
    negatives = torch.as_tensor(negative_permutation(rows, users), device=e["tau"].device)
    threshold = module.threshold_loss(e, q, negatives, **loss_kwargs(args))
    targets = module.reliability_targets(e, q)
    result = {"pair_count": len(rows), "threshold_loss": {k: float(v) for k, v in threshold.items()},
              "reliability_bce": float(module.reliability_loss(e, q, targets)),
              "bits": bit_statistics(e, q, negatives),
              "tau_enrollment_mean": float(e["tau"].mean()), "tau_query_mean": float(q["tau"].mean())}
    for modality in MODALITIES:
        result["bits"][modality]["reliability_brier"] = float(torch.stack([
            (out[f"{modality}_reliability"] - targets[modality]).square().mean() for out in (e, q)]).mean())
    return result


@torch.no_grad()
def zero_tau_baseline(cache, rows, users):
    ef, ep, qf, qp = model_inputs(cache, rows)
    e = {"face_bits": (ef >= 0).to(torch.uint8), "fingerprint_bits": (ep >= 0).to(torch.uint8)}
    q = {"face_bits": (qf >= 0).to(torch.uint8), "fingerprint_bits": (qp >= 0).to(torch.uint8)}
    negatives = torch.as_tensor(negative_permutation(rows, users), device=ef.device)
    return bit_statistics(e, q, negatives)


def save_manifests(directory, data, users, partitions, fixed_pairs, args):
    write_json(directory / "splits.json", {"split_seed": args.split_seed, **partitions,
               "synthetic_pairing": True, "real_multimodal_same_person_correspondence": False,
               "fingerprints_reused_within_split": True})
    user_rows = [{"virtual_user_index": i, **asdict(user)} for i, user in enumerate(users)]
    write_csv(directory / "virtual_users.csv", list(user_rows[0]), user_rows)
    sample_rows = []
    face_lookup = {u.face_person: u.split for u in users}
    finger_lookup = {u.fingerprint_person: u.split for u in users}
    for modality in MODALITIES:
        for identity, indices in data.groups[modality].items():
            person = identity if modality == "face" else identity[0]
            split = face_lookup[person] if modality == "face" else finger_lookup[person]
            for i in indices:
                sample_rows.append({"modality": modality, "embedding_row": i, "file": data.files[modality][i],
                                    "person_id": person, "finger_id": "" if modality == "face" else identity[1],
                                    "split": split})
    write_csv(directory / "samples.csv", list(sample_rows[0]), sample_rows)
    for split, rows in fixed_pairs.items():
        records = [{"virtual_user_index": int(row[USER]),
                    "enrollment_face": data.files["face"][int(row[EF])],
                    "query_face": data.files["face"][int(row[QF])],
                    "enrollment_fingerprint": data.files["fingerprint"][int(row[EP])],
                    "query_fingerprint": data.files["fingerprint"][int(row[QP])]}
                   for row in rows]
        write_csv(directory / f"{split}_pairs.csv", list(records[0]), records)


def train_stage(module, model, cache, data, users, val_pairs, stage, args, directory, log):
    epochs = args.threshold_epochs if stage == "threshold" else args.reliability_epochs
    lr = args.threshold_lr if stage == "threshold" else args.reliability_lr
    future = args.reliability_epochs if stage == "threshold" else 0
    model.set_training_stage(stage)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    # Every parameter in the opposite path must remain bit-for-bit unchanged.
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    select = ((lambda metrics: metrics["bits"]["selection_hamming_gap"]) if stage == "threshold"
              else (lambda metrics: metrics["reliability_bce"]))
    best_path = directory / f"best_{stage}.pt"
    initial = evaluate(module, model, cache, val_pairs, users, args)
    best, best_epoch, stale = select(initial), 0, 0
    best_metrics = initial
    checkpoint(model, best_path)
    log.record({"stage": stage, "epoch": 0, "validation": initial, "selection_score": best, "best": True})
    log.message(f"\n[{stage}] 시작 | 최대 {epochs} epoch | 초기 검증 선택지표 {best:.6f}")
    progress = Progress(stage, epochs, future, args.log_interval)
    gradient_blocks = None
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            # Same data seed/epoch schedule in both stages; labels use the frozen
            # threshold model in stage two, never a moving teacher or query leak.
            pairs = make_pairs(data, users, "train", args.seed, epoch)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, 9001]))
            batches, padded = training_batches(pairs, users, args.batch_size, rng)
            val_steps = math.ceil(len(val_pairs) / args.batch_size)
            total_steps = len(batches) + val_steps
            weighted_loss, processed = 0.0, 0
            for step, (rows, negative) in enumerate(batches, start=1):
                optimizer.zero_grad(set_to_none=True)
                e, q = forward_pair(model, cache, rows)
                if stage == "threshold":
                    negative = torch.as_tensor(negative, device=args.device, dtype=torch.long)
                    loss = module.threshold_loss(e, q, negative, **loss_kwargs(args))["total"]
                else:
                    targets = module.reliability_targets(e, q)
                    loss = module.reliability_loss(e, q, targets)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite {stage} training loss at epoch {epoch}, step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), args.grad_clip,
                                                      error_if_nonfinite=True)
                if epoch == step == 1:
                    gradient_blocks = {}
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and parameter.grad is None:
                            raise RuntimeError(f"Missing gradient: {name}")
                        if parameter.grad is not None:
                            block = name.split(".")[0]
                            gradient_blocks[block] = gradient_blocks.get(block, 0.0) + float(parameter.grad.norm())
                    if not math.isfinite(float(norm)):
                        raise RuntimeError("Nonfinite gradient norm")
                optimizer.step()
                weighted_loss += float(loss.detach()) * len(rows)
                processed += len(rows)
                progress.update(epoch, step, total_steps, "train", weighted_loss / processed,
                                force=step == 1 or step == len(batches))
            metrics = evaluate(module, model, cache, val_pairs, users, args,
                               on_batch=lambda n: progress.update(epoch, len(batches) + n, total_steps,
                                                                 "validation", weighted_loss / processed))
            for name, parameter in model.named_parameters():
                if name in frozen and (parameter.grad is not None or not torch.equal(parameter.detach(), frozen[name])):
                    raise RuntimeError(f"A frozen parameter changed: {name}")
            score = select(metrics)
            improved = score < best - args.min_delta
            if improved:
                best, best_epoch, stale, best_metrics = score, epoch, 0, metrics
                checkpoint(model, best_path)
            else:
                stale += 1
            checkpoint(model, directory / f"last_{stage}.pt")
            elapsed_epoch = time.perf_counter() - epoch_start
            log.record({"stage": stage, "epoch": epoch, "train_loss": weighted_loss / processed,
                        "original_pair_count": len(pairs), "processed_pairs": processed, "supplementary_pairs": padded,
                        "validation": metrics, "selection_score": score, "best": improved,
                        "best_epoch": best_epoch, "epoch_seconds": elapsed_epoch})
            progress.finish_line()
            detail = " | ".join(f"{m}: genuine {metrics['bits'][m]['genuine_hamming_bits']:.1f} bit, "
                                f"impostor {metrics['bits'][m]['impostor_hamming_bits']:.1f} bit, "
                                f"ones {metrics['bits'][m]['ones_fraction']:.3f}" for m in MODALITIES)
            log.message(f"[{stage} {epoch}/{epochs}] train {weighted_loss / processed:.5f} | "
                        f"val score {score:.6f} | best epoch {best_epoch} | "
                        f"patience {stale}/{args.patience} | {duration(elapsed_epoch)}\n  {detail}")
            if stale >= args.patience:
                log.message(f"[{stage}] 검증 지표가 {args.patience} epoch 동안 개선되지 않아 조기 종료합니다.")
                break
    finally:
        progress.finish_line()
    if best_epoch == 0:
        log.message(f"[{stage}] 초기 모델보다 검증 선택지표가 개선되지 않았습니다. 초기 가중치를 유지합니다.")
    model.load_state_dict(module.ThresholdingModel.load_checkpoint(best_path, args.device).state_dict())
    return {"epochs_run": epoch, "best_epoch": best_epoch, "best_selection_score": best,
            "best_validation": best_metrics, "initial_validation": initial,
            "gradients_first_step": gradient_blocks, "seconds": time.perf_counter() - progress.start,
            "selection_metric": "mean(genuine_hamming_fraction - impostor_hamming_fraction)" if stage == "threshold"
                                else "genuine_pair_bit_agreement_BCE"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "feature_extractors/results/full_dataset_check")
    parser.add_argument("--output-dir", type=Path, help="Must not exist; default is a timestamped thresholding/results/train_* directory")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-epochs", type=int, default=100)
    parser.add_argument("--reliability-epochs", type=int, default=50)
    parser.add_argument("--threshold-lr", type=float, default=0.001)
    parser.add_argument("--reliability-lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--tau-bound", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--different-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10, help="Progress updates every N train/validation batches")
    args = parser.parse_args()
    for name in ("threshold_epochs", "reliability_epochs", "patience", "threads", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.batch_size < 2 or args.seed < 0 or args.split_seed < 0:
        parser.error("batch size must be >= 2; seeds must be nonnegative")
    for name in ("threshold_lr", "reliability_lr", "temperature", "tau_bound", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    for name in ("min_delta", "different_weight", "balance_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.margin) or not 0 < args.margin <= 1:
        parser.error("margin must be in (0,1]")
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT / "thresholding/results" / datetime.now().strftime("train_%Y%m%d_%H%M%S_%f"))
    for protected in (ROOT / "muct-master", ROOT / "UareU", ROOT / "feature_extractors",
                      ROOT / "BCH", ROOT / "scripts", args.embedding_dir):
        if args.output_dir.is_relative_to(protected.resolve()):
            parser.error("Output must be separate from source datasets, embeddings, and code")
    return args


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the existing environment or pass --device cpu explicitly.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Never overwrite another run or modify an existing result directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = Log(args.output_dir)
    started = time.perf_counter()
    model = None
    try:
        write_json(args.output_dir / "status.json", {"state": "running"})
        log.message(f"결과 경로: {args.output_dir}\n전체 임베딩 확인 → 사람 단위 분할 → τ 학습 → 신뢰도 학습 → 최종 시험")
        log.message("ETA는 최대 epoch 기준 추정치이며, 첫 epoch에서 보정됩니다. 조기 종료 시 더 짧아질 수 있습니다.")
        data = Embeddings(args.embedding_dir)
        users, partitions = make_users(data, args.split_seed)
        fixed_pairs = {split: make_pairs(data, users, split, args.seed, epoch=1 if split == "train" else 0)
                       for split in SPLITS}
        # Ensure validation/test negative sampling is feasible before any training.
        for split in ("validation", "test"):
            negative_permutation(fixed_pairs[split], users)
        save_manifests(args.output_dir, data, users, partitions, fixed_pairs, args)
        module = load_model_module()
        model = module.ThresholdingModel(module.ThresholdingConfig(args.temperature, args.tau_bound)).to(args.device)
        cache = {key: torch.tensor(value, device=args.device) for key, value in data.arrays.items()}
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "model_parameters": sum(p.numel() for p in model.parameters()),
                       "model_sha256": sha256(ROOT / "Thresholding Model.py"), "trainer_sha256": sha256(__file__),
                       "input_sha256": data.hashes,
                       "source_format": provenance.SOURCE_FORMAT,
                       "code_sha256": provenance.code_hashes(__file__, relative=True),
                       "purpose": "thresholding research; synthetic multimodal pairing; no FE key evaluation",
                       "train_pair_schedule": "all face images once as query each epoch; cycle all finger captures",
                       "checkpoint_resuming_supported": False})
        write_json(args.output_dir / "config.json", config)
        for split in SPLITS:
            assigned = [u for u in users if u.split == split]
            log.message(f"{split}: 얼굴 {len(assigned)}명 / 지문 {len(partitions['fingerprint_people'][split])}명 "
                        f"{len({u.finger for u in assigned})}손가락 / 등록·질의 {len(fixed_pairs[split])}쌍")
        baseline_val = zero_tau_baseline(cache, fixed_pairs["validation"], users)
        write_json(args.output_dir / "tau_zero_validation.json", baseline_val)
        summaries = {}
        for stage in ("threshold", "reliability"):
            summaries[stage] = train_stage(module, model, cache, data, users, fixed_pairs["validation"],
                                           stage, args, args.output_dir, log)
            write_json(args.output_dir / "stage_summary.json", summaries)
        model.set_training_stage("inference")
        checkpoint(model, args.output_dir / "final_model.pt")
        log.message("\n학습 및 검증 선택 완료. 고정된 시험 쌍을 한 번 평가합니다.")
        test = evaluate(module, model, cache, fixed_pairs["test"], users, args)
        baseline_test = zero_tau_baseline(cache, fixed_pairs["test"], users)
        data.assert_unchanged()
        report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
                  "stages": summaries, "test": test, "tau_zero_validation": baseline_val,
                  "tau_zero_test": baseline_test, "input_archives_unchanged": True,
                  "limitations": ["Only 7 real fingerprint participants; virtual users are not independent new people.",
                                  "Synthetic face/finger correspondence; some virtual users share a finger.",
                                  "Natural captures only; no additional image corruption augmentation.",
                                  "No FE/key-recovery or reliability calibration claim."]}
        write_json(args.output_dir / "report.json", report)
        write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"]})
        log.message(f"완료 | 총 {duration(report['seconds'])} | 최종 모델: {args.output_dir / 'final_model.pt'}")
        log.message(f"결과: {args.output_dir / 'report.json'}")
    except BaseException as error:
        if model is not None:
            try:
                checkpoint(model, args.output_dir / "interrupted.pt")
            except Exception:
                pass
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(args.output_dir / "status.json", {"state": state, "error": str(error),
                   "seconds": time.perf_counter() - started})
        log.message(f"\n실행 {state}. 이미 저장된 최적 체크포인트와 로그는 유지됩니다.")
        with (args.output_dir / "error.log").open("w", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise


if __name__ == "__main__":
    main()
