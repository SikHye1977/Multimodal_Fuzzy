"""V5: enrollment/query loss, negative loss and coordinate-head ablations.

Run through feature_extractors/run.sh -m scripts.s11_threshold_v5.train.
No extraction, dataset mutation or training happens on import / --check-only.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .model import MODEL_PATH, MODES, ThresholdingModelV5, V5Config
from scripts.common.paths import ROOT
from scripts.common.provenance import prepare_source
from scripts.common.runtime import Log, Progress, duration, sha256, write_csv, write_json
from .evaluation import evaluate, monitor_training, selection
from .losses import ARMS, DESCRIPTIONS, SETTINGS, compute_loss
from .protocol import MODEL_NAMES, batch_forward, describe_plan, make_plan, prepare_bank, validate_population

FORMAT = "thresholding-v5-heldout-query-v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-run", type=Path, default=ROOT/"thresholding/results/train_20261029_162319_486882")
    p.add_argument("--embedding-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, help="New directory under thresholding/results (or /tmp for smoke tests)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--modes", nargs="+", choices=MODES, default=["user_vector"],
                   help="Default isolates A-D improvements with user-specific 512-D thresholds")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    p.add_argument("--protocol-seed", type=int, default=17, help="Same enrollment/query plan in all modes, arms and training seeds")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-rounds", type=int, default=2)
    p.add_argument("--eval-rounds", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=.02)
    p.add_argument("--tau-bound", type=float, default=.02)
    p.add_argument("--coordinate-hidden", type=int, default=32)
    p.add_argument("--bch-t", type=int, default=32, help="Hard Hamming acceptance radius; not an actual FE acceptance test")
    p.add_argument("--safety-margin", type=int, default=4, help="Training hinge starts at t minus this value")
    p.add_argument("--bce-weight", type=float, default=.1)
    p.add_argument("--radius-weight", type=float, default=1.)
    p.add_argument("--negative-weight", type=float, default=1.)
    p.add_argument("--tau-weight", type=float, default=.001)
    p.add_argument("--negative-identities", type=int, default=8)
    p.add_argument("--hard-mix", type=float, default=.5)
    p.add_argument("--face-negative-margin", type=float, default=96.)
    p.add_argument("--fingerprint-negative-margin", type=float, default=96.)
    p.add_argument("--distance-chunk", type=int, default=128)
    p.add_argument("--min-bit-gain", type=float, default=.1)
    p.add_argument("--max-radius-drop", type=float, default=0.)
    p.add_argument("--max-far-increase", type=float, default=0.)
    p.add_argument("--max-negative-tail-drop", type=float, default=8.)
    p.add_argument("--max-auc-drop", type=float, default=.001)
    p.add_argument("--max-constant-increase", type=float, default=.02)
    p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bch-eval", action="store_true", help="Also compile the existing BCH and evaluate final genuine bit recovery")
    p.add_argument("--check-only", action="store_true", help="Read-only source/split/registration check; no model or output directory")
    return p


def validate_args(args):
    positive = ("epochs", "patience", "batch_size", "train_rounds", "eval_rounds", "lr", "temperature",
                "tau_bound", "coordinate_hidden", "negative_identities", "distance_chunk", "min_bit_gain", "grad_clip", "log_interval", "threads")
    for name in positive:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not 1 <= args.bch_t <= 511 or not 0 <= args.safety_margin < args.bch_t:
        raise ValueError("Require 1 <= t <= 511 and 0 <= safety-margin < t")
    if args.bch_eval and args.bch_t > 255:
        raise ValueError("The existing BCH implementation supports t <= 255")
    if not 0 <= args.hard_mix <= 1:
        raise ValueError("--hard-mix must be in [0,1]")
    for name in ("face_negative_margin", "fingerprint_negative_margin"):
        if not args.bch_t < getattr(args, name) <= 512:
            raise ValueError(f"{name} must exceed t and be <=512")
    for name in ("bce_weight", "radius_weight", "negative_weight", "tau_weight", "max_negative_tail_drop",
                 "max_radius_drop", "max_far_increase", "max_auc_drop", "max_constant_increase"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if args.bce_weight + args.radius_weight <= 0:
        raise ValueError("At least one positive-target loss is required")
    for values in (args.modes, args.arms, args.seeds):
        if not values or len(values) != len(set(values)):
            raise ValueError("Modes, arms and seeds must be nonempty and unique")
    if min(args.seeds+[args.protocol_seed]) < 0:
        raise ValueError("Seeds must be nonnegative")
    if not args.check_only and torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Explicitly choose --device cpu to run on CPU.")


def save_checkpoint(model, path, metadata):
    payload = {"format": FORMAT, "config": vars(model.config), "metadata": metadata,
               "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT or payload["config"]["mode"] not in MODES:
        raise ValueError("Not a V5 enrollment-only checkpoint")
    model = ThresholdingModelV5(V5Config(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["metadata"]


def code_paths():
    paths = {MODEL_PATH, ROOT/"Thresholding Model.py", ROOT/"scripts/__init__.py",
             ROOT/"scripts/common/provenance_snapshots/train_thresholding.py.txt",
             ROOT/"scripts/common/bch_bridge.cpp", ROOT/"BCH/BCH.cpp", ROOT/"BCH/BCH.hpp"}
    paths.update((ROOT/"scripts/common").glob("*.py"))
    paths.update(Path(__file__).resolve().parent.glob("*.py"))
    return sorted(paths)


def manifest_plan(output, name, plan):
    path = output/"protocols"/(name+".json")
    if path.exists():
        if json.loads(path.read_text())["sha256"] != plan["sha256"]:
            raise RuntimeError("Comparisons acquired different enrollment plans")
    else:
        write_json(path, plan)


def summary_rows(results):
    rows = []
    for result in results:
        for split, metrics in result["metrics"].items():
            for m in MODEL_NAMES:
                value = metrics[m]
                neg = value["impostor"]["all_different_identities"]
                row = {k: result[k] for k in ("mode", "arm", "seed", "best_epoch", "selected_origin")}
                row.update(split=split, modality=m,
                    genuine_count=value["genuine"]["count"], genuine_mean=value["genuine"]["mean"],
                    genuine_macro_mean=value["genuine"]["macro_mean"],
                    genuine_macro_within_t=value["genuine"]["macro_within_t"],
                    radius_FRR=1-value["genuine"]["macro_within_t"],
                    impostor_mean=neg["mean"], radius_FAR=neg["macro_within_t"],
                    different_people_radius_FAR=value["impostor"]["different_people"]["macro_within_t"],
                    auroc=value["auroc"], tau_abs_mean=value["tau_abs_mean"],
                    tau_saturated_fraction=value["tau_saturated_fraction"],
                    constant_coordinate_fraction=value["constant_coordinate_fraction"],
                    bound_lower_error_mean=value["bound_diagnostic"]["optimistic_error_lower_bound"]["macro_mean"],
                    bound_impossible_within_t=value["bound_diagnostic"]["macro_impossible_within_t"],
                    bch_exact_target_rate=value.get("bch_genuine", {}).get("exact_target_rate"))
                rows.append(row)
    return rows


def export_tables(output, results):
    rows = summary_rows(results)
    write_csv(output/"comparison.csv", list(rows[0]), rows)
    aggregates = []
    for key in sorted({(r["mode"], r["arm"], r["split"], r["modality"]) for r in rows}):
        selected = [r for r in rows if (r["mode"], r["arm"], r["split"], r["modality"]) == key]
        row = dict(zip(("mode", "arm", "split", "modality"), key))
        row["runs"] = len(selected)
        row["epoch_zero_runs"] = sum(r["best_epoch"] == 0 for r in selected)
        for metric in ("genuine_macro_mean", "genuine_macro_within_t", "radius_FRR", "radius_FAR",
                       "different_people_radius_FAR", "auroc", "tau_abs_mean", "bch_exact_target_rate",
                       "tau_saturated_fraction", "bound_lower_error_mean", "bound_impossible_within_t"):
            values = [r[metric] for r in selected if r[metric] is not None]
            row[metric+"_mean"] = float(np.mean(values)) if values else None
            row[metric+"_std"] = float(np.std(values)) if values else None
        aggregates.append(row)
    write_csv(output/"aggregate.csv", list(aggregates[0]), aggregates)
    write_json(output/"report.json", {"format": FORMAT, "results": results,
        "interpretation": "within-t / radius-FAR are Hamming proxies, not FE key recovery or cryptographic security. "
                          "Seeds repeat one existing participant split; virtual pairing is not genuine multimodal data."})


def fit_trial(data, users, args, output, mode, arm, seed, validation_plan, validation_bank,
              baseline, future_epochs, model_hash):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = SETTINGS[arm]
    model = ThresholdingModelV5(V5Config(mode=mode, architecture=settings["architecture"],
        tau_bound=args.tau_bound, temperature=args.temperature,
        coordinate_hidden=args.coordinate_hidden)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    directory = output/f"{mode}_{arm}_seed{seed}"
    directory.mkdir()
    log = Log(directory)
    metadata = {"mode": mode, "arm": arm, "seed": seed, "epoch": 0, "origin": "zero_tau_fallback",
                "architecture": settings["architecture"], "positive_source": settings["positive"],
                "negative_loss": settings["negatives"],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "model_sha256": model_hash, "validation_plan_sha256": validation_plan["sha256"]}
    best, best_epoch = baseline, 0
    save_checkpoint(model, directory/"best.pt", metadata)
    monitor_plan = make_plan(data, users, "train", args.protocol_seed, rounds=1, epoch=0)
    manifest_plan(output, "train_monitor", monitor_plan)
    monitor_bank = prepare_bank(data, monitor_plan, args.device)
    log.message(f"{DESCRIPTIONS[arm]} | parameters={metadata['parameter_count']} | mode={mode}")
    log.record({"epoch": 0, "validation": baseline, "selected": True, "origin": metadata["origin"],
        "fixed_train_monitor": monitor_training(model, data, monitor_plan, monitor_bank, args)})
    progress = Progress(directory.name, args.epochs, future_epochs, args.log_interval)
    for epoch in range(1, args.epochs+1):
        plan = make_plan(data, users, "train", args.protocol_seed, args.train_rounds, epoch)
        manifest_plan(output, f"train_epoch_{epoch:04d}", plan)
        bank = prepare_bank(data, plan, args.device)
        order_gen = torch.Generator(device=args.device).manual_seed(seed*100003+epoch)
        negative_gen = torch.Generator(device=args.device).manual_seed(seed*100019+epoch)
        order = torch.randperm(len(bank["rows"]), generator=order_gen, device=args.device)
        steps = (len(order)+args.batch_size-1)//args.batch_size
        val_steps = sum(len(v) for v in validation_plan["modalities"].values())+1
        totals, gradient_norms = {}, []
        model.train()
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            positions = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss(batch_forward(model, bank, positions), bank, positions, arm, args, negative_gen)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Non-finite training loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            if step <= 2:
                gradient_norms.append(float(norm))
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())*len(positions)/len(order)
            progress.update(epoch, step, steps+val_steps, "train", float(losses["total"].detach()), force=step == 1)
        def validation_progress(done, total):
            progress.update(epoch, steps+done, steps+total+1, "validation", totals["total"])
        current = evaluate(model, data, validation_plan, validation_bank, args, progress=validation_progress)
        monitor = monitor_training(model, data, monitor_plan, monitor_bank, args)
        progress.update(epoch, steps+val_steps, steps+val_steps, "train-monitor", totals["total"], force=True)
        accepted, reasons = selection(current, baseline, best, args)
        if accepted:
            best, best_epoch = current, epoch
            metadata.update(epoch=epoch, origin="trained")
            save_checkpoint(model, directory/"best.pt", metadata)
        progress.finish_line()
        log.record({"epoch": epoch, "loss": totals, "validation": current, "selected": accepted,
                    "selection_reasons": reasons, "first_gradient_norms": gradient_norms,
                    "training_plan_sha256": plan["sha256"], "fixed_train_monitor": monitor})
        log.message(f"epoch={epoch} loss={totals['total']:.5f} | val <=t={current['mean_macro_within_t']:.4f} "
                    f"hamming={current['mean_macro_hamming']:.2f} | best={best_epoch} "
                    f"tau(face/finger)={current['face']['tau_abs_mean']:.6f}/{current['fingerprint']['tau_abs_mean']:.6f} "
                    f"| neg-active={totals['face_negative_active_fraction']:.3f}/{totals['fingerprint_negative_active_fraction']:.3f} "
                    f"| {'selected' if accepted else '; '.join(reasons)}")
        if epoch-best_epoch >= args.patience:
            log.message(f"Early stopping at epoch {epoch}; selected epoch {best_epoch}")
            break
    save_checkpoint(model, directory/"last.pt", {**metadata, "epoch": epoch, "origin": "last_unselected"})
    selected_model, selected_meta = load_checkpoint(directory/"best.pt", args.device)
    # Persist real tau and hard bits from the reloaded selected checkpoint.
    final_val = evaluate(selected_model, data, validation_plan, validation_bank, args, directory=directory/"validation")
    if abs(final_val["mean_macro_hamming"]-best["mean_macro_hamming"]) > 1e-8:
        raise RuntimeError("Reloaded checkpoint changed the validation result")
    write_json(directory/"selection.json", {**selected_meta, "epochs_completed": epoch, "metrics": final_val})
    return {"mode": mode, "arm": arm, "seed": seed, "best_epoch": best_epoch,
            "selected_origin": selected_meta["origin"], "directory": directory.name,
            "metrics": {"validation": final_val}}


def run_experiments(data, users, args, provenance):
    validate_population(data, users)
    torch.set_num_threads(args.threads)
    validation_plan = make_plan(data, users, "validation", args.protocol_seed, args.eval_rounds)
    test_plan = make_plan(data, users, "test", args.protocol_seed, args.eval_rounds)
    output = (args.output_dir or ROOT/"thresholding/results"/datetime.now().strftime("v5_%Y%m%d_%H%M%S_%f")).resolve()
    if not (output.is_relative_to(ROOT/"thresholding/results") or output.is_relative_to(Path("/tmp"))):
        raise ValueError("Output must be a NEW directory under thresholding/results or /tmp")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output/"status.json", {"state": "running"})
    try:
        (output/"protocols").mkdir()
        files = code_paths()
        hashes = {str(p): sha256(p) for p in files}
        for path in files:
            copy = output/"source"/path.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        write_json(output/"config.json", {"format": FORMAT, "arguments": config, "provenance": provenance,
            "input_sha256": data.hashes, "code_sha256": hashes, "arms": DESCRIPTIONS, "arm_settings": SETTINGS,
            "torch_version": str(torch.__version__), "device": args.device})
        write_json(output/"samples.json", {m: [str(f) for f in data.files[m]] for m in MODEL_NAMES})
        write_json(output/"virtual_users.json", [vars(u) for u in users])
        for name, plan in (("validation", validation_plan), ("test", test_plan)):
            manifest_plan(output, name, plan)
        validation_bank = prepare_bank(data, validation_plan, args.device)
        baseline_dir = output/"tau_zero"
        baseline_dir.mkdir()
        baseline = evaluate(None, data, validation_plan, validation_bank, args, directory=baseline_dir/"validation")
        results = [{"mode": "tau_zero", "arm": "baseline", "seed": None, "best_epoch": 0,
                    "selected_origin": "baseline", "directory": "tau_zero", "metrics": {"validation": baseline}}]
        trials = list(itertools.product(args.modes, args.arms, args.seeds))
        print(f"Output: {output}\n{len(trials)} trials + tau=0 baseline. ETA assumes the epoch cap; final test/BCH time is separate.", flush=True)
        for i, (mode, arm, seed) in enumerate(trials):
            print(f"\n[{i+1}/{len(trials)}] {mode} / {arm} / seed {seed}", flush=True)
            results.append(fit_trial(data, users, args, output, mode, arm, seed, validation_plan,
                validation_bank, baseline, (len(trials)-i-1)*args.epochs, hashes[str(MODEL_PATH)]))
            export_tables(output, results)
        # Test results cannot affect any checkpoint: all training/selection is already finished.
        test_bank = prepare_bank(data, test_plan, args.device)
        bch = None
        if args.bch_eval:
            from scripts.common.bch_evaluator import BCHEvaluator
            bch = BCHEvaluator(output/"bch", args.bch_t)
        final_start = time.perf_counter()
        for i, result in enumerate(results):
            model = None if result["mode"] == "tau_zero" else load_checkpoint(output/result["directory"]/"best.pt", args.device)[0]
            result["metrics"]["test"] = evaluate(model, data, test_plan, test_bank, args,
                                                   directory=output/result["directory"]/"test", bch=bch)
            if bch is not None:
                # Final-only genuine BCH evaluation is separate from selection.
                val = evaluate(model, data, validation_plan, validation_bank, args, bch=bch)
                result["metrics"]["validation"] = val
                write_json(output/result["directory"]/"validation"/"metrics.json", val)
            remaining = (time.perf_counter()-final_start)/(i+1)*(len(results)-i-1)
            print(f"Final test {i+1}/{len(results)} | ETA {duration(remaining)} | {result['directory']}", flush=True)
            export_tables(output, results)
        data.assert_unchanged()
        for path, original in {**hashes, **provenance.get("source_sha256", {})}.items():
            if sha256(path) != original:
                raise RuntimeError(f"Source/code changed while running: {path}")
        write_json(output/"status.json", {"state": "completed", "trials": len(trials),
                                          "elapsed_seconds": time.perf_counter()-started})
        print(f"Completed in {duration(time.perf_counter()-started)}: {output}/comparison.csv", flush=True)
        return output
    except BaseException as error:
        write_json(output/"status.json", {"state": "failed", "error": repr(error)})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    print("Checking cached embeddings, original participant splits and source hashes...", flush=True)
    data, users, _, source_config, source_hashes = prepare_source(args)
    validate_population(data, users)
    for split in ("train", "validation", "test"):
        rounds = args.train_rounds if split == "train" else args.eval_rounds
        plan = make_plan(data, users, split, args.protocol_seed, rounds, int(split == "train"))
        # Also check that every five-sample center is valid, without creating a model.
        prepare_bank(data, plan, "cpu")
        print(f"{split}: {json.dumps(describe_plan(plan), ensure_ascii=False)}", flush=True)
    print(f"Planned: {len(args.modes)*len(args.arms)*len(args.seeds)} trials; 5 enrollment captures, held-out single queries; t={args.bch_t}", flush=True)
    if args.check_only:
        data.assert_unchanged()
        print("Check passed. No training, model creation or result directory.", flush=True)
        return
    run_experiments(data, users, args, {"source_run": str(args.source_run), "source_sha256": source_hashes,
                    "split_seed": source_config["split_seed"], "virtual_pairing": True})


if __name__ == "__main__":
    main()
