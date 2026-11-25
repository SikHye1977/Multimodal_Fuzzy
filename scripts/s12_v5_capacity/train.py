"""Compare V5-D hidden widths with identical losses and tau-only diagnostics.

No extraction/training on import or --check-only. Test is evaluated only after
all checkpoints are fixed. Diagnostics do NOT silently change the objective.
"""
import argparse
from datetime import datetime
import itertools
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from scripts.common.paths import ROOT
from scripts.common.provenance import prepare_source
from scripts.common.runtime import Log, Progress, duration, sha256, write_csv, write_json
from scripts.s11_threshold_v5 import train as v5
from scripts.s11_threshold_v5.evaluation import monitor_training, selection
from scripts.s11_threshold_v5.losses import compute_loss
from scripts.s11_threshold_v5.model import MODEL_PATH, ThresholdingModelV5, V5Config
from scripts.s11_threshold_v5.protocol import (
    MODEL_NAMES, batch_forward, describe_plan, make_plan, prepare_bank, validate_population)
from .diagnostics import AnchorShiftControl, evaluate_diagnostics

FORMAT = "thresholding-v5-capacity-diagnostics-v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Reuse the V5 loss/protocol defaults without exposing unrelated A/B/C modes.
    p.set_defaults(**vars(v5.parser().parse_args([])))
    p.set_defaults(arms=["D"], modes=["user_vector"])
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--source-run", type=Path)
    p.add_argument("--embedding-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--early-stopping", action="store_true",
                   help="Opt in to V5 patience-based stopping; default gives every width the full epoch budget")
    for name in ("protocol-seed", "epochs", "patience", "batch-size", "train-rounds", "eval-rounds",
                 "bch-t", "safety-margin", "negative-identities", "distance-chunk", "log-interval", "threads"):
        p.add_argument("--"+name, type=int)
    for name in ("lr", "temperature", "tau-bound", "bce-weight", "radius-weight", "negative-weight",
                 "tau-weight", "hard-mix", "face-negative-margin", "fingerprint-negative-margin",
                 "min-bit-gain", "max-radius-drop", "max-far-increase", "max-negative-tail-drop",
                 "max-auc-drop", "max-constant-increase", "grad-clip"):
        p.add_argument("--"+name, type=float)
    p.add_argument("--bch-eval", action="store_true", help="Final genuine AND tau-only BCH bit recovery")
    p.add_argument("--check-only", action="store_true", help="Read-only input/split checks; no model/training/output")
    return p


def validate_args(args):
    v5.validate_args(args)
    if not args.widths or len(set(args.widths)) != len(args.widths) or min(args.widths) < 1:
        raise ValueError("Widths must be unique positive integers")


def code_paths():
    return sorted(set(v5.code_paths()) | set(Path(__file__).resolve().parent.glob("*.py")))


def new_output(path, prefix):
    path = (path or ROOT/"thresholding/results"/datetime.now().strftime(prefix+"_%Y%m%d_%H%M%S_%f")).resolve()
    if not (path.is_relative_to(ROOT/"thresholding/results") or path.is_relative_to(Path("/tmp"))):
        raise ValueError("Output must be a NEW directory under thresholding/results or /tmp")
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(output):
    hashes = {str(p): sha256(p) for p in code_paths()}
    for name in hashes:
        path = Path(name)
        dest = output/"source"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    return hashes


def summary_rows(results):
    rows = []
    for r in results:
        for split, metrics in r["metrics"].items():
            for m in MODEL_NAMES:
                v = metrics[m]
                row = {k: r[k] for k in ("condition", "width", "seed", "checkpoint", "epoch", "parameter_count")}
                row.update(split=split, modality=m, genuine_count=v["genuine"]["count"],
                    genuine_macro_hamming=v["genuine"]["macro_mean"],
                    genuine_within_t=v["genuine"]["macro_within_t"],
                    radius_FRR=1-v["genuine"]["macro_within_t"],
                    impostor_hamming=v["impostor"]["all_different_identities"]["macro_mean"],
                    impostor_p05=v["impostor"]["all_different_identities"]["p05"],
                    radius_FAR=v["impostor"]["all_different_identities"]["macro_within_t"],
                    different_people_FAR=v["impostor"]["different_people"]["macro_within_t"],
                    auroc=v["auroc"], tau_abs_mean=v["tau_abs_mean"],
                    tau_saturated_fraction=v["tau_saturated_fraction"],
                    genuine_bch_exact=v.get("bch_genuine", {}).get("exact_target_rate"))
                for rule, d in v["tau_only"]["rules"].items():
                    row[rule+"_hamming"] = d["macro_mean"]
                    row[rule+"_within_t"] = d["macro_within_t"]
                    row[rule+"_count"] = d["count"]
                    row[rule+"_within_t_count"] = d["within_t_count"]
                    row[rule+"_bch_exact"] = d.get("bch", {}).get("exact_target_rate")
                rows.append(row)
    return rows


def export(output, results, **extra):
    rows = summary_rows(results)
    write_csv(output/"comparison.csv", list(rows[0]), rows)
    aggregate = []
    keys = ("condition", "checkpoint", "split", "modality")
    numeric = [k for k in rows[0] if k not in (*keys, "width", "seed", "epoch", "parameter_count")]
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        chosen = [r for r in rows if tuple(r[k] for k in keys) == group]
        out = dict(zip(keys, group))
        out.update(runs=len(chosen), width=chosen[0]["width"], parameter_count=chosen[0]["parameter_count"])
        for field in numeric:
            values = [r[field] for r in chosen if r[field] is not None]
            out[field+"_mean"] = float(np.mean(values)) if values else None
            out[field+"_std"] = float(np.std(values)) if values else None
        aggregate.append(out)
    write_csv(output/"aggregate.csv", list(aggregate[0]), aggregate)
    write_json(output/"report.json", {"format": FORMAT, "results": results, **extra,
        "interpretation": "Tau-only probes are fixed diagnostics, not security proofs or selection losses. "
        "last is an unselected, predeclared endpoint; test never selects checkpoints. "
        "Within-t is a Hamming proxy; BCH evaluates exact anchor bits, not FE keys. "
        "The anchor_shift control deliberately reveals its anchor."})


def record(condition, width, seed, checkpoint, epoch, parameters, directory, metrics):
    return dict(condition=condition, width=width, seed=seed, checkpoint=checkpoint, epoch=epoch,
                parameter_count=parameters, directory=directory, metrics=metrics)


def fit_trial(data, users, args, output, width, seed, val_plan, val_bank, baseline, future_epochs, model_hash):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ThresholdingModelV5(V5Config(mode="user_vector", architecture="coordinate_mlp",
        coordinate_hidden=width, tau_bound=args.tau_bound, temperature=args.temperature)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    directory = output/f"width{width}_seed{seed}"
    directory.mkdir()
    log = Log(directory)
    metadata = dict(mode="user_vector", arm="D", width=width, seed=seed, epoch=0,
                    origin="zero_tau_fallback", parameter_count=sum(p.numel() for p in model.parameters()),
                    model_sha256=model_hash, validation_plan_sha256=val_plan["sha256"])
    v5.save_checkpoint(model, directory/"best.pt", metadata)
    best, best_epoch = baseline, 0
    monitor_plan = make_plan(data, users, "train", args.protocol_seed, 1, 0)
    v5.manifest_plan(output, "train_monitor", monitor_plan)
    monitor_bank = prepare_bank(data, monitor_plan, args.device)
    log.record({"epoch": 0, "validation": baseline, "selected": True,
                "fixed_train_monitor": monitor_training(model, data, monitor_plan, monitor_bank, args)})
    log.message(f"width={width}; parameters={metadata['parameter_count']}; diagnostics do not affect training/selection")
    progress = Progress(directory.name, args.epochs, future_epochs, args.log_interval)
    for epoch in range(1, args.epochs+1):
        plan = make_plan(data, users, "train", args.protocol_seed, args.train_rounds, epoch)
        v5.manifest_plan(output, f"train_epoch_{epoch:04d}", plan)
        bank = prepare_bank(data, plan, args.device)
        order_gen = torch.Generator(device=args.device).manual_seed(seed*100003+epoch)
        negative_gen = torch.Generator(device=args.device).manual_seed(seed*100019+epoch)
        order = torch.randperm(len(bank["rows"]), generator=order_gen, device=args.device)
        steps = (len(order)+args.batch_size-1)//args.batch_size
        val_steps = sum(len(x) for x in val_plan["modalities"].values())+1
        totals, gradients = {}, []
        model.train()
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            positions = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(batch_forward(model, bank, positions), bank, positions, "D", args, negative_gen)
            if not bool(torch.isfinite(loss["total"])):
                raise FloatingPointError("Nonfinite loss")
            loss["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            if step <= 2:
                gradients.append(float(norm))
            optimizer.step()
            for key, value in loss.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())*len(positions)/len(order)
            progress.update(epoch, step, steps+val_steps, "train", float(loss["total"]), force=step == 1)
        def val_progress(done, total):
            progress.update(epoch, steps+done, steps+total+1, "validation", totals["total"])
        current = evaluate_diagnostics(model, data, val_plan, val_bank, args, progress=val_progress)
        monitor = monitor_training(model, data, monitor_plan, monitor_bank, args)
        progress.update(epoch, steps+val_steps, steps+val_steps, "diagnostics", totals["total"], force=True)
        accepted, reasons = selection(current, baseline, best, args)
        if accepted:
            best, best_epoch = current, epoch
            v5.save_checkpoint(model, directory/"best.pt", {**metadata, "epoch": epoch, "origin": "trained"})
        progress.finish_line()
        log.record({"epoch": epoch, "loss": totals, "validation": current, "selected": accepted,
                    "selection_reasons": reasons, "first_gradient_norms": gradients,
                    "training_plan_sha256": plan["sha256"], "fixed_train_monitor": monitor})
        probe = [current[m]["tau_only"]["rules"]["negative_sign"]["macro_within_t"] for m in MODEL_NAMES]
        log.message(f"epoch={epoch} loss={totals['total']:.5f} | genuine <=t={current['mean_macro_within_t']:.4f} "
                    f"HD={current['mean_macro_hamming']:.2f} | tau-only <=t(face/finger)={probe[0]:.4f}/{probe[1]:.4f} "
                    f"| best={best_epoch} | {'selected' if accepted else '; '.join(reasons)}")
        if args.early_stopping and epoch-best_epoch >= args.patience:
            log.message(f"Early stopping at {epoch}; best={best_epoch}. last.pt remains available for diagnostics.")
            break
    v5.save_checkpoint(model, directory/"last.pt", {**metadata, "epoch": epoch, "origin": "last_unselected"})
    rows = []
    for checkpoint in ("best", "last"):
        restored, meta = v5.load_checkpoint(directory/(checkpoint+".pt"), args.device)
        metrics = evaluate_diagnostics(restored, data, val_plan, val_bank, args,
                                       directory=directory/checkpoint/"validation")
        if checkpoint == "best" and abs(metrics["mean_macro_hamming"]-best["mean_macro_hamming"]) > 1e-8:
            raise RuntimeError("Reloaded checkpoint changed validation metrics")
        rows.append(record(f"width{width}", width, seed, checkpoint, meta["epoch"],
                           metadata["parameter_count"], directory.name, {"validation": metrics}))
    write_json(directory/"selection.json", {"best_epoch": best_epoch, "epochs_completed": epoch,
        "rule": "Unchanged V5 genuine/impostor guards; tau-only diagnostics are not a selection constraint"})
    return rows


def run_experiments(data, users, args, provenance):
    validate_population(data, users)
    torch.set_num_threads(args.threads)
    output = new_output(args.output_dir, "v5_capacity")
    started = time.perf_counter()
    write_json(output/"status.json", {"state": "running"})
    try:
        hashes = snapshot(output)
        write_json(output/"config.json", {"format": FORMAT,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "code_sha256": hashes, "input_sha256": data.hashes, "provenance": provenance,
            "torch_version": str(torch.__version__)})
        (output/"protocols").mkdir()
        val_plan = make_plan(data, users, "validation", args.protocol_seed, args.eval_rounds)
        test_plan = make_plan(data, users, "test", args.protocol_seed, args.eval_rounds)
        for name, plan in (("validation", val_plan), ("test", test_plan)):
            v5.manifest_plan(output, name, plan)
        val_bank = prepare_bank(data, val_plan, args.device)
        results = []
        for name, model in (("tau_zero", None), ("anchor_shift", AnchorShiftControl(args.tau_bound))):
            metrics = evaluate_diagnostics(model, data, val_plan, val_bank, args,
                                           directory=output/name/"control"/"validation")
            results.append(record(name, None, None, "control", 0, 0, name, {"validation": metrics}))
        baseline = results[0]["metrics"]["validation"]
        trials = list(itertools.product(args.widths, args.seeds))
        print(f"Output: {output}\n{len(trials)} trials + two nontrained controls; best and last evaluated separately.", flush=True)
        for i, (width, seed) in enumerate(trials):
            print(f"\n[{i+1}/{len(trials)}] width={width}, seed={seed}", flush=True)
            results.extend(fit_trial(data, users, args, output, width, seed, val_plan, val_bank,
                baseline, (len(trials)-i-1)*args.epochs, hashes[str(MODEL_PATH)]))
            export(output, results)
        # Both endpoint types are declared before training. No test-driven selection.
        test_bank = prepare_bank(data, test_plan, args.device)
        bch = None
        if args.bch_eval:
            from scripts.common.bch_evaluator import BCHEvaluator
            bch = BCHEvaluator(output/"bch", args.bch_t)
        final_start = time.perf_counter()
        for i, row in enumerate(results):
            model = (None if row["condition"] == "tau_zero" else AnchorShiftControl(args.tau_bound)) if row["checkpoint"] == "control" else v5.load_checkpoint(
                output/row["directory"]/(row["checkpoint"]+".pt"), args.device)[0]
            base = output/row["directory"]/row["checkpoint"]
            row["metrics"]["test"] = evaluate_diagnostics(model, data, test_plan, test_bank, args,
                                                           directory=base/"test", bch=bch)
            if bch is not None:
                row["metrics"]["validation"] = evaluate_diagnostics(model, data, val_plan, val_bank, args, bch=bch)
                write_json(base/"validation/metrics.json", row["metrics"]["validation"])
            remaining = (time.perf_counter()-final_start)/(i+1)*(len(results)-i-1)
            print(f"Final test {i+1}/{len(results)} | ETA {duration(remaining)} | {row['directory']}/{row['checkpoint']}", flush=True)
            export(output, results)
        data.assert_unchanged()
        for name, digest in {**hashes, **provenance.get("source_sha256", {})}.items():
            if sha256(name) != digest:
                raise RuntimeError(f"Source changed during run: {name}")
        write_json(output/"status.json", {"state": "completed", "trials": len(trials),
                                          "elapsed_seconds": time.perf_counter()-started})
        print(f"Completed: {output}/comparison.csv", flush=True)
        return output
    except BaseException as error:
        write_json(output/"status.json", {"state": "failed", "error": repr(error)})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    data, users, _, source_config, source_hashes = prepare_source(args)
    validate_population(data, users)
    for split in ("train", "validation", "test"):
        plan = make_plan(data, users, split, args.protocol_seed,
                         args.train_rounds if split == "train" else args.eval_rounds, int(split == "train"))
        prepare_bank(data, plan, "cpu")
        print(f"{split}: {describe_plan(plan)}", flush=True)
    print(f"Widths={args.widths}; seeds={args.seeds}; {len(args.widths)*len(args.seeds)} trials.", flush=True)
    if args.check_only:
        data.assert_unchanged()
        print("Check passed; no model, training or output directory created.")
        return
    run_experiments(data, users, args, {"source_run": str(args.source_run), "source_sha256": source_hashes,
                                     "split_seed": source_config["split_seed"], "virtual_pairing": True})


if __name__ == "__main__":
    main()
