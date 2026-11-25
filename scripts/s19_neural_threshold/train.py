"""S19: A/B/C/D neural architecture and recovery-objective comparison.

No extraction or environment changes. --check-only never trains or writes.
Actual BCH and four independent public-information attackers are on by default.
"""
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.common.paths import ROOT
from scripts.common.provenance import prepare_source
from scripts.common.runtime import duration, sha256, write_json
from scripts.s11_threshold_v5 import train as v5
from scripts.s11_threshold_v5.protocol import MODEL_NAMES, describe_plan, make_plan, prepare_bank, validate_population
from scripts.s12_v5_capacity.diagnostics import AnchorShiftControl, evaluate_diagnostics
from scripts.s12_v5_capacity.train import new_output
from scripts.s16_quantile_thresholding import attacks
from scripts.s16_quantile_thresholding.c1 import control, export_payload, row_directory
from .learning import fit_trial
from .losses import ARMS
from .model import load_checkpoint
from .reporting import FORMAT, attack_rows, export


def parser():
    p = v5.parser()
    p.description = __doc__
    p.set_defaults(coordinate_hidden=128, bch_eval=True)
    # No early stopping: equal training budgets, with immutable best/last roles.
    for flag in ("--patience", "--modes"):
        action = p._option_string_actions[flag]
        p._remove_action(action)
        for group in p._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
        for option in action.option_strings:
            p._option_string_actions.pop(option)
    p.set_defaults(patience=20, modes=["user_vector"])
    for name, default in (("warmup-epochs", 10), ("set-context", 16), ("set-hidden", 64),
                          ("attack-rounds", 4), ("attack-epochs", 50), ("attack-batch-size", 64)):
        p.add_argument("--"+name, type=int, default=default)
    for name, default in (("boundary-width", 4.), ("boundary-weight", 1.), ("keep-weight", 1.),
                          ("attack-lr", .001)):
        p.add_argument("--"+name, type=float, default=default)
    p.add_argument("--no-bch-eval", action="store_false", dest="bch_eval",
                   help="Skip actual BCH explicitly; hard-distance metrics remain")
    return p


def validate_args(args):
    v5.validate_args(args)
    for name in ("set_context", "set_hidden", "attack_rounds", "attack_epochs", "attack_batch_size",
                 "boundary_width", "boundary_weight", "keep_weight", "attack_lr"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 1 <= args.warmup_epochs < args.epochs:
        raise ValueError("Require 1 <= warmup-epochs < epochs")
    if args.bch_t > 255:
        raise ValueError("Public BCH sketch implementation requires t<=255")
    if args.bch_eval and shutil.which("g++") is None:
        raise RuntimeError("Actual BCH requires existing g++; no installation is performed")


def snapshot(output):
    paths = set(v5.code_paths())
    paths.update((ROOT/"scripts/s12_v5_capacity").glob("*.py"))
    paths.update(p for p in (ROOT/"scripts/s16_quantile_thresholding").glob("*.py") if p.name != "train.py")
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.add(Path(__file__).parent/"README.md")
    hashes = {str(path): sha256(path) for path in sorted(paths)}
    for name in hashes:
        path = Path(name)
        destination = output/"source"/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return hashes


def endpoint(directory, label, output, args):
    path = directory/(label+".pt")
    model, meta = load_checkpoint(path)
    location = str(directory.relative_to(output))
    return {"id": location.replace("/", "_")+"_"+label, "kind": "checkpoint", "path": str(path),
            "directory": location, "condition": meta["condition"], "seed": meta["seed"],
            "checkpoint": label, "epoch": meta["epoch"], "selected_origin": meta["origin"],
            "architecture": meta["architecture"], "objective": meta["objective"],
            "parameter_count": meta["parameter_count"], "metrics": {},
            "width": model.config.coordinate_hidden if meta["architecture"] == "coordinate" else model.config.set_hidden,
            "checkpoint_sha256": sha256(path),
            "storage": {m: {"payload_bits": 512*32, "encoding": "512 float32 thresholds"} for m in MODEL_NAMES}}


def restore(row, args):
    if row["kind"] == "tau_zero":
        return None
    if row["kind"] == "leak_control":
        return AnchorShiftControl(args.tau_bound).to(args.device)
    return load_checkpoint(row["path"], args.device)[0]


def check_matched_warmup(metadata, output):
    comparisons = []
    for left, right in (("A", "C"), ("B", "D")):
        seeds = sorted({seed for arm, seed in metadata if arm == left} & {seed for arm, seed in metadata if arm == right})
        for seed in seeds:
            a, b = metadata[left, seed], metadata[right, seed]
            for field in ("initial_state_digest", "warmup_state_digest"):
                if a[field] != b[field]:
                    raise RuntimeError(f"Unmatched {left}/{right}, seed {seed}: {field}")
            comparisons.append({"left": left, "right": right, "seed": seed,
                                "initial_state_digest": a["initial_state_digest"],
                                "warmup_state_digest": a["warmup_state_digest"]})
    write_json(output/"matched_warmup.json", {"passed": True, "comparisons": comparisons})


def run_experiments(data, users, args, provenance):
    validate_args(args)
    validate_population(data, users)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = new_output(args.output_dir, "neural_abcd")
    began = time.perf_counter()
    write_json(output/"status.json", {"state": "running"})
    try:
        hashes = snapshot(output)
        write_json(output/"config.json", {"format": FORMAT,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "arms": ARMS, "input_sha256": data.hashes, "code_sha256": hashes,
            "provenance": provenance, "torch_version": str(torch.__version__)})
        write_json(output/"samples.json", {m: [str(f) for f in data.files[m]] for m in MODEL_NAMES})
        write_json(output/"virtual_users.json", [vars(u) for u in users])
        (output/"protocols").mkdir()
        # Test plan is bookkeeping only until every threshold and attacker is frozen.
        plans = {s: make_plan(data, users, s, args.protocol_seed, args.eval_rounds) for s in ("validation", "test")}
        plans["attack_train"] = make_plan(data, users, "train", args.protocol_seed, args.attack_rounds, 100000)
        for name, plan in plans.items():
            v5.manifest_plan(output, name, plan)
        val_bank = prepare_bank(data, plans["validation"], args.device)
        baseline = evaluate_diagnostics(None, data, plans["validation"], val_bank, args)
        results = [control("tau_zero", args), control("leak_control", args)]
        trials = list(itertools.product(args.seeds, args.arms))
        metadata = {}
        print(f"Output: {output}\n{len(trials)} neural runs; each {args.epochs} epochs including "
              f"{args.warmup_epochs} common warm-up epochs. Best and last both audited.\n"
              "ETA is phase-specific: neural training, independent attacks, final BCH evaluation.", flush=True)
        for index, (seed, arm) in enumerate(trials):
            print(f"\nNeural run {index+1}/{len(trials)}: {arm}, seed={seed}", flush=True)
            directory, meta = fit_trial(data, users, args, output, arm, seed, plans["validation"],
                val_bank, baseline, (len(trials)-index-1)*args.epochs)
            metadata[arm, seed] = meta
            results.extend(endpoint(directory, label, output, args) for label in ("best", "last"))
            export(output, results, [])
        check_matched_warmup(metadata, output)

        banks = {"validation": attacks.add_sketches(val_bank, args.bch_t),
                 "attack_train": attacks.add_sketches(prepare_bank(data, plans["attack_train"], args.device), args.bch_t)}
        attack_results, fitted = [], {}
        leak = restore(results[1], args)
        qualification = attacks.qualification(attacks.freeze_dataset(leak, banks["validation"], args), args)
        write_json(output/"audit_qualification.json", qualification)
        for index, row in enumerate(results):
            print(f"\nPublic-information audit {index+1}/{len(results)}: {row['id']}", flush=True)
            model = restore(row, args)
            train = attacks.freeze_dataset(model, banks["attack_train"], args)
            val = attacks.freeze_dataset(model, banks["validation"], args)
            fitted[row["id"]] = attacks.fit_attackers(train, val, args,
                row_directory(output, row)/"attackers", row["seed"] or 0,
                (len(results)-index-1)*len(attacks.SPECS))
            if row["kind"] == "leak_control":
                checks = [torch.load(p, map_location="cpu", weights_only=True)["validation"]["mean_hamming"]
                          for p in fitted[row["id"]]]
                if min(checks) != 0:
                    raise RuntimeError("Fresh attackers failed to detect the deliberate sign leak")
                qualification["fresh_attacker_exact_control_recovery"] = True
                write_json(output/"audit_qualification.json", qualification)
        barrier = {"model_checkpoints": {r["path"]: sha256(r["path"]) for r in results if r["path"]},
                   "reference_checkpoints": {str(p): sha256(p) for p in (output/"models").glob("*/reference.pt")},
                   "attack_checkpoints": {str(p): sha256(p) for values in fitted.values() for p in values},
                   "test_evaluation_started": False}
        write_json(output/"selection_complete.json", barrier)
        # No model fitting, reference fitting, attacker fitting or selection below.
        bch = None
        if args.bch_eval:
            from scripts.common.bch_evaluator import BCHEvaluator
            bch = BCHEvaluator(output/"bch", args.bch_t)
        banks["test"] = attacks.add_sketches(prepare_bank(data, plans["test"], args.device), args.bch_t)
        prior = attacks.estimate_prior(attacks.freeze_dataset(None, banks["attack_train"], args))
        final_start = time.perf_counter()
        for index, row in enumerate(results):
            model = restore(row, args)
            for split in ("validation", "test"):
                destination = row_directory(output, row)/split
                metrics = evaluate_diagnostics(model, data, plans[split], banks[split], args,
                                               directory=destination, bch=bch)
                row["metrics"][split] = metrics
                export_payload(model, banks[split], row, destination, args)
                dataset = attacks.freeze_dataset(model, banks[split], args)
                fixed = {}
                for name, bits in attacks.fixed_predictions(dataset, prior).items():
                    scores = attacks.score_bits(bits, dataset, args.bch_t, bch)
                    attack_results.extend(attack_rows(row, split, name, "helper", scores))
                    fixed[name] = bits.cpu().numpy().astype(np.uint8)
                np.savez_compressed(destination/"fixed_attack_bits.npz", **fixed)
                for path in fitted[row["id"]]:
                    attacker, saved = attacks.load_attacker(path, args.device)
                    scores, bits = attacks.evaluate_attacker(attacker, dataset, args, bch)
                    write_json(path.parent/(split+".json"), scores)
                    np.savez_compressed(path.parent/(split+"_predictions.npz"), bits=bits.cpu().numpy().astype(np.uint8))
                    attack_results.extend(attack_rows(row, split, saved["kind"], saved["view"], scores,
                                                     saved["seed"], saved["epoch"]))
            export(output, results, attack_results)
            elapsed = time.perf_counter()-final_start
            print(f"Final evaluation {index+1}/{len(results)} | elapsed {duration(elapsed)} | "
                  f"phase ETA {duration(elapsed/(index+1)*(len(results)-index-1))}", flush=True)
        data.assert_unchanged()
        checked = {**hashes, **provenance.get("source_sha256", {}), **barrier["model_checkpoints"],
                   **barrier["reference_checkpoints"], **barrier["attack_checkpoints"]}
        for path, digest in checked.items():
            if sha256(path) != digest:
                raise RuntimeError(f"Input/source/frozen checkpoint changed: {path}")
        write_json(output/"status.json", {"state": "completed", "neural_trials": len(trials),
            "endpoints": len(results), "fresh_attackers": sum(map(len, fitted.values())),
            "verified_hashes": len(checked), "elapsed_seconds": time.perf_counter()-began})
        print(f"Completed: {output}/comparison.csv and attacks.csv", flush=True)
        return output
    except BaseException as error:
        write_json(output/"status.json", {"state": "failed", "error": repr(error), "elapsed_seconds": time.perf_counter()-began})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    data, users, _, config, hashes = prepare_source(args)
    validate_population(data, users)
    for name, split, rounds, epoch in (("train_example", "train", args.train_rounds, 1),
        ("attack_train", "train", args.attack_rounds, 100000), ("validation", "validation", args.eval_rounds, 0),
        ("test", "test", args.eval_rounds, 0)):
        plan = make_plan(data, users, split, args.protocol_seed, rounds, epoch)
        prepare_bank(data, plan, "cpu")
        print(f"{name}: {describe_plan(plan)}", flush=True)
    if args.check_only:
        data.assert_unchanged()
        print("Check passed: no model, training, evaluation results or output directory created.", flush=True)
        return
    return run_experiments(data, users, args, {"source_run": str(args.source_run), "source_sha256": hashes,
                                              "split_seed": config["split_seed"], "virtual_pairing": True})


if __name__ == "__main__":
    main()
