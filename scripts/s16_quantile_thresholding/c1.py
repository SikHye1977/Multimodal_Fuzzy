"""Standalone C1: four-bit quantile helpers, baseline and public-helper audit.

No dataset changes, extraction or experiment execution on import/--check-only.
C1 numerical routines, split plans and attacker defaults match the original
S14-S16 C1 condition. Other experiment packages are not required.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.common.paths import ROOT
from scripts.common.provenance import prepare_source, source_code_paths
from scripts.common.runtime import duration, sha256, write_json
from . import attacks
from .diagnostics import AnchorShiftControl, evaluate_diagnostics
from .model import FORMAT as MODEL_FORMAT, QuantileConfig, QuantileThreshold
from .protocol import MODEL_NAMES, batch_forward, describe_plan, make_plan, prepare_bank, validate_population
from .reporting import FORMAT, attack_rows, export, record

SOURCE_FILES = ("__init__.py", "model.py", "protocol.py", "evaluation.py", "diagnostics.py",
                "sketch.py", "attacks.py", "reporting.py", "c1.py", "test_c1.py", "README.md")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-run", type=Path, default=ROOT/"thresholding/results/train_20261029_162319_486882",
                   help="Completed bootstrap/source run providing the original participant split")
    p.add_argument("--embedding-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, help="NEW directory under thresholding/results or /tmp")
    p.add_argument("--device", default="cuda")
    for name, default in (("protocol-seed", 17), ("batch-size", 32), ("eval-rounds", 1),
                          ("fit-rounds", 4), ("attack-rounds", 4), ("attack-epochs", 50),
                          ("attack-batch-size", 64), ("distance-chunk", 128), ("threads", 4),
                          ("log-interval", 10), ("quadrature", 16), ("bch-t", 32)):
        p.add_argument("--"+name, type=int, default=default)
    for name, default in (("tau-bound", .02), ("temperature", .02), ("attack-lr", 1e-3), ("grad-clip", 5.)):
        p.add_argument("--"+name, type=float, default=default)
    p.add_argument("--no-bch-eval", action="store_false", dest="bch_eval", default=True,
                   help="Explicitly skip actual BCH; retain radius diagnostics")
    p.add_argument("--check-only", action="store_true", help="Read-only source/data/split checks; no model or output")
    return p


def validate_args(args):
    for key in ("batch_size", "eval_rounds", "fit_rounds", "attack_rounds", "attack_epochs",
                "attack_batch_size", "distance_chunk", "threads", "log_interval", "tau_bound",
                "temperature", "attack_lr", "grad_clip"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if args.protocol_seed < 0 or not 1 <= args.bch_t <= 255 or args.quadrature < 4:
        raise ValueError("Need nonnegative seed, BCH t in [1,255], and quadrature >=4")
    if args.bch_eval and shutil.which("g++") is None:
        raise RuntimeError("Actual BCH evaluation needs existing g++; nothing is installed automatically")
    if not args.check_only and torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu")


def new_output(path):
    path = (path or ROOT/"thresholding/results"/datetime.now().strftime("c1_%Y%m%d_%H%M%S_%f")).resolve()
    if not (path.is_relative_to(ROOT/"thresholding/results") or path.is_relative_to(Path("/tmp"))):
        raise ValueError("Output must be a NEW directory under thresholding/results or /tmp")
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(output):
    paths = set(source_code_paths(ROOT/"scripts/bootstrap/train.py"))
    paths.update(Path(__file__).parent/name for name in SOURCE_FILES)
    hashes = {str(path): sha256(path) for path in sorted(paths)}
    for name in hashes:
        path = Path(name)
        destination = output/"source"/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return hashes


def restore_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != MODEL_FORMAT or saved["config"]["helper_bits"] != 4:
        raise ValueError("Expected a four-bit C1 checkpoint")
    model = QuantileThreshold(QuantileConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved["metadata"]


def fit_c1(args, output, plans, banks):
    directory = output/"models/C1"
    directory.mkdir(parents=True)
    print("C1: fitting TRAIN population/noise and 512 x 16 lookup tables...", flush=True)
    began = time.perf_counter()
    model = QuantileThreshold(QuantileConfig(helper_bits=4, tau_bound=args.tau_bound,
        temperature=args.temperature, quadrature=args.quadrature)).to(args.device)
    diagnostics = model.fit(banks["fit"], plans["fit"])
    write_json(directory/"population_fit.json", {"fit_plan_sha256": plans["fit"]["sha256"],
        "diagnostics": diagnostics, "interpretation": "Estimated marginal Gaussian CDF/noise; no joint zero-leakage claim. "
        "Inactive/degenerate coordinates use tau=0 and constant helper. All variants use the common tau bound."})
    path = directory/"fitted.pt"
    save_checkpoint(model, path, {"condition": "C1", "seed": None, "epoch": 0,
        "parameter_count": 0, "origin": "train_population_fit",
        "fit_plan_sha256": plans["fit"]["sha256"], "validation_plan_sha256": plans["validation"]["sha256"]})
    print(f"C1 population/table fit complete | elapsed {duration(time.perf_counter()-began)}", flush=True)
    return endpoint("C1", None, "fitted", path, output, model.storage())


def save_checkpoint(model, path, metadata):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(model.checkpoint(metadata), temp)
    temp.replace(path)


def endpoint(condition, seed, label, path, output, storage):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    directory = str(path.parent.relative_to(output))
    row = record(condition, None, seed, label,
                          meta["epoch"], meta["parameter_count"], directory, {})
    row.update(kind="checkpoint", path=str(path.resolve()), storage=storage,
               id=directory.replace("/", "_")+"_"+label, checkpoint_sha256=sha256(path))
    return row


def control(kind, args):
    row = record(kind, None, None, "control", 0, 0, "controls/"+kind, {})
    row.update(id=kind, kind=kind, path=None,
               storage={m: {"payload_bits": 0 if kind == "tau_zero" else 16384,
                            "encoding": "none" if kind == "tau_zero" else "deliberately leaking float32 tau"}
                        for m in MODEL_NAMES})
    return row


def restore(row, args):
    if row["kind"] == "tau_zero":
        return None
    if row["kind"] == "leak_control":
        return AnchorShiftControl(args.tau_bound).to(args.device)
    return restore_checkpoint(row["path"], args.device)[0]


def row_directory(output, row):
    return output/row["directory"]/row["checkpoint"]


@torch.no_grad()
def export_payload(model, bank, row, destination, args):
    """Persist deployable values separately from sensitive offline targets."""
    arrays = {}
    for start in range(0, len(bank["rows"]), args.batch_size):
        positions = torch.arange(start, min(start+args.batch_size, len(bank["rows"])), device=args.device)
        mapping = bank["rows"][positions].cpu().numpy()
        out = None if model is None else batch_forward(model, bank, positions)
        for col, (m, name) in enumerate(MODEL_NAMES.items()):
            if model is None:
                payload = torch.empty(len(positions), 0, device=args.device)
            elif name+"_payload" in out:
                payload = out[name+"_payload"]
                restored = model.tau_from_payload(payload, name)
                if not torch.equal(restored, out[name+"_tau"]):
                    raise RuntimeError("Quantile storage failed exact tau reconstruction")
            else:
                payload = out[name+"_tau"]
            value = payload.cpu().numpy()
            if m not in arrays:
                arrays[m] = np.zeros((len(bank[m]["target"]), value.shape[1]), dtype=value.dtype)
            actual_bits = payload.shape[1]*payload.element_size()*8
            if actual_bits != row["storage"][m]["payload_bits"]:
                raise RuntimeError(f"Incorrect physical payload size for {m}: {actual_bits}")
            arrays[m][mapping[:, col]] = value
    np.savez_compressed(destination/"public_payload.npz", **arrays)
    write_json(destination/"storage.json", {"modalities": row["storage"], "bch_sketch_bits_per_modality": 20*args.bch_t,
        "interpretation": "Logical per-user payload excludes NPZ headers and globally shared model state. "
        "BCH syndrome storage bits are not independent constraints. Diagnostic target/query files are NOT public helpers."})


def run_experiments(data, users, args, provenance):
    validate_args(args)
    validate_population(data, users)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = new_output(args.output_dir)
    began = time.perf_counter()
    write_json(output/"status.json", {"state": "running"})
    try:
        hashes = snapshot(output)
        write_json(output/"config.json", {"format": FORMAT, "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "input_sha256": data.hashes, "code_sha256": hashes, "provenance": provenance,
            "torch_version": str(torch.__version__)})
        (output/"protocols").mkdir()
        plans = {s: make_plan(data, users, s, args.protocol_seed, args.eval_rounds) for s in ("validation", "test")}
        plans["fit"] = make_plan(data, users, "train", args.protocol_seed, args.fit_rounds, 200000)
        plans["attack_train"] = make_plan(data, users, "train", args.protocol_seed, args.attack_rounds, 100000)
        for name, plan in plans.items():
            write_json(output/"protocols"/(name+".json"), plan)
        banks = {name: attacks.add_sketches(prepare_bank(data, plans[name], args.device), args.bch_t)
                 for name in ("fit", "attack_train", "validation")}
        results = [control("tau_zero", args), control("leak_control", args)]
        attack_results, fitted_attacks = [], {}
        print(f"Output: {output}\nC1 + tau=0 + leak control; ETA is phase-specific. "
              "All model and independent-attacker fits finish before final test.", flush=True)

        # Positive control validates both fixed-rule wiring and learned attacker wiring.
        leak_row = results[1]
        leak = restore(leak_row, args)
        leak_train = attacks.freeze_dataset(leak, banks["attack_train"], args)
        leak_val = attacks.freeze_dataset(leak, banks["validation"], args)
        qualification = attacks.qualification(leak_val, args)
        paths = attacks.fit_attackers(leak_train, leak_val, args, row_directory(output, leak_row)/"attackers", 0)
        learned = []
        for path in paths:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            learned.append(saved["validation"]["mean_hamming"])
        if min(learned) != 0:
            raise RuntimeError("Fresh attackers failed the deliberate sign-leak control")
        qualification["fresh_attacker_exact_control_recovery"] = True
        write_json(output/"audit_qualification.json", qualification)
        fitted_attacks[leak_row["id"]] = paths
        print("Audit qualification passed: fixed sign rule and a fresh attacker recover all control bits.", flush=True)
        del leak_train, leak_val

        results.append(fit_c1(args, output, plans, banks))

        remaining = [row for row in results if row["id"] not in fitted_attacks]
        print(f"Fitting independent attackers for {len(remaining)} remaining endpoints ({len(attacks.SPECS)} each).", flush=True)
        for index, row in enumerate(remaining):
            model = restore(row, args)
            train = attacks.freeze_dataset(model, banks["attack_train"], args)
            val = attacks.freeze_dataset(model, banks["validation"], args)
            fitted_attacks[row["id"]] = attacks.fit_attackers(train, val, args,
                row_directory(output, row)/"attackers", row["seed"] or 0, (len(remaining)-index-1)*len(attacks.SPECS))
        # No more threshold fitting, distribution fitting or attacker selection below.
        write_json(output/"selection_complete.json", {"model_checkpoints": {
            r["path"]: sha256(r["path"]) for r in results if r["path"]},
            "attack_checkpoints": {str(p): sha256(p) for paths in fitted_attacks.values() for p in paths},
            "test_evaluation_started": False})
        bch = None
        if args.bch_eval:
            from scripts.common.bch_evaluator import BCHEvaluator
            bch = BCHEvaluator(output/"bch", args.bch_t)
        banks["test"] = attacks.add_sketches(prepare_bank(data, plans["test"], args.device), args.bch_t)
        # Population-prior attack uses TRAIN targets, even on validation/test.
        prior = attacks.estimate_prior(attacks.freeze_dataset(None, banks["attack_train"], args))
        final_start = time.perf_counter()
        for index, row in enumerate(results):
            model = restore(row, args)
            base = row_directory(output, row)
            for split in ("validation", "test"):
                destination = base/split
                metrics = evaluate_diagnostics(model, data, plans[split], banks[split], args,
                                               directory=destination, bch=bch)
                row["metrics"][split] = metrics
                export_payload(model, banks[split], row, destination, args)
                dataset = attacks.freeze_dataset(model, banks[split], args)
                predictions = {}
                for name, bits in attacks.fixed_predictions(dataset, prior).items():
                    scores = attacks.score_bits(bits, dataset, args.bch_t, bch)
                    attack_results.extend(attack_rows(row, split, name, "helper", scores))
                    predictions[name] = bits.cpu().numpy().astype(np.uint8)
                np.savez_compressed(destination/"fixed_attack_bits.npz", **predictions)
                for path in fitted_attacks[row["id"]]:
                    attacker, saved = attacks.load_attacker(path, args.device)
                    scores, bits = attacks.evaluate_attacker(attacker, dataset, args, bch)
                    write_json(path.parent/(split+".json"), scores)
                    np.savez_compressed(path.parent/(split+"_predictions.npz"),
                        bits=bits.cpu().numpy().astype(np.uint8), identities=dataset["identities"].cpu().numpy())
                    attack_results.extend(attack_rows(row, split, saved["kind"], saved["view"], scores, saved["seed"], saved["epoch"]))
            export(output, results, attack_results)
            elapsed = time.perf_counter()-final_start
            print(f"Final evaluation {index+1}/{len(results)} | phase ETA~ "
                  f"{duration(elapsed/(index+1)*(len(results)-index-1))}", flush=True)
        data.assert_unchanged()
        barrier = json.loads((output/"selection_complete.json").read_text())
        checked = {**hashes, **provenance.get("source_sha256", {}),
                   **barrier["model_checkpoints"], **barrier["attack_checkpoints"]}
        for path, digest in checked.items():
            if sha256(path) != digest:
                raise RuntimeError(f"Source/input/frozen checkpoint changed during run: {path}")
        write_json(output/"status.json", {"state": "completed", "endpoints": len(results),
            "fresh_attackers": sum(map(len, fitted_attacks.values())), "elapsed_seconds": time.perf_counter()-began})
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
    descriptions = (("fit", "train", args.fit_rounds, 200000),
                    ("attack_train", "train", args.attack_rounds, 100000),
                    ("validation", "validation", args.eval_rounds, 0),
                    ("test", "test", args.eval_rounds, 0))
    for name, split, rounds, epoch in descriptions:
        plan = make_plan(data, users, split, args.protocol_seed, rounds, epoch)
        prepare_bank(data, plan, "cpu")
        print(f"{name}: {describe_plan(plan)}", flush=True)
    print(f"C1: 4 bits/coordinate; BCH t={args.bch_t}; actual BCH={args.bch_eval}", flush=True)
    if args.check_only:
        data.assert_unchanged()
        print("Check passed: no model, training or output directory created.", flush=True)
        return
    return run_experiments(data, users, args, {"source_run": str(args.source_run), "source_sha256": hashes,
        "split_seed": config["split_seed"], "virtual_pairing": True})


if __name__ == "__main__":
    main()
