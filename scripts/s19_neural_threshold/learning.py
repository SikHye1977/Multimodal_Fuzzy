"""Matched warm-up and fixed-teacher loss comparison; validation-only selection."""
from copy import deepcopy
import random

import numpy as np
import torch

from scripts.common.runtime import Log, Progress, write_json
from scripts.s11_threshold_v5.evaluation import collect_tau, distance_summary, selection
from scripts.s11_threshold_v5.protocol import MODEL_NAMES, batch_forward, make_plan, prepare_bank
from scripts.s11_threshold_v5.train import manifest_plan
from scripts.s12_v5_capacity.diagnostics import evaluate_diagnostics
from .losses import ARMS, compute_loss, recovery_terms
from .model import Config, NeuralThreshold, load_checkpoint, save_checkpoint, state_digest


@torch.no_grad()
def monitor(model, reference, bank, plan, args):
    """Fixed TRAIN episodes: comparable hard distances and positive losses.

    The reference is zero until warm-up ends, then the immutable warm-up model.
    Radius/boundary/BCE are always reported, regardless of the active objective.
    """
    taus = collect_tau(model, bank, args.batch_size)
    refs = collect_tau(reference, bank, args.batch_size)
    result = {}
    for m in MODEL_NAMES:
        datum = bank[m]
        tau = torch.from_numpy(taus[m]).to(datum["target"])
        ref = torch.from_numpy(refs[m]).to(tau)
        logits = (datum["queries"]-tau[:, None])/args.temperature
        bits = logits >= 0
        target, mask = datum["target"], datum["query_mask"]
        distance = (bits != target[:, None]).sum(-1).float()
        d0 = ((datum["queries"] >= ref[:, None]) != target[:, None]).sum(-1).float()
        boundary, keep, protected = recovery_terms(distance, d0, mask, args.bch_t, args.boundary_width)
        from scripts.s11_threshold_v5.losses import masked_mean
        bce = masked_mean(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target[:, None].expand_as(logits), reduction="none").mean(-1), mask)
        radius = masked_mean((torch.relu(distance-(args.bch_t-args.safety_margin))/args.bch_t).square(), mask)
        ids = datum["identity"][:, None].expand_as(mask)[mask].cpu().numpy()
        d, previous = distance[mask].cpu().numpy(), d0[mask].cpu().numpy()
        # Each identity has the same number of registrations, with query means
        # taken inside each registration. Virtual row duplication is excluded.
        result[m] = {"query": distance_summary(d, ids, ids, args.bch_t),
            "reference_query": distance_summary(previous, ids, ids, args.bch_t),
            "bce": float(bce.mean()), "radius": float(radius.mean()),
            "boundary": float(boundary.mean()), "keep": float(keep.mean()),
            "reference_successes": int((previous <= args.bch_t).sum()),
            "gained": int(((previous > args.bch_t) & (d <= args.bch_t)).sum()),
            "lost": int(((previous <= args.bch_t) & (d > args.bch_t)).sum()),
            "protected_fraction": float(masked_mean(protected.float(), mask).mean())}
    return result


def select(candidate, baseline, best, args):
    accepted, reasons = selection(candidate, baseline, best, args)
    for m in MODEL_NAMES:
        for rule in ("negative_sign", "nonnegative_sign", "zero_query"):
            current = candidate[m]["tau_only"]["rules"][rule]["macro_within_t"]
            original = baseline[m]["tau_only"]["rules"][rule]["macro_within_t"]
            if current > original+1e-12:
                reasons.append(f"{m}: {rule} public-tau recovery increase")
    return accepted and not reasons, reasons


def fit_trial(data, users, args, output, arm, seed, val_plan, val_bank, baseline, future_epochs):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = ARMS[arm]
    model = NeuralThreshold(Config(architecture=settings["architecture"],
        coordinate_hidden=args.coordinate_hidden, set_context=args.set_context, set_hidden=args.set_hidden,
        tau_bound=args.tau_bound, temperature=args.temperature)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    directory = output/"models"/f"{arm}_seed{seed}"
    directory.mkdir(parents=True)
    log = Log(directory)
    meta = {"condition": arm, "seed": seed, "epoch": 0, "origin": "zero_tau_fallback",
            **settings, "parameter_count": sum(p.numel() for p in model.parameters()),
            "initial_state_digest": state_digest(model), "warmup_epochs": args.warmup_epochs,
            "validation_plan_sha256": val_plan["sha256"]}
    save_checkpoint(model, directory/"best.pt", meta)
    best, best_epoch, reference = baseline, 0, None
    monitor_plan = make_plan(data, users, "train", args.protocol_seed, 1, 0)
    manifest_plan(output, "train_monitor", monitor_plan)
    monitor_bank = prepare_bank(data, monitor_plan, args.device)
    log.record({"epoch": 0, "phase": "initial", "validation": baseline, "selected": True,
                "fixed_train": monitor(model, reference, monitor_bank, monitor_plan, args)})
    log.message(f"{arm}: {settings} | parameters={meta['parameter_count']} | common warm-up={args.warmup_epochs}")
    progress = Progress(directory.name+" [threshold phase]", args.epochs, future_epochs, args.log_interval)
    for epoch in range(1, args.epochs+1):
        improved = settings["objective"] == "boundary_keep" and epoch > args.warmup_epochs
        plan = make_plan(data, users, "train", args.protocol_seed, args.train_rounds, epoch)
        manifest_plan(output, f"train_epoch_{epoch:04d}", plan)
        bank = prepare_bank(data, plan, args.device)
        order_gen = torch.Generator(device=args.device).manual_seed(seed*100003+epoch)
        negative_gen = torch.Generator(device=args.device).manual_seed(seed*100019+epoch)
        order = torch.randperm(len(bank["rows"]), generator=order_gen, device=args.device)
        steps = (len(order)+args.batch_size-1)//args.batch_size
        validation_steps = sum(len(v) for v in val_plan["modalities"].values())+1
        totals, norms = {}, []
        model.train()
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            positions = order[start:start+args.batch_size]
            with torch.no_grad():
                ref = batch_forward(reference, bank, positions) if reference is not None else None
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss(batch_forward(model, bank, positions), bank, positions, args, negative_gen,
                                  improved=improved, reference=ref)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Nonfinite neural loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())*len(positions)/len(order)
            progress.update(epoch, step, steps+validation_steps, "train", float(losses["total"].detach()), force=step == 1)
        if epoch == args.warmup_epochs:
            reference = deepcopy(model).eval().requires_grad_(False)
            meta["warmup_state_digest"] = state_digest(reference)
            save_checkpoint(reference, directory/"reference.pt", {**meta, "epoch": epoch, "origin": "fixed_warmup_reference"})
        current = evaluate_diagnostics(model, data, val_plan, val_bank, args,
            progress=lambda done, total: progress.update(epoch, steps+done, steps+total+1, "validation", totals["total"]))
        fixed = monitor(model, reference, monitor_bank, monitor_plan, args)
        accepted, reasons = select(current, baseline, best, args)
        if accepted:
            best, best_epoch = current, epoch
            origin = "post_warmup_new_loss" if improved else ("warmup_old_loss" if epoch <= args.warmup_epochs else "old_loss")
            save_checkpoint(model, directory/"best.pt", {**meta, "epoch": epoch, "origin": origin})
        progress.update(epoch, steps+validation_steps, steps+validation_steps, "fixed TRAIN monitor", totals["total"], force=True)
        progress.finish_line()
        log.record({"epoch": epoch, "phase": "boundary_keep" if improved else "radius", "loss": totals,
                    "validation": current, "selected": accepted, "selection_reasons": reasons,
                    "gradient_norm_mean": float(np.mean(norms)), "gradient_norm_max": max(norms),
                    "training_plan_sha256": plan["sha256"], "fixed_train": fixed,
                    "monitor_reference": "warmup" if reference is not None else "tau_zero"})
        gains = "/".join(f"{m}:{fixed[m]['gained']} gained,{fixed[m]['lost']} lost" for m in MODEL_NAMES)
        log.message(f"epoch={epoch}/{args.epochs} | loss={totals['total']:.5f} | "
            f"VAL <=t={current['mean_macro_within_t']:.4f} HD={current['mean_macro_hamming']:.2f} | "
            f"best={best_epoch} | {gains} | {'selected' if accepted else '; '.join(reasons)}")
        if reference is not None and state_digest(reference) != meta["warmup_state_digest"]:
            raise RuntimeError("Frozen warm-up reference was modified")
    save_checkpoint(model, directory/"last.pt", {**meta, "epoch": args.epochs, "origin": "last_unselected"})
    selected, selected_meta = load_checkpoint(directory/"best.pt", args.device)
    reloaded = evaluate_diagnostics(selected, data, val_plan, val_bank, args)
    if abs(reloaded["mean_macro_hamming"]-best["mean_macro_hamming"]) > 1e-8:
        raise RuntimeError("Checkpoint reload changed validation output")
    write_json(directory/"selection.json", {**meta, "best_epoch": best_epoch,
        "selected_origin": selected_meta["origin"], "epochs_completed": args.epochs,
        "new_loss_selected": selected_meta["origin"] == "post_warmup_new_loss",
        "warmup_teacher_is_not_validation_selected": True,
        "final_reference_coverage": {m: fixed[m]["protected_fraction"] for m in MODEL_NAMES}})
    return directory, meta
