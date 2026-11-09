"""Original C1 fixed/learned public-helper diagnostics.

Only public values enter attackers; targets supervise TRAIN and score held-out
splits. Attack failure is not a security proof. No algebraic decoder attack is
implemented here; joint_mlp_helper_sketch only feeds syndrome bits to an MLP.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.common.runtime import Log, Progress
from .evaluation import distance_summary
from .protocol import MODEL_NAMES, batch_forward
from .sketch import public_sketch

SPECS = (("coordinate_linear", "helper"), ("coordinate_mlp", "helper"),
         ("joint_mlp", "helper"), ("joint_mlp", "helper_sketch"))


def add_sketches(bank, t):
    for m in MODEL_NAMES:
        bank[m]["sketch"] = public_sketch(bank[m]["target"], t)
    return bank


def public_batch(outputs, bank, positions, bound, detach=False):
    rows = bank["rows"][positions]
    taus, extra, coordinates, targets, weights, identities, sketches = [], [], [], [], [], [], []
    for column, (m, name) in enumerate(MODEL_NAMES.items()):
        tau = outputs[name+"_tau"]/bound
        aux = outputs.get(name+"_public_aux", outputs.get(name+"_code_values"))
        if aux is not None:
            extra.append(aux.float())
        coord_aux = outputs.get(name+"_coordinate_aux", torch.zeros_like(tau))
        taus.append(tau)
        coordinates.append(torch.stack((tau, coord_aux), -1))
        targets.append(bank[m]["target"][rows[:, column]])
        sketches.append(2*bank[m]["sketch"][rows[:, column]]-1)
        weights.append(bank[m]["weights"][positions])
        identities.append(bank[m]["identity"][rows[:, column]])
    result = {"helper": torch.cat(taus+extra, 1), "coordinate": torch.stack(coordinates, 1),
              "tau": torch.stack(taus, 1), "sketch": torch.cat(sketches, 1),
              "target": torch.cat(targets, 1).detach(), "weights": torch.stack(weights, 1).detach(),
              "identities": torch.stack(identities, 1)}
    return {k: v.detach() for k, v in result.items()} if detach else result


@torch.no_grad()
def freeze_dataset(model, bank, args):
    if model is not None:
        model.eval()
    parts = []
    for start in range(0, len(bank["rows"]), args.batch_size):
        positions = torch.arange(start, min(start+args.batch_size, len(bank["rows"])), device=bank["rows"].device)
        if model is None:
            outputs = {name+"_tau": torch.zeros(len(positions), 512, device=positions.device)
                       for name in MODEL_NAMES.values()}
        else:
            outputs = batch_forward(model, bank, positions)
        parts.append(public_batch(outputs, bank, positions, args.tau_bound, detach=True))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def estimate_prior(dataset):
    target = dataset["target"].reshape(-1, 2, 512)
    weights = dataset["weights"]
    return ((target*weights[..., None]).sum(0)/weights.sum(0)[:, None]).clamp(.001, .999).detach()


def bce_by_modality(logits, dataset):
    values = F.binary_cross_entropy_with_logits(logits, dataset["target"], reduction="none").reshape(-1, 2, 512).mean(2)
    return (values*dataset["weights"]).mean(0)


class AttackNet(nn.Module):
    def __init__(self, kind, view, helper_dim, prior, width=128):
        super().__init__()
        if (kind, view) not in SPECS:
            raise ValueError("Unknown attacker specification")
        self.kind, self.view, self.helper_dim, self.width = kind, view, helper_dim, width
        self.register_buffer("prior", prior.detach().clone())
        if kind.startswith("coordinate"):
            heads = []
            for _ in range(2):
                if kind == "coordinate_linear":
                    head = nn.Linear(3, 1)
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                    # Known negative-sign leak is a starting hypothesis; train can reverse it.
                    with torch.no_grad():
                        head.weight[0, 0] = -1.
                else:
                    head = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
                heads.append(head)
            self.heads = nn.ModuleList(heads)
        else:
            self.network = None  # Exact syndrome width supplied explicitly by initialize_joint.

    def initialize_joint(self, sketch_dim):
        if self.kind == "joint_mlp":
            dimension = self.helper_dim+(sketch_dim if self.view == "helper_sketch" else 0)
            self.network = nn.Sequential(nn.Linear(dimension, self.width), nn.GELU(),
                                         nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, 1024))
        return self

    def forward(self, helper, coordinate, sketch):
        if self.kind.startswith("coordinate"):
            prior = (2*self.prior-1)[None, :, :, None].expand(len(helper), -1, -1, -1)
            x = torch.cat((coordinate, prior), -1)
            return torch.cat([head(x[:, i]).squeeze(-1) for i, head in enumerate(self.heads)], 1)
        x = torch.cat((helper, sketch), 1) if self.view == "helper_sketch" else helper
        return self.network(x)


def predict(attacker, dataset):
    # Deliberately pass only these three public input tensors.
    return attacker(dataset["helper"], dataset["coordinate"], dataset["sketch"])


def subset(dataset, indices):
    return {k: v[indices] for k, v in dataset.items()}


def fixed_predictions(dataset, prior):
    tau = dataset["tau"].flatten(1)
    return {"negative_sign": tau < 0, "nonnegative_sign": tau >= 0,
            "zero_query": tau <= 0,
            "population_prior": (prior.flatten()[None] >= .5).expand(len(tau), -1),
            "all_zero": torch.zeros_like(tau, dtype=torch.bool),
            "all_one": torch.ones_like(tau, dtype=torch.bool)}


def score_bits(bits, dataset, t, bch=None):
    bits = torch.as_tensor(bits).detach().cpu().numpy().astype(np.uint8)
    target = dataset["target"].cpu().numpy().astype(np.uint8)
    ids = dataset["identities"].cpu().numpy()
    result = {}
    for col, m in enumerate(MODEL_NAMES):
        pred, truth = bits[:, col*512:(col+1)*512], target[:, col*512:(col+1)*512]
        distance = (pred != truth).sum(1)
        scores = distance_summary(distance, ids[:, col], ids[:, col], t)
        scores["exact_bits"] = float(np.mean(distance == 0))
        if bch is not None:
            outcomes = bch.recover(truth, pred)
            scores["bch_exact_macro"] = float(np.mean([np.mean(outcomes[ids[:, col] == identity] == 1)
                                                        for identity in np.unique(ids[:, col])]))
        result[m] = scores
    result["mean_within_t"] = float(np.mean([result[m]["macro_within_t"] for m in MODEL_NAMES]))
    result["mean_hamming"] = float(np.mean([result[m]["macro_mean"] for m in MODEL_NAMES]))
    return result


@torch.no_grad()
def evaluate_attacker(attacker, dataset, args, bch=None):
    attacker.eval()
    logits = []
    for start in range(0, len(dataset["target"]), args.attack_batch_size):
        logits.append(predict(attacker, subset(dataset, slice(start, start+args.attack_batch_size))))
    logits = torch.cat(logits)
    bits = logits >= 0
    result = score_bits(bits, dataset, args.bch_t, bch)
    result["bce"] = float(bce_by_modality(logits, dataset).mean())
    return result, bits


def selection_key(metrics):
    return metrics["mean_within_t"], -metrics["mean_hamming"], -metrics["bce"]


def fit_attackers(train, val, args, directory, seed, future_jobs=0):
    directory = Path(directory)
    paths = []
    prior = estimate_prior(train)
    for i, (kind, view) in enumerate(SPECS):
        attack_seed = seed+2000000+i*1009
        torch.manual_seed(attack_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(attack_seed)
        attacker = AttackNet(kind, view, train["helper"].shape[1], prior).initialize_joint(train["sketch"].shape[1]).to(args.device)
        opt = torch.optim.Adam(attacker.parameters(), lr=args.attack_lr)
        dest = directory/(kind+"_"+view)
        dest.mkdir(parents=True, exist_ok=False)
        log = Log(dest)
        best, _ = evaluate_attacker(attacker, val, args)
        best_epoch = 0
        state = {k: v.detach().cpu().clone() for k, v in attacker.state_dict().items()}
        progress = Progress(directory.parent.name+"/"+dest.name+" [attack phase]", args.attack_epochs,
                            (future_jobs+len(SPECS)-i-1)*args.attack_epochs, args.log_interval)
        generator = torch.Generator(device=args.device).manual_seed(attack_seed)
        for epoch in range(1, args.attack_epochs+1):
            order = torch.randperm(len(train["target"]), device=args.device, generator=generator)
            steps = (len(order)+args.attack_batch_size-1)//args.attack_batch_size
            attacker.train()
            for step, start in enumerate(range(0, len(order), args.attack_batch_size), 1):
                batch = subset(train, order[start:start+args.attack_batch_size])
                opt.zero_grad(set_to_none=True)
                loss = bce_by_modality(predict(attacker, batch), batch).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite attacker loss")
                loss.backward()
                nn.utils.clip_grad_norm_(attacker.parameters(), args.grad_clip, error_if_nonfinite=True)
                opt.step()
                progress.update(epoch, step, steps+1, "train", float(loss.detach()), force=step == 1)
            current, _ = evaluate_attacker(attacker, val, args)
            if selection_key(current) > selection_key(best):
                best, best_epoch = current, epoch
                state = {k: v.detach().cpu().clone() for k, v in attacker.state_dict().items()}
            log.record({"epoch": epoch, "validation": current, "best_epoch": best_epoch})
            progress.update(epoch, steps+1, steps+1, "validation", current["bce"], force=True)
            progress.finish_line()
        path = dest/"best.pt"
        torch.save({"kind": kind, "view": view, "helper_dim": train["helper"].shape[1],
                    "sketch_dim": train["sketch"].shape[1], "prior": prior.cpu(), "width": 128,
                    "seed": attack_seed, "epoch": best_epoch, "state_dict": state, "validation": best}, path)
        paths.append(path)
    return paths


def load_attacker(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = AttackNet(saved["kind"], saved["view"], saved["helper_dim"], saved["prior"], saved["width"])
    model.initialize_joint(saved["sketch_dim"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved


def qualification(control_dataset, args):
    """Gate the audit on a deliberately leaking control, never on test labels."""
    negative = fixed_predictions(control_dataset, estimate_prior(control_dataset))["negative_sign"]
    metrics = score_bits(negative, control_dataset, args.bch_t)
    if any(metrics[m]["macro_mean"] != 0 for m in MODEL_NAMES):
        raise RuntimeError("Audit failed to recover the deliberate sign-leak control")
    return {"passed": True, "rule": "negative-sign must recover every control bit", "metrics": metrics}
