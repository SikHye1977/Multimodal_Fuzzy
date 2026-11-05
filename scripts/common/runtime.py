"""Logging, serialization, source hashing and the original model loader."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


from .paths import ROOT


def load_model_module():
    path = ROOT / "Thresholding Model.py"
    spec = importlib.util.spec_from_file_location("thresholding_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(model, path):
    temp = path.with_suffix(".tmp")
    model.save_checkpoint(temp, purpose="research_training_not_validated_for_cryptographic_enrollment")
    temp.replace(path)


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "측정 중"
    seconds = max(0, int(math.ceil(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    """ETA includes observed train/validation time and assumes the epoch caps."""
    def __init__(self, stage, epochs, future_epochs, interval):
        self.stage, self.epochs, self.future_epochs = stage, epochs, future_epochs
        self.interval = interval
        self.start = time.perf_counter()
        self.line_open = False
        self.tty = sys.stdout.isatty()

    def update(self, epoch, step, steps, phase, loss, force=False):
        if not force and step % self.interval and step != steps:
            return
        completed = epoch - 1 + step / steps
        elapsed = time.perf_counter() - self.start
        mean_epoch = elapsed / completed if completed > 0 else None
        remaining = mean_epoch * (self.epochs - completed) if mean_epoch else None
        total_remaining = remaining + mean_epoch * self.future_epochs if mean_epoch else None
        loss_text = f"{loss:.5f}" if loss is not None else "-"
        message = (f"{self.stage} | epoch {epoch}/{self.epochs} | {phase} {step}/{steps} "
                   f"({100 * step / steps:5.1f}%) | loss {loss_text} | 경과 {duration(elapsed)} "
                   f"| 단계 ETA {duration(remaining)} | 전체 ETA~ {duration(total_remaining)}")
        if self.tty:
            print("\r\033[2K" + message, end="", flush=True)
            self.line_open = True
        else:
            print(message, flush=True)

    def finish_line(self):
        if self.line_open:
            print(flush=True)
            self.line_open = False


class Log:
    def __init__(self, directory):
        self.path = directory / "training.log"
        self.history = directory / "history.jsonl"

    def message(self, text):
        print(text, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def record(self, data):
        with self.history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
