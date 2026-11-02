"""Evaluation: accuracy, log loss, Brier, ECE, score MAE and latency."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .data import to_device
from .schema import QTYPES, TYPE_IDS


@dataclass
class Collected:
    probs: list[list[float]]
    logits: list[list[float]]
    targets: list[list[float]]
    qtypes: list[int]
    options: list[list[str]]
    refs: list[tuple[int, str]]


@torch.inference_mode()
def collect(model: Any, loader: Iterable[dict[str, Any]], device: Any, use_temperature: bool = True) -> Collected:
    model.eval()
    out = Collected([], [], [], [], [], [])
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(**batch)
        scaled = model.calibrate_logits(logits, batch["qtype"], batch["opt_mask"]) if use_temperature else logits
        probs = torch.softmax(scaled, -1).cpu()
        for i, opts in enumerate(batch["options"]):
            n = len(opts)
            out.probs.append(probs[i, :n].tolist())
            out.logits.append(logits[i, :n].float().cpu().tolist())
            out.targets.append(batch["target"][i, :n].cpu().tolist())
            out.qtypes.append(int(batch["qtype"][i]))
            out.options.append(opts)
            out.refs.append(batch["refs"][i])
    return out


def _level_values(options: list[str]) -> list[float]:
    try:
        return [float(o) for o in options]
    except ValueError:
        return [float(i) for i in range(len(options))]


def expected_ece(conf: list[float], correct: list[bool], bins: int = 15) -> float:
    if not conf:
        return float("nan")
    total, ece = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            ece += len(idx) / total * abs(acc - avg)
    return ece


def summarize(c: Collected, bins: int = 15) -> dict[str, Any]:
    groups: dict[str, list[int]] = {"all": list(range(len(c.probs)))}
    for name, tid in TYPE_IDS.items():
        idx = [i for i, t in enumerate(c.qtypes) if t == tid]
        if idx:
            groups[name] = idx
    report: dict[str, Any] = {}
    for name, idx in groups.items():
        if not idx:
            continue
        conf, correct, nll, brier, mae = [], [], [], [], []
        for i in idx:
            p, t = c.probs[i], c.targets[i]
            pred = max(range(len(p)), key=p.__getitem__)
            gold = max(range(len(t)), key=t.__getitem__)
            conf.append(p[pred])
            correct.append(pred == gold)
            nll.append(-sum(ti * math.log(max(pi, 1e-12)) for pi, ti in zip(p, t)))
            brier.append(sum((pi - ti) ** 2 for pi, ti in zip(p, t)))
            if c.qtypes[i] == TYPE_IDS["score"]:
                v = _level_values(c.options[i])
                mae.append(abs(sum(a * b for a, b in zip(p, v)) - sum(a * b for a, b in zip(t, v))))
        entry = {
            "n": len(idx),
            "accuracy": sum(correct) / len(idx),
            "nll": sum(nll) / len(idx),
            "brier": sum(brier) / len(idx),
            "ece": expected_ece(conf, correct, bins),
        }
        if mae:
            entry["score_mae"] = sum(mae) / len(mae)
        report[name] = entry
    return report


def evaluate(model: Any, loader: Iterable[dict[str, Any]], device: Any, use_temperature: bool = True) -> dict[str, Any]:
    return summarize(collect(model, loader, device, use_temperature))


def benchmark_latency(fn: Any, n: int = 50, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "n": n,
        "p50_ms": statistics.median(times),
        "p95_ms": times[min(n - 1, int(0.95 * n))],
        "mean_ms": sum(times) / n,
    }


def format_report(report: dict[str, Any]) -> str:
    keys = ["n", "accuracy", "nll", "brier", "ece", "score_mae"]
    lines = [f"{'type':<8}" + "".join(f"{k:>11}" for k in keys)]
    for name in ["all", *QTYPES]:
        if name in report:
            row = report[name]
            cells = []
            for k in keys:
                v = row.get(k)
                cells.append(f"{'-':>11}" if v is None else (f"{v:>11d}" if isinstance(v, int) else f"{v:>11.4f}"))
            lines.append(f"{name:<8}" + "".join(cells))
    return "\n".join(lines)
