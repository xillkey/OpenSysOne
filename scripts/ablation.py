"""Ablation: OpenSysOne defaults vs Laya-style design choices, one change at a time.

    python examples/routing/make_data.py
    opensysone make-tiny-backbone --corpus examples/routing/data/train.jsonl --out runs/routing-backbone \\
        --hidden-size 128 --layers 4 --heads 2
    python scripts/ablation.py --seeds 0 1 --out runs/ablation

Every variant trains on the same data with the same backbone init and budget;
finished runs are skipped, so the script can be resumed. Results go to
``<out>/results.json`` and ``<out>/results.md``.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from opensysone.calibrate import calibrate
from opensysone.config import load_config
from opensysone.data import DecisionDataset, collate
from opensysone.metrics import collect, expected_ece, summarize
from opensysone.model import DecisionModel
from opensysone.predictor import Predictor
from opensysone.schema import iter_jsonl, load_jsonl
from opensysone.serializer import Serializer
from opensysone.trainer import train

LAYA_HEAD = ["head.pooling=marker", "head.head_layers=2", "head.type_embedding=true", "head.scorer=shared"]
VARIANTS: dict[str, list[str]] = {
    "base": [],
    "marker_pool": ["head.pooling=marker"],
    "marker_pool+2L": ["head.pooling=marker", "head.head_layers=2", "head.type_embedding=true"],
    "span_pool+2L": ["head.head_layers=2", "head.type_embedding=true"],
    "laya_head": LAYA_HEAD,
    "question_first": ["data.order=question_first"],
    "mask_marker": ["data.marker=mask"],
    "bool_as_options": ["data.bool_as_options=true"],
    "no_shuffle": ["data.shuffle_options=false"],
    "parallel": ["data.parallel_options=true"],
    "rlcd": ["loss.name=rlcd"],
    "laya_like": LAYA_HEAD + ["data.order=question_first", "data.marker=mask", "data.bool_as_options=true",
                              "data.shuffle_options=false", "loss.name=rlcd"],
}


def base_overrides(data: Path, backbone: Path, epochs: int) -> list[str]:
    return [
        f"backbone.model={backbone}",
        f"data.train={data / 'train.jsonl'}",
        f"data.dev={data / 'dev.jsonl'}",
        "data.max_length=128", "data.option_token_budget=8", "data.max_options=16",
        f"train.epochs={epochs}", "train.batch_size=32", "train.lr=5e-4", "train.head_lr=1e-3",
        "train.llrd=1.0", "train.warmup_ratio=0.05", "train.device=cpu", "train.log_every=100",
        "train.early_stop_patience=4", "train.calibrate=true",
    ]


def loader(model: DecisionModel, path: Path, bs: int = 128) -> DataLoader:
    ser = Serializer(model.tokenizer, model.serializer_config)
    ds = DecisionDataset(load_jsonl(path), ser)
    return DataLoader(ds, batch_size=bs, collate_fn=functools.partial(collate, pad_id=ser.pad_id))


def choice_ece(model: DecisionModel, path: Path) -> float:
    c = collect(model, loader(model, path), "cpu", use_temperature=True)
    conf, ok = [], []
    for p, t, qt in zip(c.probs, c.targets, c.qtypes):
        if qt == 0:
            pi = max(range(len(p)), key=p.__getitem__)
            conf.append(p[pi])
            ok.append(pi == max(range(len(t)), key=t.__getitem__))
    return expected_ece(conf, ok)


def order_consistency(pred: Predictor, path: Path, n: int = 300, seed: int = 0) -> dict[str, float]:
    """Re-ask each choice question with its options permuted: same answer? how far did the probabilities move?"""
    rng = random.Random(seed)
    base, perm_reqs, perms = [], [], []
    for _, obj in list(iter_jsonl(path))[:n]:
        q = obj["questions"]["intent"]
        opts = list(q["options"])
        p = list(range(len(opts)))
        while p == sorted(p):
            rng.shuffle(p)
        mk = lambda o: {"state": obj["state"], "questions": {"c": {"type": "choice", "text": q["text"], "options": o}}}
        base.append(mk(opts))
        perm_reqs.append(mk([opts[i] for i in p]))
        perms.append(p)
    a = pred.decide_many(base)
    b = pred.decide_many(perm_reqs)
    agree, tv = 0, []
    for ra, rb in zip(a, b):
        pa, pb = ra["results"]["c"]["probabilities"], rb["results"]["c"]["probabilities"]
        agree += ra["results"]["c"]["answer"] == rb["results"]["c"]["answer"]
        tv.append(0.5 * sum(abs(pa[k] - pb[k]) for k in pa))
    return {"agree": agree / len(a), "mean_tv": statistics.fmean(tv), "max_tv": max(tv)}


def latency(pred: Predictor, path: Path, n: int = 60) -> float:
    _, req = next(iter_jsonl(path))
    for _ in range(5):
        pred.decide(req)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        pred.decide(req)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def evaluate_run(run: Path, data: Path) -> dict[str, Any]:
    final = run / "final"
    model = DecisionModel.from_pretrained(final, "cpu")
    out: dict[str, Any] = {}
    for split in ("test_seen", "test_unseen"):
        rep = summarize(collect(model, loader(model, data / f"{split}.jsonl"), "cpu", use_temperature=True))
        out[split] = rep
    # same weights, temperatures refit per (type, option-count bucket) on dev
    bucketed = copy.deepcopy(model)
    calibrate(bucketed, loader(bucketed, data / "dev.jsonl"), "cpu", by_options=True)
    out["choice_ece_seen"] = {"by_type": choice_ece(model, data / "test_seen.jsonl"),
                              "by_options": choice_ece(bucketed, data / "test_seen.jsonl")}
    pred = Predictor(model, "cpu")
    out["order"] = order_consistency(pred, data / "test_seen.jsonl")
    out["latency_p50_ms"] = latency(pred, data / "test_seen.jsonl")
    out["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    meta = json.loads((run / "metrics.json").read_text())
    out["steps"] = meta["steps"]
    return out


def mean_sd(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.fmean(xs):.3f}±{statistics.stdev(xs):.3f}"


def table(results: dict[str, dict[str, dict[str, Any]]]) -> str:
    cols = [
        ("choice acc seen", lambda r: r["test_seen"]["choice"]["accuracy"]),
        ("choice acc unseen", lambda r: r["test_unseen"]["choice"]["accuracy"]),
        ("score acc", lambda r: r["test_seen"]["score"]["accuracy"]),
        ("bool acc", lambda r: r["test_seen"]["bool"]["accuracy"]),
        ("NLL seen", lambda r: r["test_seen"]["all"]["nll"]),
        ("NLL unseen", lambda r: r["test_unseen"]["all"]["nll"]),
        ("choice ECE", lambda r: r["choice_ece_seen"]["by_type"]),
        ("ECE k-bucket", lambda r: r["choice_ece_seen"]["by_options"]),
        ("order agree", lambda r: r["order"]["agree"]),
        ("order TV", lambda r: r["order"]["mean_tv"]),
        ("p50 ms", lambda r: r["latency_p50_ms"]),
    ]
    lines = ["| variant | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, seeds in results.items():
        rows = list(seeds.values())
        lines.append(f"| {name} | " + " | ".join(mean_sd([f(r) for r in rows]) for _, f in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/routing/data")
    ap.add_argument("--backbone", default="runs/routing-backbone")
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--only", nargs="*", help="run only these variants")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results: dict[str, dict[str, Any]] = json.loads(res_path.read_text()) if res_path.exists() else {}
    names = a.only or list(VARIANTS)
    for name in names:
        for seed in a.seeds:
            run = out / name / f"seed{seed}"
            if str(seed) in results.get(name, {}):
                continue
            t0 = time.time()
            if not (run / "final" / "opensysone.json").exists():
                cfg = load_config(None, base_overrides(data, Path(a.backbone), a.epochs) + VARIANTS[name]
                                  + [f"train.seed={seed}", f"train.output_dir={run}"])
                train(cfg)
            r = evaluate_run(run, data)
            r["minutes"] = (time.time() - t0) / 60
            results.setdefault(name, {})[str(seed)] = r
            res_path.write_text(json.dumps(results, indent=2))
            print(f"[done] {name} seed{seed} {r['minutes']:.1f} min  choice seen/unseen "
                  f"{r['test_seen']['choice']['accuracy']:.3f}/{r['test_unseen']['choice']['accuracy']:.3f}  "
                  f"order agree {r['order']['agree']:.3f}", flush=True)
    ordered = {n: results[n] for n in VARIANTS if n in results}
    (out / "results.md").write_text(table(ordered) + "\n")
    print(table(ordered))


if __name__ == "__main__":
    main()
