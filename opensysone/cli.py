"""Command line: ``opensysone <command> ...``"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _loader(model: Any, path: str, batch_size: int) -> Any:
    from torch.utils.data import DataLoader

    from .data import DecisionDataset, collate
    from .schema import load_jsonl
    from .serializer import Serializer

    ser = Serializer(model.tokenizer, model.serializer_config)
    ds = DecisionDataset(load_jsonl(path), ser)
    return DataLoader(ds, batch_size=batch_size, collate_fn=functools.partial(collate, pad_id=ser.pad_id))


def cmd_train(a: argparse.Namespace) -> None:
    from .config import load_config
    from .trainer import train

    cfg = load_config(a.config, a.set)
    res = train(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2, ensure_ascii=False))


def cmd_calibrate(a: argparse.Namespace) -> None:
    from .calibrate import calibrate
    from .model import DecisionModel
    from .trainer import resolve_device

    dev = resolve_device(a.device)
    model = DecisionModel.from_pretrained(a.model, dev)
    temps = calibrate(model, _loader(model, a.data, a.batch_size), dev, by_options=a.by_options)
    out = Path(a.out or a.model)
    meta = json.loads((Path(a.model) / "opensysone.json").read_text())
    own = {"format", "version", "backbone", "serializer", "head", "temperature", "temperature_by_options"}
    extra = {k: v for k, v in meta.items() if k not in own}
    model.save_pretrained(out, extra=extra)
    print(json.dumps({"temperature": temps, "saved_to": str(out)}, indent=2))


def cmd_eval(a: argparse.Namespace) -> None:
    from .metrics import benchmark_latency, evaluate, format_report
    from .model import DecisionModel
    from .predictor import Predictor
    from .schema import iter_jsonl
    from .trainer import resolve_device

    dev = resolve_device(a.device)
    model = DecisionModel.from_pretrained(a.model, dev)
    report = evaluate(model, _loader(model, a.data, a.batch_size), dev, use_temperature=not a.no_temperature)
    print(format_report(report))
    if a.latency:
        _, first = next(iter_jsonl(a.data))
        pred = Predictor(model, dev)
        report["latency_single_request"] = benchmark_latency(lambda: pred.decide(first), n=a.latency)
        print("latency (one request, all its questions):", json.dumps(report["latency_single_request"]))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _predictor(a: argparse.Namespace) -> Any:
    if getattr(a, "onnx", False):
        from .onnx_export import OnnxPredictor

        return OnnxPredictor(a.model)
    from .predictor import Predictor

    return Predictor.from_pretrained(a.model, a.device)


def cmd_predict(a: argparse.Namespace) -> None:
    from .schema import iter_jsonl

    pred = _predictor(a)
    if a.input.endswith(".jsonl"):
        reqs = [obj for _, obj in iter_jsonl(a.input)]
        for out in pred.decide_many(reqs, batch_size=a.batch_size):
            print(json.dumps(out, ensure_ascii=False))
    else:
        text = sys.stdin.read() if a.input == "-" else Path(a.input).read_text(encoding="utf-8")
        print(json.dumps(pred.decide(json.loads(text)), indent=2, ensure_ascii=False))


def cmd_export(a: argparse.Namespace) -> None:
    from .onnx_export import export_onnx

    print(export_onnx(a.model, a.out, a.opset))


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    from .serve import create_app

    app = create_app(_predictor(a), a.max_batch, a.max_wait_ms)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def cmd_validate(a: argparse.Namespace) -> None:
    from collections import Counter

    from .schema import SchemaError, iter_jsonl, parse_example

    types, opts, errors, n = Counter(), Counter(), [], 0
    for lineno, obj in iter_jsonl(a.data):
        try:
            ex = parse_example(obj)
        except SchemaError as e:
            errors.append(f"line {lineno}: {e}")
            continue
        n += 1
        for q in ex.questions:
            types[q.type] += 1
            opts[len(q.options)] += 1
            if not q.has_label:
                types[f"{q.type} (unlabelled)"] += 1
    print(json.dumps({"examples": n, "questions_by_type": types, "option_counts": dict(sorted(opts.items())),
                      "errors": len(errors)}, indent=2, ensure_ascii=False))
    for e in errors[:20]:
        print("  ", e)
    if errors:
        sys.exit(1)


def cmd_tiny(a: argparse.Namespace) -> None:
    from .tiny import make_tiny_backbone

    print(make_tiny_backbone(a.corpus, a.out, a.vocab_size, a.hidden_size, a.layers, a.heads, a.max_position))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("opensysone", description="Train and serve System-1 decision models on any encoder.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train", help="train a decision model from a YAML config")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. train.lr=2e-5")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("calibrate", help="fit per-type temperatures on held-out data")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", help="output dir (default: overwrite --model)")
    s.add_argument("--by-options", action="store_true", help="also fit per (type, option-count bucket)")
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=64)
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("eval", help="accuracy / NLL / Brier / ECE (+ latency)")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--no-temperature", action="store_true", help="report uncalibrated probabilities")
    s.add_argument("--latency", type=int, default=0, metavar="N", help="also time N single requests")
    s.add_argument("--json", help="write the report to this file")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("predict", help="run a request (.json, '-' for stdin) or a batch (.jsonl)")
    s.add_argument("--model", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--onnx", action="store_true", help="--model is an exported ONNX directory")
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=64)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("export-onnx", help="export to ONNX (outputs calibrated probabilities)")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--opset", type=int, default=17)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("serve", help="serve POST /v1/systemone")
    s.add_argument("--model", required=True)
    s.add_argument("--onnx", action="store_true")
    s.add_argument("--device", default="auto")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-batch", type=int, default=32)
    s.add_argument("--max-wait-ms", type=float, default=5.0)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("validate-data", help="check a JSONL file against the schema")
    s.add_argument("--data", required=True)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("make-tiny-backbone", help="random small BERT + tokenizer trained on your JSONL")
    s.add_argument("--corpus", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--vocab-size", type=int, default=8000)
    s.add_argument("--hidden-size", type=int, default=128)
    s.add_argument("--layers", type=int, default=2)
    s.add_argument("--heads", type=int, default=2)
    s.add_argument("--max-position", type=int, default=512)
    s.set_defaults(fn=cmd_tiny)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
