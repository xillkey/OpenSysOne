"""Inference: Jev-style ``{state, questions}`` request -> typed answers with probabilities.

Request::

    {"state": "text or JSON object",
     "questions": {"team": {"type": "choice", "question": "Which team?", "options": ["billing", "tech"]},
                   "urgency": {"type": "score", "question": "How urgent?", "min": 1, "max": 5},
                   "refund": {"type": "bool", "question": "Asks for a refund?"}}}

Response::

    {"results": {"team": {"type": "choice", "answer": "billing", "confidence": 0.93,
                          "probabilities": {"billing": 0.93, "tech": 0.07}},
                 "urgency": {..., "answer": "4", "expected": 3.8},
                 "refund": {..., "answer": "yes", "probability": 0.88}},
     "model": "...", "latency_ms": 12.4}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import MODEL_KEYS, DecisionDataset, Item, collate
from .schema import Example, SchemaError, TYPE_IDS, parse_example
from .serializer import Serializer


def _level_value(options: list[str], probs: list[float]) -> float | None:
    try:
        values = [float(o) for o in options]
    except ValueError:
        return None
    return float(sum(v * p for v, p in zip(values, probs)))


def format_result(qtype: str, options: list[str], probs: list[float]) -> dict[str, Any]:
    best = int(np.argmax(probs))
    res: dict[str, Any] = {
        "type": qtype,
        "answer": options[best],
        "confidence": float(probs[best]),
        "probabilities": {o: float(p) for o, p in zip(options, probs)},
    }
    if qtype == "bool":
        res["probability"] = float(probs[1])
    elif qtype == "score":
        res["expected"] = _level_value(options, probs)
    return res


class _BasePredictor:
    serializer: Serializer
    name: str

    def _items(self, requests: list[dict[str, Any]]) -> tuple[list[Example], list[Item]]:
        examples = [parse_example(r, with_label=False) for r in requests]
        ds = DecisionDataset(examples, self.serializer, require_labels=False)
        expected = sum(len(e.questions) for e in examples)
        if len(ds) != expected:
            # re-encode to surface the exact error to the caller
            for e in examples:
                for q in e.questions:
                    self.serializer.encode(e.state, q)
            raise SchemaError("some questions could not be encoded")
        return examples, [ds[i] for i in range(len(ds))]

    def _probs(self, batch: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def decide_many(self, requests: list[dict[str, Any]], batch_size: int = 64) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        examples, items = self._items(requests)
        results: list[dict[str, Any]] = [{"results": {}} for _ in examples]
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            probs = self._probs(collate(chunk, self.serializer.pad_id))
            for it, row in zip(chunk, probs):
                ei, qid = it.ref
                qtype = next(t for t, i in TYPE_IDS.items() if i == it.qtype)
                results[ei]["results"][qid] = format_result(qtype, it.options, row[: it.n_options].tolist())
        ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r["model"] = self.name
            r["latency_ms"] = round(ms, 3)
        return results

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.decide_many([request])[0]


class Predictor(_BasePredictor):
    """PyTorch predictor that loads an OpenSysOne model directory."""

    def __init__(self, model: Any, device: str | torch.device = "cpu", name: str = "opensysone"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.serializer = Serializer(model.tokenizer, model.serializer_config)
        self.name = name

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "auto") -> "Predictor":
        from .model import DecisionModel
        from .trainer import resolve_device

        dev = resolve_device(device)
        name = "/".join(Path(path).resolve().parts[-2:])
        return cls(DecisionModel.from_pretrained(path, dev), dev, name=name)

    @torch.inference_mode()
    def _probs(self, batch: dict[str, Any]) -> np.ndarray:
        t = {k: batch[k].to(self.device) for k in MODEL_KEYS}
        logits = self.model(**t)
        return torch.softmax(self.model.calibrate_logits(logits, t["qtype"], t["opt_mask"]), -1).float().cpu().numpy()
