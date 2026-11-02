"""Export a trained model to ONNX (calibrated probabilities as output) and run it with onnxruntime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import MODEL_KEYS, DecisionDataset, collate
from .model import DecisionModel
from .schema import parse_example
from .predictor import _BasePredictor
from .serializer import Serializer, SerializerConfig

INPUTS = list(MODEL_KEYS)


class _Exportable(nn.Module):
    def __init__(self, model: DecisionModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, q_pos, opt_pos, opt_end, opt_mask, qtype,  # type: ignore[no-untyped-def]
                position_ids, option_ids):
        mask = opt_mask.bool()
        logits = self.model(input_ids, attention_mask, q_pos, opt_pos, opt_end, mask, qtype,
                            position_ids=position_ids, option_ids=option_ids)
        return torch.softmax(self.model.calibrate_logits(logits, qtype, mask), dim=-1)


_DUMMY = {
    "state": "dummy state text used only to trace the graph",
    "questions": {
        "c": {"type": "choice", "text": "pick one", "options": ["first option", "second", "third one here"]},
        "b": {"type": "bool", "text": "is it true?"},
    },
}


def export_onnx(model_dir: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DecisionModel.from_pretrained(model_dir, "cpu")
    ser = Serializer(model.tokenizer, model.serializer_config)
    ds = DecisionDataset([parse_example(_DUMMY, with_label=False)], ser, require_labels=False)
    batch = collate([ds[i] for i in range(len(ds))], ser.pad_id)
    dummy = tuple(batch[k] for k in INPUTS)
    path = out_dir / "model.onnx"
    kwargs: dict[str, Any] = dict(
        input_names=INPUTS,
        output_names=["probabilities"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "q_pos": {0: "batch"},
            "opt_pos": {0: "batch", 1: "options"},
            "opt_end": {0: "batch", 1: "options"},
            "opt_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"},
            "position_ids": {0: "batch", 1: "seq"},
            "option_ids": {0: "batch", 1: "seq"},
            "probabilities": {0: "batch", 1: "options"},
        },
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(_Exportable(model).eval(), dummy, str(path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the ``dynamo`` argument
            torch.onnx.export(_Exportable(model).eval(), dummy, str(path), **kwargs)
    # tokenizer + metadata travel with the onnx file
    model.tokenizer.save_pretrained(out_dir / "tokenizer")
    meta = json.loads((model_dir / "opensysone.json").read_text())
    meta["runtime"] = "onnx"
    (out_dir / "opensysone.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


class OnnxPredictor(_BasePredictor):
    def __init__(self, onnx_dir: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_dir = Path(onnx_dir)
        meta = json.loads((onnx_dir / "opensysone.json").read_text())
        tok = AutoTokenizer.from_pretrained(onnx_dir / "tokenizer")
        self.serializer = Serializer(tok, SerializerConfig.from_dict(meta["serializer"]))
        self.session = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=providers or ["CPUExecutionProvider"])
        self.name = "/".join(onnx_dir.resolve().parts[-2:]) + " (onnx)"
        self.input_names = [i.name for i in self.session.get_inputs()]  # unused inputs are pruned on export

    def _probs(self, batch: dict[str, Any]) -> np.ndarray:
        feeds = {k: batch[k].numpy() for k in self.input_names}
        return self.session.run(None, feeds)[0]
