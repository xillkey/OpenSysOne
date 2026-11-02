"""Serializer layouts, head variants, RLCD loss, bucketed calibration and parallel option layout."""

import json

import pytest
import torch

from opensysone.backbone import build_backbone
from opensysone.config import load_config
from opensysone.data import DecisionDataset, collate
from opensysone.model import DecisionModel, HeadConfig, rlcd_loss
from opensysone.schema import parse_example
from opensysone.serializer import Serializer, SerializerConfig

REQ = {"state": "my parcel never arrived and I want my money back", "questions": {
    "c": {"type": "choice", "text": "which team", "options": ["billing team", "shipping", "account access", "tech"]},
    "s": {"type": "score", "text": "how urgent", "min": 1, "max": 5},
    "b": {"type": "bool", "text": "refund asked"}}}


@pytest.fixture(scope="module")
def backbone(tiny_backbone):
    return build_backbone({"name": "hf", "model": str(tiny_backbone)})


@pytest.mark.parametrize("order", ["state_first", "question_first"])
@pytest.mark.parametrize("marker", ["opt", "mask"])
@pytest.mark.parametrize("bool_as_options", [False, True])
def test_layouts_put_markers_where_they_say(backbone, order, marker, bool_as_options):
    cfg = SerializerConfig(order=order, marker=marker, bool_as_options=bool_as_options, max_length=64)
    ser = Serializer(backbone.tokenizer, cfg)
    ex = parse_example(REQ, with_label=False)
    for q in ex.questions:
        enc = ser.encode("word " * 300, q)
        assert len(enc.input_ids) == 64
        assert enc.input_ids[enc.q_pos] == ser.q_id
        assert all(enc.input_ids[p] == ser.marker_id for p in enc.opt_pos)
        n = 0 if (q.type == "bool" and not bool_as_options) else len(q.options)
        assert len(enc.opt_pos) == n
        if order == "question_first":
            assert enc.q_pos == 1
    assert ser.marker_id == (backbone.tokenizer.mask_token_id if marker == "mask" else ser.opt_id)


def test_marker_tokens_in_text_are_neutralised(backbone):
    ser = Serializer(backbone.tokenizer, SerializerConfig(marker="mask"))
    ex = parse_example({"state": "a [MASK] b [OPT] c [Q]", "questions": {"c": {"options": ["x [MASK]", "y"]}}})
    enc = ser.encode(ex.state, ex.questions[0])
    assert enc.input_ids.count(ser.marker_id) == 2 and enc.input_ids.count(ser.q_id) == 1


def _probs(model, ser, req):
    ex = parse_example(req, with_label=False)
    ds = DecisionDataset([ex], ser, require_labels=False)
    batch = collate([ds[i] for i in range(len(ds))], ser.pad_id)
    with torch.no_grad():
        logits = model(**batch)
        return torch.softmax(model.calibrate_logits(logits, batch["qtype"], batch["opt_mask"]), -1)[0]


@pytest.mark.parametrize("pooling", ["span", "marker"])
def test_parallel_layout_is_order_invariant(backbone, pooling):
    cfg = SerializerConfig(parallel_options=True, order="question_first")
    model = DecisionModel(backbone, cfg, HeadConfig(pooling=pooling, head_layers=1, type_embedding=True)).eval()
    ser = Serializer(backbone.tokenizer, cfg)
    opts = ["billing team", "shipping", "account access", "tech"]
    perm = [2, 0, 3, 1]
    base = {"state": REQ["state"], "questions": {"c": {"type": "choice", "text": "which", "options": opts}}}
    shuffled = {"state": REQ["state"], "questions": {"c": {"type": "choice", "text": "which",
                                                          "options": [opts[i] for i in perm]}}}
    p, q = _probs(model, ser, base), _probs(model, ser, shuffled)
    assert torch.allclose(p[perm], q, atol=1e-5)


def test_sequential_layout_is_not_order_invariant(backbone):
    """Sanity check that the test above can fail: the default layout does depend on order."""
    cfg = SerializerConfig()
    model = DecisionModel(backbone, cfg).eval()
    ser = Serializer(backbone.tokenizer, cfg)
    opts = ["billing team", "shipping", "account access", "tech"]
    perm = [2, 0, 3, 1]
    mk = lambda o: {"state": REQ["state"], "questions": {"c": {"type": "choice", "text": "which", "options": o}}}
    p, q = _probs(model, ser, mk(opts)), _probs(model, ser, mk([opts[i] for i in perm]))
    assert not torch.allclose(p[perm], q, atol=1e-5)


def test_rlcd_loss_gradients_point_to_target():
    torch.manual_seed(0)
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = rlcd_loss(logits, target, mask, torch.tensor([0]), torch.tensor([True]), sigma=0.3, samples=64)
    loss.backward()
    assert logits.grad[0, 1] < 0 and logits.grad[0, 0] > 0 and logits.grad[0, 2] > 0


def test_temperature_buckets_roundtrip(backbone, tmp_path):
    model = DecisionModel(backbone)
    model.set_temperatures({"choice": 1.5, "score": 0.9, "bool": 1.2}, {"choice:3-5": 2.0})
    model.save_pretrained(tmp_path)
    meta = json.loads((tmp_path / "opensysone.json").read_text())
    assert meta["temperature_by_options"] == {"choice:3-5": 2.0}
    again = DecisionModel.from_pretrained(tmp_path)
    logits = torch.zeros(2, 6)
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    out = again.calibrate_logits(logits + 1.0, torch.tensor([0, 0]), mask)
    assert out[0, 0].item() == pytest.approx(0.5) and out[1, 0].item() == pytest.approx(1 / 1.5)


@pytest.mark.parametrize("overrides", [
    ["head.pooling=marker", "head.head_layers=2", "head.type_embedding=true", "head.scorer=shared",
     "data.order=question_first", "data.marker=mask", "data.bool_as_options=true", "loss.name=rlcd",
     "train.calibrate_by_options=true"],
    ["data.parallel_options=true", "head.head_layers=1"],
])
def test_variant_trains_and_exports(toy_data, tiny_backbone, tmp_path, overrides):
    pytest.importorskip("onnxruntime")
    from opensysone.onnx_export import OnnxPredictor, export_onnx
    from opensysone.predictor import Predictor
    from opensysone.trainer import train

    cfg = load_config(None, [
        f"backbone.model={tiny_backbone}", f"data.train={toy_data / 'train.jsonl'}",
        f"data.dev={toy_data / 'dev.jsonl'}", "data.max_length=96", "data.option_token_budget=6",
        f"train.output_dir={tmp_path / 'run'}", "train.max_steps=20", "train.batch_size=32", "train.lr=1e-3",
        "train.llrd=1.0", "train.device=cpu", "train.log_every=1000", *overrides,
    ])
    res = train(cfg)
    assert torch.isfinite(torch.tensor(res["dev"]["all"]["nll"]))
    final = tmp_path / "run" / "final"
    export_onnx(final, tmp_path / "onnx")
    a = Predictor.from_pretrained(final, "cpu").decide(REQ)["results"]
    b = OnnxPredictor(tmp_path / "onnx").decide(REQ)["results"]
    for qid in a:
        pa, pb = a[qid]["probabilities"], b[qid]["probabilities"]
        assert max(abs(pa[k] - pb[k]) for k in pa) < 1e-4, qid
