"""Train -> calibrate -> evaluate -> predict -> serve -> ONNX on a tiny random backbone (CPU, ~1 min)."""

import json

import numpy as np
import pytest

from opensysone.config import load_config
from opensysone.trainer import train


@pytest.fixture(scope="module")
def trained(toy_data, tiny_backbone, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = load_config(None, [
        f"backbone.model={tiny_backbone}",
        f"data.train={toy_data / 'train.jsonl'}",
        f"data.dev={toy_data / 'dev.jsonl'}",
        "data.max_length=96", "data.option_token_budget=6",
        f"train.output_dir={out}", "train.max_steps=60", "train.batch_size=32",
        "train.lr=1e-3", "train.llrd=1.0", "train.device=cpu", "train.eval_every=30", "train.log_every=1000",
    ])
    result = train(cfg)
    return out, result


def test_training_writes_calibrated_model(trained):
    out, result = trained
    assert (out / "final" / "opensysone.json").exists()
    meta = json.loads((out / "final" / "opensysone.json").read_text())
    assert set(meta["temperature"]) == {"choice", "score", "bool"}
    assert all(0.05 <= t <= 20 for t in meta["temperature"].values())
    assert result["dev"]["all"]["n"] == 360
    assert np.isfinite(result["dev"]["all"]["nll"])


def test_predictor_response_shape(trained):
    from opensysone.predictor import Predictor

    pred = Predictor.from_pretrained(trained[0] / "final", "cpu")
    req = {"state": "my package has not arrived yet, I want a refund.", "questions": {
        "team": {"type": "choice", "question": "Which team?", "options": ["billing", "shipping"]},
        "urgency": {"type": "score", "question": "How urgent?", "min": 1, "max": 5},
        "refund": {"type": "bool", "question": "Refund?"}}}
    res = pred.decide(req)["results"]
    assert res["team"]["answer"] in ("billing", "shipping")
    assert sum(res["team"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert 1 <= res["urgency"]["expected"] <= 5
    assert 0 <= res["refund"]["probability"] <= 1 and res["refund"]["answer"] in ("yes", "no")


def test_server(trained):
    from fastapi.testclient import TestClient

    from opensysone.predictor import Predictor
    from opensysone.serve import create_app

    app = create_app(Predictor.from_pretrained(trained[0] / "final", "cpu"))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        r = client.post("/v1/systemone", json={"state": "x", "questions": {"b": {"type": "bool", "question": "?"}}})
        assert r.status_code == 200 and "b" in r.json()["results"]
        bad = client.post("/v1/systemone", json={"state": "x", "questions": {"c": {"type": "choice", "options": ["a"]}}})
        assert bad.status_code == 422


def test_onnx_matches_torch(trained, tmp_path):
    pytest.importorskip("onnxruntime")
    from opensysone.onnx_export import OnnxPredictor, export_onnx
    from opensysone.predictor import Predictor

    export_onnx(trained[0] / "final", tmp_path)
    req = {"state": "我登录不了账号，非常紧急！", "questions": {
        "team": {"type": "choice", "question": "哪个团队？", "options": ["billing", "account", "technical"]},
        "urgency": {"type": "score", "question": "多紧急？", "min": 1, "max": 5},
        "refund": {"type": "bool", "question": "退款？"}}}
    a = Predictor.from_pretrained(trained[0] / "final", "cpu").decide(req)["results"]
    b = OnnxPredictor(tmp_path).decide(req)["results"]
    for qid in a:
        pa, pb = a[qid]["probabilities"], b[qid]["probabilities"]
        assert pa.keys() == pb.keys()
        assert max(abs(pa[k] - pb[k]) for k in pa) < 1e-4
