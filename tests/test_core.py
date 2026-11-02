import pytest
import torch

from opensysone.backbone import build_backbone
from opensysone.model import DecisionModel, decision_loss
from opensysone.schema import SchemaError, parse_example
from opensysone.serializer import Serializer, SerializerConfig


@pytest.fixture(scope="module")
def backbone(tiny_backbone):
    return build_backbone({"name": "hf", "model": str(tiny_backbone)})


def test_special_tokens_added(backbone):
    tok = backbone.tokenizer
    assert tok.convert_tokens_to_ids("[Q]") != tok.unk_token_id
    assert backbone.encoder.get_input_embeddings().weight.shape[0] == len(tok)


def test_serializer_positions_and_truncation(backbone):
    ser = Serializer(backbone.tokenizer, SerializerConfig(max_length=40, option_token_budget=3))
    ex = parse_example({"state": "word " * 200, "questions": {
        "c": {"type": "choice", "text": "pick", "options": ["billing team", "shipping", "account"]}}})
    enc = ser.encode(ex.state, ex.questions[0])
    assert len(enc.input_ids) == 40  # state truncated, question and options kept
    assert enc.input_ids[enc.q_pos] == ser.q_id
    assert [enc.input_ids[p] for p in enc.opt_pos] == [ser.opt_id] * 3
    assert enc.opt_end[:-1] == enc.opt_pos[1:] and enc.opt_end[-1] == len(enc.input_ids) - 1
    assert enc.input_ids[-1] == ser.sep[0]


def test_serializer_rejects_too_many_options(backbone):
    ser = Serializer(backbone.tokenizer, SerializerConfig(max_options=3))
    ex = parse_example({"state": "x", "questions": {"c": {"options": list("abcd")}}})
    with pytest.raises(SchemaError):
        ser.encode(ex.state, ex.questions[0])


def test_bool_has_no_option_markers(backbone):
    ser = Serializer(backbone.tokenizer)
    ex = parse_example({"state": "x", "questions": {"b": {"type": "bool", "text": "ok?"}}})
    assert ser.encode(ex.state, ex.questions[0]).opt_pos == []


def test_loss_is_proper_and_masked():
    mask = torch.tensor([[True, True, True, False]])
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    qtype = torch.tensor([1])  # score -> RPS on
    has = torch.tensor([True])
    good = torch.tensor([[-9.0, 9.0, -9.0, -1e4]])
    assert decision_loss(good, target, mask, qtype, has) < 1e-3
    # truth = level 0: predicting level 1 vs level 2 has the same log loss, but RPS ranks the far miss worse
    t0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    near = torch.tensor([[-9.0, 9.0, -9.0, -1e4]])
    far = torch.tensor([[-9.0, -9.0, 9.0, -1e4]])
    l_near = decision_loss(near, t0, mask, qtype, has)
    l_far = decision_loss(far, t0, mask, qtype, has)
    assert l_far > l_near
    # with RPS off (choice type) they are equal
    choice = torch.tensor([0])
    assert decision_loss(near, t0, mask, choice, has) == pytest.approx(decision_loss(far, t0, mask, choice, has).item())
    # unlabeled rows contribute nothing
    assert decision_loss(far, t0, mask, qtype, torch.tensor([False])) == 0


def test_bool_logits_are_sigmoid(backbone):
    model = DecisionModel(backbone).eval()
    ids = torch.tensor([[2, 10, 3, backbone.tokenizer.convert_tokens_to_ids("[Q]"), 11, 3]])
    out = model(ids, torch.ones_like(ids), torch.tensor([3]), torch.zeros(1, 2, dtype=torch.long),
                torch.zeros(1, 2, dtype=torch.long), torch.tensor([[True, True]]), torch.tensor([2]))
    assert out.shape == (1, 2) and out[0, 0] == 0
