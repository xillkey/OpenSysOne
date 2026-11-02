import pytest

from opensysone.schema import SchemaError, parse_example, parse_label


def test_hard_and_soft_labels():
    opts = ["a", "b", "c"]
    assert parse_label("b", opts, "choice") == [0.0, 1.0, 0.0]
    assert parse_label(2, opts, "choice") == [0.0, 0.0, 1.0]
    assert parse_label({"a": 3, "c": 1}, opts, "choice") == [0.75, 0.0, 0.25]
    assert parse_label([1, 1, 2], opts, "choice") == [0.25, 0.25, 0.5]
    assert parse_label(None, opts, "choice") is None


def test_bool_labels():
    for yes in (True, "yes", "是", 1, {"yes": 1.0}):
        assert parse_label(yes, ["no", "yes"], "bool") == [0.0, 1.0]
    assert parse_label(0.3, ["no", "yes"], "bool") == pytest.approx([0.7, 0.3])
    with pytest.raises(SchemaError):
        parse_label("maybe", ["no", "yes"], "bool")


def test_score_options_and_label_by_value():
    ex = parse_example({"state": "x", "questions": {"u": {"type": "rating", "min": 0, "max": 3, "label": 2}}})
    q = ex.questions[0]
    assert q.type == "score" and q.options == ["0", "1", "2", "3"]
    assert q.label == [0.0, 0.0, 1.0, 0.0]


def test_json_state_and_question_list():
    ex = parse_example({"state": {"b": 1, "a": "中"}, "questions": [{"id": "q", "type": "noul", "question": "?"}]})
    assert ex.state == '{"a": "中", "b": 1}'
    assert ex.questions[0].qid == "q" and ex.questions[0].type == "bool"


@pytest.mark.parametrize("bad", [
    {"questions": {}},
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"q": {"type": "choice", "options": ["a"]}}},
    {"state": "x", "questions": {"q": {"type": "choice", "options": ["a", "a"]}}},
    {"state": "x", "questions": {"q": {"type": "weird", "options": ["a", "b"]}}},
    {"state": "x", "questions": {"q": {"type": "choice", "options": ["a", "b"], "label": "z"}}},
])
def test_rejects_bad_examples(bad):
    with pytest.raises(SchemaError):
        parse_example(bad)
