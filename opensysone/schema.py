"""Data schema: a decision example is a *state* plus a set of typed questions.

JSONL line format (one example per line)::

    {
      "id": "optional-id",
      "state": "free text"  |  {"any": "json object"},
      "questions": {
        "category": {"type": "choice", "text": "Which team?", "options": ["billing", "tech"], "label": "billing"},
        "urgency":  {"type": "score",  "text": "How urgent?", "min": 1, "max": 5, "label": 4},
        "refund":   {"type": "bool",   "text": "Asks for a refund?", "label": true}
      }
    }

Labels may be hard (option name / index / bool) or soft (a probability list or
an ``{option: prob}`` dict), so teacher-model distillation works out of the box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

QTYPES = ("choice", "score", "bool")
TYPE_IDS = {name: i for i, name in enumerate(QTYPES)}
BOOL_OPTIONS = ["no", "yes"]

_TYPE_ALIASES = {
    "choice": "choice", "select": "choice", "classify": "choice", "category": "choice",
    "score": "score", "scale": "score", "rating": "score", "ordinal": "score",
    "bool": "bool", "boolean": "bool", "yesno": "bool", "yes_no": "bool",
    "binary": "bool", "noul": "bool",
}
_TRUE = {"yes", "true", "1", "y", "是", "对"}
_FALSE = {"no", "false", "0", "n", "否", "不是"}


class SchemaError(ValueError):
    """Raised when an example or a request does not match the schema."""


def normalize_type(raw: str | None) -> str:
    key = str(raw or "choice").strip().lower()
    if key not in _TYPE_ALIASES:
        raise SchemaError(f"unknown question type {raw!r}; expected one of {QTYPES}")
    return _TYPE_ALIASES[key]


@dataclass
class Question:
    qid: str
    type: str
    text: str
    options: list[str]
    label: list[float] | None = None  # soft target over ``options``

    @property
    def type_id(self) -> int:
        return TYPE_IDS[self.type]

    @property
    def has_label(self) -> bool:
        return self.label is not None


@dataclass
class Example:
    state: str
    questions: list[Question]
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _options_for(qtype: str, raw: dict[str, Any]) -> list[str]:
    if qtype == "bool":
        return list(BOOL_OPTIONS)
    if qtype == "score":
        if raw.get("options"):
            opts = [str(o) for o in raw["options"]]
        else:
            lo, hi = int(raw.get("min", 1)), int(raw.get("max", 5))
            if hi <= lo:
                raise SchemaError(f"score question needs max > min, got {lo}..{hi}")
            opts = [str(i) for i in range(lo, hi + 1)]
    else:
        opts = [str(o) for o in raw.get("options") or []]
    if len(opts) < 2:
        raise SchemaError(f"{qtype} question needs at least 2 options")
    if len(set(opts)) != len(opts):
        raise SchemaError(f"duplicate options: {opts}")
    return opts


def _bool_prob(label: Any) -> float:
    if isinstance(label, bool):
        return 1.0 if label else 0.0
    if isinstance(label, (int, float)):
        p = float(label)
        if not 0.0 <= p <= 1.0:
            raise SchemaError(f"bool probability must be in [0, 1], got {label}")
        return p
    if isinstance(label, str):
        key = label.strip().lower()
        if key in _TRUE:
            return 1.0
        if key in _FALSE:
            return 0.0
        raise SchemaError(f"cannot read bool label {label!r}")
    if isinstance(label, dict):
        if "yes" in label:
            return float(label["yes"]) / max(float(label["yes"]) + float(label.get("no", 0.0)), 1e-12)
        if "no" in label:
            return 1.0 - float(label["no"])
    if isinstance(label, (list, tuple)) and len(label) == 2:
        no, yes = float(label[0]), float(label[1])
        return yes / max(no + yes, 1e-12)
    raise SchemaError(f"cannot read bool label {label!r}")


def parse_label(label: Any, options: list[str], qtype: str) -> list[float] | None:
    """Turn any supported label form into a normalized distribution over options."""
    if label is None:
        return None
    k = len(options)
    if qtype == "bool":
        p = _bool_prob(label)
        return [1.0 - p, p]
    if isinstance(label, dict):
        unknown = set(map(str, label)) - set(options)
        if unknown:
            raise SchemaError(f"label mentions unknown options {sorted(unknown)}")
        vec = [float(label.get(o, 0.0)) for o in options]
    elif isinstance(label, (list, tuple)):
        if len(label) != k:
            raise SchemaError(f"soft label has {len(label)} values but there are {k} options")
        vec = [float(x) for x in label]
    else:
        key = str(label)
        if key in options:
            vec = [1.0 if o == key else 0.0 for o in options]
        elif isinstance(label, int) and not isinstance(label, bool) and 0 <= label < k:
            vec = [1.0 if i == label else 0.0 for i in range(k)]
        else:
            raise SchemaError(f"label {label!r} is not one of the options {options}")
    if any(x < 0 for x in vec):
        raise SchemaError("label probabilities must be non-negative")
    total = sum(vec)
    if total <= 0:
        raise SchemaError("label probabilities sum to zero")
    return [x / total for x in vec]


def parse_question(qid: str, raw: dict[str, Any], with_label: bool = True) -> Question:
    if not isinstance(raw, dict):
        raise SchemaError(f"question {qid!r} must be an object")
    qtype = normalize_type(raw.get("type"))
    text = raw.get("text", raw.get("question", raw.get("prompt", "")))
    options = _options_for(qtype, raw)
    label = parse_label(raw.get("label"), options, qtype) if with_label else None
    return Question(qid=str(qid), type=qtype, text=str(text), options=options, label=label)


def _iter_raw_questions(raw: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(raw, dict):
        yield from ((str(k), v) for k, v in raw.items())
    elif isinstance(raw, list):
        for i, q in enumerate(raw):
            qid = q.get("id", q.get("qid", f"q{i}")) if isinstance(q, dict) else f"q{i}"
            yield str(qid), q
    else:
        raise SchemaError("'questions' must be an object or a list")


def parse_example(obj: dict[str, Any], with_label: bool = True) -> Example:
    if "state" not in obj:
        raise SchemaError("example is missing 'state'")
    questions = [parse_question(qid, q, with_label) for qid, q in _iter_raw_questions(obj.get("questions", {}))]
    if not questions:
        raise SchemaError("example has no questions")
    return Example(
        state=state_to_text(obj["state"]),
        questions=questions,
        id=obj.get("id"),
        meta={k: v for k, v in obj.items() if k not in {"state", "questions", "id"}},
    )


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith("//"):
                yield lineno, json.loads(line)


def load_jsonl(path: str | Path, strict: bool = True) -> list[Example]:
    examples: list[Example] = []
    for lineno, obj in iter_jsonl(path):
        try:
            examples.append(parse_example(obj))
        except SchemaError as e:
            if strict:
                raise SchemaError(f"{path}:{lineno}: {e}") from e
    return examples


def load_many(paths: str | Path | Iterable[str | Path]) -> list[Example]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Example] = []
    for p in paths:
        out.extend(load_jsonl(p))
    return out
