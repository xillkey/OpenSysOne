"""Turn (state, question) into one encoder input sequence.

Default layout (one sequence per question)::

    [CLS] state ... [SEP] [Q] question [OPT] option_1 [OPT] option_2 ... [SEP]

Options (``SerializerConfig``):

* ``order="question_first"`` puts the question block first and the state last
  (``[CLS] [Q] question [OPT] ... [SEP] state [SEP]``), the order Laya uses.
* ``marker="mask"`` marks options with the tokenizer's own ``[MASK]`` token
  instead of the new ``[OPT]`` token, so MLM-pretrained encoders start from a
  meaningful embedding.
* ``bool_as_options=True`` renders yes/no questions as two options
  (``no: ...`` / ``yes: ...``) scored like a choice, instead of a sigmoid on ``[Q]``.
* ``parallel_options=True`` gives every option the same position ids and
  returns ``option_ids`` so the backbone can stop options attending to each
  other: the encoding no longer depends on option order.

Each option gets a fixed token budget, the question gets its own budget and the
state is truncated last, so options are never cut off by a long state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import Question, SchemaError

Q_TOKEN = "[Q]"
OPT_TOKEN = "[OPT]"
SPECIAL_TOKENS = [Q_TOKEN, OPT_TOKEN]
ORDERS = ("state_first", "question_first")
MARKERS = ("opt", "mask")


@dataclass
class SerializerConfig:
    max_length: int = 512
    option_token_budget: int = 16
    question_token_budget: int = 96
    max_options: int = 32
    order: str = "state_first"
    marker: str = "opt"
    bool_as_options: bool = False
    bool_option_text: list[str] = field(
        default_factory=lambda: ["no: the statement does not hold", "yes: the statement holds"]
    )
    parallel_options: bool = False

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {self.order!r}")
        if self.marker not in MARKERS:
            raise ValueError(f"marker must be one of {MARKERS}, got {self.marker!r}")
        if len(self.bool_option_text) != 2:
            raise ValueError("bool_option_text needs exactly two strings: [no, yes]")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SerializerConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Encoded:
    input_ids: list[int]
    q_pos: int
    opt_pos: list[int]   # index of each option marker
    opt_end: list[int]   # exclusive end of each option span (marker + its text)
    position_ids: list[int] | None = None  # only with parallel_options
    option_ids: list[int] | None = None    # 0 = shared token, s + 1 = token of option s


def parallel_layout(opt_pos: list[int], opt_end: list[int], length: int) -> tuple[list[int], list[int]]:
    """Every option span starts at the same position; tokens after the options continue after the longest."""
    if not opt_pos:
        return list(range(length)), [0] * length
    start = opt_pos[0]
    position_ids, option_ids = list(range(start)), [0] * start
    for s, (a, b) in enumerate(zip(opt_pos, opt_end)):
        position_ids += range(start, start + b - a)
        option_ids += [s + 1] * (b - a)
    after = start + max(b - a for a, b in zip(opt_pos, opt_end))
    rest = length - len(position_ids)
    position_ids += range(after, after + rest)
    option_ids += [0] * rest
    return position_ids, option_ids


class Serializer:
    def __init__(self, tokenizer: Any, config: SerializerConfig | None = None):
        self.tok = tokenizer
        self.cfg = config or SerializerConfig()
        unk = getattr(tokenizer, "unk_token_id", None)
        self.q_id = tokenizer.convert_tokens_to_ids(Q_TOKEN)
        self.opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
        if self.q_id is None or self.opt_id is None or unk in (self.q_id, self.opt_id):
            raise ValueError("tokenizer is missing [Q]/[OPT]; build it through a Backbone adapter")
        if self.cfg.marker == "mask":
            if tokenizer.mask_token_id is None:
                raise ValueError("marker='mask' needs a tokenizer with a mask token")
            self.marker_id = tokenizer.mask_token_id
        else:
            self.marker_id = self.opt_id
        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.prefix = [cls_id] if cls_id is not None else []
        self.sep = [sep_id] if sep_id is not None else []
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # text must never smuggle in a marker token
        self._strip = [t for t in {Q_TOKEN, OPT_TOKEN, tokenizer.mask_token} if t]

    def _ids(self, text: str, limit: int | None = None) -> list[int]:
        for t in self._strip:
            text = text.replace(t, " ")
        ids = self.tok(text, add_special_tokens=False, verbose=False)["input_ids"] if text else []
        return ids[:limit] if limit is not None else ids

    def has_option_spans(self, question: Question) -> bool:
        return question.type != "bool" or self.cfg.bool_as_options

    def encode(self, state: str, question: Question, options: list[str] | None = None) -> Encoded:
        cfg = self.cfg
        options = question.options if options is None else options
        if question.type == "bool" and cfg.bool_as_options:
            options = list(cfg.bool_option_text)
        if question.type != "bool" and len(options) > cfg.max_options:
            raise SchemaError(
                f"question {question.qid!r} has {len(options)} options (max_options={cfg.max_options}); "
                "split it hierarchically or raise max_options"
            )
        block = [self.q_id] + self._ids(question.text, cfg.question_token_budget)
        rel_opt: list[int] = []
        rel_end: list[int] = []
        if self.has_option_spans(question):
            for opt in options:
                rel_opt.append(len(block))
                block += [self.marker_id] + self._ids(opt, cfg.option_token_budget)
                rel_end.append(len(block))
        fixed = len(self.prefix) + 2 * len(self.sep) + len(block)
        if fixed > cfg.max_length:
            raise SchemaError(
                f"question {question.qid!r} needs {fixed} tokens before any state text "
                f"(max_length={cfg.max_length}); reduce options or option_token_budget"
            )
        state_ids = self._ids(state)[: cfg.max_length - fixed]
        if cfg.order == "question_first":
            offset = len(self.prefix)
            ids = self.prefix + block + self.sep + state_ids + self.sep
        else:
            offset = len(self.prefix) + len(state_ids) + len(self.sep)
            ids = self.prefix + state_ids + self.sep + block + self.sep
        opt_pos = [offset + r for r in rel_opt]
        opt_end = [offset + r for r in rel_end]
        enc = Encoded(input_ids=ids, q_pos=offset, opt_pos=opt_pos, opt_end=opt_end)
        if cfg.parallel_options:
            enc.position_ids, enc.option_ids = parallel_layout(opt_pos, opt_end, len(ids))
        return enc
