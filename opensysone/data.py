"""Dataset + collate: one training item per (example, question)."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .schema import Example, Question, SchemaError
from .serializer import Serializer

log = logging.getLogger(__name__)


@dataclass
class Item:
    input_ids: list[int]
    q_pos: int
    opt_pos: list[int]
    opt_end: list[int]
    position_ids: list[int] | None
    option_ids: list[int] | None
    n_options: int
    qtype: int
    target: list[float] | None
    ref: tuple[int, str]  # (example index, question id)
    options: list[str]


class DecisionDataset(Dataset):
    """Expands examples into question-level items.

    ``shuffle_options`` permutes the options of ``choice`` questions every time
    an item is read, which teaches permutation invariance (score levels and bool
    keep their order).
    """

    def __init__(
        self,
        examples: list[Example],
        serializer: Serializer,
        shuffle_options: bool = False,
        require_labels: bool = True,
        seed: int = 0,
    ):
        self.serializer = serializer
        self.shuffle_options = shuffle_options
        self.rng = random.Random(seed)
        self.index: list[tuple[int, Question]] = []
        self.examples = examples
        skipped = 0
        for ei, ex in enumerate(examples):
            for q in ex.questions:
                if require_labels and not q.has_label:
                    continue
                try:
                    serializer.encode(ex.state, q)
                except SchemaError as e:
                    skipped += 1
                    log.warning("skipping %s/%s: %s", ex.id or ei, q.qid, e)
                    continue
                self.index.append((ei, q))
        if skipped:
            log.warning("skipped %d questions that do not fit the serializer limits", skipped)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Item:
        ei, q = self.index[i]
        options, target = q.options, q.label
        if self.shuffle_options and q.type == "choice":
            perm = list(range(len(options)))
            self.rng.shuffle(perm)
            options = [options[j] for j in perm]
            target = [target[j] for j in perm] if target is not None else None
        enc = self.serializer.encode(self.examples[ei].state, q, options)
        return Item(
            input_ids=enc.input_ids,
            q_pos=enc.q_pos,
            opt_pos=enc.opt_pos,
            opt_end=enc.opt_end,
            position_ids=enc.position_ids,
            option_ids=enc.option_ids,
            n_options=2 if q.type == "bool" else len(options),
            qtype=q.type_id,
            target=target,
            ref=(ei, q.qid),
            options=options,
        )


def collate(items: list[Item], pad_id: int) -> dict[str, Any]:
    b = len(items)
    t = max(len(it.input_ids) for it in items)
    k = max(2, max(it.n_options for it in items))
    input_ids = torch.full((b, t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    opt_pos = torch.zeros((b, k), dtype=torch.long)
    opt_end = torch.zeros((b, k), dtype=torch.long)
    opt_mask = torch.zeros((b, k), dtype=torch.bool)
    position_ids = torch.arange(t, dtype=torch.long).repeat(b, 1)
    option_ids = torch.zeros((b, t), dtype=torch.long)
    target = torch.zeros((b, k), dtype=torch.float)
    has_target = torch.zeros(b, dtype=torch.bool)
    for i, it in enumerate(items):
        n = len(it.input_ids)
        input_ids[i, :n] = torch.tensor(it.input_ids)
        attention_mask[i, :n] = 1
        if it.position_ids is not None:
            position_ids[i, :n] = torch.tensor(it.position_ids)
            option_ids[i, :n] = torch.tensor(it.option_ids)
        if it.opt_pos:
            opt_pos[i, : len(it.opt_pos)] = torch.tensor(it.opt_pos)
            opt_end[i, : len(it.opt_end)] = torch.tensor(it.opt_end)
        opt_mask[i, : it.n_options] = True
        if it.target is not None:
            target[i, : it.n_options] = torch.tensor(it.target)
            has_target[i] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "q_pos": torch.tensor([it.q_pos for it in items], dtype=torch.long),
        "opt_pos": opt_pos,
        "opt_end": opt_end,
        "opt_mask": opt_mask,
        "position_ids": position_ids,
        "option_ids": option_ids,
        "qtype": torch.tensor([it.qtype for it in items], dtype=torch.long),
        "target": target,
        "has_target": has_target,
        "refs": [it.ref for it in items],
        "options": [it.options for it in items],
    }


MODEL_KEYS = ("input_ids", "attention_mask", "q_pos", "opt_pos", "opt_end", "opt_mask", "qtype", "position_ids", "option_ids")
TENSOR_KEYS = MODEL_KEYS + ("target", "has_target")


def to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {k: (v.to(device) if k in TENSOR_KEYS else v) for k, v in batch.items()}
