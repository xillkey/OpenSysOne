"""Build a small randomly initialised BERT encoder + WordPiece tokenizer from your own corpus.

Useful for offline smoke tests, CI, and for training a decision model from
scratch on a narrow domain. For real workloads start from a pretrained encoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import iter_jsonl, state_to_text


def _texts_from_jsonl(paths: Iterable[str | Path]) -> Iterator[str]:
    for path in paths:
        for _, obj in iter_jsonl(path):
            yield state_to_text(obj.get("state", ""))
            qs = obj.get("questions", {})
            for q in (qs.values() if isinstance(qs, dict) else qs):
                yield str(q.get("text", q.get("question", "")))
                yield " ".join(map(str, q.get("options", [])))


def make_tiny_backbone(
    corpus: list[str | Path],
    out_dir: str | Path,
    vocab_size: int = 8000,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    max_position: int = 512,
    seed: int = 0,
) -> Path:
    import torch
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = list(_texts_from_jsonl(corpus))
    wp = BertWordPieceTokenizer(lowercase=True, handle_chinese_chars=True, strip_accents=False)
    wp.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=1,
                           special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=wp._tokenizer,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]",
        model_max_length=max_position,
    )
    if len(tok) < 50:
        raise RuntimeError(f"tokenizer vocabulary is suspiciously small ({len(tok)}); check the corpus")
    tok.save_pretrained(out)

    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=len(tok), hidden_size=hidden_size, num_hidden_layers=num_layers,
        num_attention_heads=num_heads, intermediate_size=4 * hidden_size,
        max_position_embeddings=max_position,
    )
    BertModel(cfg, add_pooling_layer=False).save_pretrained(out)
    (out / "TINY_BACKBONE.json").write_text(json.dumps({"corpus": [str(c) for c in corpus], "texts": len(texts)}))
    return out
