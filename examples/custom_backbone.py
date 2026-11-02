"""Example plugin: register your own backbone.

Use it from a config::

    plugins: [examples.custom_backbone]     # importable module path
    backbone:
      name: mean-pooled-hf                   # the name registered below
      model: answerdotai/ModernBERT-base

Any class works as long as it follows the ``Backbone`` contract (tokenizer with
[Q]/[OPT], ``hidden_size``, ``forward(ids, mask) -> [B, T, H]``, ``save(dir)``).
This one subclasses the HF adapter and mixes a little global context into every
token, just to show where custom logic goes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from opensysone.backbone import HFEncoder, register_backbone


@register_backbone("mean-pooled-hf")
class MeanPooledHF(HFEncoder):
    def __init__(self, model: str, mix: float = 0.1, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.mix = mix

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = super().forward(input_ids, attention_mask, **kwargs)  # kwargs: position_ids / option_ids
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
        return h + self.mix * mean

    def save(self, directory: str | Path) -> dict[str, Any]:
        kwargs = super().save(directory)
        # the returned dict is stored in opensysone.json and used to rebuild the backbone
        return {**kwargs, "name": "mean-pooled-hf", "mix": self.mix}
