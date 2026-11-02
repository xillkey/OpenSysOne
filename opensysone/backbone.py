"""Swappable encoder backbones.

Everything above this file only relies on the small ``Backbone`` contract:

* ``tokenizer``            a Hugging Face style tokenizer containing [Q] and [OPT]
* ``hidden_size``          int
* ``forward(ids, mask)``   -> last hidden states ``[batch, seq, hidden]``
* ``forward(ids, mask, position_ids=..., option_ids=...)`` only when the
  serializer uses ``parallel_options`` (set ``supports_parallel = True``)
* ``save(dir)`` / ``build_backbone({"name": ..., "model": dir})`` round trip

The built-in ``hf`` adapter wraps any bidirectional encoder on the Hugging Face
Hub (ModernBERT, mmBERT, DeBERTa-v3, XLM-R, BERT, E5/BGE encoders ...). To plug
in something else, subclass ``Backbone`` and decorate it with
``@register_backbone("my-name")``, then list your module under ``plugins`` in
the training config.
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .serializer import SPECIAL_TOKENS

log = logging.getLogger(__name__)

BACKBONES: dict[str, type["Backbone"]] = {}
_LAYER_RE = re.compile(r"(?:^|\.)(?:layer|layers|block|blocks|h)\.(\d+)\.")


def register_backbone(name: str) -> Callable[[type["Backbone"]], type["Backbone"]]:
    def deco(cls: type["Backbone"]) -> type["Backbone"]:
        BACKBONES[name] = cls
        cls.registry_name = name
        return cls

    return deco


def import_plugins(modules: list[str] | None) -> None:
    """Import plugin modules (dotted paths); the current directory is importable."""
    if modules and os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    for mod in modules or []:
        importlib.import_module(mod)


def build_backbone(cfg: dict[str, Any]) -> "Backbone":
    cfg = dict(cfg)
    import_plugins(cfg.pop("plugins", None))
    name = cfg.pop("name", "hf")
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(BACKBONES)}")
    return BACKBONES[name](**cfg)


class Backbone(nn.Module):
    registry_name = "base"
    tokenizer: Any
    hidden_size: int

    supports_parallel = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        option_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def save(self, directory: str | Path) -> dict[str, Any]:
        """Write weights + tokenizer and return the kwargs that rebuild this backbone from ``directory``."""
        raise NotImplementedError  # pragma: no cover

    def layer_depth(self, param_name: str) -> int | None:
        """Depth used for layer-wise LR decay: 0 = embeddings, i+1 = block i, None = top."""
        if "embed" in param_name:
            return 0
        m = _LAYER_RE.search(param_name)
        return int(m.group(1)) + 1 if m else None


@register_backbone("hf")
class HFEncoder(Backbone):
    """Any Hugging Face bidirectional encoder loaded with ``AutoModel``."""

    def __init__(
        self,
        model: str,
        lora: dict[str, Any] | None = None,
        train_embeddings: bool = True,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        revision: str | None = None,
        **_: Any,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_name = model
        self.trust_remote_code = trust_remote_code
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision)
        if getattr(config, "is_decoder", False) and not getattr(config, "is_encoder_decoder", False):
            log.warning("%s looks like a causal decoder; OpenSysOne expects a bidirectional encoder", model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision)
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "revision": revision}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if getattr(config, "is_encoder_decoder", False):
            raise ValueError(
                f"{model} is an encoder-decoder model; register a custom backbone "
                "(e.g. wrapping T5EncoderModel) instead of using the 'hf' adapter"
            )
        encoder = AutoModel.from_pretrained(model, config=config, **kwargs)

        missing = [t for t in SPECIAL_TOKENS if t not in self.tokenizer.get_vocab()]
        if missing:
            self.tokenizer.add_tokens(missing, special_tokens=True)
            encoder.resize_token_embeddings(len(self.tokenizer))
            self._init_new_embeddings(encoder, missing)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token

        if gradient_checkpointing:
            encoder.gradient_checkpointing_enable()
        self.lora_enabled = bool(lora)
        if lora:
            encoder = self._wrap_lora(encoder, lora, train_embeddings)
        self.encoder = encoder
        self.hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "d_model"))

    def _init_new_embeddings(self, encoder: nn.Module, new_tokens: list[str]) -> None:
        emb = encoder.get_input_embeddings().weight
        with torch.no_grad():
            known = emb[: len(self.tokenizer) - len(new_tokens)]
            mean, std = known.mean(0), known.std(0)
            gen = torch.Generator().manual_seed(1234)
            for tok in new_tokens:
                idx = self.tokenizer.convert_tokens_to_ids(tok)
                noise = torch.randn(emb.shape[1], generator=gen).to(emb)
                emb[idx] = mean + 0.5 * std * noise  # distinct vectors for [Q] and [OPT]

    def _wrap_lora(self, encoder: nn.Module, lora: dict[str, Any], train_embeddings: bool) -> nn.Module:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise ImportError("LoRA needs `pip install opensysone[lora]`") from e
        cfg = LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=lora.get("target_modules", "all-linear"),
            bias="none",
        )
        encoder = get_peft_model(encoder, cfg)
        if train_embeddings:  # let the new [Q]/[OPT] rows learn
            encoder.get_input_embeddings().weight.requires_grad_(True)
        return encoder

    supports_parallel = True

    def _parallel_mask(self, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                       option_ids: torch.Tensor) -> Any:
        """Boolean [B, 1, T, T] mask: every real token, except tokens of a *different* option."""
        keys = attention_mask.bool()[:, None, :]
        other = (option_ids[:, :, None] > 0) & (option_ids[:, None, :] > 0) \
            & (option_ids[:, :, None] != option_ids[:, None, :])
        full = (keys & ~other)[:, None]
        cfg = getattr(self.encoder, "config", None)
        if getattr(cfg, "model_type", "") == "modernbert":  # per-layer-type masks (transformers >= 5)
            window = getattr(cfg, "local_attention", None)
            masks = {"full_attention": full}
            if window:
                near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= window // 2
                masks["sliding_attention"] = full & near[:, None]
            return masks
        return full

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        option_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if option_ids is None:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            mask = self._parallel_mask(attention_mask, position_ids, option_ids)
            out = self.encoder(input_ids=input_ids, attention_mask=mask, position_ids=position_ids)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def save(self, directory: str | Path) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.encoder
        if self.lora_enabled:
            model = copy.deepcopy(model).merge_and_unload()  # ship plain weights
        model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        return {"name": "hf", "trust_remote_code": self.trust_remote_code}
