"""Decision heads, losses and the full model (backbone + heads + calibration)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import Backbone, build_backbone
from .schema import TYPE_IDS
from .serializer import SerializerConfig

FORMAT_NAME = "opensysone"
FORMAT_VERSION = 2
NEG_INF = -1e4  # safe in fp16/bf16
CHOICE, SCORE, BOOL = TYPE_IDS["choice"], TYPE_IDS["score"], TYPE_IDS["bool"]
QTYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}

# temperature buckets by option count (same split as Laya)
K_BUCKETS = ("2", "3-5", "6-10", "11+")
TEMP_MIN, TEMP_MAX = 0.05, 20.0


def k_bucket_index(k: torch.Tensor) -> torch.Tensor:
    return (k > 2).long() + (k > 5).long() + (k > 10).long()


def k_bucket_name(k: int) -> str:
    return K_BUCKETS[(k > 2) + (k > 5) + (k > 10)]


POOLINGS = ("span", "marker")
SCORERS = ("per_type", "shared")


@dataclass
class HeadConfig:
    """Decision head options.

    * ``pooling``        ``span`` = mean over [marker + option tokens]; ``marker`` = the marker position only
    * ``head_layers``    extra transformer layers on top of the backbone (Laya uses 2)
    * ``type_embedding`` add a learned question-type embedding to every position before the head
    * ``scorer``         ``per_type`` = separate MLPs for choice / score over ``[h_opt, h_q, h_opt*h_q]``;
                         ``shared`` = one MLP over ``h_opt`` for every type (Laya style)
    """

    pooling: str = "span"
    head_layers: int = 0
    type_embedding: bool = False
    scorer: str = "per_type"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {self.scorer!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "HeadConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(d_in: int, d_hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, 1))


class HeadLayer(nn.Module):
    """Pre-norm transformer layer written with plain ops so it exports to ONNX with dynamic shapes."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.n_heads = max(1, d // 64)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = (t.unflatten(-1, (self.n_heads, -1)).transpose(1, 2) for t in self.qkv(self.norm1(x)).chunk(3, -1))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :],
                                              dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.out(attn.transpose(1, 2).flatten(-2)))
        return x + self.drop(self.ff(self.norm2(x)))


class DecisionHead(nn.Module):
    """Reads [Q] / option hidden states and returns logits over options.

    Every type returns ``[batch, K]`` logits, so loss, calibration and serving
    share one code path. ``bool`` questions either carry two option spans
    (``bool_as_options``) and are scored like a choice, or use
    ``z = MLP([h_q, h_cls])`` with logits ``[0, z]`` (softmax == sigmoid(z)).
    """

    def __init__(self, hidden_size: int, cfg: HeadConfig, bool_as_options: bool = False):
        super().__init__()
        self.cfg = cfg
        self.bool_as_options = bool_as_options
        d, p = hidden_size, cfg.dropout
        self.dropout = nn.Dropout(p)
        self.type_emb = nn.Embedding(len(TYPE_IDS), d) if cfg.type_embedding else None
        self.layers = nn.ModuleList([HeadLayer(d, p) for _ in range(cfg.head_layers)])
        if cfg.scorer == "shared":
            self.shared = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))
        else:
            self.choice_mlp = _mlp(3 * d, d, p)
            self.score_mlp = _mlp(3 * d, d, p)
        if not bool_as_options:
            self.bool_mlp = _mlp(2 * d, d, p)

    def forward(
        self,
        hidden: torch.Tensor,          # [B, T, H]
        attention_mask: torch.Tensor,  # [B, T]
        q_pos: torch.Tensor,           # [B]
        opt_pos: torch.Tensor,         # [B, K]  start of each option span (marker)
        opt_end: torch.Tensor,         # [B, K]  exclusive end of each option span
        opt_mask: torch.Tensor,        # [B, K] bool
        qtype: torch.Tensor,           # [B]
    ) -> torch.Tensor:
        if self.type_emb is not None:
            hidden = hidden + self.type_emb(qtype)[:, None, :]
        key_mask = attention_mask.bool()
        for layer in self.layers:
            hidden = layer(hidden, key_mask)
        hidden = self.dropout(hidden)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        h_q = hidden[rows, q_pos]                                                       # [B, H]
        if self.cfg.pooling == "span":
            # mean over each option span via prefix sums (ONNX friendly, no loops)
            csum = torch.cat([torch.zeros_like(hidden[:, :1]), hidden.cumsum(1)], dim=1)  # [B, T+1, H]
            span_sum = csum[rows[:, None], opt_end] - csum[rows[:, None], opt_pos]
            h_opt = span_sum / (opt_end - opt_pos).clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        else:
            h_opt = hidden[rows[:, None], opt_pos]                                      # [B, K, H]

        qt = qtype[:, None]
        if self.cfg.scorer == "shared":
            logits = self.shared(h_opt).squeeze(-1)
        else:
            h_qk = h_q[:, None, :].expand_as(h_opt)
            feats = torch.cat([h_opt, h_qk, h_opt * h_qk], dim=-1)
            choice = self.choice_mlp(feats).squeeze(-1)
            score = self.score_mlp(feats).squeeze(-1)
            logits = torch.where(qt == SCORE, score, choice)  # bool-as-options uses the choice scorer
        if not self.bool_as_options:
            z = self.bool_mlp(torch.cat([h_q, hidden[:, 0]], dim=-1))                   # [B, 1]
            k = opt_pos.size(1)
            bool_logits = torch.cat([torch.zeros_like(z), z, torch.zeros_like(z).expand(-1, k - 2)], dim=-1)
            logits = torch.where(qt == BOOL, bool_logits, logits)
        return logits.float().masked_fill(~opt_mask, NEG_INF)


def soft_ce(logits: torch.Tensor, target: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
    """Per-row soft-target cross-entropy."""
    return -(target * F.log_softmax(logits.float(), -1) * opt_mask.float()).sum(-1)


def decision_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    opt_mask: torch.Tensor,
    qtype: torch.Tensor,
    has_target: torch.Tensor,
    rps_weight: float = 0.5,
    brier_weight: float = 0.0,
    type_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Strictly proper scoring rules on soft targets.

    * log loss (soft cross-entropy) for every type
    * + ranked probability score for ``score`` questions (penalises far-off levels more)
    * + optional Brier score
    """
    p = torch.softmax(logits.float(), -1)
    mask = opt_mask.float()
    loss = soft_ce(logits, target, opt_mask)
    if brier_weight:
        loss = loss + brier_weight * (((p - target) ** 2) * mask).sum(-1)
    if rps_weight:
        cdf_gap = (p.cumsum(-1) - target.cumsum(-1)) ** 2 * mask
        levels = (mask.sum(-1) - 1).clamp_min(1)
        loss = loss + rps_weight * cdf_gap.sum(-1) / levels * (qtype == SCORE).float()
    w = torch.tensor(type_weights, device=logits.device, dtype=torch.float)[qtype] * has_target.float()
    return (loss * w).sum() / w.sum().clamp_min(1e-8)


def proper_reward(q: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor, mask: torch.Tensor,
                  w_sph: float = 0.75, w_rps: float = 1.0, log_floor: float = -9.21) -> torch.Tensor:
    """Log score + w_sph * spherical score - w_rps * RPS (score questions). q: [S, B, K], target: [B, K]."""
    q = q * mask
    log_score = (target * torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    k = mask.sum(-1).clamp_min(2)
    rps = (((q.cumsum(-1) - target.cumsum(-1)) ** 2) * mask).sum(-1) / (k - 1)
    return r - w_rps * rps * (qtype == SCORE).float()


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    opt_mask: torch.Tensor,
    qtype: torch.Tensor,
    has_target: torch.Tensor,
    sigma: float,
    samples: int = 4,
    w_sph: float = 0.75,
    w_rps: float = 1.0,
    type_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Laya-style RLCD objective: REINFORCE on Gaussian-perturbed logits with a proper-scoring reward, + soft CE.

    Zero-mean noise (projected onto each row's options) is added to the detached logits; each
    sample is rewarded by ``proper_reward``; group-normalised advantages weight the Gaussian
    log-density of the sample.
    """
    logits = logits.float()
    fmask = opt_mask.float()
    k = fmask.sum(-1, keepdim=True)
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    with torch.no_grad():
        q = torch.softmax(z.masked_fill(~opt_mask, NEG_INF), -1)
        r = proper_reward(q, target, qtype, fmask, w_sph, w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)      # [S, B]
    w = torch.tensor(type_weights, device=logits.device)[qtype] * has_target.float()   # [B]
    pg = -(adv * logp * w).sum() / (samples * w.sum().clamp_min(1e-8))
    ce = (soft_ce(logits, target, opt_mask) * w).sum() / w.sum().clamp_min(1e-8)
    return pg + ce


class DecisionModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        serializer_config: SerializerConfig | None = None,
        head_config: HeadConfig | dict[str, Any] | None = None,
        backbone_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.serializer_config = serializer_config or SerializerConfig()
        if not isinstance(head_config, HeadConfig):
            head_config = HeadConfig.from_dict(head_config)
        self.head_config = head_config
        if self.serializer_config.parallel_options and not getattr(backbone, "supports_parallel", False):
            raise ValueError(f"backbone {type(backbone).__name__} does not support parallel_options")
        self.head = DecisionHead(backbone.hidden_size, head_config, self.serializer_config.bool_as_options)
        self.backbone_config = dict(backbone_config or {})
        # temperature per question type (fallback) and per (type, option-count bucket)
        self.register_buffer("temperature", torch.ones(len(TYPE_IDS)))
        self.register_buffer("temperature_k", torch.ones(len(TYPE_IDS), len(K_BUCKETS)))

    @property
    def tokenizer(self) -> Any:
        return self.backbone.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        q_pos: torch.Tensor,
        opt_pos: torch.Tensor,
        opt_end: torch.Tensor,
        opt_mask: torch.Tensor,
        qtype: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        option_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if self.serializer_config.parallel_options:
            hidden = self.backbone(input_ids, attention_mask, position_ids=position_ids, option_ids=option_ids)
        else:
            hidden = self.backbone(input_ids, attention_mask)
        return self.head(hidden, attention_mask, q_pos, opt_pos, opt_end, opt_mask, qtype)

    def calibrate_logits(self, logits: torch.Tensor, qtype: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        bucket = k_bucket_index(opt_mask.sum(-1))
        t = self.temperature_k.to(logits.device)[qtype, bucket].clamp_min(1e-3)[:, None]
        return logits / t

    def set_temperatures(self, by_type: dict[str, float], by_options: dict[str, float] | None = None) -> None:
        for name, tid in TYPE_IDS.items():
            if name in by_type:
                self.temperature[tid] = float(by_type[name])
            self.temperature_k[tid] = self.temperature[tid]
        for key, value in (by_options or {}).items():
            name, bucket = key.split(":")
            self.temperature_k[TYPE_IDS[name], K_BUCKETS.index(bucket)] = float(value)

    def temperatures_by_options(self) -> dict[str, float]:
        """Buckets whose temperature differs from the type fallback."""
        out = {}
        for name, tid in TYPE_IDS.items():
            for bi, b in enumerate(K_BUCKETS):
                t = float(self.temperature_k[tid, bi])
                if not math.isclose(t, float(self.temperature[tid]), rel_tol=1e-6):
                    out[f"{name}:{b}"] = t
        return out

    # ------------------------------------------------------------------ io
    def save_pretrained(self, directory: str | Path, extra: dict[str, Any] | None = None) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        bb_kwargs = self.backbone.save(directory / "backbone")
        head_state = {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}
        save_file(head_state, str(directory / "head.safetensors"))
        meta = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "backbone": {
                **bb_kwargs,
                "source": self.backbone_config.get("model"),
                "plugins": list(self.backbone_config.get("plugins") or []),
            },
            "serializer": self.serializer_config.to_dict(),
            "head": {**self.head_config.to_dict(), "hidden_size": self.backbone.hidden_size},
            "temperature": {name: float(self.temperature[i]) for name, i in TYPE_IDS.items()},
            "temperature_by_options": self.temperatures_by_options(),
            **(extra or {}),
        }
        (directory / "opensysone.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str | torch.device = "cpu") -> "DecisionModel":
        from safetensors.torch import load_file

        directory = Path(directory)
        meta = json.loads((directory / "opensysone.json").read_text())
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{directory} is not an OpenSysOne model directory")
        bb_cfg = {k: v for k, v in meta["backbone"].items() if k != "source"}
        bb_cfg["model"] = str(directory / "backbone")
        backbone = build_backbone(bb_cfg)
        model = cls(
            backbone,
            SerializerConfig.from_dict(meta["serializer"]),
            HeadConfig.from_dict(meta.get("head")),
            backbone_config={"model": meta["backbone"].get("source"), "plugins": meta["backbone"].get("plugins")},
        )
        model.head.load_state_dict(load_file(str(directory / "head.safetensors")))
        model.set_temperatures(meta.get("temperature", {}), meta.get("temperature_by_options"))
        return model.to(device).eval()
