"""Training configuration (YAML -> dataclasses) with ``--set a.b=c`` overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import HeadConfig


@dataclass
class DataConfig:
    train: list[str] = field(default_factory=list)
    dev: list[str] = field(default_factory=list)
    max_length: int = 512
    option_token_budget: int = 16
    question_token_budget: int = 96
    max_options: int = 32
    shuffle_options: bool = True
    order: str = "state_first"        # or question_first
    marker: str = "opt"               # or mask (tokenizer's [MASK])
    bool_as_options: bool = False
    parallel_options: bool = False


@dataclass
class LossConfig:
    name: str = "ce"  # ce = soft CE + RPS (+ Brier); rlcd = Laya-style REINFORCE on noisy logits + soft CE
    rps_weight: float = 0.5
    brier_weight: float = 0.0
    type_weights: dict[str, float] = field(default_factory=lambda: {"choice": 1.0, "score": 1.0, "bool": 1.0})
    # rlcd only
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0


@dataclass
class TrainConfig:
    output_dir: str = "runs/opensysone"
    epochs: int = 3
    max_steps: int = 0  # >0 overrides epochs (smoke tests)
    batch_size: int = 16
    eval_batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-5
    head_lr: float = 1e-3
    llrd: float = 0.9  # layer-wise LR decay; 1.0 disables it
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    device: str = "auto"
    seed: int = 42
    log_every: int = 20
    eval_every: int = 0  # steps; 0 = once per epoch
    early_stop_patience: int = 3
    calibrate: bool = True
    calibrate_by_options: bool = False  # extra temperatures per (type, option-count bucket)
    num_workers: int = 0


@dataclass
class Config:
    backbone: dict[str, Any] = field(default_factory=lambda: {"name": "hf", "model": "answerdotai/ModernBERT-base"})
    plugins: list[str] = field(default_factory=list)
    head: HeadConfig = field(default_factory=HeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path, value = item.split("=", 1)
        node = raw
        keys = path.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _parse_scalar(value)
    return raw


def _coerce(value: Any, type_name: str, key: str) -> Any:
    # YAML reads "1e-3" (no dot) as a string; coerce scalars to the declared field type
    if isinstance(value, str) and type_name in ("float", "int"):
        try:
            return float(value) if type_name == "float" else int(value)
        except ValueError as e:
            raise ValueError(f"{key} must be a {type_name}, got {value!r}") from e
    if type_name == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    fields = {f.name: str(f.type) for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: _coerce(v, fields[k], f"{cls.__name__}.{k}") for k, v in raw.items()})  # type: ignore[operator]


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    base = Path(".")
    if path:
        base = Path(path).parent
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = apply_overrides(raw, overrides)
    data = dict(raw.get("data") or {})
    for key in ("train", "dev"):
        val = data.get(key) or []
        val = [val] if isinstance(val, str) else list(val)
        # relative data paths are resolved against the config file
        data[key] = [str(p) if Path(p).is_absolute() or Path(p).exists() else str(base / p) for p in val]
    cfg = Config(
        backbone=dict(raw.get("backbone") or Config().backbone),
        plugins=list(raw.get("plugins") or []),
        head=_build(HeadConfig, raw.get("head")),
        data=_build(DataConfig, data),
        loss=_build(LossConfig, raw.get("loss")),
        train=_build(TrainConfig, raw.get("train")),
    )
    return cfg
