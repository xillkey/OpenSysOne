"""Training loop: LLRD AdamW, warmup schedule, mixed precision, early stopping, calibration."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .backbone import build_backbone, import_plugins
from .calibrate import calibrate
from .config import Config
from .data import DecisionDataset, collate, to_device
from .metrics import evaluate, format_report
from .model import DecisionModel, decision_loss, rlcd_loss
from .schema import QTYPES, load_many
from .serializer import Serializer, SerializerConfig

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name: str, device: torch.device) -> torch.dtype | None:
    if name == "fp32" or device.type == "cpu":
        return None
    if name == "fp16":
        return torch.float16
    if name in ("bf16", "auto"):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if name == "auto" and device.type == "cuda" else None
    raise ValueError(f"unknown precision {name!r}")


def build_optimizer(model: DecisionModel, cfg: Config) -> torch.optim.Optimizer:
    tc = cfg.train
    depths: dict[str, int | None] = {}
    for name, p in model.backbone.named_parameters():
        if p.requires_grad:
            depths[name] = model.backbone.layer_depth(name)
    n_layers = max([d for d in depths.values() if d is not None] or [0])
    groups: dict[tuple[float, bool], list[torch.nn.Parameter]] = {}
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        depth = depths[name]
        depth = n_layers + 1 if depth is None else depth
        scale = tc.llrd ** (n_layers + 1 - depth)
        decay = p.ndim > 1 and "norm" not in name.lower()
        groups.setdefault((tc.lr * scale, decay), []).append(p)
    param_groups = [
        {"params": ps, "lr": lr, "weight_decay": tc.weight_decay if decay else 0.0}
        for (lr, decay), ps in groups.items()
    ]
    head_decay = [p for n, p in model.head.named_parameters() if p.ndim > 1]
    head_no_decay = [p for n, p in model.head.named_parameters() if p.ndim <= 1]
    param_groups += [
        {"params": head_decay, "lr": tc.head_lr, "weight_decay": tc.weight_decay},
        {"params": head_no_decay, "lr": tc.head_lr, "weight_decay": 0.0},
    ]
    for g in param_groups:
        g["initial_lr"] = g["lr"]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tc = cfg.train
        set_seed(tc.seed)
        import_plugins(cfg.plugins)
        self.device = resolve_device(tc.device)
        self.amp_dtype = resolve_precision(tc.precision, self.device)

        log.info("loading backbone %s", cfg.backbone)
        bb_cfg = {**cfg.backbone, "plugins": list(cfg.plugins)}
        backbone = build_backbone(bb_cfg)
        dc = cfg.data
        self.ser_cfg = SerializerConfig(
            max_length=dc.max_length, option_token_budget=dc.option_token_budget,
            question_token_budget=dc.question_token_budget, max_options=dc.max_options,
            order=dc.order, marker=dc.marker, bool_as_options=dc.bool_as_options,
            parallel_options=dc.parallel_options,
        )
        self.serializer = Serializer(backbone.tokenizer, self.ser_cfg)
        self.model = DecisionModel(backbone, self.ser_cfg, cfg.head, bb_cfg).to(self.device)
        if cfg.loss.name not in ("ce", "rlcd"):
            raise ValueError(f"loss.name must be 'ce' or 'rlcd', got {cfg.loss.name!r}")

        if not dc.train:
            raise ValueError("data.train is empty")
        self.train_ds = DecisionDataset(load_many(dc.train), self.serializer, dc.shuffle_options, seed=tc.seed)
        self.dev_ds = DecisionDataset(load_many(dc.dev), self.serializer) if dc.dev else None
        coll = functools.partial(collate, pad_id=self.serializer.pad_id)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=tc.batch_size, shuffle=True, collate_fn=coll, num_workers=tc.num_workers
        )
        self.dev_loader = (
            DataLoader(self.dev_ds, batch_size=tc.eval_batch_size, collate_fn=coll, num_workers=tc.num_workers)
            if self.dev_ds
            else None
        )
        self.out = Path(tc.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.type_weights = tuple(float(cfg.loss.type_weights.get(t, 1.0)) for t in QTYPES)

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _dev_score(self) -> tuple[float, dict[str, Any]]:
        assert self.dev_loader is not None
        report = evaluate(self.model, self.dev_loader, self.device, use_temperature=False)
        return report["all"]["nll"], report

    def _loss(self, logits: torch.Tensor, batch: dict[str, Any], progress: float) -> torch.Tensor:
        lc = self.cfg.loss
        args = (logits, batch["target"], batch["opt_mask"], batch["qtype"], batch["has_target"])
        if lc.name == "rlcd":
            sigma = lc.sigma_start + (lc.sigma_end - lc.sigma_start) * min(1.0, progress)
            return rlcd_loss(*args, sigma=sigma, samples=lc.rl_samples, w_sph=lc.w_sph, w_rps=lc.w_rps,
                             type_weights=self.type_weights)
        return decision_loss(*args, lc.rps_weight, lc.brier_weight, self.type_weights)

    def save(self, directory: Path, extra: dict[str, Any] | None = None) -> None:
        self.model.save_pretrained(directory, extra={"train_config": self.cfg.to_dict(), **(extra or {})})

    def train(self) -> dict[str, Any]:
        cfg, tc = self.cfg, self.cfg.train
        steps_per_epoch = math.ceil(len(self.train_loader) / tc.grad_accum)
        total = tc.max_steps if tc.max_steps > 0 else steps_per_epoch * tc.epochs
        warmup = int(total * tc.warmup_ratio)
        opt = build_optimizer(self.model, cfg)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total))
        scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        eval_every = tc.eval_every or steps_per_epoch
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "train items=%d dev items=%d steps=%d trainable params=%.1fM device=%s amp=%s",
            len(self.train_ds), len(self.dev_ds) if self.dev_ds else 0, total, n_train / 1e6,
            self.device, self.amp_dtype,
        )

        best, bad_evals, step, history = float("inf"), 0, 0, []
        best_dir = self.out / "best"
        t0 = time.time()
        running = 0.0
        self.model.train()
        done = False
        while not done:
            for micro, batch in enumerate(self.train_loader):
                batch = to_device(batch, self.device)
                with self._autocast():
                    logits = self.model(**batch)
                loss = self._loss(logits, batch, step / max(1, total - 1)) / tc.grad_accum
                (scaler.scale(loss) if scaler else loss).backward()
                running += loss.item()
                if (micro + 1) % tc.grad_accum:
                    continue
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % tc.log_every == 0:
                    log.info("step %d/%d loss %.4f lr %.2e %.1fs", step, total, running / tc.log_every,
                             sched.get_last_lr()[0], time.time() - t0)
                    running = 0.0
                if self.dev_loader and (step % eval_every == 0 or step == total):
                    nll, report = self._dev_score()
                    history.append({"step": step, **report["all"]})
                    log.info("eval step %d\n%s", step, format_report(report))
                    if nll < best - 1e-4:
                        best, bad_evals = nll, 0
                        self.save(best_dir, {"best_step": step})
                    else:
                        bad_evals += 1
                        if tc.early_stop_patience and bad_evals >= tc.early_stop_patience:
                            log.info("early stopping at step %d", step)
                            done = True
                    self.model.train()
                if step >= total or done:
                    done = True
                    break

        if not self.dev_loader:
            log.warning("no dev set: saving the last step uncalibrated (temperature = 1)")
            self.save(self.out / "final", {"best_step": step})
            return {"steps": step, "history": history}

        # reload best weights, calibrate on dev, write the final model
        self.model = DecisionModel.from_pretrained(best_dir, self.device)
        result: dict[str, Any] = {"steps": step, "best_dev_nll": best, "history": history}
        if tc.calibrate:
            temps = calibrate(self.model, self.dev_loader, self.device, by_options=tc.calibrate_by_options)
            log.info("temperatures %s", temps)
            result["temperature"] = temps
        final = evaluate(self.model, self.dev_loader, self.device, use_temperature=True)
        log.info("final dev (calibrated)\n%s", format_report(final))
        result["dev"] = final
        self.save(self.out / "final", {"dev_metrics": final})
        (self.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result


def train(cfg: Config) -> dict[str, Any]:
    return Trainer(cfg).train()
