"""Post-hoc temperature scaling: one temperature per question type, optionally per (type, option-count bucket)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

from .metrics import collect
from .model import K_BUCKETS, TEMP_MAX, TEMP_MIN, k_bucket_name
from .schema import TYPE_IDS

log = logging.getLogger(__name__)


def fit_temperature(logits: list[list[float]], targets: list[list[float]], steps: int = 200) -> float:
    """Minimise soft-target NLL over a single scalar temperature (optimised in log space)."""
    k = max(len(x) for x in logits)
    lg = torch.full((len(logits), k), -1e4)
    tg = torch.zeros((len(logits), k))
    for i, (l, t) in enumerate(zip(logits, targets)):
        lg[i, : len(l)] = torch.tensor(l)
        tg[i, : len(t)] = torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = -(tg * torch.log_softmax(lg / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(TEMP_MIN, TEMP_MAX))


def calibrate(
    model: Any,
    loader: Iterable[dict[str, Any]],
    device: Any,
    min_items: int = 20,
    by_options: bool = False,
) -> dict[str, Any]:
    """Fit and store temperatures on a held-out loader.

    Returns ``{"by_type": {...}, "by_options": {...}}``. With ``by_options`` every
    (type, option-count bucket) with at least ``min_items`` items gets its own
    temperature; other buckets fall back to the type temperature.
    """
    c = collect(model, loader, device, use_temperature=False)
    by_type: dict[str, float] = {}
    for name, tid in TYPE_IDS.items():
        idx = [i for i, t in enumerate(c.qtypes) if t == tid]
        if len(idx) < min_items:
            by_type[name] = float(model.temperature[tid])
            if idx:
                log.warning("only %d %s items; keeping temperature %.3f", len(idx), name, by_type[name])
            continue
        by_type[name] = fit_temperature([c.logits[i] for i in idx], [c.targets[i] for i in idx])
    fitted: dict[str, float] = {}
    if by_options:
        for name, tid in TYPE_IDS.items():
            for bucket in K_BUCKETS:
                idx = [i for i, t in enumerate(c.qtypes)
                       if t == tid and k_bucket_name(len(c.logits[i])) == bucket]
                if len(idx) >= min_items:
                    fitted[f"{name}:{bucket}"] = fit_temperature([c.logits[i] for i in idx], [c.targets[i] for i in idx])
    model.set_temperatures(by_type, fitted)
    return {"by_type": by_type, "by_options": fitted}
