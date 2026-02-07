from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

@dataclass
class CheckpointState:
    best_metric: float | None = None
    best_epoch: int | None = None

def save_state_dict(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def update_best(
    state: CheckpointState,
    model: nn.Module,
    metric_value: float,
    epoch: int,
    best_path: Path,
    lower_is_better: bool = True,
) -> bool:
    improved = False
    if state.best_metric is None:
        improved = True
    else:
        improved = metric_value < state.best_metric if lower_is_better else metric_value > state.best_metric

    if improved:
        state.best_metric = float(metric_value)
        state.best_epoch = int(epoch)
        save_state_dict(model, best_path)

    return improved