from __future__ import annotations
from pathlib import Path
import torch

def save_metrics(path: Path, loss: list[float], acc: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "loss": loss,
            "accuracy": acc,
            "epochs": list(range(1, len(loss) + 1)),
        },
        path,
    )