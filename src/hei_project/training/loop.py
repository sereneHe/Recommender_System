from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    acc: float


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    logits = logits.squeeze()
    preds = (logits > 0).to(y.dtype)
    correct = (preds == y).sum().item()
    return correct / y.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> EpochMetrics:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits.squeeze(), batch_y.float())

        if not math.isfinite(loss.item()):
            raise ValueError(f"Non-finite loss detected: {loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = (logits.squeeze() > 0).to(batch_y.dtype)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total
    return EpochMetrics(loss=avg_loss, acc=avg_acc)


def fit(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    on_epoch_end: Callable[[int, EpochMetrics], None] | None = None,
) -> tuple[list[float], list[float]]:
    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, dataloader, optimizer, criterion)
        loss_hist.append(metrics.loss)
        acc_hist.append(metrics.acc)

        logger.debug(
            f"Epoch {epoch}/{epochs} | Loss: {metrics.loss:.4f} | "
            f"Accuracy: {metrics.acc * 100:.2f}%"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

    return loss_hist, acc_hist