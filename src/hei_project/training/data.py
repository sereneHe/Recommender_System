from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainData:
    x: torch.Tensor
    y: torch.Tensor


def load_train_tensors(processed_dir: Path) -> TrainData:
    data_file = processed_dir / "train.pt"
    logger.debug(f"Loading training data from: {data_file}")

    try:
        x_train, y_train = torch.load(data_file)
    except FileNotFoundError as e:
        logger.error(
            f"Training data not found at {data_file}. "
            f"Please run preprocessing first."
        )
        raise e

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Shape mismatch: x has {x_train.shape[0]} rows, y has {y_train.shape[0]}"
        )

    logger.info(
        f"\n{'=' * 10} Data Sanity Check {'=' * 10}"
        f"\n   Input Shape (x): {tuple(x_train.shape)}"
        f"\n   Target Shape (y): {tuple(y_train.shape)}"
        f"\n   First 2 Targets: {y_train[:2].tolist()}"
        f"\n{'=' * 39}\n"
    )

    return TrainData(x=x_train, y=y_train)


def make_train_dataloader(
    data: TrainData, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = TensorDataset(data.x, data.y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)