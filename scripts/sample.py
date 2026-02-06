from pathlib import Path

from loguru import logger
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_sample_data() -> None:
    """
    Create a small CoDiet sample tensor dataset for tests and quick checks.
    """
    prep_path = PROJECT_ROOT / "data" / "processed" / "prep_data.pkl"
    save_path = PROJECT_ROOT / "tests" / "sample_data.pt"

    if not prep_path.exists():
        logger.error(f"{prep_path} not found. Run CoDiet prepare step first.")
        return

    df = pd.read_pickle(prep_path)
    numeric_cols = [c for c in df.columns if c not in {"ID", "VISIT"}]
    if "GLU (mg/dL)" not in numeric_cols:
        logger.error("Expected target column 'GLU (mg/dL)' in prep_data.")
        return

    feat_cols = [c for c in numeric_cols if c != "GLU (mg/dL)"]
    if not feat_cols:
        logger.error("No numeric feature columns found in prep_data.")
        return

    sample_df = df[feat_cols + ["GLU (mg/dL)"]].dropna().head(8)
    sample_x = torch.tensor(sample_df[feat_cols].to_numpy(dtype=float), dtype=torch.float32)
    sample_y = torch.tensor(sample_df["GLU (mg/dL)"].to_numpy(dtype=float), dtype=torch.float32)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((sample_x, sample_y), save_path)
    logger.success(f"Created {save_path} with shape: {sample_x.shape}")


if __name__ == "__main__":
    create_sample_data()
