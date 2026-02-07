import torch
from pathlib import Path
from loguru import logger
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_sample_data():
    """Create a small sample dataset from the full data for testing purposes."""
    full_data_path = PROJECT_ROOT / "data" / "raw" / "bcw.csv"
    save_path = PROJECT_ROOT / "tests" / "sample_data.pt"

    if not full_data_path.exists():
        logger.error(f"! {full_data_path} not found. Run data.py first!")
        return

    # 1. Load the raw data
    df = pd.read_csv(full_data_path)

    # 2. Mimic preprocessing
    features = df.drop(columns=['id', 'diagnosis'], errors='ignore')
    # Remove trailing empty columns if any
    features = features.iloc[:, :30]

    # 3. Take a small slice
    sample_x = torch.tensor(features.iloc[:5].values, dtype=torch.float32)

    # Simple encoding for diagnosis (M=1, B=0)
    sample_y = torch.tensor(
        df['diagnosis'].iloc[:5].map({'M': 1, 'B': 0}).values,
        dtype=torch.long
    )

    # 4. Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((sample_x, sample_y), save_path)

    logger.success(f"Created {save_path} with shape: {sample_x.shape}")


if __name__ == "__main__":
    create_sample_data()
