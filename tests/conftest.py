from pathlib import Path
import sys
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session):
    """Runs once before all tests. Bootstraps the test data."""
    # Ensure the code can be imported even if running from inside /tests
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    csv_path = PROJECT_ROOT / "data" / "raw" / "bcw.csv"
    sample_path = PROJECT_ROOT / "tests" / "sample_data.pt"

    logger.info(f"Checking for raw data at: {csv_path}")

    if not csv_path.exists():
        logger.error(f"CRITICAL: Raw data NOT found at {csv_path}. Check .gitignore!")
        return

    if not sample_path.exists():
        logger.info(f"Generating missing test data at {sample_path}")
        try:
            from scripts.sample import create_sample_data
            # Force the script to use the correct paths if it allows arguments
            create_sample_data()
        except Exception as e:
            logger.error(f"Failed to create sample data: {e}")
