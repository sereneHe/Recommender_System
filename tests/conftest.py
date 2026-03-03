from pathlib import Path
import types
import sys
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def pytest_sessionstart(session):
    """Runs once before all tests. Ensures the current src layout is importable."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    if "mlflow" not in sys.modules:
        sys.modules["mlflow"] = types.SimpleNamespace(
            log_text=lambda *args, **kwargs: None,
            set_tracking_uri=lambda *args, **kwargs: None,
            set_experiment=lambda *args, **kwargs: None,
        )

    logger.info(f"Using project root: {PROJECT_ROOT}")
    logger.info(f"Using source root: {SRC_ROOT}")
