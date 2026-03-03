from pathlib import Path

from hei_project.hei.compute_tools import get_pca

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "hei_project" / "config" / "config.yaml"


def test_current_config_file_exists():
    assert CONFIG_PATH.exists(), f"Expected config at {CONFIG_PATH}"


def test_data_root_matches_current_workspace_layout():
    data_dir = PROJECT_ROOT / "data"
    assert data_dir.exists(), f"Expected data directory at {data_dir}"
    assert not (PROJECT_ROOT / "datasets").exists(), "Old datasets/ layout should not be required here"


def test_compute_tools_exports_pca_function():
    assert callable(get_pca)
