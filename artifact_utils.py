from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_ARTIFACT_NAMES = {
    "config.yaml",
    "constraint_counts.csv",
    "constraint_metadata.csv",
    "constraint_stat_audit.csv",
    "cv_fold_metrics.csv",
    "ci_window_filter_audit.csv",
    "cv_errors.yaml",
    "cv_score_normalizers.yaml",
    "cv_validation_history.yaml",
    "validation_history.yaml",
    "w_constraint_audit.csv",
}


def get_run_output_dir() -> Path | None:
    output_dir = os.environ.get("RUN_OUTPUT_DIR")
    if not output_dir:
        return None
    return Path(output_dir)


def write_text_artifact(name: str, text: str) -> Path | None:
    if name not in _ALLOWED_ARTIFACT_NAMES:
        return None
    output_dir = get_run_output_dir()
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def write_yaml_artifact(name: str, data: Any) -> Path | None:
    if name not in _ALLOWED_ARTIFACT_NAMES:
        return None
    output_dir = get_run_output_dir()
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
