from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_ARTIFACT_NAMES = {
    "config.yaml",
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
