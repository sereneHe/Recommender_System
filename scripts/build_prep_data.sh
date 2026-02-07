#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/data/processed}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd

from hei_project.hei.data_helper import load_all_data

out_dir = Path(os.environ["OUT_DIR"]) if "OUT_DIR" in os.environ else Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

food_feats, non_food_feats, prep_data = load_all_data()

prep_path = out_dir / "prep_data.pkl"
food_path = out_dir / "food_feats.json"
non_food_path = out_dir / "non_food_feats.json"

prep_data.to_pickle(prep_path)
food_path.write_text(json.dumps(list(food_feats), ensure_ascii=False, indent=2), encoding="utf-8")
non_food_path.write_text(json.dumps(list(non_food_feats), ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[OK] prep_data: {prep_path} shape={prep_data.shape}")
print(f"[OK] food_feats: {food_path} n={len(food_feats)}")
print(f"[OK] non_food_feats: {non_food_path} n={len(non_food_feats)}")
PY
