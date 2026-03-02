#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN_DEFAULT="${ROOT_DIR}/.venv/bin/python"
RESULTS_DIR_DEFAULT="${ROOT_DIR}/reports/results"
FIG_DIR_DEFAULT="${ROOT_DIR}/reports/figures"

resolve_python() {
  local py_bin="${PY_BIN:-$PY_BIN_DEFAULT}"
  if [[ ! -x "${py_bin}" ]]; then
    py_bin="$(command -v python3)"
  fi
  echo "${py_bin}"
}

run_plot() {
  local py_bin="$1"
  local results_dir="${RESULTS_DIR:-$RESULTS_DIR_DEFAULT}"
  local fig_dir="${FIG_DIR:-$FIG_DIR_DEFAULT}"
  local n_select_features="${N_SELECT_FEATURES:-5}"
  mkdir -p "${fig_dir}"

  PYTHONPATH="${ROOT_DIR}/src" \
  RESULTS_DIR="${results_dir}" \
  FIG_DIR="${fig_dir}" \
  N_SELECT_FEATURES="${n_select_features}" \
  "${py_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import pickle as pkl
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

results_dir = Path(os.environ["RESULTS_DIR"])
fig_dir = Path(os.environ["FIG_DIR"])
n_select_features = int(os.environ["N_SELECT_FEATURES"])


def get_results(path: Path, trgt_names: np.ndarray | None = None):
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            res_dict = pkl.load(f)
    else:
        res_dict = json.loads(path.read_text(encoding="utf-8"))

    if trgt_names is None:
        trgt_names = np.array([k for k in res_dict.keys() if k != "whtr(waist-height_ratio)"])
    pred_vals = np.array([(1 - res_dict[k][2][-1]) * 100 for k in trgt_names])
    return trgt_names, pred_vals, res_dict


def load_pattern(pattern: str):
    files = [f for f in os.listdir(results_dir) if re.match(pattern, f)]
    if len(files) == 0:
        return None
    trgt_names, _, _ = get_results(results_dir / files[0], trgt_names=None)
    mats = []
    for file in sorted(files):
        _, pred_vals, _ = get_results(results_dir / file, trgt_names=trgt_names)
        mats.append(pred_vals[:, None])
    return np.concatenate(mats, axis=1)


no_lip_perm = load_pattern(rf"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_True_\d+\.pkl")
no_lip_no_perm = load_pattern(rf"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_False_\d+\.pkl")
lip_perm = load_pattern(rf"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_True_\d+\.pkl")
lip_no_perm = load_pattern(rf"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_False_\d+\.pkl")

for name, arr in {
    "no_lip_perm": no_lip_perm,
    "no_lip_no_perm": no_lip_no_perm,
    "lip_perm": lip_perm,
    "lip_no_perm": lip_no_perm,
}.items():
    if arr is None:
        raise FileNotFoundError(f"Missing pattern results for {name} in {results_dir}")

combined_list = [no_lip_perm, no_lip_no_perm, lip_perm, lip_no_perm]
names_lst = ["permute no lipids", "no lipids", "permute with lipids", "with lipids"]

# target order from reference (with lipids, no perm)
ref_file = sorted(results_dir.glob(rf"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_False_*.pkl"))[0]
trgt_names, _, _ = get_results(ref_file)

sidx = np.argsort(combined_list[-1].mean(axis=1))[::-1]
trgt_names = trgt_names[sidx]
combined_list = [c[sidx] for c in combined_list]
means_lst = [c.mean(axis=1) for c in combined_list]
stds_lst = [c.std(axis=1) for c in combined_list]

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(trgt_names))
n_bars = len(means_lst)
bar_width = 0.8 / n_bars
offsets = np.linspace(-(n_bars - 1) * bar_width / 2, (n_bars - 1) * bar_width / 2, n_bars)

for idx, (label, means, errors) in enumerate(zip(names_lst, means_lst, stds_lst)):
    x_positions = x_pos + offsets[idx]
    ax.bar(
        x_positions,
        means,
        bar_width,
        alpha=0.8,
        yerr=errors,
        capsize=5,
        label=label,
        error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.7},
    )

ax.set_xticks(x_pos)
ax.set_xticklabels([str(k)[:20] for k in trgt_names], rotation=90, fontsize=12)
ax.set_ylabel("Var Reduction (%)", fontsize=12)
ax.set_title("Risk Prediction From Biomarkers (5 features max)", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
out = fig_dir / "ncd_risk_incremental_5feat.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Figure saved: {out}")
PY
}

main() {
  local py_bin
  py_bin="$(resolve_python)"
  run_plot "${py_bin}"
}

main "$@"
