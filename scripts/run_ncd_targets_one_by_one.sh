#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

SEEDS="${SEEDS:-13042}"
N_RUNS="${N_RUNS:-40}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"
MODEL_NAME="${MODEL_NAME:-XGB}"
CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE:-mse_builtin}"
TARGETS="${TARGETS:-GLU (mg/dL),HDL (mg/dL),LDL (mg/dL),TRIG (mg/dL),HbA1c (%),Systolic Blood Pressure (mm Hg),Diastolic Blood Pressure (mm Hg),CRP (mg/dL)}"
FINAL_RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/reports/results}"
TMP_RESULTS_DIR="${TMP_RESULTS_DIR:-${ROOT_DIR}/reports/results/.tmp_target_runs}"

mkdir -p "${FINAL_RESULTS_DIR}" "${TMP_RESULTS_DIR}"

target_done_all_cases() {
  local target="$1"
  local seed="$2"
  local nfeat="$3"
  FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR}" TARGET="${target}" SEED="${seed}" NFEAT="${nfeat}" \
  "${PY_BIN}" - <<'PY'
import os, pickle
from pathlib import Path
target = os.environ["TARGET"]
seed = os.environ["SEED"]
nfeat = os.environ["NFEAT"]
base = Path(os.environ["FINAL_RESULTS_DIR"])
names = [
    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{nfeat}_permute_False_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{nfeat}_permute_True_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{nfeat}_permute_False_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{nfeat}_permute_True_{seed}.pkl",
]
for n in names:
    p = base / n
    if not p.exists():
        raise SystemExit(1)
    with p.open("rb") as f:
        d = pickle.load(f)
    if target not in d:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

merge_tmp_into_final() {
  local seed="$1"
  local nfeat="$2"
  FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR}" TMP_RESULTS_DIR="${TMP_RESULTS_DIR}" SEED="${seed}" NFEAT="${nfeat}" \
  "${PY_BIN}" - <<'PY'
import os, pickle
from pathlib import Path
final_dir = Path(os.environ["FINAL_RESULTS_DIR"])
tmp_dir = Path(os.environ["TMP_RESULTS_DIR"])
seed = os.environ["SEED"]
nfeat = os.environ["NFEAT"]
names = [
    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{nfeat}_permute_False_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{nfeat}_permute_True_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{nfeat}_permute_False_{seed}.pkl",
    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{nfeat}_permute_True_{seed}.pkl",
]
for n in names:
    src = tmp_dir / n
    if not src.exists():
        continue
    with src.open("rb") as f:
        src_d = pickle.load(f)
    dst = final_dir / n
    if dst.exists():
        with dst.open("rb") as f:
            dst_d = pickle.load(f)
    else:
        dst_d = {}
    dst_d.update(src_d)
    with dst.open("wb") as f:
        pickle.dump(dst_d, f)
PY
}

IFS=',' read -r -a targets_arr <<< "${TARGETS}"
IFS=',' read -r -a seeds_arr <<< "${SEEDS}"

for seed in "${seeds_arr[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  for target in "${targets_arr[@]}"; do
    target="$(echo "${target}" | xargs)"
    if target_done_all_cases "${target}" "${seed}" "${N_SELECT_FEATURES}"; then
      echo "[SKIP] seed=${seed} target=${target} already complete"
      continue
    fi
    echo "[RUN] seed=${seed} target=${target}"
    rm -f "${TMP_RESULTS_DIR}"/NCD_analysis_incremental_feat_lipids_*_maxfeat_"${N_SELECT_FEATURES}"_permute_*_"${seed}".pkl
    env \
      MODEL_NAME="${MODEL_NAME}" \
      CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE}" \
      N_SELECT_FEATURES="${N_SELECT_FEATURES}" \
      N_RUNS="${N_RUNS}" \
      TARGETS="${target}" \
      SEEDS="${seed}" \
      RESULTS_DIR="${TMP_RESULTS_DIR}" \
      ./scripts/NCD_analysis_food_and_conditioning.sh
    merge_tmp_into_final "${seed}" "${N_SELECT_FEATURES}"
    echo "[DONE] seed=${seed} target=${target}"
  done
done

echo "[OK] all targets processed"
