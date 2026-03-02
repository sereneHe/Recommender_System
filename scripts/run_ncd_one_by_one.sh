#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS="${SEEDS:-13042}"
N_RUNS="${N_RUNS:-40}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"
TARGETS="${TARGETS:-GLU (mg/dL),HDL (mg/dL),LDL (mg/dL),TRIG (mg/dL),HbA1c (%),Systolic Blood Pressure (mm Hg),Diastolic Blood Pressure (mm Hg),CRP (mg/dL)}"

for USE in False True; do
  for PERM in False True; do
    seed_tag="${SEEDS%%,*}"
    out_file="${ROOT_DIR}/reports/results/NCD_analysis_incremental_feat_lipids_${USE}_maxfeat_${N_SELECT_FEATURES}_permute_${PERM}_${seed_tag}.pkl"
    if [[ -f "${out_file}" ]]; then
      echo "[SKIP] already exists: ${out_file}"
      continue
    fi
    echo "[START] seeds=${SEEDS} use_lipids=${USE} permute=${PERM} $(date)"
    env \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
      NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}" \
      VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}" \
      XGB_N_JOBS="${XGB_N_JOBS:-1}" \
      XGB_TREE_METHOD="${XGB_TREE_METHOD:-hist}" \
      SEEDS="${SEEDS}" \
      N_RUNS="${N_RUNS}" \
      N_SELECT_FEATURES="${N_SELECT_FEATURES}" \
      TARGETS="${TARGETS}" \
      USE_LIPIDS_VALUES="${USE}" \
      PERMUTE_VALUES="${PERM}" \
      ./scripts/NCD_analysis_food_and_conditioning.sh
    echo "[DONE] seeds=${SEEDS} use_lipids=${USE} permute=${PERM} $(date)"
  done
done

./scripts/intake_analysis_plot_results_v2.sh
