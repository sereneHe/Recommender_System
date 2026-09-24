#!/usr/bin/env bash
# Evidence-tree batch 3: A7 NN-favourable mechanisms (synthetic).
#
# Each mechanism runs as its own cohort (et_a7_<mech>_v1) so the evidence
# builder groups arms correctly and never mixes one data family with another.
#
# Submit:
#   qsub -l walltime=24:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/batch3_nn_synthetic.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

MECHANISMS="${MECHANISMS:-smooth_additive compositional highdim_smooth periodic temporal_smooth}"

for mech in ${MECHANISMS}; do
  echo
  echo "################################################################################"
  echo "### A8 mechanism=${mech}"
  echo "################################################################################"
  EVIDENCE_BATCH_ID="et_a8_${mech}_v1" MECHANISM="${mech}" \
    bash "${SCRIPT_DIR}/A8_nn_favorable_synthetic.sh"
done

echo
echo "=== A8 batch complete (mechanisms: ${MECHANISMS}) ==="
