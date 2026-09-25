#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
. "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config-cluster"


${CMD} experiment="CDS_RECOMMENDER" solver="mark_with_cc, mark" problem="cds_05abbf, cds_8b69ap, cds_06dabk, cds_efagg9, cds_fh49gg, cds_8d8575, cds_gg6ebt, cds_dd359m, cds_ff667m, cds_0h99b7, cds_2h66b7, cds_8a87ag, cds_nn2a8g, cds_6a516f"
