#!/usr/bin/env bash
# Evidence-tree hypothesis H1: CE-only vs matched NN (synthetic ER).
#
# This is a NEW cohort, not a continuation of the legacy priority/pilot runs.
# It fixes the comparison to a single variable: the reference is the matched NN
# with no constraint, the candidate adds only the conditional-expectation (CE)
# penalty.  Optimizer, A4 graph estimation and hyper-parameter tuning are
# deliberately NOT mixed into this cohort.
#
# Protocol (experiment_registry.yaml -> hypotheses.H1):
#   reference : EV:H1:nn        (constrained=false, no CE, no W)
#   candidate : EV:H1:ce_only   (EV_CE_BASE; the only difference is CE)
#   seeds     : graph 42-51 x noise 101-102 = 20 graph x noise units
#   fold-W    : "same" (W/DAG is a frozen nuisance, both arms must share it)
#
# Submit:
#   EVIDENCE_BATCH_ID=et_h1_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/H1_mechanism.sh,EVIDENCE_BATCH_ID=et_h1_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="H1"
EV_SCOPE="synthetic/ER"
# Seed table frozen in the registry.  Set BEFORE sourcing _common.sh so its
# defaults never leak in.
export GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
export NOISE_SEEDS="${NOISE_SEEDS:-101 102}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "H1" "CE-only vs matched NN (batch=${EVIDENCE_BATCH_ID})"

# Refuse any seed outside the frozen H1 table: a cohort must not silently run a
# seed the protocol did not pre-register.
"${PYTHON_BIN}" - "${GRAPH_SEEDS}" "${NOISE_SEEDS}" <<'PY'
import sys
sys.path.insert(0, "scripts")
from experiment_registry import load, hypothesis, normalize_seed_table

graph = [int(x) for x in sys.argv[1].split()]
noise = [int(x) for x in sys.argv[2].split()]
table = normalize_seed_table(hypothesis(load(), "H1")["seed_table"])
bad = ([g for g in graph if g not in table["graph_seeds"]]
       + [n for n in noise if n not in table["noise_seeds"]])
if bad or graph != table["graph_seeds"] or noise != table["noise_seeds"]:
    raise SystemExit(
        "H1 blocked: GRAPH_SEEDS and NOISE_SEEDS must exactly match the "
        f"frozen registry table; got {graph} x {noise}, expected "
        f"{table['graph_seeds']} x {table['noise_seeds']}"
    )
print(f"H1 seed table ok: {len(graph)} graphs x {len(noise)} noise = {len(graph) * len(noise)} units")
PY

# --- reference: matched NN, no constraint -----------------------------------
run_arm "H1" "nn" "${EV_COMMON[@]}" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

# --- candidate: CE-only (W off); the CE penalty is the only change ----------
run_arm "H1" "ce_only" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}"

echo "=== H1 CE-only vs NN complete: batch=${EVIDENCE_BATCH_ID} ==="
