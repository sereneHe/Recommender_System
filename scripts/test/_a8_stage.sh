#!/usr/bin/env bash
# Shared A8 stage policy for Task 12-16 (source this, do not execute).
#
#   A8_STAGE = smoke | pilot | confirm | locked
#
# The stage changes ONLY the independent-unit count and the arm set -- never
# n_samples, the NN structure, batch size, training budget, lr, the data
# mechanism, or the true DAG.  A smaller stage must be the SAME problem as the
# larger one, otherwise it proves nothing about the larger one.
#
# Seed partition (declared up front; stages are disjoint on purpose):
#   smoke   : graph 42        x noise 101
#   pilot   : graph 42-44     x noise 101-102        (development)
#   confirm : graph 52-61     x noise 101-102        (confirmation, new)
#   locked  : graph 1001-1010 x noise 501-502        (F0 reserved table)

A8_STAGE="${A8_STAGE:-smoke}"
case "${A8_STAGE}" in
  smoke)   _A8_GS="42";                                               _A8_NS="101";     _A8_ARMS="nn0,nn_ce_true,nn_w_true,nn_w_ce_true,xgb100" ;;
  pilot)   _A8_GS="42 43 44";                                         _A8_NS="101 102"; _A8_ARMS="nn0,nn_ce_true,xgb100" ;;
  confirm) _A8_GS="52 53 54 55 56 57 58 59 60 61";                    _A8_NS="101 102"; _A8_ARMS="nn0,nn_ce_true,xgb100" ;;
  locked)  _A8_GS="1001 1002 1003 1004 1005 1006 1007 1008 1009 1010"; _A8_NS="501 502"; _A8_ARMS="nn0,nn_ce_true,xgb100" ;;
  *) echo "A8_STAGE must be one of smoke|pilot|confirm|locked (got ${A8_STAGE})." >&2
     return 2 2>/dev/null || exit 2 ;;
esac

export A8_STAGE
export GRAPH_SEEDS="${GRAPH_SEEDS:-${_A8_GS}}"
export NOISE_SEEDS="${NOISE_SEEDS:-${_A8_NS}}"
export A8_ARMS="${A8_ARMS:-${_A8_ARMS}}"

# Task 16 (temporal) may NOT run CE/W arms before a lag-aware oracle exists:
# without it, a CE number cannot be interpreted as CE being right or wrong.
if [[ "${A8_MECHANISM:-}" == "temporal_smooth" ]]; then
  export A8_ARMS="nn0,xgb100"
fi
