#!/usr/bin/env bash
# Evidence-tree axis F0: frozen selection + locked holdout.
#
# NOT IMPLEMENTED YET.  A real F0 runner must:
#   1. freeze the selected configuration and every threshold;
#   2. fix a calendar cutoff before its first execution;
#   3. fit transformations and the DAG only on data before the cutoff;
#   4. refit each frozen configuration exactly once;
#   5. report raw MSE, fold-normalised MSE, MAE, per-target results,
#      constraint counts/violations, runtime and failure rate.
#
# This script intentionally refuses to run until that runner exists, so an
# accidental submission cannot be mistaken for a final held-out result.

EV_AXIS="F0"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "F0 locked holdout is not implemented yet." >&2
echo "Create a dedicated freeze + locked-holdout runner before submitting F0." >&2
exit 2
