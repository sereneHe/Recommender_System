#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROBLEMS="FRED_16country_monthly/industry_eu_est,FRED_16country_monthly/industry_eu_fin" \
  bash "${SCRIPT_DIR}/test_fred_pair.bash"
