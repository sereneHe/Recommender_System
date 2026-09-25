#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROBLEMS="FRED_16country_monthly/industry_eu_ltu,FRED_16country_monthly/industry_eu_lux" \
  bash "${SCRIPT_DIR}/test_fred_pair.bash"
