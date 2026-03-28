#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}

CMD="python3 run_experiments.py --multirun --config-name=config"
BASE="/Users/xiaoyuhe/Recommender_Pavel/src/industry/experiments_conf/problem"

PROBLEMS=$(python3 - <<'PY'
from pathlib import Path

base = Path("/Users/xiaoyuhe/Recommender_Pavel/src/industry/experiments_conf/problem")
paths = sorted(base.glob("9countries/*.yaml")) + sorted(base.glob("16countries/*.yaml"))
problem_names = [str(path.relative_to(base).with_suffix("")).replace("\\", "/") for path in paths]
print(",".join(problem_names))
PY
)

echo "Running industry problems: ${PROBLEMS}"
${CMD} experiment="recommender_industry" solver="hc_predictor" problem="${PROBLEMS}"
