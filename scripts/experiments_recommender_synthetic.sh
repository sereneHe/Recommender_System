#!/usr/bin/env bash

# Run the ordinary recommender experiment on both synthetic graph families
# and write a compact CSV with the same columns as tmp_summary_check/*.csv.
#
# Examples:
#   bash scripts/experiments_recommender_synthetic.sh
#   SOLVER=hc_predictor_ce SYNTHETIC_SEED=42 \
#     bash scripts/experiments_recommender_synthetic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE="${HC_WEIBULL_GAUSSIANIZE:-0}"
export HC_WEIBULL_MODE="${HC_WEIBULL_MODE:-rand}"

source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CONFIG_NAME="${CONFIG_NAME:-config}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-SYNTHETIC_RECOMMENDER_ER_SF}"
SOLVER="${SOLVER:-hc_predictor}"
PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-120}"
SYNTHETIC_RECALCULATE_DAG="${SYNTHETIC_RECALCULATE_DAG:-true}"
MULTIRUN_DIR="${MULTIRUN_DIR:-multirun}"
SUMMARY_DIR="${SUMMARY_DIR:-results/synthetic_summary}"

mkdir -p "${SUMMARY_DIR}"

echo "Synthetic ER/SF: solver=${SOLVER}, backend=${HC_CONSTRAINT_BACKEND},"
echo "  seed=${SYNTHETIC_SEED}, recalculate_dag=${SYNTHETIC_RECALCULATE_DAG}"

EXTRA_OVERRIDES=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.recalculate_dag=${SYNTHETIC_RECALCULATE_DAG}"
  "++problem.seed=${SYNTHETIC_SEED}"
  "solver.random_state=${SYNTHETIC_SEED}"
  "solver.cv_random_state=$((SYNTHETIC_SEED + 10000))"
  "solver.validation_random_state=$((SYNTHETIC_SEED + 20000))"
)

if [[ "${SYNTHETIC_RECALCULATE_DAG}" == "true" ]]; then
  EXTRA_OVERRIDES+=("solver.knowledge_graph_filename=null")
fi

"${PYTHON_BIN}" run_experiments.py \
  --multirun \
  --config-name="${CONFIG_NAME}" \
  experiment="${EXPERIMENT_NAME}" \
  solver="${SOLVER}" \
  problem="${PROBLEMS}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"

# Convert the finished Hydra runs into the same compact schema used by
# tmp_summary_check.  Like experiments_recommender_industry.sh, the Hydra
# output directory is supplied by the caller (the PBS launcher passes it in
# ``$@``); the newest finished run for each graph is selected.
"${PYTHON_BIN}" - "${MULTIRUN_DIR}" "${SUMMARY_DIR}" "${REPO_ROOT}" "${SOLVER}" "${EXPERIMENT_NAME}" <<'PY'
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

multirun_dir = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
repo_root = Path(sys.argv[3]).resolve()
solver_name = sys.argv[4]
experiment_name = sys.argv[5]
summary_dir.mkdir(parents=True, exist_ok=True)

columns = [
    "problem_file",
    "target",
    "selected features",
    "train_mean",
    "test_mean",
    "run_id",
    "start_time",
    "runtime",
]
timestamp_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")

def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else ""

def run_duration(log_text):
    stamps = [
        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        for m in timestamp_re.finditer(log_text)
    ]
    return (stamps[-1] - stamps[0]).total_seconds() if len(stamps) >= 2 else ""

rows = []
config_paths = sorted(
    multirun_dir.glob("**/config.yaml"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
seen_problem_files = set()
for config_path in config_paths:
    run_dir = config_path.parent
    log_path = run_dir / "run_experiments.log"
    errors_path = run_dir / "cv_errors.yaml"
    if not log_path.exists() or not errors_path.exists():
        continue
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Experiment Finished" not in log_text:
        continue
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    solver = cfg.get("solver", {}) or {}
    problem = cfg.get("problem", {}) or {}
    if cfg.get("experiment") != experiment_name:
        continue
    if solver.get("name") != solver_name:
        continue
    errors = yaml.safe_load(errors_path.read_text(encoding="utf-8")) or {}
    train_errs = errors.get("train_errs", []) or []
    test_errs = errors.get("test_errs", []) or []
    if not train_errs or not test_errs:
        continue
    selected_path = run_dir / "selected_features.txt"
    if selected_path.exists():
        selected = selected_path.read_text(encoding="utf-8").strip()
        selected = selected.strip("[]").replace("'", "").replace('"', "")
    else:
        selected = ", ".join(str(v) for v in problem.get("features", []))
    graph_type = str(problem.get("graph_type", "unknown")).lower()
    problem_file = f"synthetic_{graph_type}"
    if problem_file in seen_problem_files:
        continue
    start_match = timestamp_re.search(log_text)
    start_time = ""
    if start_match:
        start_time = datetime.strptime(
            start_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
        ).isoformat()
    try:
        run_id = str(run_dir.resolve().relative_to(repo_root))
    except ValueError:
        run_id = str(run_dir.resolve())
    rows.append({
        "problem_file": problem_file,
        "target": problem.get("target", ""),
        "selected features": selected,
        "train_mean": mean(train_errs),
        "test_mean": mean(test_errs),
        "run_id": run_id,
        "start_time": start_time,
        "runtime": run_duration(log_text),
    })
    seen_problem_files.add(problem_file)

rows.sort(key=lambda row: row["problem_file"])
combined = summary_dir / f"synthetic_er_sf_{solver_name}.csv"
with combined.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

for graph_type in ("er", "sf"):
    graph_rows = [row for row in rows if row["problem_file"] == f"synthetic_{graph_type}"]
    output = summary_dir / f"synthetic_{graph_type}_{solver_name}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(graph_rows)

print(f"Wrote {len(rows)} synthetic summary rows to {combined}")
if len(rows) == 0:
    raise SystemExit("No completed synthetic runs were found in the sweep output.")
PY

echo "Synthetic summary files: ${SUMMARY_DIR}/synthetic_er_sf_${SOLVER}.csv"
