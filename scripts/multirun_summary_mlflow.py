from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write summary tables for one industry problem-group directory."
    )
    parser.add_argument("--group", required=True, help="Problem group directory name under experiments_conf/problem.")
    parser.add_argument("--reports-dir", default="reports", help="Directory where summary tables will be written.")
    parser.add_argument("--experiment", default="recommender_industry", help="MLflow experiment name.")
    parser.add_argument("--solver", default=None, help="Optional solver name filter, for example hc_predictor or mark.")
    parser.add_argument("--output-name", default=None, help="Optional output CSV basename without extension.")
    parser.add_argument(
        "--source",
        choices=("auto", "multirun", "mlflow"),
        default="auto",
        help="Read local Hydra multirun logs, MLflow runs, or try multirun then MLflow.",
    )
    parser.add_argument("--multirun-dir", default="multirun", help="Hydra multirun directory to scan.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_group_configs(group_dir: Path) -> tuple[str, str, list[tuple[str, str]]]:
    configs = []
    data_path = None
    frequency = None
    for path in sorted(group_dir.glob("industry_eu_*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        target = cfg["target"]
        configs.append((path.stem, target))
        data_path = cfg["data_path"]
        frequency = cfg["frequency"]
    if data_path is None or frequency is None:
        raise ValueError(f"No industry_eu_*.yaml files found in {group_dir}")
    return data_path, frequency, configs


_RUN_RESULT_RE = re.compile(
    r"Run result summary: "
    r"target=(?P<target>[^,]+), "
    r"selected_features=(?P<selected_features>\[.*?\]), "
    r"train_error=(?P<train_error>[^,]+), "
    r"test_error=(?P<test_error>[^,]+), "
    r"runtime=(?P<runtime>[0-9.eE+-]+)s, "
    r"wall_runtime=(?P<wall_runtime>[0-9.eE+-]+)s"
)


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_selected_features(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip().strip("[]")
    if isinstance(parsed, (list, tuple)):
        return ", ".join(str(item) for item in parsed)
    return str(parsed)


def _parse_result_summary(log_path: Path) -> dict[str, object] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(_RUN_RESULT_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "target": match.group("target"),
        "selected features": _format_selected_features(match.group("selected_features")),
        "train_error": _as_float(match.group("train_error")),
        "test_err": _as_float(match.group("test_error")),
        "runtime": _as_float(match.group("runtime")),
        "wall_runtime": _as_float(match.group("wall_runtime")),
    }


def _read_legacy_text_metric(run_dir: Path, filename: str) -> float | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return _as_float(path.read_text(encoding="utf-8").strip())


def _read_legacy_selected_features(run_dir: Path) -> str:
    selected_txt = run_dir / "selected_features.txt"
    if selected_txt.exists():
        return selected_txt.read_text(encoding="utf-8").strip().replace("[", "").replace("]", "")
    selected_yaml = run_dir / "selected_features.yaml"
    if selected_yaml.exists():
        data = yaml.safe_load(selected_yaml.read_text(encoding="utf-8")) or {}
        features = data.get("selected_best_features")
        if isinstance(features, list):
            return ", ".join(str(feature) for feature in features)
    return ""


def _run_start_time(run_dir: Path) -> str:
    try:
        date_part = run_dir.parent.parent.name
        time_part = run_dir.parent.name.replace("-", ":")
        return f"{date_part}T{time_part}"
    except IndexError:
        return ""


def _multirun_sort_key(run_dir: Path) -> tuple[str, str, int]:
    try:
        run_num = int(run_dir.name)
    except ValueError:
        run_num = -1
    return (run_dir.parent.parent.name, run_dir.parent.name, run_num)


def latest_multirun_runs_for_group(
    multirun_dir: Path,
    data_path: str,
    frequency: str,
    targets: list[str],
    solver_name: str | None = None,
) -> pd.DataFrame:
    target_set = set(targets)
    found_rows = []
    seen_targets: set[str] = set()
    run_dirs = sorted(
        (path.parent for path in multirun_dir.glob("*/*/*/config.yaml")),
        key=_multirun_sort_key,
        reverse=True,
    )

    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        log_path = run_dir / "run_experiments.log"
        if not log_path.exists():
            continue

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        problem = cfg.get("problem", {})
        solver = cfg.get("solver", {})
        target = problem.get("target")
        if problem.get("data_path") != data_path or problem.get("frequency") != frequency:
            continue
        if solver_name is not None and solver.get("name") != solver_name:
            continue
        if target not in target_set or target in seen_targets:
            continue

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Experiment Finished" not in log_text:
            continue

        summary = _parse_result_summary(log_path) or {}
        train_error = summary.get("train_error")
        test_err = summary.get("test_err")
        if train_error is None:
            train_error = _read_legacy_text_metric(run_dir, "train_error.txt")
        if test_err is None:
            test_err = _read_legacy_text_metric(run_dir, "test_err.txt")
        if train_error is None or test_err is None:
            continue

        selected_features = str(summary.get("selected features") or _read_legacy_selected_features(run_dir))
        runtime = summary.get("runtime")
        if runtime is None:
            runtime = _read_legacy_text_metric(run_dir, "runtime.txt")

        seen_targets.add(target)
        found_rows.append(
            {
                "target": target,
                "run_id": str(run_dir),
                "start_time": _run_start_time(run_dir),
                "train_error": train_error,
                "test_err": test_err,
                "selected features": selected_features,
                "runtime": runtime,
            }
        )
        if seen_targets == target_set:
            break

    result = pd.DataFrame(found_rows)
    if result.empty:
        raise ValueError(f"No matching multirun target runs found for {data_path}")
    return result


def latest_runs_for_group(
    experiment_name: str,
    data_path: str,
    frequency: str,
    targets: list[str],
    solver_name: str | None = None,
) -> pd.DataFrame:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"params.problem.data_path = '{data_path}' "
            f"and params.problem.frequency = '{frequency}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )
    if runs.empty:
        raise ValueError(f"No MLflow runs found for data_path={data_path} frequency={frequency}")

    found_rows = []
    seen_targets: set[str] = set()
    target_set = set(targets)
    for _, row in runs.iterrows():
        target = row.get("params.problem.target")
        solver = row.get("params.solver.name")
        if solver_name is not None and solver != solver_name:
            continue
        if row.get("status") != "FINISHED":
            continue
        if pd.isna(row.get("metrics.train_error")) or pd.isna(row.get("metrics.test_err")):
            continue
        if target not in target_set or target in seen_targets:
            continue
        seen_targets.add(target)
        found_rows.append(
            {
                "target": target,
                "run_id": row.get("run_id"),
                "start_time": row.get("start_time"),
                "train_error": row.get("metrics.train_error"),
                "test_err": row.get("metrics.test_err"),
                "selected_features_param": row.get("params.selected_features"),
            }
        )
        if seen_targets == target_set:
            break

    result = pd.DataFrame(found_rows)
    if result.empty:
        raise ValueError(f"No matching target runs found for {data_path}")
    return result


def selected_features_for_run(artifact_uri: str) -> str:
    if artifact_uri.startswith("file://"):
        artifact_root = Path(artifact_uri.removeprefix("file://"))
    else:
        artifact_root = Path(artifact_uri)
    selected_path = artifact_root / "selected_features.txt"
    if not selected_path.exists():
        return ""
    return selected_path.read_text(encoding="utf-8").strip().replace("[", "").replace("]", "")


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = []
    for idx, header in enumerate(headers):
        cell_width = max([len(header)] + [len(row[idx]) for row in rows])
        widths.append(cell_width)

    def fmt_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    parts = [fmt_row(headers), sep]
    parts.extend(fmt_row(row) for row in rows)
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    root = repo_root()
    group_dir = root / "experiments_conf" / "problem" / args.group
    reports_dir = root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path, frequency, configs = load_group_configs(group_dir)
    stems_by_target = {target: stem for stem, target in configs}
    targets = [target for _, target in configs]

    if args.source in {"auto", "multirun"}:
        try:
            runs = latest_multirun_runs_for_group(
                root / args.multirun_dir,
                data_path,
                frequency,
                targets,
                solver_name=args.solver,
            )
        except ValueError:
            if args.source == "multirun":
                raise
            runs = None
    else:
        runs = None

    if runs is None:
        runs = latest_runs_for_group(args.experiment, data_path, frequency, targets, solver_name=args.solver)
        client = MlflowClient()
        runs["artifact_uri"] = runs["run_id"].map(lambda run_id: client.get_run(run_id).info.artifact_uri)
        runs["selected features"] = runs["artifact_uri"].map(selected_features_for_run)
        runs["selected features"] = runs["selected features"].where(
            runs["selected features"].astype(str).str.len() > 0,
            runs["selected_features_param"].fillna(""),
        )
        runs["runtime"] = runs["run_id"].map(lambda run_id: client.get_run(run_id).data.metrics.get("runtime"))

    runs["problem_file"] = runs["target"].map(stems_by_target)
    runs["train_mean"] = runs["train_error"]
    runs["test_mean"] = runs["test_err"]
    parsed_start_time = pd.to_datetime(runs["start_time"], utc=True, errors="coerce")
    formatted_start_time = parsed_start_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    formatted_start_time = formatted_start_time.str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    runs["start_time"] = formatted_start_time.where(parsed_start_time.notna(), runs["start_time"].astype(str))
    runs = runs[
        [
            "problem_file",
            "target",
            "selected features",
            "train_mean",
            "test_mean",
            "run_id",
            "start_time",
            "runtime",
        ]
    ]
    runs = runs.sort_values("problem_file").reset_index(drop=True)

    output_name = args.output_name or args.group
    csv_path = reports_dir / f"{output_name}.csv"
    runs.to_csv(csv_path, index=False)
    print(csv_path)


if __name__ == "__main__":
    main()
