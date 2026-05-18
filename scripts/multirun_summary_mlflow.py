from __future__ import annotations

import argparse
from pathlib import Path

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

    runs = latest_runs_for_group(args.experiment, data_path, frequency, targets, solver_name=args.solver)
    client = MlflowClient()
    runs["problem_file"] = runs["target"].map(stems_by_target)
    runs["artifact_uri"] = runs["run_id"].map(lambda run_id: client.get_run(run_id).info.artifact_uri)
    runs["selected features"] = runs["artifact_uri"].map(selected_features_for_run)
    runs["train_mean"] = runs["train_error"]
    runs["test_mean"] = runs["test_err"]
    runs["runtime"] = runs["run_id"].map(lambda run_id: client.get_run(run_id).data.metrics.get("runtime"))
    runs["start_time"] = pd.to_datetime(runs["start_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    runs["start_time"] = runs["start_time"].str.replace(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
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
