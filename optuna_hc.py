#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import optuna
except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
    raise SystemExit(
        "optuna is not installed in the current environment. "
        "Install it first, for example: `.venv/bin/python -m pip install optuna`"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent
RUN_EXPERIMENTS = REPO_ROOT / "run_experiments.py"
MLRUNS_DIR = REPO_ROOT / "mlruns"
REPORTS_DIR = REPO_ROOT / "reports" / "optuna_hc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HC recommender hyperparameters with Optuna."
    )
    parser.add_argument("--problem", default="industry_eu_aut")
    parser.add_argument("--solver", default="hc_predictor_old")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=None,
        help="Trials to run per optimization batch. Defaults to --trials.",
    )
    parser.add_argument(
        "--max-total-trials",
        type=int,
        default=None,
        help="Maximum total trials when adaptive continuation is enabled. Defaults to --trials.",
    )
    parser.add_argument("--study-name", default="optuna_hc_constrained_true")
    parser.add_argument(
        "--storage",
        default=f"sqlite:///{(REPORTS_DIR / 'optuna_hc.db').as_posix()}",
        help="Optuna storage URL.",
    )
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--optuna-seed", type=int, default=None)
    parser.add_argument(
        "--constrained",
        choices=["true", "false"],
        default="true",
        help="Whether to optimize constrained or unconstrained HC.",
    )
    parser.add_argument(
        "--experiment",
        default="recommender_industry_optuna_hc",
        help="MLflow experiment name used by trial runs.",
    )
    parser.add_argument(
        "--reference-summary",
        default=str(
            REPO_ROOT
            / "reports"
            / "old result"
            / "industry_summary_9countries_hc_2026-03-28_11-29-34.csv"
        ),
        help="CSV used as the old-result reference baseline.",
    )
    parser.add_argument(
        "--close-ratio",
        type=float,
        default=0.05,
        help="Treat current best as 'close enough' when within this relative ratio of the reference test_mean.",
    )
    return parser.parse_args()


def parse_sweep_dir(stdout: str) -> Path:
    match = re.search(r"sweep output dir : (multirun/\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2})", stdout)
    if not match:
        raise RuntimeError("Could not parse Hydra sweep output directory from stdout.")
    return REPO_ROOT / match.group(1)


def find_artifact_dir(work_dir: Path) -> Path:
    for work_dir_txt in MLRUNS_DIR.glob("*/*/artifacts/work_dir.txt"):
        try:
            if work_dir_txt.read_text().strip() == str(work_dir):
                return work_dir_txt.parent
        except OSError:
            continue
    raise FileNotFoundError(f"Could not find MLflow artifacts for work dir: {work_dir}")


def mean_from_cv_errors(cv_errors_path: Path) -> tuple[float, float]:
    data = yaml.safe_load(cv_errors_path.read_text()) or {}
    train_mean = data.get("train_mean")
    test_mean = data.get("test_mean")

    if train_mean is None:
        train_errs = data.get("train_errs")
        if isinstance(train_errs, list) and train_errs:
            train_mean = statistics.mean(train_errs)
    if test_mean is None:
        test_errs = data.get("test_errs")
        if isinstance(test_errs, list) and test_errs:
            test_mean = statistics.mean(test_errs)

    if train_mean is None or test_mean is None:
        raise RuntimeError(f"Could not derive train/test means from {cv_errors_path}")
    return float(train_mean), float(test_mean)


def selected_features(artifact_dir: Path) -> str:
    sf_path = artifact_dir / "selected_features.yaml"
    if not sf_path.exists():
        return ""
    data = yaml.safe_load(sf_path.read_text()) or {}
    for key in ("best_features", "selected_best_features"):
        value = data.get(key)
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if value is not None:
            return str(value)
    return ""


def reference_test_mean(args: argparse.Namespace) -> float | None:
    summary_path = Path(args.reference_summary)
    if not summary_path.exists():
        return None
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    target = args.problem.replace("industry_eu_", "").upper()
    for row in rows:
        if row.get("target", "").upper() == target:
            return float(row["test_mean"])
    return None


def is_good_enough(best_value: float, reference_value: float | None, close_ratio: float) -> bool:
    if reference_value is None:
        return False
    threshold = reference_value * (1.0 + close_ratio)
    return best_value <= threshold


def run_trial(args: argparse.Namespace, trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 3e-1, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
    depth = trial.suggest_int("depth", 1, 3)
    n_select_features = trial.suggest_int("N_SELECT_FEATURES", 3, 7)
    n_outer = trial.suggest_int("n_outer", 5, 15)
    n_inner = trial.suggest_int("n_inner", 25, 150, step=25)
    rho0 = trial.suggest_float("rho0", 1e-4, 1.0, log=True) if args.constrained == "true" else 0.0
    lambda_update_rate = trial.suggest_float("lambda_update_rate", 1e-2, 1.0, log=True)
    lambda1 = trial.suggest_float("lambda1", 1e-4, 1.0, log=True)
    lambda2 = trial.suggest_float("lambda2", 1e-4, 1.0, log=True)

    cmd = [
        sys.executable,
        str(RUN_EXPERIMENTS),
        "--multirun",
        "--config-name=config",
        f"experiment={args.experiment}",
        f"solver={args.solver}",
        f"problem={args.problem}",
        f"solver.constrained={args.constrained}",
        f"solver.learning_rate={learning_rate}",
        f"solver.weight_decay={weight_decay}",
        f"solver.hidden_dim={hidden_dim}",
        f"solver.depth={depth}",
        f"solver.N_SELECT_FEATURES={n_select_features}",
        f"solver.n_outer={n_outer}",
        f"solver.n_inner={n_inner}",
        f"solver.rho0={rho0}",
        f"solver.lambda_update_rate={lambda_update_rate}",
        f"solver.lambda1={lambda1}",
        f"solver.lambda2={lambda2}",
    ]

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        trial.set_user_attr("stdout_tail", completed.stdout[-2000:])
        trial.set_user_attr("stderr_tail", completed.stderr[-2000:])
        raise RuntimeError(
            f"Trial failed with return code {completed.returncode}.\n"
            f"STDERR:\n{completed.stderr[-2000:]}"
        )

    sweep_dir = parse_sweep_dir(completed.stdout)
    work_dir = sweep_dir / "0"
    artifact_dir = find_artifact_dir(work_dir)
    cv_errors_path = artifact_dir / "cv_errors.yaml"
    train_mean, test_mean = mean_from_cv_errors(cv_errors_path)

    trial.set_user_attr("work_dir", str(work_dir))
    trial.set_user_attr("artifact_dir", str(artifact_dir))
    trial.set_user_attr("train_mean", train_mean)
    trial.set_user_attr("test_mean", test_mean)
    trial.set_user_attr("best_features", selected_features(artifact_dir))
    trial.set_user_attr("w_est_path", str(artifact_dir / "W_est.csv"))

    return test_mean


def save_reports(study: optuna.Study, report_prefix: Path) -> None:
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "n_trials": len(study.trials),
    }
    (report_prefix.with_suffix(".json")).write_text(json.dumps(summary, indent=2))

    with (report_prefix.with_suffix(".csv")).open("w", newline="") as f:
        fieldnames = [
            "trial_number",
            "state",
            "value",
            "learning_rate",
            "weight_decay",
            "hidden_dim",
            "depth",
            "N_SELECT_FEATURES",
            "n_outer",
            "n_inner",
            "rho0",
            "lambda_update_rate",
            "lambda1",
            "lambda2",
            "train_mean",
            "test_mean",
            "best_features",
            "work_dir",
            "artifact_dir",
            "w_est_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row: dict[str, Any] = {
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "train_mean": trial.user_attrs.get("train_mean"),
                "test_mean": trial.user_attrs.get("test_mean"),
                "best_features": trial.user_attrs.get("best_features"),
                "work_dir": trial.user_attrs.get("work_dir"),
                "artifact_dir": trial.user_attrs.get("artifact_dir"),
                "w_est_path": trial.user_attrs.get("w_est_path"),
            }
            row.update(trial.params)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sampler = None
    if args.optuna_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    batch_trials = args.batch_trials or args.trials
    max_total_trials = args.max_total_trials or args.trials
    reference_value = reference_test_mean(args)

    while len(study.trials) < max_total_trials:
        remaining = max_total_trials - len(study.trials)
        study.optimize(
            lambda trial: run_trial(args, trial),
            n_trials=min(batch_trials, remaining),
            timeout=args.timeout,
        )
        if len(study.trials) == 0:
            break
        if is_good_enough(study.best_value, reference_value, args.close_ratio):
            break

    report_prefix = REPORTS_DIR / args.study_name
    save_reports(study, report_prefix)
    print(f"Best test_mean: {study.best_value}")
    print(f"Best params: {study.best_params}")
    if reference_value is not None:
        print(f"Reference test_mean: {reference_value}")
        print(f"Close-enough threshold: {reference_value * (1.0 + args.close_ratio)}")
    print(f"Saved reports to: {report_prefix.with_suffix('.json')} and {report_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
