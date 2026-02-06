from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from hei_project.data import load_all_data
from hei_project.train import train_recommender
from hei_project.visualize import get_results


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    return int(v)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hei_project.notebook_env",
        description=(
            "Notebook variable compatibility CLI. "
            "Variables previously set in notebooks can now be provided via flags or env vars."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser(
        "prepare",
        help="build_prep_data.ipynb style data preparation",
    )
    p_prepare.add_argument(
        "--do-prep",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEI_DO_PREP", True),
        help="Notebook DO_PREP. Env: HEI_DO_PREP (default: true).",
    )
    p_prepare.add_argument(
        "--data-dir",
        type=Path,
        default=_env_path("HEI_RAW_DATA_DIR", "data/raw"),
        help="Raw data root. Env: HEI_RAW_DATA_DIR (default: data/raw).",
    )
    p_prepare.add_argument(
        "--cache-dir",
        type=Path,
        default=_env_path("HEI_CACHE_DIR", "data/cache"),
        help="Cache dir (preferred if exists). Env: HEI_CACHE_DIR (default: data/cache).",
    )
    p_prepare.add_argument(
        "--prep-out-path",
        type=Path,
        default=_env_path("HEI_PREP_OUT", "data/processed/prep_data.pkl"),
        help="Output prep_data path. Env: HEI_PREP_OUT (default: data/processed/prep_data.pkl).",
    )

    p_train = sub.add_parser(
        "train",
        help="NCD_analysis_food_and_conditioning.ipynb style training",
    )
    p_train.add_argument(
        "--processed-dir",
        type=Path,
        default=_env_path("HEI_PROCESSED_DIR", "data/processed"),
        help="Dir with prep_data.pkl and feature json files. Env: HEI_PROCESSED_DIR.",
    )
    p_train.add_argument(
        "--model-dir",
        type=Path,
        default=_env_path("HEI_MODEL_DIR", "models/recommender"),
        help="Output model dir. Env: HEI_MODEL_DIR.",
    )
    p_train.add_argument(
        "--report-dir",
        type=Path,
        default=_env_path("HEI_REPORT_DIR", "reports"),
        help="Output report dir. Env: HEI_REPORT_DIR.",
    )
    p_train.add_argument(
        "--model-name",
        choices=["REG", "XGB"],
        default=_env_str("HEI_MODEL_NAME", "XGB"),
        help="Notebook model_name. Env: HEI_MODEL_NAME (REG|XGB, default: XGB).",
    )
    p_train.add_argument(
        "--custom-objective",
        choices=["lagrange", "mse_builtin"],
        default=_env_str("HEI_CUSTOM_OBJECTIVE", "lagrange"),
        help="Objective. Env: HEI_CUSTOM_OBJECTIVE (default: lagrange).",
    )
    p_train.add_argument(
        "--n-select-features",
        type=int,
        default=_env_int("HEI_N_SELECT_FEATURES", 5),
        help="Notebook N_SELECT_FEATURES. Env: HEI_N_SELECT_FEATURES (default: 5).",
    )
    p_train.add_argument(
        "--n-runs",
        type=int,
        default=_env_int("HEI_N_RUNS", 40),
        help="Notebook n_runs. Env: HEI_N_RUNS (default: 40).",
    )
    p_train.add_argument(
        "--targets",
        type=str,
        default=_env_str("HEI_TARGETS", "GLU (mg/dL)"),
        help="Notebook target_columns (comma-separated). Env: HEI_TARGETS.",
    )
    p_train.add_argument(
        "--seed",
        type=int,
        default=_env_int("HEI_SEED", 42),
        help="Random seed. Env: HEI_SEED (default: 42).",
    )

    p_patterns = sub.add_parser(
        "patterns",
        help="intake_analysis_plot_results_v2.ipynb style pattern summary",
    )
    p_patterns.add_argument(
        "--pkl-outs-dir",
        type=Path,
        default=_env_path("HEI_PKL_OUTS_DIR", "pkl_outs"),
        help="Notebook directory variable. Env: HEI_PKL_OUTS_DIR (default: pkl_outs).",
    )
    p_patterns.add_argument(
        "--patterns",
        type=str,
        default=_env_str("HEI_PATTERNS", r".*\.pkl$"),
        help="Regex patterns (comma-separated). Env: HEI_PATTERNS.",
    )

    return p


def _cmd_prepare(args: argparse.Namespace) -> int:
    if not args.do_prep:
        print("DO_PREP is false, skipping data preparation.")
        return 0
    bundle = load_all_data(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        prep_out_path=args.prep_out_path,
    )
    print(f"Saved prep_data: {args.prep_out_path}")
    print(f"Rows={bundle.prep_data.shape[0]}, Cols={bundle.prep_data.shape[1]}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    train_recommender(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        model_name=args.model_name,
        custom_objective=args.custom_objective,
        n_select_features=args.n_select_features,
        n_runs=args.n_runs,
        targets=args.targets,
        seed=args.seed,
    )
    print("Training complete.")
    return 0


def _cmd_patterns(args: argparse.Namespace) -> int:
    pattern_list = [p.strip() for p in args.patterns.split(",") if p.strip()]
    res = get_results(args.pkl_outs_dir, patterns=pattern_list)
    if not res:
        print("No matched result files.")
        return 0
    for pattern, arr in res.items():
        means = arr.mean(axis=1) if arr.ndim == 2 else np.asarray(arr)
        print(f"{pattern}: shape={arr.shape}, mean={float(np.nanmean(means)):.3f}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "patterns":
        return _cmd_patterns(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
