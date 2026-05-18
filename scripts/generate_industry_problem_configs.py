from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hydra problem YAML files from a processed industry table."
    )
    parser.add_argument("--csv", required=True, help="Processed CSV path relative to repo root or absolute path.")
    parser.add_argument("--output-dir", required=True, help="Directory where problem YAML files will be written.")
    parser.add_argument("--name", default="industry_eu", help="Problem name stored in each YAML.")
    parser.add_argument("--frequency", default="M", help="Problem frequency stored in each YAML.")
    parser.add_argument("--impute", default="none", help="Problem imputation mode.")
    parser.add_argument(
        "--dropna-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Value for dropna_selected in generated YAML files.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _yaml_path(output_dir: Path, name: str, target: str) -> Path:
    return output_dir / f"{name}_{target.lower()}.yaml"


def main() -> None:
    args = parse_args()
    csv_path = _resolve_path(args.csv)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=1)
    countries = [column for column in df.columns if column != "date"]

    csv_ref = args.csv if not Path(args.csv).is_absolute() else str(csv_path)

    for target in countries:
        features = [country for country in countries if country != target]
        data = {
            "name": args.name,
            "data_path": csv_ref,
            "frequency": args.frequency,
            "impute": args.impute,
            "dropna_selected": args.dropna_selected,
            "target": target,
            "features": features,
        }
        yaml_path = _yaml_path(output_dir, args.name, target)
        yaml_path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(yaml_path)


if __name__ == "__main__":
    main()
