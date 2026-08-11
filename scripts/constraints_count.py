#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "FRED" / "FRED_16country_monthly"
LEGACY_DEFAULT_CE_REPORT = (
    PROJECT_ROOT / "reports" / "FRED" / "old" / "FRED_16country_monthly_hc_predictor_ce_ori.csv"
)
INCLUDED_METHODS = {
    "hc_predictor_ce",
    "hc_predictor_ce_ceThreshold_0.05",
    "hc_predictor_ce_milpLambda_0.05",
    "hc_predictor_ce_shielded_collider",
    "hc_predictor_ce_shielded_collider_limit",
}

CE_SUMMARY_RE = re.compile(
    r"Conditional expectation constraints:.*?"
    r"independent=(?P<independent>\d+), dependent=(?P<dependent>\d+)"
)
CI_SUMMARY_RE = re.compile(r"Initial CI penalty:.*?constraints=(?P<count>\d+)")
RELATION_RE = re.compile(r"CI constraint \d+: relation=(?P<relation>independent|dependent)\b")


def method_from_report(path: Path, report_dir: Path) -> str:
    prefix = f"{report_dir.name}_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem


def resolve_run_dir(run_id: str, project_root: Path) -> Path:
    run_path = Path(run_id)
    return run_path if run_path.is_absolute() else project_root / run_path


def count_constraints_in_log(log_path: Path) -> tuple[int, int, str]:
    if not log_path.is_file():
        return 0, 0, "missing_log"

    text = log_path.read_text(encoding="utf-8", errors="replace")
    ce_match = CE_SUMMARY_RE.search(text)
    if ce_match:
        return (
            int(ce_match.group("independent")),
            int(ce_match.group("dependent")),
            "ce_summary",
        )

    ci_match = CI_SUMMARY_RE.search(text)
    if ci_match and int(ci_match.group("count")) == 0:
        return 0, 0, "ci_summary"

    independent = 0
    dependent = 0
    seen_constraint_indices: set[int] = set()
    constraint_re = re.compile(
        r"CI constraint (?P<index>\d+): relation=(?P<relation>independent|dependent)\b"
    )
    for match in constraint_re.finditer(text):
        index = int(match.group("index"))
        if index in seen_constraint_indices:
            break
        seen_constraint_indices.add(index)
        if match.group("relation") == "independent":
            independent += 1
        else:
            dependent += 1

    if seen_constraint_indices:
        return independent, dependent, "constraint_lines"
    return 0, 0, "no_constraint_log"


def build_constraint_tables(
    report_dir: Path = DEFAULT_REPORT_DIR,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    report_pattern = f"{report_dir.name}_*.csv"

    for report_path in sorted(report_dir.glob(report_pattern)):
        report = pd.read_csv(report_path)
        if not {"target", "run_id"}.issubset(report.columns):
            continue

        method = method_from_report(report_path, report_dir)
        if method not in INCLUDED_METHODS:
            continue
        for row in report.itertuples(index=False):
            run_id = str(getattr(row, "run_id"))
            run_dir = resolve_run_dir(run_id, project_root)
            independent, dependent, source = count_constraints_in_log(
                run_dir / "run_experiments.log"
            )
            detail_rows.append(
                {
                    "method": method,
                    "target": str(getattr(row, "target")),
                    "independence_constraints": independent,
                    "dependent_constraints": dependent,
                    "total_constraints": independent + dependent,
                    "run_id": run_id,
                    "count_source": source,
                }
            )

    if LEGACY_DEFAULT_CE_REPORT.is_file():
        report = pd.read_csv(LEGACY_DEFAULT_CE_REPORT)
        if "target" in report.columns:
            for row in report.itertuples(index=False):
                detail_rows.append(
                    {
                        "method": "hc_predictor_ce",
                        "target": str(getattr(row, "target")),
                        "independence_constraints": 120,
                        "dependent_constraints": 0,
                        "total_constraints": 120,
                        "run_id": "legacy_default_ce",
                        "count_source": "legacy_default_ce",
                    }
                )

    if not detail_rows:
        raise RuntimeError(f"No report CSVs with target/run_id found in {report_dir}")

    detail = pd.DataFrame(detail_rows).sort_values(["method", "target"]).reset_index(drop=True)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            independence_min=("independence_constraints", "min"),
            independence_max=("independence_constraints", "max"),
            dependence_min=("dependent_constraints", "min"),
            dependence_max=("dependent_constraints", "max"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return summary, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count independence and dependent constraints for each report method."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary CSV path (default: REPORT_DIR/constraints_count.csv).",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help="Per-target CSV path (default: REPORT_DIR/constraints_count_by_target.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    output = args.output or report_dir / "constraints_count.csv"
    detail_output = args.detail_output or report_dir / "constraints_count_by_target.csv"

    summary, detail = build_constraint_tables(report_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    detail.to_csv(detail_output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary: {output}")
    print(f"Saved detail:  {detail_output}")


if __name__ == "__main__":
    main()
