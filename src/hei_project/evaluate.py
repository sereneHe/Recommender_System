from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import numpy as np
import typer

app = typer.Typer()


def _var_reduction_from_ratio(ratio: float | None) -> float:
    if ratio is None:
        return float("nan")
    return (1.0 - float(ratio)) * 100.0


def summarize_recommender_results(
    report_path: Path = Path("reports/recommender_training_results.json"),
    output_path: Path = Path("reports/recommender_eval_summary.json"),
) -> None:
    """
    Summarize CoDiet recommender results into a compact evaluation report.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError(f"Invalid or empty results payload in {report_path}")

    targets_summary: dict[str, dict[str, float | int]] = {}
    reductions: list[float] = []

    for target, target_data in results.items():
        if not isinstance(target_data, dict):
            continue
        final_test_ratio = target_data.get("final_test_ratio")
        final_train_ratio = target_data.get("final_train_ratio")
        test_reduction = _var_reduction_from_ratio(final_test_ratio if isinstance(final_test_ratio, (int, float)) else None)
        train_reduction = _var_reduction_from_ratio(
            final_train_ratio if isinstance(final_train_ratio, (int, float)) else None
        )
        selected_features = target_data.get("selected_features", [])
        n_selected = len(selected_features) if isinstance(selected_features, list) else 0
        targets_summary[target] = {
            "n_selected_features": n_selected,
            "train_var_reduction_pct": train_reduction,
            "test_var_reduction_pct": test_reduction,
        }
        if np.isfinite(test_reduction):
            reductions.append(test_reduction)

    summary = {
        "report_path": str(report_path),
        "n_targets": len(targets_summary),
        "mean_test_var_reduction_pct": float(np.mean(reductions)) if reductions else float("nan"),
        "targets": targets_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"CoDiet evaluation summary saved to: {output_path}")


@app.command()
def evaluate_results(
    report_path: Path = Path("reports/recommender_training_results.json"),
    output_path: Path = Path("reports/recommender_eval_summary.json"),
) -> None:
    summarize_recommender_results(report_path=report_path, output_path=output_path)


if __name__ == "__main__":
    app()
