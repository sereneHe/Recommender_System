from __future__ import annotations

import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
from loguru import logger
import pandas as pd
import typer

from hei_project.model import run_feature_selection


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_full_features(prep_data: pd.DataFrame, food_feats: list[str], non_food_feats: list[str]) -> list[str]:
    full_feats: list[str] = []
    full_feats += list(food_feats)
    full_feats += [
        "age",
        "gender_numeric",
        "stress_index",
        "fatigue_index",
        "mean_hrt",
        "site_continental",
        "weight",
        "height",
        "GMWI",
        "microbiome_Shannon",
    ]
    full_feats += [c for c in non_food_feats if "dbs_rbc_lip" in c]
    full_feats += [c for c in prep_data.columns if "microb_clean15_" in c]
    full_feats = _dedup_keep_order(full_feats)
    full_feats = [c for c in full_feats if c in prep_data.columns]
    return full_feats


def train_recommender(
    processed_dir: Path = Path("data/processed"),
    model_dir: Path = Path("models/recommender"),
    report_dir: Path = Path("reports"),
    model_name: str = "XGB",
    custom_objective: str = "lagrange",
    n_select_features: int = 5,
    n_runs: int = 20,
    targets: str = "GLU (mg/dL)",
    seed: int = 42,
) -> None:
    """
    Train CoDiet nutrition recommenders and save per-target artifacts + summary.
    """
    prep_path = Path(processed_dir) / "prep_data.pkl"
    food_path = Path(processed_dir) / "food_feats.json"
    non_food_path = Path(processed_dir) / "non_food_feats.json"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing {prep_path}")
    if not food_path.exists() or not non_food_path.exists():
        raise FileNotFoundError(f"Missing feature list files in {processed_dir}")

    prep_data = pd.read_pickle(prep_path)
    food_feats = json.loads(food_path.read_text(encoding="utf-8"))
    non_food_feats = json.loads(non_food_path.read_text(encoding="utf-8"))

    full_feats = _build_full_features(prep_data, food_feats, non_food_feats)
    target_cols = [t.strip() for t in str(targets).split(",") if t.strip()]
    for t in target_cols:
        if t not in prep_data.columns:
            raise KeyError(f"Target '{t}' not found in prep_data")

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    for target_col in target_cols:
        curr_feats, curr_train_errs, curr_test_errs, best_final = run_feature_selection(
            prep_data,
            model_name=model_name,
            custom_objective=custom_objective,
            target_col=target_col,
            n_runs=int(n_runs),
            n_features=int(n_select_features),
            full_features=full_feats,
            seed=int(seed),
        )

        safe_target = (
            target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("%", "pct")
        )
        model_path = model_dir / f"{safe_target}_model.joblib"
        if best_final is not None:
            joblib.dump(best_final.model, model_path)

        results[target_col] = {
            "selected_features": curr_feats,
            "train_var_ratio_history": curr_train_errs,
            "test_var_ratio_history": curr_test_errs,
            "final_train_mse": None if best_final is None else best_final.train_mse,
            "final_test_mse": None if best_final is None else best_final.test_mse,
            "final_test_ratio": None if best_final is None else best_final.test_ratio,
            "final_train_ratio": None if best_final is None else best_final.train_ratio,
            "final_kendalltau": None if best_final is None else best_final.test_kendalltau,
            "model_path": str(model_path) if best_final is not None else None,
        }
        logger.info(
            f"Target={target_col}: selected={len(curr_feats)}, "
            f"last_test_ratio={curr_test_errs[-1] if curr_test_errs else 'N/A'}"
        )

    summary = {
        "model_name": model_name,
        "custom_objective": custom_objective,
        "n_select_features": int(n_select_features),
        "n_runs": int(n_runs),
        "targets": target_cols,
        "full_feats": full_feats,
        "results": results,
    }
    out_json = Path(report_dir) / "recommender_training_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"CoDiet training finished. Summary written to {out_json}")


if __name__ == "__main__":
    typer.run(train_recommender)
