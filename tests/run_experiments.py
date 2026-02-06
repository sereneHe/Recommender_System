from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hei_project.data import load_all_data  # noqa: E402
from hei_project.train import train_recommender  # noqa: E402
from hei_project.visualize import plot_grouped_var_reduction  # noqa: E402


DEFAULT_CFG: dict[str, Any] = {
    "paths": {
        "data_dir": "datasets/raw",
        "cache_dir": "data/cache",
        "processed_dir": "data/processed",
        "model_dir": "models/recommender",
        "reports_dir": "reports",
        "figures_dir": "reports/figures",
    },
    "solver": {
        "model_name": "XGB",
        "custom_objective": "lagrange",
        "n_select_features": 5,
        "n_runs": 20,
        "seed": 42,
    },
    "targets": {
        "columns": ["GLU (mg/dL)"],
    },
}


def _as_path(v: str) -> Path:
    p = Path(v)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _parse_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node: dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def _parse_overrides(argv: list[str]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    for token in argv:
        if "=" not in token:
            continue
        k, raw = token.split("=", 1)
        _set_nested(cfg, k, _parse_value(raw))
    return cfg


def _ensure_cache_from_raw(data_dir: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "hei.pkl": data_dir / "HEI" / "hei.pkl",
        "blood_data.pkl": data_dir / "UpdatedDataFromSara" / "blood_data.pkl",
        "body_comp.pkl": data_dir / "body_composition" / "body_comp.pkl",
        "average_expenditure.pkl": data_dir / "energy_expenditure" / "average_expenditure.pkl",
    }
    missing: list[str] = []
    for name, src in mapping.items():
        dst = cache_dir / name
        if src.exists():
            shutil.copy2(src, dst)
        elif not dst.exists():
            missing.append(str(src))
    if missing:
        raise FileNotFoundError(f"Missing required CoDiet source files: {missing}")


def _write_feature_lists(processed_dir: Path, food_feats: list[str], non_food_feats: list[str]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "food_feats.json").write_text(json.dumps(food_feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (processed_dir / "non_food_feats.json").write_text(
        json.dumps(non_food_feats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _extract_plot_arrays(report_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = report_payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("No target results found in recommender_training_results.json")

    targets: list[str] = []
    train_vals: list[float] = []
    test_vals: list[float] = []
    for target, info in results.items():
        if not isinstance(info, dict):
            continue
        train_ratio = info.get("final_train_ratio")
        test_ratio = info.get("final_test_ratio")
        train_reduction = (1.0 - float(train_ratio)) * 100.0 if isinstance(train_ratio, (int, float)) else np.nan
        test_reduction = (1.0 - float(test_ratio)) * 100.0 if isinstance(test_ratio, (int, float)) else np.nan
        targets.append(str(target))
        train_vals.append(train_reduction)
        test_vals.append(test_reduction)

    return np.asarray(targets), np.asarray(train_vals)[:, None], np.asarray(test_vals)[:, None]


def _plot_selected_feature_counts(report_payload: dict[str, Any], save_path: Path) -> None:
    results = report_payload.get("results", {})
    if not isinstance(results, dict) or not results:
        return
    names: list[str] = []
    counts: list[int] = []
    for target, info in results.items():
        if not isinstance(info, dict):
            continue
        feats = info.get("selected_features", [])
        names.append(str(target)[:24])
        counts.append(len(feats) if isinstance(feats, list) else 0)

    if not names:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(names)), counts, color="#4B7BEC", alpha=0.9)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Selected Feature Count")
    ax.set_title("CoDiet Selected Features by Target", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else argv
    cfg = _parse_overrides(argv)

    data_dir = _as_path(str(cfg["paths"]["data_dir"]))
    cache_dir = _as_path(str(cfg["paths"]["cache_dir"]))
    processed_dir = _as_path(str(cfg["paths"]["processed_dir"]))
    model_dir = _as_path(str(cfg["paths"]["model_dir"]))
    reports_dir = _as_path(str(cfg["paths"]["reports_dir"]))
    figures_dir = _as_path(str(cfg["paths"]["figures_dir"]))

    solver = cfg.get("solver", {})
    targets_cfg = cfg.get("targets", {}).get("columns", ["GLU (mg/dL)"])
    if isinstance(targets_cfg, str):
        targets = [targets_cfg]
    else:
        targets = [str(t) for t in targets_cfg]

    print("[run_experiments] step=read_and_integrate_data")
    _ensure_cache_from_raw(data_dir, cache_dir)
    bundle = load_all_data(data_dir=data_dir, cache_dir=cache_dir, prep_out_path=processed_dir / "prep_data.pkl")
    _write_feature_lists(processed_dir, bundle.food_feats, bundle.non_food_feats)
    print(
        f"[run_experiments] prep_data_shape={bundle.prep_data.shape}, "
        f"food_feats={len(bundle.food_feats)}, non_food_feats={len(bundle.non_food_feats)}"
    )

    print("[run_experiments] step=train_model")
    train_recommender(
        processed_dir=processed_dir,
        model_dir=model_dir,
        report_dir=reports_dir,
        model_name=str(solver.get("model_name", "XGB")),
        custom_objective=str(solver.get("custom_objective", "lagrange")),
        n_select_features=int(solver.get("n_select_features", 5)),
        n_runs=int(solver.get("n_runs", 20)),
        targets=",".join(targets),
        seed=int(solver.get("seed", 42)),
    )

    report_path = reports_dir / "recommender_training_results.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    print("[run_experiments] step=export_features_and_plots")
    selected_features_by_target = {
        t: payload.get("results", {}).get(t, {}).get("selected_features", []) for t in payload.get("targets", [])
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "selected_features_by_target.json").write_text(
        json.dumps(selected_features_by_target, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trgt_names, train_arr, test_arr = _extract_plot_arrays(payload)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped_var_reduction(
        combined_list=[train_arr, test_arr],
        trgt_names=trgt_names,
        names_lst=["train", "test"],
        title="CoDiet Target Var Reduction (Train vs Test)",
        figure_name="codiet_train_test_var_reduction.png",
        save_dir=figures_dir,
        show_plot=False,
    )
    _plot_selected_feature_counts(payload, figures_dir / "codiet_selected_feature_count.png")

    print("[run_experiments] done")
    print(f"[run_experiments] report={report_path}")
    print(f"[run_experiments] features={reports_dir / 'selected_features_by_target.json'}")
    print(f"[run_experiments] figures_dir={figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
