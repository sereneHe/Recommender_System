from pathlib import Path
import json
import pickle as pkl
import re
import os

import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_pca(
    prep_data: pd.DataFrame,
    objective_df: pd.DataFrame,
    pca_objective_name: str,
    n_dims: int = 5,
    label_col: str | None = "GLU (mg/dL)",
    do_plot: bool = False,
    figure_name: str | None = None,
) -> pd.DataFrame:
    ids = objective_df["ID"].to_numpy()
    x_orig = objective_df.drop(columns=["ID"]).to_numpy()
    x_scaled = StandardScaler().fit_transform(x_orig)
    if label_col is not None and label_col in prep_data.columns:
        y = prep_data[label_col].to_numpy()
    else:
        y = np.ones(x_scaled.shape[0])

    pca = PCA(n_components=min(n_dims, x_scaled.shape[1], x_scaled.shape[0]), random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    if do_plot and x_pca.shape[1] >= 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sc = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
        plt.colorbar(sc)
        plt.title(f"PCA {pca_objective_name}: PC1 vs PC2")
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.title("Explained variance ratio")
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", pca_objective_name.strip()) or "pca"
        out_name = figure_name or f"pca_{safe_name}.png"
        figures_dir = Path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / out_name, dpi=150, bbox_inches="tight")
        plt.show()

    pca_df = pd.DataFrame(x_pca, columns=[f"{pca_objective_name}.pca_{i}" for i in range(x_pca.shape[1])])
    pca_df.insert(0, "ID", ids)
    return pca_df


def _extract_var_reduction_from_result_file(path: Path, target_order: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            obj = pkl.load(f)
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], dict):
        res = obj["results"]
        targets = target_order if target_order is not None else list(res.keys())
        vals: list[float] = []
        for t in targets:
            hist = res.get(t, {}).get("test_var_ratio_history", [])
            vals.append((1.0 - float(hist[-1])) * 100.0 if hist else np.nan)
        return targets, np.asarray(vals, dtype=float)

    if isinstance(obj, dict):
        targets = target_order if target_order is not None else list(obj.keys())
        vals: list[float] = []
        for t in targets:
            try:
                vals.append((1.0 - float(obj[t][2][-1])) * 100.0)
            except Exception:
                vals.append(np.nan)
        return targets, np.asarray(vals, dtype=float)

    raise ValueError(f"Unsupported result file format: {path}")


def _load_pattern_batch(pkl_outs_dir: Path, pattern: str, target_order: list[str]) -> np.ndarray | None:
    files = sorted([p for p in pkl_outs_dir.iterdir() if p.is_file() and re.match(pattern, p.name)])
    if not files:
        return None
    runs: list[np.ndarray] = []
    for file_path in files:
        _, vals = _extract_var_reduction_from_result_file(file_path, target_order=target_order)
        runs.append(vals[:, None])
    return np.concatenate(runs, axis=1) if runs else None


def _get_results_batch(pkl_outs_dir: Path, patterns: list[str]) -> dict[str, np.ndarray]:
    seed_files = sorted([p for p in pkl_outs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pkl"])
    if not seed_files:
        return {}
    target_order, _ = _extract_var_reduction_from_result_file(seed_files[0], target_order=None)
    out: dict[str, np.ndarray] = {}
    for pattern in patterns:
        arr = _load_pattern_batch(pkl_outs_dir, pattern, target_order)
        if arr is not None:
            out[pattern] = arr
    return out


def load_pattern(
    pkl_outs_dir_or_pattern: Path | str,
    pattern_or_directory: str | Path | None = None,
    target_order: list[str] | None = None,
) -> np.ndarray | None:
    """
    Compatible wrapper.
    New style:
      load_pattern(pkl_outs_dir=Path(...), pattern="...", target_order=[...])
    Notebook style:
      load_pattern(pattern="...", directory="./pkl_outs")
    """
    if target_order is not None:
        pkl_outs_dir = Path(pkl_outs_dir_or_pattern)
        pattern = str(pattern_or_directory)
        return _load_pattern_batch(pkl_outs_dir, pattern, target_order)

    pattern = str(pkl_outs_dir_or_pattern)
    directory = Path("." if pattern_or_directory is None else pattern_or_directory)
    if not directory.exists():
        return None
    files = sorted([f for f in os.listdir(directory) if re.match(pattern, f)])
    if not files:
        return None

    target_names, _, _ = get_results(directory / files[0], trgt_names=None)
    runs: list[np.ndarray] = []
    for file_name in files:
        _, pred_vals, _ = get_results(directory / file_name, trgt_names=target_names)
        runs.append(pred_vals[:, None])
    return np.concatenate(runs, axis=1) if runs else None


def get_results(
    pkl_outs_dir_or_fname: Path | str,
    patterns: list[str] | None = None,
    trgt_names: list[str] | np.ndarray | None = None,
) -> dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Compatible wrapper.
    New style:
      get_results(Path("pkl_outs"), [pattern1, pattern2]) -> dict[pattern, ndarray]
    Notebook style:
      get_results("one_result.pkl", trgt_names=None) -> (trgt_names, pred_vals, raw_dict)
    """
    path = Path(pkl_outs_dir_or_fname)
    if patterns is not None and path.is_dir():
        return _get_results_batch(path, patterns)

    if patterns is not None and trgt_names is None and path.is_file():
        # Backward compatibility for notebook positional call:
        # get_results(fname, trgt_names)
        trgt_names = patterns

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            res_dict = pkl.load(f)
    else:
        res_dict = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(res_dict, dict):
        raise ValueError(f"Unsupported result payload in {path}")

    if isinstance(res_dict.get("results"), dict):
        nested = res_dict["results"]
        if trgt_names is None:
            names = np.array(list(nested.keys()))
        else:
            names = np.asarray(trgt_names)
        pred_vals = np.array(
            [
                (1.0 - float(nested.get(k, {}).get("test_var_ratio_history", [np.nan])[-1])) * 100.0
                for k in names
            ],
            dtype=float,
        )
        return names, pred_vals, nested

    if trgt_names is None:
        names = np.array([k for k in res_dict.keys() if k != "whtr(waist-height_ratio)"])
    else:
        names = np.asarray(trgt_names)
    pred_vals = np.array([(1.0 - float(res_dict[k][2][-1])) * 100.0 for k in names], dtype=float)
    return names, pred_vals, res_dict


def plot_grouped_var_reduction(
    combined_list: list[np.ndarray],
    trgt_names: list[str] | np.ndarray,
    names_lst: list[str],
    *,
    title: str = "Risk Prediction From Biomarkers (5 features max)",
    figure_name: str = "ncd_risk_incremental_5feat.png",
    save_dir: Path = Path("reports/figures"),
    show_plot: bool = True,
) -> Path:
    """
    Plot grouped bar chart for variable reduction results and save to reports/figures.
    Each array in combined_list is expected to be shape [n_targets, n_runs].
    """
    if not combined_list:
        raise ValueError("combined_list is empty")
    if len(names_lst) != len(combined_list):
        raise ValueError("names_lst length must match combined_list length")

    arrays = [np.asarray(arr) for arr in combined_list]
    arrays = [arr[:, None] if arr.ndim == 1 else arr for arr in arrays]
    n_targets = arrays[0].shape[0]
    if any(arr.ndim != 2 or arr.shape[0] != n_targets for arr in arrays):
        raise ValueError("All arrays in combined_list must be 2D with the same target dimension")

    trgt_names_arr = np.asarray(trgt_names)
    if trgt_names_arr.shape[0] != n_targets:
        raise ValueError("trgt_names length must match number of targets")

    sidx = np.argsort(arrays[-1].mean(axis=1))[::-1]
    trgt_names_arr = trgt_names_arr[sidx]
    arrays = [arr[sidx] for arr in arrays]

    means_lst = [arr.mean(axis=1) for arr in arrays]
    stds_lst = [arr.std(axis=1) for arr in arrays]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(trgt_names_arr))

    n_bars = len(means_lst)
    bar_width = 0.8 / n_bars
    offsets = np.linspace(-(n_bars - 1) * bar_width / 2, (n_bars - 1) * bar_width / 2, n_bars)

    for idx, (label, means, errors) in enumerate(zip(names_lst, means_lst, stds_lst)):
        x_positions = x_pos + offsets[idx]
        ax.bar(
            x_positions,
            means,
            bar_width,
            alpha=0.8,
            yerr=errors,
            capsize=5,
            label=label,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.7},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k)[:20] for k in trgt_names_arr], rotation=90, fontsize=12)
    ax.set_ylabel("Var Reduction (%)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / figure_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    logger.success(f"Grouped var reduction figure saved to: {output_path}")
    return output_path


def plot_nolip_lip_var_reduction(
    food_pred: np.ndarray,
    food_pred_lipids: np.ndarray,
    trgt_names: list[str] | np.ndarray,
    *,
    title: str = "Risk Prediction From Biomarkers",
    figure_name: str = "ncd_risk_incremental_nolip_lip.png",
    save_dir: Path = Path("reports/figures"),
    show_plot: bool = True,
) -> Path:
    """
    Plot side-by-side bars for no-lipids vs with-lipids predictions.
    """
    food_pred_arr = np.asarray(food_pred, dtype=float).reshape(-1)
    food_pred_lipids_arr = np.asarray(food_pred_lipids, dtype=float).reshape(-1)
    trgt_names_arr = np.asarray(trgt_names)

    if food_pred_arr.shape[0] != food_pred_lipids_arr.shape[0]:
        raise ValueError("food_pred and food_pred_lipids must have the same length")
    if trgt_names_arr.shape[0] != food_pred_arr.shape[0]:
        raise ValueError("trgt_names length must match prediction length")

    sidx = np.argsort(food_pred_lipids_arr)[::-1]
    food_pred_lipids_arr = food_pred_lipids_arr[sidx]
    trgt_names_arr = trgt_names_arr[sidx]
    food_pred_arr = food_pred_arr[sidx]

    means_lst = [food_pred_arr, food_pred_lipids_arr]
    stds_lst = [np.zeros(len(food_pred_arr))] * 2
    names_lst = ["no lipids", "with lipids"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(trgt_names_arr))

    n_bars = len(means_lst)
    bar_width = 0.8 / n_bars
    offsets = np.linspace(-(n_bars - 1) * bar_width / 2, (n_bars - 1) * bar_width / 2, n_bars)

    for idx, (label, means, errors) in enumerate(zip(names_lst, means_lst, stds_lst)):
        x_positions = x_pos + offsets[idx]
        ax.bar(
            x_positions,
            means,
            bar_width,
            alpha=0.8,
            yerr=errors,
            capsize=5,
            label=label,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.7},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k)[:20] for k in trgt_names_arr], rotation=90, fontsize=12)
    ax.set_ylabel("Var Reduction (%)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / figure_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    logger.success(f"No-lipids vs with-lipids figure saved to: {output_path}")
    return output_path


def print_res_dict(res_dict: dict, header: str = "") -> None:
    # New batch format: {pattern: np.ndarray}
    if res_dict and all(isinstance(v, np.ndarray) for v in res_dict.values()):
        for name, arr in res_dict.items():
            means = arr.mean(axis=1) if arr.ndim == 2 else np.asarray(arr)
            print(f"{name}: shape={arr.shape}, mean={float(np.nanmean(means)):.3f}")
        return

    # Notebook format: {target: [features, train_hist, test_hist, ...]}
    for trgt in sorted(res_dict.keys()):
        entry = res_dict[trgt]
        try:
            feats = entry[0]
            test_reductions = 100 * (1 - np.array(entry[2]))
        except Exception:
            print(f"{header} - {trgt}: unsupported entry format")
            continue
        print(f"{header} - {('=== ' + str(trgt))[:25]:25} - Total Var Reduction")
        for feat, value in zip(feats, test_reductions):
            print(f"{str(feat)[:20]:20} - {float(value):.2f}")
        print()


def print_res_dict_compact(res_dict: dict[str, np.ndarray]) -> None:
    for name, arr in res_dict.items():
        means = arr.mean(axis=1) if arr.ndim == 2 else np.asarray(arr)
        print(f"{name}: shape={arr.shape}, mean={float(np.nanmean(means)):.3f}")


if __name__ == "__main__":
    raise SystemExit(
        "No standalone visualize CLI. Use notebook_env and plotting helpers in hei_project.visualize."
    )
