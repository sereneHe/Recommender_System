from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.stats import kendalltau, spearmanr


@dataclass
class HEIDataBundle:
    hei_data: pd.DataFrame
    blood_data: pd.DataFrame
    body_comp: pd.DataFrame
    average_expenditure: pd.DataFrame
    prep_data: pd.DataFrame
    food_feats: list[str]
    non_food_feats: list[str]


def average_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated visits per subject; keeps numeric columns only."""
    if "ID" not in df.columns:
        return df.copy()
    grouped = df.groupby("ID", as_index=False).mean(numeric_only=True)
    grouped = grouped.drop(columns=["VISIT"], errors="ignore")
    return grouped


def clean(v: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mark-ori compatible cleaner:
    cast to float and keep only rows where both vectors are finite.
    """
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if v_arr.shape[0] != u_arr.shape[0]:
        raise ValueError("v and u must have the same length")
    mask = ~np.isnan(v_arr) & ~np.isnan(u_arr)
    return v_arr[mask], u_arr[mask]


def corr(u: np.ndarray, v: np.ndarray, corr_type: str | None = None) -> float:
    """
    Correlation helper from mark-ori utilities.
    Supported types: 'spearmanr', 'kendalltau', 'pearson'.
    """
    u_arr, v_arr = clean(u, v)
    if u_arr.size == 0:
        raise ValueError("No valid samples after cleaning")

    if corr_type == "spearmanr":
        val = spearmanr(u_arr, v_arr).statistic
    elif corr_type == "kendalltau":
        val = kendalltau(u_arr, v_arr).statistic
    elif corr_type == "pearson":
        val = np.corrcoef(u_arr, v_arr)[0, 1]
    else:
        raise ValueError("corr_type must be one of: 'spearmanr', 'kendalltau', 'pearson'")
    return float(val)


def perm_test_pval(
    u: np.ndarray,
    v: np.ndarray,
    corr_type: str | None = None,
    n_permutes: int = 1000,
    seed: int | None = None,
) -> float:
    """Permutation-test p-value for a chosen correlation metric."""
    if n_permutes <= 0:
        raise ValueError("n_permutes must be > 0")

    u_arr, v_arr = clean(u, v)
    ref_value = corr(u_arr, v_arr, corr_type=corr_type)
    rng = np.random.default_rng(seed=seed)
    res = np.array([corr(rng.permutation(u_arr), v_arr, corr_type=corr_type) for _ in range(n_permutes)])

    if ref_value > 0:
        cnt = int((res >= ref_value).sum())
    else:
        cnt = int((res <= ref_value).sum())
    return float(cnt / n_permutes)


def split_visits_from_visit_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split columns by name pattern used in mark-ori notebooks:
    columns containing 'visit_3' vs all other non-ID columns.
    """
    if "ID" not in df.columns:
        raise KeyError("ID column is required")

    all_columns = df.columns.tolist()
    visit_3_columns = [col for col in all_columns if "visit_3" in str(col)]
    df1 = df[["ID"] + visit_3_columns].copy()

    other_columns = [col for col in all_columns if "visit_3" not in str(col) and col != "ID"]
    df2 = df[["ID"] + other_columns].copy()
    return df1, df2


def split_visits_from_column(
    df: pd.DataFrame,
    lst1: list[int] | list[str],
    lst2: list[int] | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows by VISIT labels and optionally average when multiple visits are selected.
    """
    if "VISIT" not in df.columns:
        raise KeyError("VISIT column is required")

    def _slice(visits: list[int] | list[str]) -> pd.DataFrame:
        part = df[df["VISIT"].isin(visits)].copy()
        if len(visits) > 1:
            return average_visits(part)
        return part.drop(columns=["VISIT"], errors="ignore").reset_index(drop=True)

    return _slice(lst1), _slice(lst2)


def _read_pickle_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_pickle(path)


def load_all_data(
    data_dir: Path = Path("./data/raw"),
    cache_dir: Path = Path("./data/cache"),
    prep_out_path: Path | None = Path("./data/processed/prep_data.pkl"),
) -> HEIDataBundle:
    """
    Project-HEI style data loader/merger.
    Expects prebuilt cache pickles:
      hei.pkl, blood_data.pkl, body_comp.pkl, average_expenditure.pkl
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        root = cache_dir
    else:
        root = data_dir

    hei_data = _read_pickle_if_exists(root / "hei.pkl")
    blood_data = _read_pickle_if_exists(root / "blood_data.pkl")
    body_comp = _read_pickle_if_exists(root / "body_comp.pkl")
    average_expenditure = _read_pickle_if_exists(root / "average_expenditure.pkl")

    # Normalize to per-subject rows for safer merge behavior.
    hei_avg = average_visits(hei_data)
    blood_avg = average_visits(blood_data)
    body_avg = average_visits(body_comp)
    expend_avg = average_visits(average_expenditure)

    dfs_to_merge = [hei_avg, blood_avg, body_avg, expend_avg]
    prep_data = dfs_to_merge[0]
    for d in dfs_to_merge[1:]:
        # The source tables often share non-key columns (e.g. VISIT).
        # Keep ID as merge key and drop overlapping non-ID columns from the right side.
        d = d.drop(columns=["VISIT"], errors="ignore")
        overlap = (set(prep_data.columns) & set(d.columns)) - {"ID"}
        if overlap:
            d = d.drop(columns=sorted(overlap), errors="ignore")
        prep_data = pd.merge(prep_data, d, on="ID", how="inner")

    food_feats = [c for c in prep_data.columns if c.startswith("food_")]
    non_food_feats = [c for c in prep_data.columns if c not in {"ID", "VISIT"} and c not in food_feats]

    if prep_out_path is not None:
        out = Path(prep_out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        prep_data.to_pickle(out)
        logger.info(f"Saved merged prep_data to {out}")

    return HEIDataBundle(
        hei_data=hei_data,
        blood_data=blood_data,
        body_comp=body_comp,
        average_expenditure=average_expenditure,
        prep_data=prep_data,
        food_feats=food_feats,
        non_food_feats=non_food_feats,
    )


def prepare_codiet(
    data_dir: Path = Path("./datasets/raw"),
    cache_dir: Path = Path("./data/cache"),
    prep_out_path: Path = Path("./data/processed/prep_data.pkl"),
) -> None:
    bundle = load_all_data(data_dir=data_dir, cache_dir=cache_dir, prep_out_path=prep_out_path)
    logger.success(
        f"Prepared CoDiet merged dataset: rows={bundle.prep_data.shape[0]}, cols={bundle.prep_data.shape[1]}"
    )


if __name__ == "__main__":
    import typer
    typer.run(prepare_codiet)
