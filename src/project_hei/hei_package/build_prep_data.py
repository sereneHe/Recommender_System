# hei_package/build_prep_data_gpt.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import reduce
import pandas as pd
from hei_package.mk_utils import average_visits
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Optional, Callable

# Optional Hydra support
try:
    import hydra
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

def _pickle_load(path: Path, fallback: Optional[Callable] = None) -> pd.DataFrame:
    """Load a pickle file, optionally fallback to a generator function if not found."""
    path = '/Users/xiaoyuhe/Recommender_System/data/process/' + path
    # Ensure the path argument is converted to a Path object
    if not isinstance(path, Path):
        path = Path(path)

    if path.exists():
        with path.open("rb") as f:
            return pd.read_pickle(f)
    elif fallback is not None:
        return fallback()
    else:
        raise FileNotFoundError(f"File not found: {path}")
    
# --- Helper functions ---
def apply_pca(df: pd.DataFrame, n_dims: int, print_shape: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    numeric_df = df.select_dtypes(include="number")
    pca = PCA(n_components=min(n_dims, numeric_df.shape[1]))
    transformed = pca.fit_transform(numeric_df)
    pca_cols = [f"{col}_pca{i+1}" for i, col in enumerate(numeric_df.columns[:min(n_dims, numeric_df.shape[1])])]
    pca_df = pd.DataFrame(transformed, index=df.index, columns=pca_cols)
    if print_shape:
        print(f"PCA applied: {df.shape} -> {pca_df.shape}")
    return pca_df


def filter_non_food(df: pd.DataFrame, non_food_prefixes: list) -> pd.DataFrame:
    if df is None:
        return df
    cols_to_keep = [c for c in df.columns if not any(c.startswith(p) for p in non_food_prefixes)]
    return df[cols_to_keep]


# --- Core prep function ---
def build_prep_data(
    data: pd.DataFrame,
    join_key: str = "ID",
    join_type: str = "inner",
    use_average_expenditure: bool = True,
    use_blood_pressure: bool = True,
    select_numeric_only: dict = None,
    use_pca: bool = False,
    pca_print: bool = False,
    pca_blocks: list = None,
    non_food_prefixes: list = None,
    use_lipidomics: bool = True,
    use_single_microbiome_alpha: bool = False,
    add_pure_ms_serum: bool = True,
    add_pure_ms_urine: bool = True,
) -> pd.DataFrame:
    """Flexible HEI data prep pipeline"""

    select_numeric_only = select_numeric_only or {}
    pca_blocks = pca_blocks or []
    non_food_prefixes = non_food_prefixes or []


    # --- Legacy prep ---
    print("[prep] --- Legacy prep ---")
    prep_hei_data = average_visits(_pickle_load("hei.pkl"))
    print(f"[prep] hei_data shape: {None if prep_hei_data is None else prep_hei_data.shape}")

    prep_average_expenditure = average_visits(_pickle_load("average_expenditure.pkl")) if use_average_expenditure else None
    print(f"[prep] average_expenditure shape: {None if prep_average_expenditure is None else prep_average_expenditure.shape}")

    prep_body_comp_data = average_visits(_pickle_load("body_comp.pkl"))
    if prep_body_comp_data is not None and select_numeric_only.get("body_comp", True):
        body_comp_numeric = prep_body_comp_data.select_dtypes(include="number")
    else:
        body_comp_numeric = prep_body_comp_data
    print(f"[prep] body_comp shape: {None if body_comp_numeric is None else body_comp_numeric.shape}")

    prep_blood_data = average_visits(_pickle_load("blood_data.pkl"))
    if prep_blood_data is not None and select_numeric_only.get("blood_data", True):
        blood_data_numeric = prep_blood_data.select_dtypes(include="number")
    else:
        blood_data_numeric = prep_blood_data
    print(f"[prep] blood_data shape: {None if blood_data_numeric is None else blood_data_numeric.shape}")

    prep_gmwi_data = average_visits(_pickle_load("gmwi_data.pkl"))
    print(f"[prep] gmwi_data shape: {None if prep_gmwi_data is None else prep_gmwi_data.shape}")

    blood_pressure = average_visits(_pickle_load("blood_pressure.pkl"))
    print(f"[prep] blood_pressure shape: {None if blood_pressure is None else blood_pressure.shape}")

    # --- New dfs ---

    # --- New dfs ---
    new_dfs_prep = {}
    print(f"[prep] new_dfs keys: {list(data_dict.get('new_dfs', {}).keys())}")
    for n, df in data_dict.get("new_dfs", {}).items():
        df_avg = average_visits(_pickle_load(f"{n}.pkl"))
        new_dfs_prep[n] = df_avg
        print(f"[prep] {n}: prep shape: {df_avg.shape}")


    # --- Merge all dfs ---
    dfs_to_merge = [
        prep_hei_data,
        prep_average_expenditure,
        prep_body_comp_data,
        prep_blood_data,
        prep_gmwi_data,
        blood_pressure,
        *new_dfs_prep.values(),
    ]
    dfs_to_merge = [df for df in dfs_to_merge if df is not None]
    print(f"[prep] merging {len(dfs_to_merge)} dataframes on key '{join_key}' with join_type '{join_type}'")
    if len(dfs_to_merge) == 0:
        print("[prep] No dataframes to merge! Returning empty DataFrame.")
        return pd.DataFrame()
    prep_data = reduce(lambda l, r: pd.merge(l, r, on=join_key, how=join_type), dfs_to_merge)
    print(f"[prep] merged prep_data shape: {prep_data.shape}")


    # --- Filter non-food columns ---
    prep_data = filter_non_food(prep_data, non_food_prefixes)
    print(f"[prep] after filter_non_food shape: {prep_data.shape}")

    # --- PCA ---
    if use_pca:
        print("[prep] PCA enabled")
        for block in pca_blocks:
            name = block.get("name")
            n_dims = block.get("n_dims", 10)
            if name in data_dict:
                print(f"[prep] applying PCA to {name} with n_dims={n_dims}")
                pca_df = apply_pca(data_dict[name], n_dims, print_shape=pca_print)
                prep_data = pd.concat([prep_data, pca_df], axis=1)
                print(f"[prep] after PCA {name} shape: {prep_data.shape}")
    else:
        print("[prep] PCA not used")

    # --- Lipidomics inclusion ---
    if use_lipidomics and "lipidomics" in data_dict:
        print("[prep] including lipidomics")
        prep_data = pd.concat([prep_data, average_visits(data_dict["lipidomics"])], axis=1)

    # --- Optional pure MS additions ---
    if add_pure_ms_serum and "ms_serum" in data_dict:
        print("[prep] including ms_serum")
        prep_data = pd.concat([prep_data, data_dict["ms_serum"]], axis=1)
    if add_pure_ms_urine and "ms_urine" in data_dict:
        print("[prep] including ms_urine")
        prep_data = pd.concat([prep_data, data_dict["ms_urine"]], axis=1)

    print(f"[prep] Total prep_data shape: {prep_data.shape}")
    return prep_data
if HYDRA_AVAILABLE:
    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def build_prep_data_hydra(cfg: DictConfig) -> pd.DataFrame:
        from data_load import load_data_dict  # Replace with your actual loader
        data_dict = load_data_dict()
        prep_cfg = cfg.prep
        return build_prep_data(
            data_dict,
            join_key=prep_cfg.get("join_key", "ID"),
            join_type=prep_cfg.get("join_type", "inner"),
            use_average_expenditure=prep_cfg.get("use_average_expenditure", True),
            use_blood_pressure=prep_cfg.get("use_blood_pressure", True),
            select_numeric_only=prep_cfg.get("select_numeric_only", {}),
            use_pca=prep_cfg.get("use_pca", False),
            pca_print=prep_cfg.get("pca_print", False),
            pca_blocks=prep_cfg.get("pca_blocks", []),
            non_food_prefixes=prep_cfg.get("non_food_prefixes", []),
            use_lipidomics=prep_cfg.get("use_lipidomics", True),
            use_single_microbiome_alpha=prep_cfg.get("use_single_microbiome_alpha", False),
        )


if __name__ == "__main__":
    from hei_package import data_load, build_prep_data

    # Load all data
    data_dict = data_load.load_all_data(Path('/Users/xiaoyuhe/Recommender_System/data/process'))

    # Build prep data
    df = build_prep_data.build_prep_data(data_dict)
    print(df.head())