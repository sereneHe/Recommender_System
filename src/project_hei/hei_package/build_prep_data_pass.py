import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

# Top-level imports for use throughout the module
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ========== YAML 配置自动加载 ==========
def _load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

_yaml_cfg = _load_yaml_config(os.path.join(os.path.dirname(__file__), '../configs/config.yaml'))
config = {
    'USE_SINGLE_MICROBIOME_ALPHA': bool(_yaml_cfg.get('prep', {}).get('use_single_microbiome_alpha', False)),
    'DO_PREP_PCA_PRINTING': bool(_yaml_cfg.get('prep', {}).get('pca_print', False)),
    'DO_PREP_USE_PCA': bool(_yaml_cfg.get('prep', {}).get('use_pca', False)),
    'DO_PREP_ADD_PURE_MS_SERUM': True,  # 可扩展为从yaml读取
    'DO_PREP_ADD_PURE_MS_URINE': True,  # 可扩展为从yaml读取
    'USE_LIPIDOMICS': bool(_yaml_cfg.get('prep', {}).get('use_lipidomics', True)),
}
# 环境变量优先
for k in config:
    env = os.environ.get(k)
    if env is not None:
        config[k] = env.lower() == 'true' if env.lower() in ['true', 'false'] else env

# ========== HYDRA 集成与用法示例 ==========
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Hydra入口：自动加载 config.yaml 并可命令行覆盖参数。
    用法示例：
        cd src/project_hei/hei_package
        python build_prep_data.py prep.use_pca=False prep.use_lipidomics=False
    """
    print("Hydra 加载配置:")
    print(OmegaConf.to_yaml(cfg))
    # 可将 cfg 传递给 build_prep_data 作为参数，或用于控制流程
    # 例如：df = build_prep_data(cfg)

# =====================
# Main callable function
# =====================
def build_prep_data(data_dict: dict = None) -> pd.DataFrame:
    """
    Merge and preprocess all HEI project data for downstream modeling.
    Supports prep_data_path if provided.
    Args:
        data_dict: dict returned by load_all_data(), optionally with 'prep_data_path'
    Returns:
        prep_data: pd.DataFrame ready for modeling
    """
    if data_dict is None:
        from hei_package.data_load import load_all_data
        data_dict = load_all_data(Path('../../data/raw'))

    def average_visits(df):
        if df is None or len(df) == 0:
            return df
        group_cols = [c for c in ['ID', 'VISIT'] if c in df.columns]
        if len(group_cols) < 2:
            print("average_visits: missing ID or VISIT column, skipping.")
            return df
        return df.groupby(group_cols).mean(numeric_only=True).reset_index()

    # --- 环境变量参数 ---
    USE_SINGLE_MICROBIOME_ALPHA = os.environ.get('USE_SINGLE_MICROBIOME_ALPHA', 'False').lower() == 'true'
    DO_PREP_PCA_PRINTING = os.environ.get('DO_PREP_PCA_PRINTING', 'False').lower() == 'true'
    DO_PREP_USE_PCA = os.environ.get('DO_PREP_USE_PCA', 'False').lower() == 'true'
    DO_PREP_ADD_PURE_MS_SERUM = os.environ.get('DO_PREP_ADD_PURE_MS_SERUM', 'True').lower() == 'true'
    DO_PREP_ADD_PURE_MS_URINE = os.environ.get('DO_PREP_ADD_PURE_MS_URINE', 'True').lower() == 'true'
    USE_LIPIDOMICS = os.environ.get('USE_LIPIDOMICS', 'True').lower() == 'true'

    # --- 支持通过环境变量指定 prep_data_path ---
    env_prep_data_path = os.environ.get('PREP_DATA_PATH')
    if env_prep_data_path:
        prep_data_path = Path(env_prep_data_path)
        print(f"[build_prep_data] 环境变量指定，直接从 {prep_data_path} 读取 prep_data")
        return pd.read_csv(prep_data_path)
    # --- 如果 data_dict 未指定，默认用 load_all_data() ---
    if data_dict is None:
        from hei_package.data_load import load_all_data
        data_dict = load_all_data(Path('../../data/raw'))
    prep_data_path = data_dict.get('prep_data_path')
    if prep_data_path is not None:
        prep_data_path = Path(prep_data_path)
        print(f"[build_prep_data] 直接从 {prep_data_path} 读取 prep_data")
        return pd.read_csv(prep_data_path)

    # --- 均值处理各类数据 ---
    body_comp = average_visits(data_dict.get('body_comp'))
    hei_data = average_visits(data_dict.get('hei_data'))
    blood_data = average_visits(data_dict.get('blood_data')) if 'blood_data' in data_dict else None
    gmwi_data = average_visits(data_dict.get('gmwi_data')) if 'gmwi_data' in data_dict else None

    prep_data = body_comp.copy() if body_comp is not None else pd.DataFrame()
    if hei_data is not None:
        prep_data = prep_data.merge(hei_data, on=["ID", "VISIT"], how="left")
    if blood_data is not None:
        prep_data = prep_data.merge(blood_data, on=["ID", "VISIT"], how="left")
    if gmwi_data is not None:
        prep_data = prep_data.merge(gmwi_data, on=["ID", "VISIT"], how="left")

    # 合并 new_dfs_prep
    new_dfs_prep = data_dict.get('new_dfs_prep', {})
    for name, df in new_dfs_prep.items():
        df_avg = average_visits(df)
        print(f"{name}: prep shape: {df_avg.shape}")
        prep_data = prep_data.merge(df_avg, on=["ID", "VISIT"], how="left")

    # 合并 aux_dfs
    aux_dfs = data_dict.get('aux_dfs', {})
    for name, df in aux_dfs.items():
        df_avg = average_visits(df)
        print(f"{name}: prep shape: {df_avg.shape}")
        if name in ["ms_serum", "ms_urine", "nmr_urine", "ms_lip", "dbs_rbc_lip"]:
            print(f"========== {name} =============== ")
        prep_data = prep_data.merge(df_avg, on=["ID", "VISIT"], how="left")

    print(f"Total prep_data shape: {prep_data.shape}")

    # Biochemical data
    biochemical_data = data_dict.get('biochemical_data')
    if biochemical_data is not None:
        prep_data = prep_data.merge(average_visits(biochemical_data), on=["ID", "VISIT"], how="left")

    # GMWI / site
    data_dir = Path(data_dict.get('data_dir', '../../data/raw'))
    gmwi_path = data_dir / 'gmwi_data.csv'
    if gmwi_path.exists():
        gmwi = pd.read_csv(gmwi_path).rename(columns={'sample_modified': 'sample_id'})
        gmwi['ID'] = gmwi['sample_id'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else np.nan)
        gmwi['VISIT'] = gmwi['sample_id'].apply(lambda x: int(re.findall(r'\d+', str(x))[1]) if len(re.findall(r'\d+', str(x))) > 1 else np.nan)
        gmwi['site_continental'] = gmwi['site']
        gmwi = gmwi[['ID', 'VISIT', 'GMWI', 'site_continental']]
        prep_data = prep_data.merge(gmwi, on=["ID", "VISIT"], how="left")

    # Microbiome Shannon
    micro_path = Path(data_dict.get('data_dir', '../../data/raw')) / 'microbiome/alpha_summary_CoDiet_total_v2.csv'
    if micro_path.exists():
        try:
            micro_df = pd.read_csv(micro_path)
            if micro_df.shape[0] > 0:
                micro_df = micro_df.rename(columns={micro_df.columns[0]: 'sample_id'})
                micro_df['ID'] = micro_df['sample_id'].apply(
                    lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else np.nan
                )
                micro_df['VISIT'] = micro_df['sample_id'].apply(
                    lambda x: int(re.findall(r'\d+', str(x))[1]) if len(re.findall(r'\d+', str(x))) > 1 else np.nan
                )
                micro_df = micro_df[['ID', 'VISIT', 'Shannon']].rename(columns={'Shannon': 'microbiome_Shannon'})
                prep_data = prep_data.merge(micro_df, on=["ID", "VISIT"], how="left")
            else:
                print("[build_prep_data] Warning: Microbiome file exists but is empty.")
        except Exception as e:
            print(f"[build_prep_data] Warning: Failed to process microbiome file: {e}")
    else:
        print("[build_prep_data] Warning: Microbiome file not found, skipping.")

    # Feature engineering: BMI
    if 'weight' in prep_data.columns and 'height' in prep_data.columns:
        prep_data['BMI'] = prep_data['weight'] / (prep_data['height'] / 100) ** 2

    # One-hot encode site_continental
    if 'site_continental' in prep_data.columns:
        prep_data = pd.get_dummies(prep_data, columns=['site_continental'], dummy_na=True)

    # TODO: Add normalization, encoding, missing value handling

    return prep_data

if __name__ == "__main__":
    main()