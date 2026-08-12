import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

# Top-level imports for use throughout the module
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce

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

    def groupby_id_visit(df):
        if df is None or len(df) == 0:
            return df
        if 'VISIT' in df.columns:
            return df.groupby(['ID', 'VISIT']).mean(numeric_only=True).reset_index()
        else:
            return df.groupby(['ID']).mean(numeric_only=True).reset_index()

    def groupby_id(df):
        if df is None or len(df) == 0:
            return df
        return df.groupby('ID').mean(numeric_only=True).reset_index()

    from hei_package.data_load import load_all_data
    data_dict = load_all_data(Path('../../data/raw'))

    prep_hei_data = groupby_id_visit(data_dict.get('hei_data'))
    print("[DEBUG] hei_data columns:", prep_hei_data.columns)
    print(f'hei_data: prep shape: {None if prep_hei_data is None else prep_hei_data.shape}')

    prep_average_expenditure = groupby_id(data_dict.get('average_expenditure')) if 'average_expenditure' in data_dict else None
    if prep_average_expenditure is not None:
        print("[DEBUG] average_expenditure columns:", prep_average_expenditure.columns)
    print(f'average_expenditure: prep shape: {None if prep_average_expenditure is None else prep_average_expenditure.shape}')

    body_comp = data_dict.get('body_comp')
    prep_body_comp_data = groupby_id(body_comp.select_dtypes(include=['number'])) if body_comp is not None else None
    if prep_body_comp_data is not None:
        print("[DEBUG] body_comp columns:", prep_body_comp_data.columns)
    print(f'body_comp: prep shape: {None if prep_body_comp_data is None else prep_body_comp_data.shape}')

    blood_data = data_dict.get('blood_data')
    prep_blood_data = groupby_id(blood_data.select_dtypes(include=['number'])) if blood_data is not None else None
    if prep_blood_data is not None:
        print("[DEBUG] blood_data columns:", prep_blood_data.columns)
    print(f'blood_data: prep shape: {None if prep_blood_data is None else prep_blood_data.shape}')

    prep_gmwi_data = groupby_id(data_dict.get('gmwi_data')) if 'gmwi_data' in data_dict else None
    if prep_gmwi_data is not None:
        print("[DEBUG] gmwi_data columns:", prep_gmwi_data.columns)
    print(f'gmwi_data: prep shape: {None if prep_gmwi_data is None else prep_gmwi_data.shape}')

    blood_pressure = groupby_id(data_dict.get('blood_pressure')) if 'blood_pressure' in data_dict else None
    if blood_pressure is not None:
        print("[DEBUG] blood_pressure columns:", blood_pressure.columns)
    print(f'blood_pressure: prep shape: {None if blood_pressure is None else blood_pressure.shape}')   # 处理 new_dfs，全部只按 ID 聚合
    new_dfs = data_dict.get('new_dfs', {})
    new_dfs_prep = {}
    for n in new_dfs:
        df = groupby_id(new_dfs[n])
        new_dfs_prep[n] = df
        # lipidomics 相关表名兼容 notebook
        if n in ['ms_lip', 'lipidomics']:
            print(f'lipidomics: prep shape: {df.shape}')
        elif n in ['dbs_rbc_lip', 'lipidomics_dbs_rbc']:
            print(f'lipidomics_dbs_rbc: prep shape: {df.shape}')
        else:
            print(f'{n}: prep shape: {df.shape}')
    print(f"new_dfs keys: {list(new_dfs_prep.keys())}")

    # 合并，主表 on ['ID','VISIT']，其余 on 'ID'
    merge_list = [prep_hei_data] + [df for df in [prep_average_expenditure, prep_body_comp_data, prep_blood_data, prep_gmwi_data, blood_pressure] if df is not None] + list(new_dfs_prep.values())
    if len(merge_list) == 0 or merge_list[0] is None:
        return pd.DataFrame()
    # 主表保留 VISIT
    main_cols = ['ID', 'VISIT'] + [c for c in merge_list[0].columns if c not in ['ID', 'VISIT']]
    merge_list[0] = merge_list[0][main_cols]
    # 其余表只保留 'ID' 和数值型（去掉 VISIT）
    for i in range(1, len(merge_list)):
        cols = ['ID'] + [c for c in merge_list[i].columns if c not in ['ID', 'VISIT']]
        merge_list[i] = merge_list[i][cols]
    # 检查唯一性
    for idx, df in enumerate(merge_list):
        if df['ID'].duplicated().any():
            print(f"Warning: Table {idx} has duplicated IDs after groupby!")
    # 依次 merge，主表 on ['ID','VISIT']，其余 on 'ID'
    prep_data = merge_list[0]
    for df in merge_list[1:]:
        prep_data = pd.merge(prep_data, df, on='ID', how='inner')

    print(f'\nTotal prep_data shape: {prep_data.shape}')
    # print(prep_data.columns)

    return prep_data

if __name__ == "__main__":
    main()