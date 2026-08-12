import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pandas as pd
import pickle
from omegaconf import DictConfig

DEFAULT_READ_DATA_DIR = Path("/Users/xiaoyuhe/Datasets/CoDiet/raw")
DEFAULT_SAVE_DATA_DIR = Path("/Users/xiaoyuhe/Datasets/CoDiet/processed")

# ------------------------
# 通用工具函数
# ------------------------
def _safe_pickle_load(path: Path, fallback: Optional[Callable] = None) -> pd.DataFrame:
    """Load a pickle file, optionally fallback to a generator function if not found."""
    if path.exists():
        df = pd.read_pickle(path)
        print(f"Loaded cached: {path} ({df.shape})")
        return df
    elif fallback is not None:
        df = fallback()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Debugging log to inspect the structure of the fallback data
        print(f"[debug] Fallback data structure: {type(df)}")
        print(f"[debug] Fallback data content: {df}")
        # Ensure the fallback returns a valid DataFrame structure
        if isinstance(df, dict):
            print(f"[warning] Fallback returned a dict. Converting to DataFrame.")
            try:
                df = pd.DataFrame({k: [v] for k, v in df.items()} if all(isinstance(v, (int, float, str)) for v in df.values()) else df, index=[0])
            except Exception as e:
                print(f"[error] Failed to convert dict to DataFrame: {e}")
                raise
        elif isinstance(df, (int, float, str)):
            print(f"[warning] Fallback returned a scalar. Wrapping in DataFrame.")
            df = pd.DataFrame({"value": [df]})
        df.to_pickle(path)
        print(f"Saved pickle: {path} ({df.shape})")
        return df
    else:
        raise FileNotFoundError(f"{path} not found and no fallback provided.")

def extract_first_int(s):
    """安全地提取字符串中的第一个整数，否则返回 pd.NA"""
    nums = re.findall(r'\d+', str(s))
    if nums:
        return int(nums[0])
    else:
        return pd.NA

def cleaned_loader(fname: Path, feat_name: str, reader_func: Callable = pd.read_csv, keep_cols: Optional[list] = None):
    """
    Load a dataset, insert ID and VISIT columns safely, optionally keep only some columns.
    Drops columns with too many NaNs (>10) by default.
    """
    try:
        df = reader_func(fname)
    except Exception as e:
        print(f"Error reading file {fname} with default reader_func: {e}")
        df = pd.read_excel(fname)
        print(f"Successfully read {fname} using pd.read_excel as fallback.")

    id_col, visit_col = df.columns[0], df.columns[1]

    df.insert(0, "ID", df[id_col].apply(extract_first_int))
    df.insert(1, "VISIT", df[visit_col].apply(extract_first_int))

    # 丢弃无法解析的行
    invalid_rows = df[df[['ID','VISIT']].isna().any(axis=1)]
    if len(invalid_rows) > 0:
        print(f"{feat_name}: Dropping {len(invalid_rows)} rows with invalid ID or VISIT")
        df = df.drop(index=invalid_rows.index)

    df = df.drop(columns=[id_col, visit_col])

    # 保留指定列
    if keep_cols is not None:
        keep_cols_safe = [c for c in keep_cols if c in df.columns]
        df = df[['ID','VISIT'] + keep_cols_safe]

    # 给特征列加前缀
    feature_cols = [c for c in df.columns if c not in ["ID", "VISIT"]]
    drop_cols = [c for c in feature_cols if df[c].isna().sum() > 10]
    if drop_cols:
        print(f"{feat_name}: Dropping cols with many nans (>10): {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)
    rename_dict = {c: f"{feat_name}_{c}" for c in feature_cols if c not in drop_cols}
    df.rename(columns=rename_dict, inplace=True)
    
    print(f"{feat_name}: final shape {df.shape}")
    return df

# Add environment variable support for data directories
def get_data_dir(env_var: str, default_path: str) -> Path:
    """Retrieve the data directory from an environment variable or use the default."""
    return Path(os.getenv(env_var, default_path))

# ------------------------
# HEI 数据
# ------------------------
def load_hei_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    hei_path = save_data_dir / "hei.pkl"

    def compute_hei() -> pd.DataFrame:
        fped = pd.read_csv(save_data_dir / "processed_fped.csv")
        diet = pd.read_csv(save_data_dir / "processed_diet.csv")
        demo = None
        from hei_package.hei import hei  # 延迟导入
        hei_data = hei(fped, diet, demo, agethresh=2, return_full_feats=True)
        hei_data.drop(columns=["DRSTZ"], inplace=True, errors='ignore')
        missing_v_seqn = [seqn for seqn in hei_data["SEQN"] if "V" not in seqn]
        hei_data.loc[hei_data["SEQN"].isin(missing_v_seqn), "SEQN"] += "_V1"
        hei_data.insert(1, "ID", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        hei_data.insert(2, "VISIT", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
        hei_data.set_index(["ID", "VISIT"], inplace=True)
        hei_data.sort_index(inplace=True)
        hei_data.drop(columns=["SEQN"], inplace=True)
        hei_data.reset_index(inplace=True)
        print(f"hei_data: {hei_data.shape}")
        return hei_data

    return _safe_pickle_load(hei_path, fallback=compute_hei)

# ------------------------
# 食物数据
# ------------------------
def load_food_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    def compute_food():
        food_data = pd.read_excel(read_data_dir / "Ashley_code_data/CoDiet Intake24 Data - Tidied up.xlsx")
        food_data.insert(1, "ID", food_data["User ID"].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
        return food_data
    return _safe_pickle_load(save_data_dir / "food_data.pkl", fallback=compute_food)

# ------------------------
# 血液和 site 数据
# ------------------------
def load_blood_site_data(read_data_dir: Path, save_data_dir: Path) -> Dict[str, pd.DataFrame]:
    site_mapping = dict(zip(['AUTH','BILBAO','CORK','ICL','UVEG'], range(1,6)))
    def compute_data():
        file = read_data_dir / "UpdatedDataFromSara/biochemical data all converted values.xlsx"
        data = pd.read_excel(file)
        data.insert(1, "ID", data["ΙD participant / Compound "].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
        data.rename(columns={"Timepoint":"VISIT"}, inplace=True)
        data['site_numeric'] = data['Site Collection'].map(site_mapping)

        site_data = data[["ID", "Site Collection"]].drop_duplicates().reset_index(drop=True)
        site_data.rename(columns={"Site Collection":"site"}, inplace=True)

        blood_data = data.drop(columns=["Site Collection","ΙD participant / Compound "]).apply(pd.to_numeric, errors='coerce')
        return {"blood_data": blood_data, "site_data": site_data}

    cached_blood = _safe_pickle_load(save_data_dir / "blood_data.pkl", fallback=lambda: compute_data()["blood_data"])
    cached_site = _safe_pickle_load(save_data_dir / "site_data.pkl", fallback=lambda: compute_data()["site_data"])
    return {"blood_data": cached_blood, "site_data": cached_site}

# ------------------------
# 能量支出
# ------------------------
def load_energy_expenditure(read_data_dir: Path, save_data_dir: Path) -> Dict[str, pd.DataFrame]:
    def compute_energy():
        energy_expenditure = pd.DataFrame()
        for file in os.listdir(read_data_dir / "energy_expenditure"):
            if file.endswith(".csv"):
                tee = pd.read_csv(read_data_dir / "energy_expenditure" / file)
                tee = tee.dropna(subset=["timepoint"])
                tee = tee[tee["sample_id"].apply(lambda x: len(re.findall(r"\d+", str(x)))==2)]
                tee.insert(1, "ID", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
                tee.insert(2, "VISIT", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", str(x))[1])))
                tee.drop(columns=["sample_id"], inplace=True)
                tee.reset_index(drop=True, inplace=True)
                energy_expenditure = pd.concat([energy_expenditure, tee], ignore_index=True)
        energy_expenditure.rename(columns={"TEE2":"TEE","TEE":"TEE_orig"}, inplace=True)
        average_expenditure = energy_expenditure.groupby(["ID","VISIT"]).agg({"TEE":"mean"}).reset_index()
        return {"energy_expenditure": energy_expenditure, "average_expenditure": average_expenditure}

    cached_ee = _safe_pickle_load(save_data_dir / "expenditure.pkl", fallback=lambda: compute_energy()["energy_expenditure"])
    cached_avg = _safe_pickle_load(save_data_dir / "average_expenditure.pkl", fallback=lambda: compute_energy()["average_expenditure"])
    return {"energy_expenditure": cached_ee, "average_expenditure": cached_avg}

# ------------------------
# 体成分数据
# ------------------------
def load_body_comp(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    body_path = save_data_dir / "body_comp.pkl"

    def fallback() -> pd.DataFrame:
        df = pd.read_excel(read_data_dir / "body_composition/BiosensorsMicrocaya_data_combined_jan2025.xlsx")
        df.dropna(how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.str.strip().str.replace(" ", "")
        df.insert(1, "ID", df["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        df.insert(2, "VISIT", df["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
        df['gender_numeric'] = df['gender'].apply(lambda x: 0 if x == 'Male' else 1)
        df.drop(columns=["sample_id", "volunteer_id", "date_of_birth", "exam_date", "recruitment_site"], inplace=True)
        print(f"body_comp: {df.shape}")
        return df

    return _safe_pickle_load(body_path, fallback=fallback)

# ------------------------
# 辅助数据加载
# ------------------------
def load_aux_dfs(read_data_dir: Path, save_data_dir: Path, nan_thresh: int = 10) -> Dict[str, pd.DataFrame]:
    dfs = {}

    # 微生物 alpha
    alpha_path = read_data_dir / "microbiome/alpha_summary_CoDiet_total_v2.csv"
    if alpha_path.exists():
        dfs['microbiome'] = cleaned_loader(alpha_path, 'microbiome', keep_cols=None)
    else:
        print("Warning: microbiome alpha file not found.")

    # scafs
    dfs['scafs'] = cleaned_loader(read_data_dir / "more_biomarkers/scafs-stool.csv", 'scafs',
                                  keep_cols=['scafs_acetate','scafs_butyrate','scafs_formate','scafs_propionate'])

    # ms urine & serum
    ms_urine_df = cleaned_loader(read_data_dir / "more_biomarkers/ms-urine.csv", 'ms_urine')
    ms_urine_df = ms_urine_df.drop(columns=[c for c in ms_urine_df.columns if 'type' in c.lower()], errors='ignore')
    dfs['ms_urine'] = ms_urine_df

    ms_serum_df = cleaned_loader(read_data_dir / "more_biomarkers/ms-serum.csv", 'ms_serum')
    ms_serum_df = ms_serum_df.drop(columns=[c for c in ms_serum_df.columns if 'type' in c.lower()], errors='ignore')
    dfs['ms_serum'] = ms_serum_df

    nmr_urine_df = cleaned_loader(read_data_dir / "UpdatedNMRLipids_12_25/unified-nmr-targeted-urine_v2.xlsx", 'nmr_urine')
    nmr_urine_df = nmr_urine_df.drop(columns=[c for c in ms_serum_df.columns if 'type' in c.lower()], errors='ignore')
    dfs['nmr_urine'] = nmr_urine_df

    # lipidomics
    lipid_path = read_data_dir / "lipidomics/lipidomics.xlsx"
    df_lip = pd.read_excel(lipid_path)
    df_lip = df_lip[df_lip["type"] == "sample"].drop(columns=["type"])
    lipidomics_df = cleaned_loader(Path(""), 'ms_lip', reader_func=lambda fn: df_lip)
    lipidomics_dbs_rbc_df = cleaned_loader(read_data_dir / "lipidomics/lipidomics-dbs-rbc.xlsx", 'dbs_rbc_lip', reader_func=pd.read_excel)
    #dfs['ms_lip']= lipidomics_df
    #dfs['dbs_rbc_lip']= lipidomics_dbs_rbc_df
    lipidomics_df = _safe_pickle_load(save_data_dir / "ms_lip.pkl", fallback=lambda: lipidomics_df)
    lipidomics_dbs_rbc_df = _safe_pickle_load(save_data_dir / "dbs_rbc_lip.pkl", fallback=lambda: lipidomics_dbs_rbc_df)


    return dfs

# ------------------------
# Gut Microbiome Wellness Index
# ------------------------
def load_gmwi_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    gmwi_path = read_data_dir / "gmwi_data.csv"
    gmwi_data = pd.read_csv(gmwi_path)
    gmwi_data = gmwi_data[gmwi_data["sample_modified"].apply(lambda x: len(re.findall(r"\d+", x)) > 0)]
    gmwi_data.insert(1, "ID", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    gmwi_data.insert(2, "VISIT", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    gmwi_data.drop(columns=['sample_original', 'sample_modified', 'site', 'HealthStatus',
                            'Visit', 'ParticipantID'], inplace=True)
    gmwi_data = _safe_pickle_load(save_data_dir / "gmwi_data.pkl", fallback=lambda: gmwi_data)
    
    return gmwi_data

# ------------------------
# Blood Pressure Data
# ------------------------
def load_blood_pressure_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    file = read_data_dir / "UpdatedDataFromSara/Blood pressure values all sites WP2.xlsx"
    data = pd.read_excel(file)
    data.insert(1, "ID", data['Participant  '].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
    blood_pressure = data.drop(columns=['Site Collection', 'hypertension/medication',
                                         'Sex', 'Participant  ', 'Age'])
    blood_pressure = _safe_pickle_load(save_data_dir / "blood_pressure.pkl", fallback=lambda: blood_pressure)
    
    return blood_pressure

# ------------------------
# Microbiome Alpha Data
# ------------------------
def load_microbiome_alpha_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    alpha_path = read_data_dir / "microbiome/alpha_summary_CoDiet_total_v2.csv"
    microbiome_alpha_df = pd.read_csv(alpha_path)
    microbiome_alpha_df = microbiome_alpha_df[microbiome_alpha_df['Unnamed: 0'].str.contains('CD_', na=False)]
    microbiome_alpha_df.insert(1, "ID", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    microbiome_alpha_df.insert(2, "VISIT", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    microbiome_alpha_df = microbiome_alpha_df.drop(columns=['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 5'])
    orig_cols = [(c, 'microbiome_' + c) for c in microbiome_alpha_df.columns if c not in ['ID', 'VISIT']]
    microbiome_alpha_df = microbiome_alpha_df.rename(columns=dict(orig_cols))
    microbiome_alpha_df = _safe_pickle_load(save_data_dir / "microbiome_alpha.pkl", fallback=lambda: microbiome_alpha_df)
    
    return microbiome_alpha_df

# ------------------------
# Lipidomics Data
# ------------------------
def load_lipidomics_data(read_data_dir: Path, save_data_dir: Path) -> pd.DataFrame:
    lipid_path = read_data_dir / "lipidomics/lipidomics.xlsx"
    df = pd.read_excel(lipid_path)
    df = df[df['type'] == 'sample']
    df.drop(columns=['type'], inplace=True)
    lipidomics_df = cleaned_loader(Path(""), 'ms_lip', reader_func=lambda fn: df)
    lipidomics_df = _safe_pickle_load(save_data_dir / "lipidomics.pkl", fallback=lambda: lipidomics_df)
    
    return lipidomics_df


# ------------------------
# 全量数据加载
# ------------------------
def load_all_data(
    read_data_dir: Path, 
    save_data_dir: Optional[Path]=None,
    nan_thresh: int = 10  # Default value for nan_thresh
) -> Dict[str, Any]:
    if save_data_dir is None:
        save_data_dir = DEFAULT_SAVE_DATA_DIR

    print(f"Process ID: {os.getpid()}")
    data = {}

    # HEI
    data['hei_data'] = load_hei_data(read_data_dir, save_data_dir)
    # 体成分
    data['body_comp'] = load_body_comp(read_data_dir, save_data_dir)
    # 辅助数据
    data['new_dfs'] = load_aux_dfs(read_data_dir, save_data_dir, nan_thresh=nan_thresh)
    data['blood_pressure'] = load_blood_pressure_data(read_data_dir, save_data_dir)
    data['blood_site'] = load_blood_site_data(read_data_dir, save_data_dir)
    data['gmwi_data'] = load_gmwi_data(read_data_dir, save_data_dir)
    data['microbiome_alpha'] = load_microbiome_alpha_data(read_data_dir, save_data_dir)
    data['lipidomics'] = load_lipidomics_data(read_data_dir, save_data_dir)
    # 食物数据
    data['food_data'] = load_food_data(read_data_dir, save_data_dir)
    # 能量支出
    energy_data = load_energy_expenditure(read_data_dir, save_data_dir)
    data['energy_expenditure'] = energy_data['energy_expenditure']
    data['average_expenditure'] = energy_data['average_expenditure']

    microbiome_cl_df = pd.read_csv(read_data_dir/"derived/microbiome_4_clusters.csv")
    microbiome_cl_df = microbiome_cl_df.drop(columns=['Unnamed: 0'])
    data['microbiome_cl'] = microbiome_cl_df
    microbiome_cl_df = _safe_pickle_load(save_data_dir/"microbiome_cl_df.pkl", fallback=lambda: microbiome_cl_df)
    
    microbiome_phyl_cl_df = pd.read_csv(read_data_dir / "derived/microbiome_phylumn4_clusters.csv")
    microbiome_phyl_cl_df = microbiome_phyl_cl_df.drop(columns=['Unnamed: 0'])
    data['microbiome_phyl_cl'] = microbiome_phyl_cl_df
    microbiome_phyl_cl_df = _safe_pickle_load(save_data_dir/"microbiome_phyl_cl_df.pkl", fallback=lambda: microbiome_phyl_cl_df)
    
    microbiome_embedding_df = pd.read_csv(read_data_dir/"derived/microbiome_embedding_20.csv")
    microbiome_clean15_df = pd.read_csv(read_data_dir/"derived/microbiome_clean15.csv")
    data['microbiome_embedding'] = microbiome_embedding_df
    microbiome_embedding_df = _safe_pickle_load(save_data_dir/"microbiome_embedding_df.pkl", fallback=lambda: microbiome_embedding_df)
    
    # 安全 per-kg 归一化 HEI 食物特征
    hei_data = data['hei_data']
    body_comp = data['body_comp']

    weight_col = None
    for col in body_comp.columns:
        if col.lower() in ['weight','weight_kg','bodyweight','body_weight']:
            weight_col = col
            break

    if weight_col:
        food_feats = [c for c in hei_data.columns if c.startswith('food_')]
        merged = pd.merge(hei_data, body_comp[['ID','VISIT',weight_col]], on=['ID','VISIT'], how='left')
        for feat in food_feats:
            merged[feat + '_perkg'] = merged.apply(
                lambda row: row[feat] / row[weight_col] if pd.notna(row[weight_col]) else pd.NA,
                axis=1
            )
        data['hei_data'] = merged
        print(f"HEI food features safely normalized per kg using '{weight_col}'")
    else:
        print("Warning: No weight column found in body_comp, skipping per-kg normalization.")

    return data


if __name__ == "__main__":
    data = load_all_data(DEFAULT_READ_DATA_DIR, DEFAULT_SAVE_DATA_DIR)
