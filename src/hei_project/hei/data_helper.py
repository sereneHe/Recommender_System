import re
import os
import pickle
import pandas as pd
from hei_project.hei.hei import hei


SEED = 2227070966
# create a manual site to number map, for consistency
SITE_MAPPING = dict(zip(['AUTH', 'BILBAO', 'CORK', 'ICL', 'UVEG'], range(1, 6)))

BASE_PATH = "."
#BASE_PATH = os.path.join("C:\\", "Users", "petrr", "Documents", "codiet", "recommender", "codiet_analysis", "src")
DATA_PATH = os.path.join(BASE_PATH, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")


# try renorming to body weight instead of TKAL

def read_HEI():
    if os.path.exists(os.path.join(RAW_DATA_PATH, "HEI", "hei.pkl")):
        return pd.read_pickle(os.path.join(RAW_DATA_PATH, "HEI", "hei.pkl"))

    fped = pd.read_csv(os.path.join(RAW_DATA_PATH, "HEI", "processed_fped.csv"))
    diet = pd.read_csv(os.path.join(RAW_DATA_PATH, "HEI", "processed_diet.csv"))
    # Demo data from ashley code contains repetitions, resulting in repetitions in the hei data
    hei_data = hei(fped, diet, None, agethresh=2, return_full_feats=True)

    # Currently remove DRSTZ column, as it represents day
    hei_data.drop(columns=["DRSTZ"], inplace=True)

    # hei_data.rename(columns = {'DRSTZ':'timepoint'}, inplace=True)

    missing_v_seqn = [seqn for seqn in hei_data["SEQN"] if "V" not in seqn]
    print(f"SEQN missing 'V': {missing_v_seqn}")
    hei_data.loc[hei_data["SEQN"].isin(missing_v_seqn), "SEQN"] += "_V1"
    hei_data.insert(1, "ID", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    hei_data.insert(2, "VISIT", hei_data["SEQN"].apply(lambda x: int(re.findall(r"\d+", x)[1])))

    hei_data.set_index(["ID", "VISIT"], inplace=True)
    hei_data.sort_index(inplace=True)
    hei_data.drop(columns=["SEQN"], inplace=True)

    # Average over days

    hei_data = hei_data.groupby(level=["ID", "VISIT"]).mean().reset_index()

    food_feats_orig = ["TKCAL", "WHOLEFRT", "MONOPOLY", "ALLMEAT", "SEAPLANT", "ADDSUGC", "SOLFATC", "TALCO",
                       "T_F_TOTAL", "T_G_WHOLE", "T_D_TOTAL", "TSFAT", "TSODI", "T_G_REFINED", "EMPTYCAL10",
                       "T_V_TOTAL", "T_V_DRKGR", "T_V_LEGUMES"
                       ]

    hei_data = hei_data[food_feats_orig + ['ID', 'VISIT']]

    rename_dict = dict([(n, 'food_' + n) for n in food_feats_orig])
    hei_data = hei_data.rename(columns=rename_dict)

    hei_data.to_pickle(os.path.join(RAW_DATA_PATH, "HEI", "hei.pkl"))
    return hei_data

def read_food_data():
    if os.path.exists(os.path.join(BASE_PATH, "data", "food_data", "food_data.pkl")):
        return pd.read_pickle(os.path.join(BASE_PATH, "data", "food_data", "food_data.pkl"))

    food_data = pd.read_excel(os.path.join(BASE_PATH, "Ashley_code", "CoDiet Intake24 Data - Tidied up.xlsx"))
    food_data.insert(1, "ID", food_data["User ID"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    food_data.to_pickle(os.path.join(BASE_PATH, "data", "food_data", "food_data.pkl"))
    return food_data

def read_body_comp():
    if os.path.exists(os.path.join(RAW_DATA_PATH, "body_composition", "body_comp.pkl")):
        return pd.read_pickle(os.path.join(RAW_DATA_PATH, "body_composition", "body_comp.pkl"))

    body_comp = pd.read_excel(os.path.join(RAW_DATA_PATH, "body_composition", "BiosensorsMicrocaya_data_combined_jan2025.xlsx"))
    body_comp.dropna(how="all", inplace=True)
    body_comp.reset_index(drop=True, inplace=True)
    body_comp.columns = body_comp.columns.str.strip()
    body_comp.columns = body_comp.columns.str.replace(" ", "")
    body_comp.insert(1, "ID", body_comp["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    body_comp.insert(2, "VISIT", body_comp["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    limit_columns = [col for col in body_comp.columns if "limit" in col.lower()]
    single_value_columns = [col for col in body_comp.columns if body_comp[col].nunique() == 1]
    to_drop = ["sample_id", "volunteer_id", "date_of_birth", "exam_date", "recruitment_site"]
    body_comp.drop(columns=limit_columns + single_value_columns + to_drop, inplace=True)
    body_comp.to_pickle(os.path.join(RAW_DATA_PATH, "body_composition", "body_comp.pkl"))
    body_comp['gender_numeric'] = body_comp['gender'].apply(lambda x: 0 if x == 'Male' else 1)
    return body_comp

def read_blood_data():
    if os.path.exists(os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "blood_data.pkl")):
        return pd.read_pickle(os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "blood_data.pkl"))

    file = os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "biochemical data all converted values.xlsx")

    data = pd.read_excel(file)

    data.insert(1, "ID", data["ΙD participant / Compound "].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
    data.rename(columns={"Timepoint": "VISIT"}, inplace=True)

    data['site_numeric'] = data['Site Collection'].map(SITE_MAPPING)

    site_data = data[["ID", "Site Collection"]].copy()
    site_data.drop_duplicates(inplace=True)
    site_data.reset_index(drop=True, inplace=True)
    site_data.rename(columns={'Site Collection': 'site'}, inplace=True)

    blood_data = data.drop(columns=["Site Collection", "ΙD participant / Compound "]).reset_index(drop=True)
    blood_data = blood_data.apply(pd.to_numeric, errors='coerce')

    with open(os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "blood_data.pkl"), "wb") as f:
        pickle.dump(blood_data, f)
    with open(os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "site_data.pkl"), "wb") as f:
        pickle.dump(site_data, f)

    return blood_data

def read_average_expenditure():
    if os.path.exists(os.path.join(RAW_DATA_PATH, "energy_expenditure", "average_expenditure.pkl")):
        return pd.read_pickle(os.path.join(RAW_DATA_PATH, "energy_expenditure", "average_expenditure.pkl"))

    energy_expenditure = pd.DataFrame()
    for file in os.listdir(os.path.join(RAW_DATA_PATH, "energy_expenditure")):
        if file.endswith(".csv"):
            tee = pd.read_csv(os.path.join(RAW_DATA_PATH, "energy_expenditure", file))

            tee = tee.dropna(subset=["timepoint"])

            # some entries don't have visit specified. remove them.
            tee = tee[tee["sample_id"].apply(lambda x: len(re.findall(r"\d+", x)) == 2)]

            tee.insert(1, "ID", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
            tee.insert(2, "VISIT", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", x)[1])))

            tee.drop(columns=["sample_id"], inplace=True)
            tee.reset_index(drop=True, inplace=True)
            if energy_expenditure.empty:
                energy_expenditure = tee
            else:
                energy_expenditure = pd.concat([energy_expenditure, tee], ignore_index=True)

    energy_expenditure.rename(columns={"TEE2": "TEE", "TEE": "TEE_orig"}, inplace=True)
    energy_expenditure["timepoint"].astype(int)
    average_expenditure = energy_expenditure.groupby(["ID", "VISIT"]).agg({"TEE": "mean"}).reset_index()

    average_expenditure.to_pickle(os.path.join(RAW_DATA_PATH, "energy_expenditure", "average_expenditure.pkl"))
    energy_expenditure.to_pickle(os.path.join(RAW_DATA_PATH, "energy_expenditure", "expenditure.pkl"))
    return average_expenditure

def load_all_data():

    hei_data = read_HEI()
    blood_data = read_blood_data()
    body_comp = read_body_comp()
    body_comp['gender_numeric'] = body_comp['gender'].apply(lambda x: 0 if x == 'Male' else 1)
    average_expenditure = read_average_expenditure()

    ##### Gut Microbiome Wellness Index
    gmwi_data = pd.read_csv(os.path.join(RAW_DATA_PATH, "gmwi_data.csv"))

    gmwi_data = gmwi_data[gmwi_data["sample_modified"].apply(lambda x: len(re.findall(r"\d+", x)) > 0)]

    gmwi_data.insert(1, "ID", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    gmwi_data.insert(2, "VISIT", gmwi_data["sample_modified"].apply(lambda x: int(re.findall(r"\d+", x)[1])))

    gmwi_data.drop(columns=['sample_original', 'sample_modified', 'site', 'HealthStatus',
                            'Visit', 'ParticipantID'], inplace=True)

    ##### Blood Pressure
    data = pd.read_excel(os.path.join(RAW_DATA_PATH, "UpdatedDataFromSara", "Blood pressure values all sites WP2.xlsx"))
    data.insert(1, "ID", data['Participant  '].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
    blood_pressure = data.drop(columns=['Site Collection', 'hypertension/medication',
                                        'Sex', 'Participant  ', 'Age'
                                        ])

    ##### microbiome alpha
    microbiome_alpha_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "microbiome", "alpha_summary_CoDiet_total_v2.csv"))
    microbiome_alpha_df = microbiome_alpha_df[microbiome_alpha_df['Unnamed: 0'].str.contains('CD_', na=False)]
    microbiome_alpha_df.insert(1, "ID", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
    microbiome_alpha_df.insert(2, "VISIT", microbiome_alpha_df["Unnamed: 0"].apply(lambda x: int(re.findall(r"\d+", x)[1])))
    microbiome_alpha_df = microbiome_alpha_df.drop(columns=['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 5'])

    orig_cols = [(c, 'microbiome' + '_' + c) for c in microbiome_alpha_df.columns if c not in ['ID', 'VISIT']]
    microbiome_alpha_df = microbiome_alpha_df.rename(columns=dict(orig_cols))


    def cleaned_loader(fname, feat_name, reader_func=pd.read_csv):
        df = reader_func(fname)

        df.insert(0, "ID", df["patient"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        df.insert(1, "VISIT", df["visit"].apply(lambda x: int(re.findall(r"\d+", x)[0])))
        df = df.drop(columns=['patient', 'visit'])

        orig_cols = [(c, feat_name + '_' + c) for c in df.columns if c not in ['ID', 'VISIT']]
        df = df.rename(columns=dict(orig_cols))

        cols_with_many_nans = list(df.columns[df.isnull().sum() > 10])
        if len(cols_with_many_nans) > 0:
            print(f'{feat_name}: Dropping cols with many nans: {cols_with_many_nans}. ')
        df = df.drop(columns=cols_with_many_nans)

        print(f'{feat_name}: {df.shape}')

        return df


    ######### more dfs

    ## NOTE: there should not be a column prefix name that is a substring of another
    ## column prefix name

    scafs_df = cleaned_loader(os.path.join(RAW_DATA_PATH, "more_biomarkers", "scafs-stool.csv"), 'scafs')
    scafs_df = scafs_df[['ID', 'VISIT', 'scafs_acetate', 'scafs_butyrate', 'scafs_formate', 'scafs_propionate']]

    ms_urine_df = cleaned_loader(os.path.join(RAW_DATA_PATH, "more_biomarkers", "ms-urine.csv"), 'ms_urine')
    ms_urine_df = ms_urine_df.drop(columns=['ms_urine_type', 'ms_urine_sample-type'])

    ms_serum_df = cleaned_loader(os.path.join(RAW_DATA_PATH, "more_biomarkers", "ms-serum.csv"), 'ms_serum')
    ms_serum_df = ms_serum_df.drop(columns=['ms_serum_type', 'ms_serum_sample-type'])

    nmr_urine_df = cleaned_loader(
        os.path.join(RAW_DATA_PATH, "UpdatedNMRLipids_12_25", "unified-nmr-targeted-urine_v2.xlsx"), 'nmr_urine',
        reader_func=pd.read_excel
    )

    # the 'nmr_urine_1-methyladenosine' column is all zeros
    nmr_urine_df.drop(columns=['nmr_urine_site', 'nmr_urine_1-methyladenosine'], inplace=True)


    df = pd.read_excel(os.path.join(RAW_DATA_PATH, "lipidomics", "lipidomics.xlsx"))
    df = df[df['type'] == 'sample']
    df.drop(columns=['type'], inplace=True)
    lipidomics_df = cleaned_loader('', 'ms_lip', reader_func=lambda fn: df)

    lipidomics_dbs_rbc_df = cleaned_loader(os.path.join(RAW_DATA_PATH, "lipidomics", "lipidomics-dbs-rbc.xlsx"),
                                           'dbs_rbc_lip',
                                           reader_func=pd.read_excel)

    microbiome_cl_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "derived", "microbiome_4_clusters.csv"))
    microbiome_cl_df = microbiome_cl_df.drop(columns=['Unnamed: 0'])

    microbiome_phyl_cl_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "derived", "microbiome_phylumn4_clusters.csv"))
    microbiome_phyl_cl_df = microbiome_phyl_cl_df.drop(columns=['Unnamed: 0'])

    microbiome_embedding_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "derived", "microbiome_embedding_20.csv"))
    microbiome_clean15_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "derived", "microbiome_clean15.csv"))

    df_pairs = [
        ('microbiome', microbiome_alpha_df),
        ('scafs', scafs_df),
        ('ms_serum', ms_serum_df),
        ('ms_urine', ms_urine_df),
        ('nmr_urine', nmr_urine_df),
        ('lipidomics', lipidomics_df),
        ('lipidomics_dbs_rbc', lipidomics_dbs_rbc_df),
        ('microbiome_4_cl', microbiome_cl_df),
        ('microbiome_phyl4_cl', microbiome_phyl_cl_df),
        ('microbiome_embedding', microbiome_embedding_df),
        ('microbiome_clean15', microbiome_clean15_df)
    ]

    new_dfs = dict(df_pairs)
    new_dfs_names = [p[0] for p in df_pairs]

    ### BUILD PREP DATA

    def average_visits(df):
        df = df.groupby("ID").mean()
        df.drop(columns=["VISIT"], inplace=True)
        df.reset_index(inplace=True)
        return df

    USE_LIPIDOMICS = True

    from functools import reduce

    # note some subjects, like 377, only have one visit, and their values are strange.

    prep_hei_data = average_visits(hei_data)
    prep_average_expenditure = average_visits(average_expenditure)
    prep_body_comp_data = average_visits(body_comp.select_dtypes(include=['number']))
    prep_blood_data = average_visits(blood_data.select_dtypes(include=['number']))
    prep_gmwi_data = average_visits(gmwi_data)

    new_dfs_prep = {}
    for n in new_dfs_names:
        df = average_visits(new_dfs[n])
        new_dfs_prep[n] = df
        print(f'{n}: prep shape: {df.shape}')
        # print(df.columns)

    prep_data = reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'),
                       [prep_hei_data, prep_average_expenditure, prep_body_comp_data, prep_blood_data,
                        prep_gmwi_data, blood_pressure] + list(new_dfs_prep.values())
                       )

    print(f'\nTotal prep_data shape: {prep_data.shape}')

    prep_data['site_continental'] = prep_data['site_numeric'].isin(
        [SITE_MAPPING[s] for s in ['AUTH', 'BILBAO', 'UVEG']]
    ).astype(int)

    base_feats = ['age', 'gender_numeric', 'weight', 'height', 'stress_index',
                  'fatigue_index', 'mean_hrt', 'site_continental'
                  ]

    prep_data['normed_TEE'] = prep_data['TEE'] / prep_data['lean_mass_of_trunk']

    non_food_feats = ['normed_TEE', 'GMWI']
    for name in ['microbiome', 'scafs', 'ms_serum.pca', 'ms_urine.pca',
                 'nmr_urine', 'nmr_urine.pca'
                 ]:
        non_food_feats += [c for c in prep_data.columns if name + '_' in c]

    LIPIDOMICS_COLUMNS = [c for c in prep_data.columns if 'ms_lip' + '_' in c]
    LIPIDOMICS_COLUMNS += [c for c in prep_data.columns if 'dbs_rbc_lip' + '_' in c]
    non_food_feats += LIPIDOMICS_COLUMNS

    food_feats = [c for c in prep_data if 'food_' in c[:5]]
    for c in food_feats:
        prep_data['normed_' + c] = prep_data[c] / prep_data['lean_mass_of_trunk']

    food_feats = [c for c in prep_data if 'normed_food_' in c] + ['food_TKCAL']

    M = {'base': base_feats, 'non_food': non_food_feats, 'food': food_feats}
    get_features_from_composition = lambda lst_names: [fn for lst_name in lst_names for fn in M[lst_name]]

    print(prep_data['gender_numeric'])

    return food_feats, non_food_feats, prep_data
