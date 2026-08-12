# Recommender_System — HEI / Multimodal Health Modeling

This repository contains code and configuration for **Healthy Eating Index (HEI)** analysis and **multimodal health-data modeling**.

- Main experiment entrypoint: `src/project_hei/run_experiment.py`
- Hydra configs: `src/project_hei/configs/`
- Raw data (not committed): `data/raw/`

## Data Sources (Raw)

The project uses multiple raw modalities. The table below summarizes where they come from (folder/file hints) and the typical variables they contain.

| Modality | 📁 Data Sources / File Hints | 🔑 Key Dimensions (Typical Variables) |
| --- | --- | --- |
| **🍽️ Food / Diet** | `food_data/` (e.g. *Diet data … Intake 24.xlsx*); `Ashley_code_data/` (Intake24 tidied outputs) | Daily / per-meal intake, food groups, macro- & micronutrients, potential inputs for HEI or other diet quality scores |
| **🏃 Physical Activity** | `activity/activity.csv` | Activity duration, intensity, type/category, frequency (field definitions based on table schema) |
| **🔥 Energy Expenditure** | `energy_expenditure/*.csv`; `PAL&TEE Calculation.docx`; `Activity_Summarize Data_All.xlsx` | Physical Activity Level (PAL), Total Energy Expenditure (TEE), site-level or activity-level energy expenditure summaries |
| **⚖️ Body Composition** | `body_composition/*.xlsx`; `body_composition/biosensors.csv` | Body weight, body fat–related measures, and other biosensor-derived physiological indicators |
| **🩸 Blood Biochemistry / Blood Pressure / Biomarkers** | `blood_biochemistry/bloodbiochemistry.xlsx`; `UpdatedDataFromSara/Blood pressure*.xlsx`; `UpdatedDataFromSara/biochemical data*.xlsx`; `more_biomarkers/*.csv` | Clinical biochemistry markers, blood pressure measures, and targeted or untargeted biomarker panels |
| **🦠 Microbiome** | `microbiome/*metaphlan*.csv`; `microbiome/*alpha_summary*.csv`; `microbiome/*beta_summary*.csv`; corresponding files under `Liam_microbiome/` | Taxonomic abundance profiles, prevalence/detection metrics, alpha diversity indices, beta diversity summaries |
| **🧪 Metabolomics (NMR / MS)** | `UpdatedNMRIsabelNieves/*.xls(x)`; `UpdatedNMRLipids_12_25/*.xlsx/.ods`; `more_biomarkers/ms-serum.csv`; `more_biomarkers/ms-urine.csv`; `more_biomarkers/nmr-targeted-serum.csv` | Quantitative metabolite concentrations from serum and urine (NMR targeted; MS-based serum/urine) |
| **🧬 Lipidomics** | `lipidomics/lipidomics.xlsx`; `lipidomics/lipidomics-dbs-rbc.xlsx`; `UpdatedNMRLipids_12_25/lipidomics_rbc_*.xlsx`; `lipidomic_dbs_*.xlsx` | Lipid molecular species profiles, potentially stratified by DBS/RBC or sample type |
| **📷 Camera / Site-specific Data** | `camera_data/ICL.xlsx`; `camera_data/Bilbao.xlsx` | Site-specific or camera-derived variables (exact fields defined in source tables) |

Notes:
- This is a **raw-only** inventory (`data/raw/**`). Processed/derived artifacts are intentionally not documented here.
- Exact column definitions should be taken from each source file’s schema.

## Methods / Models

Experiments are configured with Hydra configs under `src/project_hei/configs/` and run via `src/project_hei/run_experiment.py`.

Configured/implemented model families include (see `src/project_hei/hei_package/models.py` and `src/project_hei/configs/experiment/*.yaml`):
- Gaussian Process Regression (GP)
- Linear Regression (REG)
- XGBoost Regressor (XGB)
- Random Forest (RF)
- Decision Tree (DT)
- Bagged kNN (config: `src/project_hei/configs/exp_bagged_knn.yaml`)

Preprocessing / feature engineering includes (configurable in `src/project_hei/configs/prep.yaml`):
- Standard scaling (via sklearn pipelines)
- Optional PCA blocks (e.g., MS serum / MS urine / NMR urine)

## Project Structure

```text
Recommender_System/
  data/
    raw/                # raw multimodal source data (not committed)
    process/            # intermediate/processed tables (not committed)
  src/
    hei.py
    project_hei/
      run_experiment.py
      configs/
      hei_package/
  tests/
```

## Quickstart (Hydra)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# If needed (Hydra runtime):
pip install hydra-core omegaconf

# Run default config
python src/project_hei/run_experiment.py

# Override experiment type (examples)
python src/project_hei/run_experiment.py experiment=GP
python src/project_hei/run_experiment.py experiment=XGB
python src/project_hei/run_experiment.py experiment=REG
```

## GCP + DVC + W&B

This repo includes `scripts/gcp_wandb_pipeline.sh` for cloud integration bootstrap.

- Fixed DVC remote bucket: `gs://serenehe_bucket_1/dvc`
- Fixed GCP project default in script: `recommender-system-hei`
- Fixed W&B defaults in script:
  - entity: `hexiaoyu-czech-technical-university-in-prague`
  - project: `Recommender System`

Run from repo root:

```bash
cd /Users/xiaoyuhe/Recommender_System

# Required for W&B login
export WANDB_API_KEY=<your_wandb_api_key>

# Optional: if you use a service-account json
# export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/key.json

./scripts/gcp_wandb_pipeline.sh
```

What this script does:
- Sets active gcloud project.
- Ensures DVC repo exists and configures default remote to `gs://serenehe_bucket_1/dvc`.
- Runs `gsutil ls gs://serenehe_bucket_1` access check.
- Performs W&B login if `WANDB_API_KEY` is provided.

## Data Policy

Raw datasets are not pushed to GitHub (see `.gitignore`). To reproduce results, place the expected source files under `data/raw/`.
