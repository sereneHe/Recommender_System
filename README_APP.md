# HEI App Guide (Non-Developers)

This guide is for users who only need to install, run, and view result figures.

## 1) Installation

Run in terminal:

```bash
cd /Users/xiaoyuhe/Recommender_System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use `uv`:

```bash
cd /Users/xiaoyuhe/Recommender_System
uv sync
```

## 2) Data Overview

This project uses multimodal health data. Below is a practical overview of raw inputs.

| Modality | Example Source Hints | Typical Variables |
| --- | --- | --- |
| Food / Diet | `data/raw/food_data/`, Intake24 files | Daily intake, meal intake, macro/micronutrients |
| Physical Activity | `data/raw/activity/activity.csv` | Activity duration, intensity, category, frequency |
| Energy Expenditure | `data/raw/energy_expenditure/*.csv` | PAL, TEE, site/activity-level summaries |
| Body Composition | `data/raw/body_composition/*.xlsx` | Weight, fat-related measures, biosensor indicators |
| Blood / Biomarkers | `data/raw/blood_biochemistry/*.xlsx`, `data/raw/more_biomarkers/*.csv` | Clinical chemistry, blood pressure, biomarker panels |
| Microbiome | `data/raw/microbiome/*` | Taxonomic abundance, alpha/beta diversity summaries |
| Metabolomics | `data/raw/more_biomarkers/ms-*.csv`, `data/raw/UpdatedNMR*/` | Serum/urine metabolite concentrations |
| Lipidomics | `data/raw/lipidomics/*.xlsx` | Lipid species profiles (DBS/RBC/sample-specific) |

Notes:
- Raw data is expected under `data/raw/**`.
- Processed artifacts are generated automatically into `data/processed/` and `reports/results/`.
- File schemas (exact column names) come from each source table and may vary by site/study.

## 3) Prepare Data

Place raw data under:

- `data/raw/`

Important files (examples):

- `data/W_est.csv.zip`
- `data/intra_nodes.txt`
- `data/raw/**` (diet/biomarker raw tables)

## 4) Run Data Prep + Analysis + Plotting

```bash
cd /Users/xiaoyuhe/Recommender_System
chmod +x scripts/*.sh
./scripts/build_prep_data.sh
SEEDS='13042,95863' ./scripts/NCD_analysis_food_and_conditioning.sh
./scripts/intake_analysis_plot_results_v2.sh
```

## 5) Output Locations

- Intermediate results (pkl/json): `reports/results/`
- Figure outputs: `reports/figures/`
- Main figure 1: `reports/figures/ncd_risk_incremental_5feat.png`
- Main figure 2: `reports/figures/ncd_risk_incremental_nolip_lip.png`

## 6) Start API (Optional)

```bash
cd /Users/xiaoyuhe/Recommender_System
source .venv/bin/activate
api
```

Default address:

- `http://127.0.0.1:8000`
- Health check: `GET /health`
