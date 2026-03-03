# Running Notes

This note replaces the old `project_hei` / `hei_package` commands and matches the current
repository layout in `Recommender_System`.

## 1. Open The Workspace

Always start from the repository root:

```bash
cd /Users/xiaoyuhe/Recommender_System
```

Do not use these old paths anymore:

- `src/project_hei`
- `src/HEI`
- `src/project_hei/configs/config.yaml`

The active package path is:

```bash
/Users/xiaoyuhe/Recommender_System/src/hei_project
```

## 2. Create And Activate The Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
```

Expected Python path:

```bash
/Users/xiaoyuhe/Recommender_System/.venv/bin/python
```

## 3. Install Dependencies

Install runtime dependencies:

```bash
uv pip install -r requirements.txt
```

Optional but recommended: install the package in editable mode so imports work without
manually setting `PYTHONPATH`.

```bash
uv pip install -e .
```

If you do not install with `-e .`, use:

```bash
export PYTHONPATH=/Users/xiaoyuhe/Recommender_System/src
```

## 4. Main Config Files

Current main config:

```bash
/Users/xiaoyuhe/Recommender_System/src/hei_project/config/config.yaml
```

There is also a top-level placeholder directory:

```bash
/Users/xiaoyuhe/Recommender_System/configs
```

For training and API, use the config under `src/hei_project/config/`.

## 5. Quick Import Checks

Check that the core loader imports correctly:

```bash
PYTHONPATH=src python -c "from hei_project.hei.data_helper import load_all_data; print(load_all_data)"
```

Check that PCA tooling imports correctly:

```bash
PYTHONPATH=src python -c "from hei_project.hei.compute_tools import get_pca; print(get_pca)"
```

## 6. PCA Smoke Test

This is the current equivalent of the old `data_compute_tools.get_pca` test:

```bash
PYTHONPATH=src python -c "import pandas as pd; import numpy as np; from hei_project.hei.compute_tools import get_pca; np.random.seed(42); prep_data = pd.DataFrame({'ID': range(1, 21), 'TRIG (mg/dL)': np.random.normal(150, 30, 20)}); objective_df = pd.DataFrame({'ID': range(1, 21), 'feat1': np.random.normal(0, 1, 20), 'feat2': np.random.normal(0, 1, 20)}); print(get_pca(prep_data, objective_df, 'test_pca', n_dims=2, label_col='TRIG (mg/dL)', do_print=False))"
```

## 7. Build `prep_data`

Use the maintained shell wrapper:

```bash
bash scripts/build_prep_data.sh
```

That script writes outputs to:

```bash
/Users/xiaoyuhe/Recommender_System/data/processed
```

You can override the output directory:

```bash
OUT_DIR=/Users/xiaoyuhe/Recommender_System/data/processed bash scripts/build_prep_data.sh
```

Direct Python equivalent:

```bash
PYTHONPATH=src python -c "import json; from pathlib import Path; from hei_project.hei.data_helper import load_all_data; out_dir = Path('data/processed'); out_dir.mkdir(parents=True, exist_ok=True); food_feats, non_food_feats, prep_data = load_all_data(); prep_data.to_pickle(out_dir / 'prep_data.pkl'); (out_dir / 'food_feats.json').write_text(json.dumps(list(food_feats), ensure_ascii=False, indent=2), encoding='utf-8'); (out_dir / 'non_food_feats.json').write_text(json.dumps(list(non_food_feats), ensure_ascii=False, indent=2), encoding='utf-8'); print(prep_data.shape)"
```

## 8. Run Training

Hydra entrypoint:

```bash
PYTHONPATH=src python -m hei_project.train
```

Example with overrides:

```bash
PYTHONPATH=src python -m hei_project.train solver.n_runs=10 solver.n_select_features=3 targets.columns='[\"GLU (mg/dL)\"]'
```

The default config loaded by training is:

```bash
src/hei_project/config/config.yaml
```

## 9. Run The API

```bash
PYTHONPATH=src uvicorn hei_project.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 10. Run Analysis Scripts

Available scripts:

- `scripts/build_prep_data.sh`
- `scripts/NCD_analysis_incremental_features.sh`
- `scripts/NCD_analysis_food_and_conditioning.sh`
- `scripts/run_ncd_incremental_full.sh`
- `scripts/run_ncd_one_by_one.sh`
- `scripts/run_ncd_targets_one_by_one.sh`
- `scripts/intake_analysis_plot_results_v2.sh`
- `scripts/runningtime.sh`

Run them from the repository root, for example:

```bash
bash scripts/runningtime.sh
```

## 11. Old Name To New Name Mapping

Use these replacements when reading older notes:

- `project_hei` -> `hei_project`
- `hei_package` -> `hei_project.hei` or a specific module under `hei_project`
- `src/project_hei/configs/config.yaml` -> `src/hei_project/config/config.yaml`
- `src/HEI/.venv/bin/python` -> `.venv/bin/python` at the repository root
- `datasets/...` -> `data/...` in the current workspace

## 12. Known Code Mismatch

There is still one stale default path inside the code:

- `src/hei_project/data.py` still references `src/project_hei/configs/config.yaml` in its default
  string values.

Treat that as legacy text. Use the current path:

```bash
src/hei_project/config/config.yaml
```
