# HEI Developer Guide

This guide is for contributors who maintain and extend the open-source codebase.

## 1) Get the Code

```bash
git clone https://github.com/sereneHe/Recommender_System.git
cd Recommender_System
```

Create a feature branch:

```bash
git checkout -b feat/your-change
```

## 2) Development Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

Or use `uv`:

```bash
uv sync --all-groups
```

## 3) Key Entry Points

- Training entry (Hydra): `src/hei_project/train.py`
- API entry (FastAPI): `src/hei_project/api.py`
- Core recommender logic: `src/hei_project/model.py`
- Data loading: `src/hei_project/hei/data_helper.py`
- Notebook-aligned script 1: `scripts/build_prep_data.sh`
- Notebook-aligned script 2: `scripts/NCD_analysis_food_and_conditioning.sh`
- Notebook-aligned script 3: `scripts/intake_analysis_plot_results_v2.sh`

## 4) Local Run

Train (default config):

```bash
cd /Users/xiaoyuhe/Recommender_System
source .venv/bin/activate
train
```

Train (override parameters):

```bash
cd /Users/xiaoyuhe/Recommender_System
source .venv/bin/activate
train solver.model_name=XGB solver.custom_objective=mse_builtin solver.n_runs=1 solver.n_select_features=5 'targets.columns=["GLU (mg/dL)"]'
```

API:

```bash
cd /Users/xiaoyuhe/Recommender_System
source .venv/bin/activate
api
```

## 5) Function Debugging

Run all tests:

```bash
cd /Users/xiaoyuhe/Recommender_System
source .venv/bin/activate
pytest -q
```

Run one file or one test function:

```bash
pytest -q tests/test_model.py
pytest -q tests/test_model.py::test_model_forward_shape
```

Debug with `pdb`:

```bash
python -m pdb src/hei_project/train.py
```

Recommended workflow for function-level debugging:

1. Add `breakpoint()` in the target function.
2. Add a minimal regression test in `tests/`.
3. Remove temporary prints after fix, keep the test case.

## 6) Code Quality

```bash
ruff check src tests
pytest -q
```

Before commit:

```bash
git status
git add -A
git commit -m "feat: your change"
git push origin feat/your-change
```
