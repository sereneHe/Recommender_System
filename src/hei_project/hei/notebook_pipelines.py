"""
notebook_pipelines.py

将所有 notebook 按 notebook 维度打包为函数
- 函数名 = notebook 名字
- 路径统一来自 model.py
- 图片输出统一由 visualize.py 接管
"""

import os
from pathlib import Path

from model import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

from visualize import _wrap_savefig, _restore_savefig


# =========================
# Utils
# =========================
def _enter_project_root():
    """保证 notebook 中相对路径行为一致"""
    os.chdir(PROJECT_ROOT)


# =========================================================
# 1. data_analysis.ipynb
# =========================================================
def data_analysis():
    NOTEBOOK_NAME = "data_analysis"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 data_analysis.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 2. data_analysis_clusters.ipynb
# =========================================================
def data_analysis_clusters():
    NOTEBOOK_NAME = "data_analysis_clusters"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 data_analysis_clusters.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 3. data_analysis_mktest_v2.ipynb
# =========================================================
def data_analysis_mktest_v2():
    NOTEBOOK_NAME = "data_analysis_mktest_v2"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 data_analysis_mktest_v2.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 4. data_analysis_mktest_expenditure.ipynb
# =========================================================
def data_analysis_mktest_expenditure():
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 data_analysis_mktest_expenditure.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 5. NCD_analysis_plot_results.ipynb
# =========================================================
def NCD_analysis_plot_results():
    NOTEBOOK_NAME = "NCD_analysis_plot_results"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 NCD_analysis_plot_results.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 6. NCD_analysis_plot_results_v2.ipynb
# =========================================================
def NCD_analysis_plot_results_v2():
    NOTEBOOK_NAME = "NCD_analysis_plot_results_v2"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 NCD_analysis_plot_results_v2.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 7. NCD_analysis_incremental_features.ipynb
# =========================================================
def NCD_analysis_incremental_features():
    NOTEBOOK_NAME = "NCD_analysis_incremental_features"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 NCD_analysis_incremental_features.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 8. NCD_analysis_food_only_plot_results_v2.ipynb
# =========================================================
def NCD_analysis_food_only_plot_results_v2():
    NOTEBOOK_NAME = "NCD_analysis_food_only_plot_results_v2"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 NCD_analysis_food_only_plot_results_v2.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 9. microbiome_process.ipynb
# =========================================================
def microbiome_process():
    NOTEBOOK_NAME = "microbiome_process"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 microbiome_process.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 10. intake_analysis_plot_results_v2.ipynb
# =========================================================
def intake_analysis_plot_results_v2():
    NOTEBOOK_NAME = "intake_analysis_plot_results_v2"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 intake_analysis_plot_results_v2.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 11. diet_metabolite_analysis.ipynb
# =========================================================
def diet_metabolite_analysis():
    NOTEBOOK_NAME = "diet_metabolite_analysis"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 diet_metabolite_analysis.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 12. nmr_urine_unify.ipynb
# =========================================================
def nmr_urine_unify():
    NOTEBOOK_NAME = "nmr_urine_unify"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 nmr_urine_unify.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 13. nmr_urine_unify_v2.ipynb
# =========================================================
def nmr_urine_unify_v2():
    NOTEBOOK_NAME = "nmr_urine_unify_v2"
    _enter_project_root()
    original = _wrap_savefig(NOTEBOOK_NAME)

    try:
        # ===== 粘贴 nmr_urine_unify_v2.ipynb 的代码 =====
        pass
    finally:
        _restore_savefig(original)


# =========================================================
# 14. build_prep_data.ipynb
# =========================================================
def build_prep_data():
    NOTEBOOK_NAME = "build_prep_data"
    _enter_project_root()

    try:
        # ===== 粘贴 build_prep_data.ipynb 的代码 =====
        pass
    finally:
        pass


# =========================================================
# Export
# =========================================================
__all__ = [
    "data_analysis",
    "data_analysis_clusters",
    "data_analysis_mktest_v2",
    "data_analysis_mktest_expenditure",
    "NCD_analysis_plot_results",
    "NCD_analysis_plot_results_v2",
    "NCD_analysis_incremental_features",
    "NCD_analysis_food_only_plot_results_v2",
    "microbiome_process",
    "intake_analysis_plot_results_v2",
    "diet_metabolite_analysis",
    "nmr_urine_unify",
    "nmr_urine_unify_v2",
    "build_prep_data",
]
