"""visualize.py

Auto-generated visualization functions extracted from uploaded notebooks.

Rules:
- Prefer original filename in plt.savefig()/fig.savefig() (output file name).
- Force output to: reports/figures/<notebook_name>/<original_filename>
- If a viz cell has no savefig call, save the current figure as:
  reports/figures/<notebook_name>/<function_name>.png
- Function arguments preserve original variable names (inferred from free variables).
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt

# Unified paths come from model.py (single source of truth)
try:
    from model import FIGURE_DIR  # recommended
except Exception:
    try:
        from model import FIGURES_DIR as FIGURE_DIR  # type: ignore
    except Exception:
        FIGURE_DIR = Path("reports/figures")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 2
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def _safe_savefig(save_func, filename, notebook_name: str, *args, **kwargs):
    outdir = Path(FIGURE_DIR) / notebook_name
    _ensure_dir(outdir)

    try:
        base = Path(filename).name
    except Exception:
        base = str(filename)

    outpath = _next_available_path(outdir / base)

    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")

    save_func(outpath, *args, **kwargs)
    return outpath


@contextmanager
def _wrap_savefig(notebook_name: str):
    """Redirect plt.savefig and Figure.savefig into FIGURE_DIR/notebook_name/"""
    import matplotlib.figure as _mf

    orig_plt_savefig = plt.savefig
    orig_fig_savefig = _mf.Figure.savefig

    def plt_savefig(fname, *args, **kwargs):
        return _safe_savefig(orig_plt_savefig, fname, notebook_name, *args, **kwargs)

    def fig_savefig(self, fname, *args, **kwargs):
        def _bound(path, *a, **k):
            return orig_fig_savefig(self, path, *a, **k)
        return _safe_savefig(_bound, fname, notebook_name, *args, **kwargs)

    plt.savefig = plt_savefig  # type: ignore
    _mf.Figure.savefig = fig_savefig  # type: ignore

    try:
        yield
    finally:
        plt.savefig = orig_plt_savefig  # type: ignore
        _mf.Figure.savefig = orig_fig_savefig  # type: ignore


def _save_if_any_fig(notebook_name: str, fname: str):
    nums = plt.get_fignums()
    if not nums:
        return None
    fig = plt.figure(nums[-1])
    outdir = Path(FIGURE_DIR) / notebook_name
    _ensure_dir(outdir)
    outpath = _next_available_path(outdir / fname)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return outpath


# =====================
# Extracted functions
# =====================

def viz_NCD_analysis_food_only_plot_results_v2_cell6(c, k, names_lst):
    """From NCD_analysis_food_only_plot_results_v2.ipynb cell 6"""
    NOTEBOOK_NAME = "NCD_analysis_food_only_plot_results_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(combined_list[-1].mean(axis = 1))[::-1]

        trgt_names = trgt_names[sidx]
        combined_list = [c[sidx] for c in combined_list]


        means_lst = [c.mean(axis = 1) for c in combined_list]
        stds_lst = [c.std(axis = 1) for c in combined_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers ', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        #plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_NCD_analysis_plot_results_ncd_risk_incremental_5feat(c, k, names_lst):
    """From NCD_analysis_plot_results.ipynb cell 6"""
    NOTEBOOK_NAME = "NCD_analysis_plot_results"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(combined_list[-1].mean(axis = 1))[::-1]

        trgt_names = trgt_names[sidx]
        combined_list = [c[sidx] for c in combined_list]


        means_lst = [c.mean(axis = 1) for c in combined_list]
        stds_lst = [c.std(axis = 1) for c in combined_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers (5 features max)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_NCD_analysis_plot_results_ncd_risk_incremental_nolip_lip(k):
    """From NCD_analysis_plot_results.ipynb cell 9"""
    NOTEBOOK_NAME = "NCD_analysis_plot_results"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(food_pred_lipids)[::-1]

        food_pred_lipids = food_pred_lipids[sidx]
        trgt_names = trgt_names[sidx]
        food_pred = food_pred[sidx]



        means_lst = [food_pred, food_pred_lipids]
        stds_lst = [np.zeros(len(food_pred))]*2
        names_lst = ['no lipids','with lipids']

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('ncd_risk_incremental_nolip_lip.png')
        plt.show()
        # --- end cell ---

def viz_NCD_analysis_plot_results_ncd_risk_incremental_5feat(c, k, names_lst):
    """From NCD_analysis_plot_results.ipynb cell 6"""
    NOTEBOOK_NAME = "NCD_analysis_plot_results"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(combined_list[-1].mean(axis = 1))[::-1]

        trgt_names = trgt_names[sidx]
        combined_list = [c[sidx] for c in combined_list]


        means_lst = [c.mean(axis = 1) for c in combined_list]
        stds_lst = [c.std(axis = 1) for c in combined_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers (5 features max)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_NCD_analysis_plot_results_ncd_risk_incremental_nolip_lip(k):
    """From NCD_analysis_plot_results.ipynb cell 9"""
    NOTEBOOK_NAME = "NCD_analysis_plot_results"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(food_pred_lipids)[::-1]

        food_pred_lipids = food_pred_lipids[sidx]
        trgt_names = trgt_names[sidx]
        food_pred = food_pred[sidx]



        means_lst = [food_pred, food_pred_lipids]
        stds_lst = [np.zeros(len(food_pred))]*2
        names_lst = ['no lipids','with lipids']

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('ncd_risk_incremental_nolip_lip.png')
        plt.show()
        # --- end cell ---

def viz_NCD_analysis_plot_results_v2_ncd_risk_incremental_5feat(c, k, names_lst):
    """From NCD_analysis_plot_results_v2.ipynb cell 4"""
    NOTEBOOK_NAME = "NCD_analysis_plot_results_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(combined_list[-1].mean(axis = 1))[::-1]

        trgt_names = trgt_names[sidx]
        combined_list = [c[sidx] for c in combined_list]


        means_lst = [c.mean(axis = 1) for c in combined_list]
        stds_lst = [c.std(axis = 1) for c in combined_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers (5 features max)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_data_analysis_cell8(evaluation_results):
    """From data_analysis.ipynb cell 8"""
    NOTEBOOK_NAME = "data_analysis"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        # %matplotlib widget
        plt.figure()
        tmp = plt.hist(evaluation_results.validation_mse["Full Data"], 50, alpha=0.5)
        tmp = plt.hist(evaluation_results.validation_mse["Predict 0 Fold Change"], 50, alpha=0.5)
        plt.show()
        plt.figure()
        sidx = np.argsort(evaluation_results.validation_mse["Full Data"])
        plt.plot(evaluation_results.validation_mse["Full Data"][sidx], "-o")
        plt.plot(evaluation_results.validation_mse["Predict 0 Fold Change"][sidx], "-o")
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_cell8.png")

def viz_data_analysis_clusters_cell8(unique_hei_data):
    """From data_analysis_clusters.ipynb cell 8"""
    NOTEBOOK_NAME = "data_analysis_clusters"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        hei_delta = unique_hei_data.groupby("ID")["HEI"].diff()

        plt.figure(figsize=(10, 6))
        hei_delta.to_frame().hist(bins=20)
        plt.xlabel("HEI Delta")
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_clusters_cell8.png")

def viz_data_analysis_clusters_wp2_intake24_summ(food_data, n, orig_diagonal, r, zz_matrix, zz_mnp):
    """From data_analysis_clusters.ipynb cell 13"""
    NOTEBOOK_NAME = "data_analysis_clusters"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        food_code_name_map_df = food_data[['Food group code','Food group (en)']].drop_duplicates().dropna()
        food_code_name_map = dict([ (int(r[0]),r[1]) for r in food_code_name_map_df.to_numpy()])
        food_codes = zz_matrix.columns.to_numpy().astype(int)

        n_show = 50
        n_trunc = 20
        font_size = 25
        sidx = np.argsort(orig_diagonal)[::-1]

        plt.figure(figsize=(30, 15))
        plt.plot(orig_diagonal[sidx][:n_show],'-o', markersize = 20)
        plt.xticks(
            range(len(sidx))[:n_show],
            [food_code_name_map[food_codes[n]][:n_trunc] for n in sidx][:n_show],
            rotation=90,
            fontsize = font_size
        )
        plt.yticks(fontsize=font_size)

        plt.grid()
        #plt.minorticks_on()

        plt.xlabel('Food Group', fontsize=font_size)
        plt.ylabel('Meal Count', fontsize=font_size)
        plt.title(f'WP2 Intake24 Summary, {zz_mnp.shape[0]} Meals', fontsize = font_size)
        plt.tight_layout()

        plt.savefig('wp2_intake24_summ.png')
        plt.show()
        # --- end cell ---

def viz_data_analysis_clusters_wp2_intake24_item_counts(person_food_counts):
    """From data_analysis_clusters.ipynb cell 16"""
    NOTEBOOK_NAME = "data_analysis_clusters"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        plt.plot((person_food_counts).sum(axis=0),'-o')
        plt.tight_layout()

        plt.savefig('wp2_intake24_item_counts.png')
        plt.show()
        # --- end cell ---

def viz_data_analysis_clusters_wp2_subject_cluster_by_food(labels, spectral_clustering):
    """From data_analysis_clusters.ipynb cell 20"""
    NOTEBOOK_NAME = "data_analysis_clusters"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(labels)
        plt.imshow(spectral_clustering.affinity_matrix_[sidx,:][:,sidx], interpolation = None)
        plt.colorbar()
        plt.savefig('wp2_subject_cluster_by_food.png')
        plt.show()
        # --- end cell ---

def viz_data_analysis_mktest_expenditure_cell11(X, Y, model):
    """From data_analysis_mktest_expenditure.ipynb cell 11"""
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        plt.figure()
        u = np.linspace(X[:,0].min(),X[:,0].max(),100)
        plt.plot(u,model.predict(u[:,None]),'-o')
        #plt.plot(u,model.predict(np.concatenate([u[:,None],u[:,None]**2,u[:,None]**3 ],axis = 1)),'-o')

        #plt.hist(base_data['heitotpro'],20, alpha = 0.5)
        plt.plot(X[:,0],Y,'x')
        plt.grid()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_mktest_expenditure_cell11.png")

def viz_data_analysis_mktest_expenditure_cell12(X, Y, model):
    """From data_analysis_mktest_expenditure.ipynb cell 12"""
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        plt.figure()
        plt.plot(model.predict(X),Y,'o')
        plt.grid()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_mktest_expenditure_cell12.png")

def viz_data_analysis_mktest_expenditure_cell14(predictor_test_corrs, predictor_train_corrs):
    """From data_analysis_mktest_expenditure.ipynb cell 14"""
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        plt.figure()
        sidx = np.argsort(predictor_test_corrs)
        #sidx = np.argsort(bench_test_errors)

        #plt.plot(np.array(bench_test_errors)[sidx],'-o', label = 'bench test err')
        #plt.plot(np.array(test_errors)[sidx],'-o', label = 'forest test err')
        plt.plot(np.array(predictor_train_corrs)[sidx],'-o')
        plt.plot(np.array(predictor_test_corrs)[sidx],'-o')

        #plt.plot(np.array(predictor_test_corr_sigs)[sidx],'-o')
        plt.legend()
        plt.grid()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_mktest_expenditure_cell14.png")

def viz_data_analysis_mktest_expenditure_cell15(X, Y, mask):
    """From data_analysis_mktest_expenditure.ipynb cell 15"""
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        u = X[:,0]
        v = Y
        sidx = np.argsort(u)
        u = u[sidx]
        v = v[sidx]

        v_orig = v.copy()
        u_orig = u.copy()

        #v = np.clip(v, -0.1, 0.1)
        #mask = np.abs(v)>=4
        #v = v[~mask]
        #u = u[~mask]

        cs = v.cumsum()
        L = 20
        v_smoother = (cs[L:] - cs[:-L])/L

        plt.figure()
        plt.plot(u[L:],v_smoother,'-o', label = 'running avrg. predictor',alpha = 0.5)
        plt.plot(u_orig[mask],v_orig[mask],'o', label = f'data (large change) ({mask.sum()})')
        plt.plot(u,v,'o', label = f'data (small change) ({v.shape[0]})', alpha = 0.5)

        plt.grid()

        #plt.xlabel('Gut Microbiome Wellnes Index V1')
        #plt.ylabel('hba1c change (V1->V3), %')

        plt.legend()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_mktest_expenditure_cell15.png")

def viz_data_analysis_mktest_expenditure_cell20(base_data):
    """From data_analysis_mktest_expenditure.ipynb cell 20"""
    NOTEBOOK_NAME = "data_analysis_mktest_expenditure"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        z = base_data[['GMWI', 'hba1c']].to_numpy()
        u,v = z[:,0],z[:,1]
        plt.figure()
        plt.plot(u,v,'o')
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_data_analysis_mktest_expenditure_cell20.png")

def viz_data_analysis_mktest_v2_cell5():
    """From data_analysis_mktest_v2.ipynb cell 5"""
    NOTEBOOK_NAME = "data_analysis_mktest_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        """
        diet_columns = [s for s in hei_data.columns]

        def show_corrs(base_data, base_delta, base_columns, target_col,
                       corr_type,
                       n_permutes = 1000,
                        n_show = 30,
                        fig_name = None,
                      ):

            title_str = f'{corr_type } correlations with {target_col}'

            corrs = np.ones(len(base_columns))
            pvals = np.ones(len(base_columns))
            #n_permutes = 1000

            for i,cb in enumerate(base_columns):

                try:
                    u,v = clean(base_data[cb].to_numpy(), base_delta[target_col].to_numpy())

                    split_seed = np.random.randint(1000000)
                    u_train, u_test, v_train, v_test = train_test_split(u,v,
                        test_size=0.3,
                        random_state=split_seed
                    )

                    assert len(v_test)>=30, f'len(v_test) <=30'

                    cls = MedianKNNRegressor(n_neighbors=20)
                    cls.fit(u_train[:,None],v_train)
                    u = cls.predict(u_test[:,None])
                    v = v_test


                    corrs[i] = corr(u,v,corr_type =  corr_type)
                    pvals[i] = perm_test_pval(u,v, corr_type = corr_type, n_permutes = n_permutes)

                    #happens when u or v is constant
                    if np.isnan(corrs[i]):
                        corrs[i] = 0
                        pvals[i] = 1.


                except Exception as e:
                    corrs[i] = 0
                    pvals[i] = 1.

                    print(f'Error at: {cb}->{target_col}: {e}')


            base_columns = np.array(base_columns)

            ok_pvals = pvals<=0.05
            corrs = corrs[ok_pvals]
            base_columns = base_columns[ok_pvals]
            pvals = pvals[ok_pvals]

            n_show = min(n_show, len(pvals))
            if n_show == 0:
                print(f'NO GOOD PVALS FOR {title_str}')
                return

            sidx = np.argsort(np.abs(corrs))[::-1]
            scorrs = corrs[sidx][:n_show]
            scols = base_columns[sidx][:n_show]
            spvals = pvals[sidx][:n_show]

            sidx = np.argsort(scorrs)[::-1]
            scorrs = scorrs[sidx]
            scols = scols[sidx]
            spvals = spvals[sidx]




            fig, ax1 = plt.subplots()

            ax1.plot(np.arange(n_show),scorrs,'b-o', label = 'corr')
            ax1.set_ylabel('corr', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax1.set_xticks(np.arange(n_show))
            ax1.set_xticklabels([s[:20] for s in scols], rotation = 90)

            ax1.plot(np.arange(n_show),np.zeros(n_show),'C1--')

            for i,label in enumerate(ax1.get_xticklabels()):
                if scols[i] in diet_columns:
                #if 'hei' in scols[i]:
                    label.set_color('green')


            ax2 = ax1.twinx()
            ax2.plot(np.arange(n_show),spvals,'r-o', label = 'pvals')
            ax2.set_ylabel(f'pvals (n_permute = {n_permutes})', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            plt.title(title_str)

            ax1.grid(True)

            plt.tight_layout()

            if fig_name is not None:
                fig.savefig(fig_name+".png", dpi=300, bbox_inches='tight')

            plt.show()

        """
        True
        # --- end cell ---

def viz_data_analysis_mktest_v2_cell7(BaggingRegressor, MedianKNNRegressor, ax1, cb, clean, cls, corr, corrs, e, fig, hei_data, i, label, meta_rnd_corrs, meta_rnd_stds, ok_pvals, p_mask, percentile_mask, pvals, res_vals, rnd_corrs, rnd_perm, rnd_pvals, rnd_res_vals, s, scols, scorrs, sidx, split_seed, spvals, srnd_corrs, srnd_pvals, title_str, train_test_split, u, u_full, u_test, u_train, v, v_full, v_test, v_train):
    """From data_analysis_mktest_v2.ipynb cell 7"""
    NOTEBOOK_NAME = "data_analysis_mktest_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        diet_columns = [s for s in hei_data.columns]
        more_special_columns = ['height','hei_GMWI', 'fatigue_index' ]

        #this version computes the correlations of a classifier with data on test
        def show_corrs(base_data, base_delta, base_columns, target_col,
                       corr_type,
                       n_permutes = 1000,
                        n_show = 30,
                        fig_name = None,
                       n_cv_rounds = 100,
                       rnd_rounds = 0
                      ):

            title_str = f'{corr_type } correlations with {target_col}'

            corrs = np.zeros(len(base_columns))
            pvals = np.ones(len(base_columns))

            rnd_corrs = np.zeros(len(base_columns))
            rnd_pvals = np.ones(len(base_columns))*0.1


            for i,cb in enumerate(base_columns):

                try:
                    u_full,v_full = clean(base_data[cb].to_numpy(), base_delta[target_col].to_numpy())

                    #remove some possible outliers
                    p_mask = percentile_mask(v_full,3,100-3)
                    u_full = u_full[p_mask]
                    v_full = v_full[p_mask]

                    res_vals = []
                    for _ in range(n_cv_rounds):



                        split_seed = np.random.randint(1000000)
                        u_train, u_test, v_train, v_test = train_test_split(u_full,v_full,
                            test_size=0.3,
                            random_state=split_seed
                        )

                        assert len(v_test)>=30, f'len(v_test) <=30'

                        #cls = MedianKNNRegressor(n_neighbors=20)
                        cls = BaggingRegressor(
                                estimator =  MedianKNNRegressor(n_neighbors=20),
                                n_estimators =  5,
                                #"random_state" : 42
                        )
                        cls.fit(u_train[:,None],v_train)
                        u = cls.predict(u_test[:,None])
                        v = v_test

                        res_vals.append(corr(u,v,corr_type =  corr_type))

                    res_vals = np.array(res_vals)
                    corrs[i] = res_vals.mean()
                    pvals[i] = res_vals.std()

                    #now the random classifier test

                    meta_rnd_corrs = []
                    meta_rnd_stds = []


                    for _ in range(rnd_rounds):
                        rnd_res_vals = []
                        rnd_perm = np.random.permutation(len(v_full))

                        for _ in range(n_cv_rounds):

                            split_seed = np.random.randint(1000000)
                            u_train, u_test, v_train, v_test = train_test_split(
                                u_full,
                                v_full[rnd_perm],
                                test_size=0.3,
                                random_state=split_seed
                            )

                            assert len(v_test)>=30, f'len(v_test) <=30'

                            cls = MedianKNNRegressor(n_neighbors=20)
                            cls.fit(u_train[:,None],v_train)
                            u = cls.predict(u_test[:,None])
                            v = v_test

                            rnd_res_vals.append(corr(u,v,corr_type =  corr_type))

                            meta_rnd_corrs.append(np.array(rnd_res_vals).mean())
                            meta_rnd_stds.append(np.array(rnd_res_vals).std())



                    if rnd_rounds > 0:
                        rnd_corrs[i] = np.array(meta_rnd_corrs).mean()
                        rnd_pvals[i] = np.array(meta_rnd_stds).mean()


                    #happens when u or v is constant
                    if np.isnan(corrs[i]):
                        corrs[i] = 0
                        pvals[i] = 1.
                        rnd_corrs[i] = 0
                        rnd_pvals[i] = 1.

                except Exception as e:
                    corrs[i] = 0
                    pvals[i] = 1.
                    rnd_corrs[i] = 0
                    rnd_pvals[i] = 1.

                    print(f'Error at: {cb}->{target_col}: {e}')


            base_columns = np.array(base_columns)

            ok_pvals = pvals<=0.35
            corrs = corrs[ok_pvals]
            base_columns = base_columns[ok_pvals]
            pvals = pvals[ok_pvals]
            rnd_corrs = rnd_corrs[ok_pvals]
            rnd_pvals = rnd_pvals[ok_pvals]

            n_show = min(n_show, len(pvals))
            if n_show == 0:
                print(f'NO GOOD PVALS FOR {title_str}')
                return

            sidx = np.argsort(corrs)[::-1]
            scorrs = corrs[sidx][:n_show]
            scols = base_columns[sidx][:n_show]
            spvals = pvals[sidx][:n_show]

            srnd_corrs = rnd_corrs[sidx][:n_show]
            srnd_pvals = rnd_pvals[sidx][:n_show]




            fig, ax1 = plt.subplots()

            ax1.plot(np.arange(n_show),scorrs,'b-o', label = 'corr')
            ax1.plot(np.arange(n_show),scorrs+spvals,'r--o', label = 'stds')
            ax1.plot(np.arange(n_show),scorrs-spvals,'r--o')


            ax1.set_ylabel('corr', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax1.set_xticks(np.arange(n_show))
            ax1.set_xticklabels([s[:20] for s in scols], rotation = 90)

            ax1.plot(np.arange(n_show),srnd_corrs,'k-o', label = 'rnd corr')
            ax1.plot(np.arange(n_show),srnd_corrs+srnd_pvals,'k--o', label = 'rnd stds')
            ax1.plot(np.arange(n_show),srnd_corrs-srnd_pvals,'k--o')


            ax1.plot(np.arange(n_show),np.zeros(n_show),'C1--')


            for i,label in enumerate(ax1.get_xticklabels()):
                if scols[i] in diet_columns:
                    label.set_color('green')
                if scols[i] in more_special_columns:
                    label.set_color('red')



            plt.title(title_str)

            ax1.grid(True)

            plt.tight_layout()

            if fig_name is not None:
                fig.savefig(fig_name+".png", dpi=300, bbox_inches='tight')

            plt.show()
        # --- end cell ---

def viz_diet_metabolite_analysis_diet_prediction_plot(composition_names, t, target_columns, test_results_by_comp_type):
    """From diet_metabolite_analysis.ipynb cell 16"""
    NOTEBOOK_NAME = "diet_metabolite_analysis"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #species = target_columns
        #penguin_means =  test_results_by_comp_type

        spacing_factor = 50 # Increase this for more space between groups
        x = np.arange(len(target_columns))*spacing_factor  # the label locations
        width = 10  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute in composition_names:
            measurement = (1.-test_results_by_comp_type[attribute])*100

            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('% Variance Reduction')
        ax.set_title('Diet Prediction By Variety of Features')
        ax.set_xticks(x + width, [t[len('normed_food_'):] for t in target_columns], rotation = 90)

        #ax.legend(bbox_to_anchor=(0.5, 1.02), loc='center', ncols=len(composition_names))

        ax.legend(loc='upper left', ncols = 4)
        ax.set_ylim(0, 24)
        plt.grid()

        plt.savefig('diet_prediction_plot.png')
        plt.show()
        # --- end cell ---

def viz_intake_analysis_plot_results_v2_cell6(c, k, names_lst):
    """From intake_analysis_plot_results_v2.ipynb cell 6"""
    NOTEBOOK_NAME = "intake_analysis_plot_results_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        sidx = np.argsort(combined_list[-1].mean(axis = 1))[::-1]

        trgt_names = trgt_names[sidx]
        combined_list = [c[sidx] for c in combined_list]


        means_lst = [c.mean(axis = 1) for c in combined_list]
        stds_lst = [c.std(axis = 1) for c in combined_list]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(trgt_names))

        #ax.bar(x_pos, pred_vals_lipids,  alpha=0.8)
        #ax.bar(x_pos, pred_vals_lipids_v1,  alpha=0.8)
        """
        ax.bar(x_pos, pv_lipids_means, alpha=0.8,
               yerr=pv_lipids_std, capsize=5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})
        """

        n_bars = len(means_lst)

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx, (label, means, errors) in enumerate(zip(names_lst,means_lst,stds_lst)):
            x_positions = x_pos + offsets[idx]

            ax.bar(x_positions, means, bar_width, alpha=0.8, yerr=errors,
                   capsize=5, label=label,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})



        ax.set_xticks(x_pos)
        ax.set_xticklabels([k[:20] for k in trgt_names], rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('Var Reduction (%)', fontsize=12)
        ax.set_title('Risk Prediction From Biomarkers (5 features max)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        #plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_microbiome_process_cell20(X, cluster_labels_orig, h_list):
    """From microbiome_process.ipynb cell 20"""
    NOTEBOOK_NAME = "microbiome_process"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #cmap = plt.cm.tab10
        #colors = [cmap(i) for i in range(20)]

        fig, ax = plt.subplots(figsize=(10, 13))
        x_pos = np.arange(len(h_list))


        n_bars = max(cluster_labels_orig)+1

        # Calculate bar width and positions
        bar_width = 0.8 / n_bars  # 0.8 to leave some space between groups

        # Generate positions for each bar set
        offsets = np.linspace(-(n_bars-1)*bar_width/2, (n_bars-1)*bar_width/2, n_bars)

        # Plot each bar set
        for idx in range(n_bars):
            x_positions = x_pos + offsets[idx]

            means = X[cluster_labels_orig == idx].mean(axis = 0)

            ax.bar(x_positions, means, bar_width, alpha=0.8,
                   capsize=5, label=f'cl: {idx}',
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7},
                   #color = colors[idx]
                  )



        ax.set_xticks(x_pos)
        ax.set_xticklabels(h_list, rotation = 90, fontsize=12)
        #ax.set_xlabel('Targets', fontsize=12)
        ax.set_ylabel('proportion', fontsize=12)
        ax.set_title('cluster means', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        #plt.savefig('ncd_risk_incremental_5feat.png')
        plt.show()
        # --- end cell ---

def viz_nmr_urine_unify_cell15(cic_df, icl_df):
    """From nmr_urine_unify.ipynb cell 15"""
    NOTEBOOK_NAME = "nmr_urine_unify"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #a few sanity checks


        numeric_cols = cic_df.select_dtypes(include='number').columns
        print(f'Got {len(numeric_cols)} numeric columns')

        cic_medians = cic_df[numeric_cols].median()
        icl_medians = icl_df[numeric_cols].median()

        plt.figure()
        plt.plot(cic_medians.index, cic_medians.values,'-o',label = 'cic')
        plt.plot(icl_medians.index, icl_medians.values,'-o',label = 'icl')
        plt.xlabel('Columns')
        plt.ylabel('Median Value')
        plt.title('Median Values Per Metabolite')
        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.legend()
        plt.grid()
        plt.tight_layout()  # Prevent label cutoff
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_nmr_urine_unify_cell15.png")

def viz_nmr_urine_unify_cell17(cic_df, icl_df):
    """From nmr_urine_unify.ipynb cell 17"""
    NOTEBOOK_NAME = "nmr_urine_unify"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #metabolite = numeric_cols[1]
        #metabolite = 'acetic-acid'
        #metabolite = 'citric-acid'
        #metabolite = 'taurine'
        #metabolite = 'arginine'
        #metabolite = 'trimethylamine'
        metabolite = 'lactic-acid'


        #metabolite = 'creatinine'


        plt.figure()
        plt.title(metabolite)

        all_vals = np.concat([cic_df[metabolite],icl_df[metabolite]])
        counts1, bins = np.histogram(all_vals, bins='auto')

        plt.hist(cic_df[metabolite],bins = bins, density=True, alpha = .5, label ='cic')

        #counts1, bins = np.histogram(icl_df[metabolite], bins='auto')
        plt.hist(icl_df[metabolite],bins = bins, density=True, alpha = .5, label = 'icl')
        plt.legend()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_nmr_urine_unify_cell17.png")

def viz_nmr_urine_unify_v2_cell18(cic_df, icl_df):
    """From nmr_urine_unify_v2.ipynb cell 18"""
    NOTEBOOK_NAME = "nmr_urine_unify_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #a few sanity checks


        numeric_cols = cic_df.select_dtypes(include='number').columns
        print(f'Got {len(numeric_cols)} numeric columns')

        cic_medians = cic_df[numeric_cols].median()
        icl_medians = icl_df[numeric_cols].median()

        plt.figure()
        plt.plot(cic_medians.index, cic_medians.values,'-o',label = 'cic')
        plt.plot(icl_medians.index, icl_medians.values,'-o',label = 'icl')
        plt.xlabel('Columns')
        plt.ylabel('Median Value')
        plt.title('Median Values Per Metabolite')
        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.legend()
        plt.grid()
        plt.tight_layout()  # Prevent label cutoff
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_nmr_urine_unify_v2_cell18.png")

def viz_nmr_urine_unify_v2_cell20(cic_df, icl_df):
    """From nmr_urine_unify_v2.ipynb cell 20"""
    NOTEBOOK_NAME = "nmr_urine_unify_v2"
    with _wrap_savefig(NOTEBOOK_NAME):
        # --- extracted from notebook cell ---
        #metabolite = numeric_cols[1]
        #metabolite = 'acetic-acid'
        #metabolite = 'citric-acid'
        #metabolite = 'taurine'
        #metabolite = 'arginine'
        #metabolite = 'trimethylamine'
        metabolite = 'lactic-acid'


        #metabolite = 'creatinine'


        plt.figure()
        plt.title(metabolite)

        all_vals = np.concatenate([cic_df[metabolite],icl_df[metabolite]])
        counts1, bins = np.histogram(all_vals, bins='auto')

        plt.hist(cic_df[metabolite],bins = bins, density=True, alpha = .5, label ='cic')

        #counts1, bins = np.histogram(icl_df[metabolite], bins='auto')
        plt.hist(icl_df[metabolite],bins = bins, density=True, alpha = .5, label = 'icl')
        plt.legend()
        plt.show()
        # --- end cell ---
        _save_if_any_fig(NOTEBOOK_NAME, "viz_nmr_urine_unify_v2_cell20.png")

__all__ = [
    "viz_NCD_analysis_food_only_plot_results_v2_cell6",
    "viz_NCD_analysis_plot_results_ncd_risk_incremental_5feat",
    "viz_NCD_analysis_plot_results_ncd_risk_incremental_nolip_lip",
    "viz_NCD_analysis_plot_results_ncd_risk_incremental_5feat",
    "viz_NCD_analysis_plot_results_ncd_risk_incremental_nolip_lip",
    "viz_NCD_analysis_plot_results_v2_ncd_risk_incremental_5feat",
    "viz_data_analysis_cell8",
    "viz_data_analysis_clusters_cell8",
    "viz_data_analysis_clusters_wp2_intake24_summ",
    "viz_data_analysis_clusters_wp2_intake24_item_counts",
    "viz_data_analysis_clusters_wp2_subject_cluster_by_food",
    "viz_data_analysis_mktest_expenditure_cell11",
    "viz_data_analysis_mktest_expenditure_cell12",
    "viz_data_analysis_mktest_expenditure_cell14",
    "viz_data_analysis_mktest_expenditure_cell15",
    "viz_data_analysis_mktest_expenditure_cell20",
    "viz_data_analysis_mktest_v2_cell5",
    "viz_data_analysis_mktest_v2_cell7",
    "viz_diet_metabolite_analysis_diet_prediction_plot",
    "viz_intake_analysis_plot_results_v2_cell6",
    "viz_microbiome_process_cell20",
    "viz_nmr_urine_unify_cell15",
    "viz_nmr_urine_unify_cell17",
    "viz_nmr_urine_unify_v2_cell18",
    "viz_nmr_urine_unify_v2_cell20",
]
