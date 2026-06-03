import ast
import os
import time
import zipfile
from os.path import join
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import yaml

from data_helper import load_all_data
from artifact_utils import write_text_artifact, write_yaml_artifact
from experiments_utils import log_system_info, log_params_from_omegaconf_dict
from recommender import run_recommender
from recommender_utils import run_feature_selection_scikit
from utils import log_exceptions, plot_heatmap

logger = logging.getLogger(__name__)


def _dag_layout(graph):
    if not nx.is_directed_acyclic_graph(graph):
        logger.warning("W adjacency graph is not a DAG; using spring layout.")
        return nx.spring_layout(graph, seed=42)

    generations = list(nx.topological_generations(graph))
    pos = {}
    for level, nodes in enumerate(generations):
        width = max(1, len(nodes) - 1)
        for idx, node in enumerate(nodes):
            x = idx - width / 2
            y = -level
            pos[node] = (x, y)
    return pos


def _plot_adjacency_dag(adjacency_labeled, output_path):
    graph = nx.from_pandas_adjacency(adjacency_labeled, create_using=nx.DiGraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    pos = _dag_layout(graph)

    plt.figure(figsize=(9, 6), dpi=300)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1700,
        node_color="#dbeafe",
        edgecolors="#1f2937",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.8,
        edge_color="#374151",
        connectionstyle="arc3,rad=0.04",
        min_source_margin=18,
        min_target_margin=22,
    )
    plt.title("MILP DAG adjacency graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

import hydra


@hydra.main(version_base=None,  config_path="./experiments_conf", config_name="config")
@log_exceptions
def start_experiment(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.environ["RUN_OUTPUT_DIR"] = output_dir
    conf_yaml = OmegaConf.to_yaml(cfg)
    print(conf_yaml)
    logger.info(f"Starting experiment in {output_dir}")
    write_text_artifact("work_dir.txt", output_dir)
    write_text_artifact("config.yaml", conf_yaml)
    log_params_from_omegaconf_dict(cfg)
    log_system_info(HydraConfig.get())

    prep_data = None
    if cfg.problem.name == "codiet":
        food_feats, non_food_feats, prep_data = load_all_data()
        with zipfile.ZipFile("./data/W_est.csv.zip") as z:
            with z.open("W_est.csv") as f:
                w_est = np.loadtxt(f, delimiter=",")
        s = Path("./data/intra_nodes.txt").read_text(encoding="utf-8").strip()
        write_text_artifact("intra_nodes.txt", s)
        row_and_col_names = ast.literal_eval(s)

    if cfg.problem.name == "codiet-select":
        prep_data = pd.read_feather(join(cfg.problem.data_path, "features.feather"))
        prep_data = prep_data.select_dtypes(include=["number"])
        row_and_col_names = prep_data.columns
        with zipfile.ZipFile(join(cfg.problem.data_path, "W_est.csv.zip")) as z:
            with z.open(f"W_est_{cfg.problem.regularization}.csv") as f:
                df = pd.read_csv(f, index_col=0, header=0)
                pd.testing.assert_index_equal(prep_data.columns, df.columns)
                w_est = df.to_numpy()

    if cfg.problem.name == "cds":
        from cds_utils import load_data

        prep_data = load_data(cfg.problem.n, cfg.problem.granularity, cfg.problem.p, cfg.problem.data_path)
    if cfg.problem.name == "Sachs":
        from sachs_utils import load_data
    if cfg.problem.name in ["cds", "Sachs"]:
        prep_data = load_data(cfg.problem.variant, cfg.problem.normalize, cfg.problem.data_path)
        with zipfile.ZipFile(join(cfg.problem.data_path, "W_est.csv.zip")) as z:
            with z.open(f"W_est_{cfg.problem.name}.csv") as f:
                w_est = np.loadtxt(f, delimiter=",")
        row_and_col_names = prep_data.columns
    if cfg.problem.name == "industry_eu":
        from industry_utils import load_data

        prep_data = load_data(
            cfg.problem.data_path,
            frequency=cfg.problem.get("frequency"),
            target=cfg.problem.get("target"),
            features=cfg.problem.get("features"),
            start_date=cfg.problem.get("start_date"),
            end_date=cfg.problem.get("end_date"),
            impute=cfg.problem.get("impute", "none"),
            dropna_selected=cfg.problem.get("dropna_selected", True),
        )
        row_and_col_names = prep_data.columns
        w_est = np.zeros((len(row_and_col_names), len(row_and_col_names)))

    print(row_and_col_names)
    start_process_time = time.process_time()
    start_wall_time = time.perf_counter()
    target_feat = cfg.problem.target
    full_feats = cfg.problem.features
    if "N_SELECT_FEATURES" in cfg.solver:
        n_select_features = cfg.solver.N_SELECT_FEATURES
    else:
        n_select_features = len(full_feats)
    current_column_names = list(prep_data.columns)
    logger.info(
        "Run data summary: target=%s, candidate_features=%d, features=%s",
        target_feat,
        len(current_column_names),
        current_column_names,
    )
    curr_feats, curr_train_errs, curr_test_errs, all_train_errs, all_test_errs, w_est = run_feature_selection_scikit(
        prep_data,
        cfg.solver.model_name,
        cfg.solver.custom_objective,
        target_feat,
        w_est,
        row_and_col_names,
        cfg.solver.n_runs,
        n_select_features,
        full_feats,
        solver_cfg=cfg.solver,
        model_factory=None,
    )

    if w_est is not None:
        intra_nodes = list(curr_feats) + [target_feat]
        w_est_labeled = pd.DataFrame(w_est, index=intra_nodes, columns=intra_nodes)
        adjacency = (np.abs(w_est) > float(cfg.solver.nonzero_threshold)).astype(int)
        adjacency_labeled = pd.DataFrame(adjacency, index=intra_nodes, columns=intra_nodes)
        np.savetxt("W_est.csv", w_est, delimiter=",")
        w_est_labeled.to_csv("W_est_labeled.csv")
        adjacency_labeled.to_csv("W_adjacency_labeled.csv")
        np.savetxt("W_adjacency.csv", adjacency, delimiter=",", fmt="%d")
        logger.info("MILP weighted W_est matrix:\n%s", w_est_labeled.to_string())
        logger.info("MILP DAG adjacency matrix:\n%s", adjacency_labeled.to_string())

    target_feat_clean = target_feat.replace(" ", "_").replace("(", "_").replace(")", "_").replace("/", "_")

    solving_duration = time.process_time() - start_process_time
    wall_duration = time.perf_counter() - start_wall_time
    logger.info(
        "Run result summary: target=%s, selected_features=%s, train_error=%s, test_error=%s, runtime=%ss, wall_runtime=%ss",
        target_feat_clean,
        list(curr_feats),
        curr_train_errs,
        curr_test_errs,
        solving_duration,
        wall_duration,
    )
    logger.info(
        "Run CV summary: train_errs=%s, test_errs=%s",
        all_train_errs.tolist(),
        all_test_errs.tolist(),
    )

    logger.info("Experiment Finished")


if __name__ == "__main__":
    start_experiment()
