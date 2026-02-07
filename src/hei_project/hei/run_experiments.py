import ast
import time
import zipfile
from os.path import join
import logging
from pathlib import Path

import numpy as np
from hydra.core.hydra_config import HydraConfig

from data_helper import load_all_data
from experiments_utils import log_system_info, log_params_from_omegaconf_dict
from recommender import run_recommender
from utils import log_exceptions

logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import log_metric, log_metrics, log_param, log_artifact, log_text, log_table


@hydra.main(version_base=None,  config_path="./experiments_conf", config_name="config")
@log_exceptions
def start_experiment(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    mlflow.set_experiment(cfg.experiment) #mlflow.set_experiment(experiment_name=cfg.experiment)
    conf_yaml = OmegaConf.to_yaml(cfg)
    print(conf_yaml)
    app_config = OmegaConf.to_container(cfg)
    with mlflow.start_run() as mlrun:
        logger.info(f'Starting experiment in {output_dir}')
        log_text(output_dir, 'work_dir.txt')
        with open(join(output_dir, 'config.yaml'), 'w') as f:
            f.write(conf_yaml)
        log_artifact(join(output_dir, 'config.yaml'))
        log_params_from_omegaconf_dict(cfg)
        log_system_info(HydraConfig.get())

        food_feats, non_food_feats, prep_data = load_all_data()
        with zipfile.ZipFile("./data/W_est.csv.zip") as z:
            with z.open("W_est.csv") as f:
                w_est = np.loadtxt(f, delimiter=",")
            # w_est = np.loadtxt("./data/W_est.csv", delimiter=",")
        mlflow.log_artifact("./data/W_est.csv.zip")
        s = Path('./data/intra_nodes.txt').read_text(encoding="utf-8").strip()
        mlflow.log_text(s, 'intra_nodes.txt')
        row_and_col_names = ast.literal_eval(s) #[x.strip() for x in s.strip("[]").split(",")]

        print(row_and_col_names)
        start_time = time.time()



        result = run_recommender(food_feats, non_food_feats, prep_data, w_est, row_and_col_names, cfg.solver.model_name, cfg.solver.custom_objective, cfg.solver.N_SELECT_FEATURES, cfg.solver.n_runs)

        print(result)

        assert len(result) == 1

        for target_feat, (curr_feats, curr_train_errs, curr_test_errs) in result.items():

            target_feat = target_feat.replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_')
            mlflow.log_text('['+", ".join(curr_feats) + "]", target_feat + "_vs_selected_feats_list.txt")

            train_err = curr_train_errs[-1]
            test_err = curr_test_errs[-1]
            log_metric(f'{target_feat}_train_err', train_err)
            log_metric(f'{target_feat}_test_err', test_err)


        solving_duration = time.time() - start_time

        log_metric('runtime', solving_duration)

        logger.info(f'Experiment Finished')


if __name__ == "__main__":
    start_experiment()
