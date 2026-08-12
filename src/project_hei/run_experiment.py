# run_experiment.py


import hydra
from omegaconf import DictConfig
from hei_package.experiment import run_experiment_from_config
from hei_package.data_preparation import build_prep_data

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    prep_data, feature_groups = build_prep_data(cfg)
    food_feats = feature_groups["food"]
    non_food_feats = feature_groups["non_food"]

    feats, train_errs, test_errs = run_experiment_from_config(
        cfg,
        prep_data,
        food_feats,
        non_food_feats,
    )

    print("Selected features:")
    print(feats)

if __name__ == "__main__":
    main()
