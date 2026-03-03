import pandas as pd
from omegaconf import OmegaConf

from hei_project.model import build_full_feats


def test_build_full_feats_filters_missing_columns_and_deduplicates():
    prep_data = pd.DataFrame(
        columns=[
            "food_a",
            "age",
            "height",
            "dbs_rbc_lip_x",
            "microb_clean15_alpha",
        ]
    )
    result = build_full_feats(
        prep_data=prep_data,
        food_feats=["food_a", "food_missing"],
        non_food_feats=["dbs_rbc_lip_x", "dbs_rbc_lip_x", "other"],
    )
    assert result == ["food_a", "age", "height", "dbs_rbc_lip_x", "microb_clean15_alpha"]


def test_build_full_feats_respects_cfg_driven_feature_selection():
    prep_data = pd.DataFrame(
        columns=["food_a", "extra_1", "weight", "dbs_rbc_lip_x", "microb_custom_beta"]
    )
    cfg = OmegaConf.create(
        {
            "features": {
                "extra_base": ["extra_1"],
                "include_anthropometrics": True,
                "anthropometrics": ["weight"],
                "extra_non_food": ["dbs_rbc_lip_x"],
                "include_dbs_rbc_lip": False,
                "include_prefixes": ["microb_custom_"],
            }
        }
    )
    result = build_full_feats(
        prep_data=prep_data,
        food_feats=["food_a"],
        non_food_feats=["dbs_rbc_lip_x"],
        cfg=cfg,
    )
    assert result == ["food_a", "extra_1", "weight", "dbs_rbc_lip_x", "microb_custom_beta"]
