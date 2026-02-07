from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np


def hei(
        fped: pd.DataFrame,
        diet: pd.DataFrame,
        demo: Optional[pd.DataFrame] = None,
        days: Optional[list[int]] = None,
        agethresh: Optional[int] = None,
        return_full_feats=False
) -> pd.DataFrame:
    """
    Calculate Healthy Eating Index (HEI) scores for NHANES participants.

    Computes HEI scores based on dietary intake data from the National Health and
    Nutrition Examination Survey (NHANES) using Food Pattern Equivalent Database (FPED).

    Parameters
    ----------
    fped : pd.DataFrame
        Food Pattern Equivalent Database data
    diet : pd.DataFrame
        Dietary data from NHANES database
    demograph : pd.DataFrame
        Demographic data from NHANES database
    agethresh : int or float, default 2
        Age threshold in years; participants younger than this value are excluded
    verbose : bool, default False
        If True, return all processed data; if False, return only ID, age, and HEI score

    Returns
    -------
    pd.DataFrame
        DataFrame containing HEI scores and relevant data

    Notes
    -----
    Implementation based on the HEI-2010 scoring algorithm:
    https://www.cnpp.usda.gov/healthyeatingindex
    """
    # Combine datasets and apply legume allocations
    dat = combo(fped, diet, demo, days, agethresh)
    dat = leg_all(dat)

    # Calculate component scores
    _calculate_adequacy_components(dat)
    _calculate_moderation_components(dat)

    # Calculate total HEI score
    component_columns = [
        "heiveg",
        "heibngrn",
        "heitotfrt",
        "heiwholefrt",
        "heiwholegrain",
        "heidairy",
        "heitotpro",
        "heiseaplantpro",
        "heifattyacid",
        "heirefgrain",
        "heisofaas",
        "heisodi",
    ]
    dat["HEI"] = dat[component_columns].sum(axis=1)

    if return_full_feats:
        return dat

    return dat[["SEQN", "DRSTZ", "RIDAGEYR", "TKCAL", "HEI"] + component_columns]


def _calculate_adequacy_components(dat):
    """Calculate HEI adequacy component scores (higher intake = higher score)."""
    # Component definitions: (column_name, numerator, min_value, max_value, max_points)
    adequacy_components = [
        ("heiveg", "lvtotal", 0, 1.1, 5),  # Total vegetables
        ("heibngrn", "lbeangrn", 0, 0.2, 5),  # Beans and greens
        ("heitotfrt", "T_F_TOTAL", 0, 0.8, 5),  # Total fruit
        ("heiwholefrt", "WHOLEFRT", 0, 0.4, 5),  # Whole fruit
        ("heiwholegrain", "T_G_WHOLE", 0, 1.5, 10),  # Whole grains
        ("heidairy", "T_D_TOTAL", 0, 1.3, 10),  # Dairy
        ("heitotpro", "lallmeat", 0, 2.5, 5),  # Total protein
        ("heiseaplantpro", "lseaplant", 0, 0.8, 5),  # Seafood and plant proteins
        ("heifattyacid", "faratio", 1.2, 2.5, 10),  # Fatty acid ratio
    ]

    # Calculate each adequacy component
    for component, numerator, min_value, max_value, max_points in adequacy_components:
        if numerator == "faratio":
            density = np.where(dat["TSFAT"] > 0, dat["MONOPOLY"] / dat["TSFAT"], max_value)
        else:
            density = dat[numerator] / (dat["TKCAL"] / 1000)
        dat[component] = max_points * (density - min_value) / (max_value - min_value)
        dat[component] = dat[component].clip(lower=0, upper=max_points)


def _calculate_moderation_components(dat):
    """Calculate HEI moderation component scores (lower intake = higher score)."""

    # Calculate sodium component
    dat["sodden"] = dat["TSODI"] / dat["TKCAL"]
    dat["heisodi"] = _calculate_reverse_score(dat["sodden"], 1.1, 2.0, 10)

    # Calculate refined grain component
    dat["refgrainnden"] = dat["T_G_REFINED"] / (dat["TKCAL"] / 1000)
    dat["heirefgrain"] = _calculate_reverse_score(dat["refgrainnden"], 1.8, 4.3, 10)

    # Calculate AAS component
    dat["sofa_perc"] = 100 * (dat["EMPTYCAL10"] / dat["TKCAL"])
    dat["heisofaas"] = _calculate_reverse_score(dat["sofa_perc"], 19, 50, 20)


def _calculate_reverse_score(values, lower_bound, upper_bound, max_points):
    """
    Calculate a reverse-scored component where lower values are better.

    Parameters
    ----------
    values : pd.Series
        Input values to score
    lower_bound : float
        Values below this threshold receive maximum points
    upper_bound : float
        Values above this threshold receive zero points
    max_points : int
        Maximum points for this component

    Returns
    -------
    pd.Series
        Calculated component scores
    """
    scores = max_points - (max_points * (values - lower_bound) / (upper_bound - lower_bound))
    scores = scores.clip(lower=0, upper=max_points)

    return scores


def leg_all(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate legumes for HEI scoring.

    Args:
        dat (pd.DataFrame): Data to be processed.

    Returns:
        pd.DataFrame: Data frame of dietary values with legumes allocated appropriately.
    """
    dat["mbmax"] = 2.5 * (dat["TKCAL"] / 1000)

    dat["meatleg"] = np.where(dat["ALLMEAT"] < dat["mbmax"], dat["T_V_LEGUMES"] * 4, 0)

    dat["needmeat"] = np.where(dat["ALLMEAT"] < dat["mbmax"], dat["mbmax"] - dat["ALLMEAT"], 0)

    dat["lallmeat"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["ALLMEAT"] + dat["meatleg"], 0)

    dat["lseaplant"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["SEAPLANT"] + dat["meatleg"], 0)

    dat["lvtotal"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["T_V_TOTAL"], 0)

    dat["lbeangrn"] = np.where(dat["meatleg"] <= dat["needmeat"], dat["T_V_DRKGR"], 0)

    dat["extrameat"] = np.where(dat["meatleg"] > dat["needmeat"], dat["meatleg"] - dat["needmeat"], 0)

    dat["extraleg"] = np.where(dat["meatleg"] > dat["needmeat"], dat["extrameat"] / 4, 0)

    dat["lallmeat"] = np.where(dat["meatleg"] > dat["needmeat"], dat["ALLMEAT"] + dat["needmeat"], dat["lallmeat"])

    dat["lseaplant"] = np.where(dat["meatleg"] > dat["needmeat"], dat["SEAPLANT"] + dat["needmeat"], dat["lseaplant"])

    dat["lvtotal"] = np.where(dat["meatleg"] > dat["needmeat"], dat["T_V_TOTAL"] + dat["extraleg"], dat["lvtotal"])

    dat["lbeangrn"] = np.where(dat["meatleg"] > dat["needmeat"], dat["T_V_DRKGR"] + dat["extraleg"], dat["lbeangrn"])

    dat["lallmeat"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["ALLMEAT"], dat["lallmeat"])

    dat["lseaplant"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["SEAPLANT"], dat["lseaplant"])

    dat["lvtotal"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["T_V_TOTAL"] + dat["T_V_LEGUMES"], dat["lvtotal"])

    dat["lbeangrn"] = np.where(dat["ALLMEAT"] >= dat["mbmax"], dat["T_V_DRKGR"] + dat["T_V_LEGUMES"], dat["lbeangrn"])

    return dat


def combo(
        fped: pd.DataFrame,
        diet: pd.DataFrame,
        demo: Optional[pd.DataFrame] = None,
        days: Optional[list[int]] = None,
        agethresh: Optional[int] = None,
) -> pd.DataFrame:
    dat = pd.merge(fped, diet, on=["SEQN", "DRSTZ"], how="left")
    if demo is not None:
        dat = pd.merge(dat, demo, on="SEQN", how="left")
    # Remove high and low calorie days per HEI guidelines
    mask_by_kcal = dat["TKCAL"].between(600, 5000)
    dat = dat[mask_by_kcal]
    # Include only specified days
    if days is not None:
        days_mask = dat["DRSTZ"].isin(days)
        dat = dat[days_mask]
    # Average across days per visit
    # dat = dat.groupby(["SEQN"], as_index=False).mean()
    # dat.drop(columns=["DRSTZ"], inplace=True)
    if agethresh:
        age_index = dat["RIDAGEYR"] >= agethresh
        dat = dat[age_index]

    dat["WHOLEFRT"] = dat["T_F_CITMLB"] + dat["T_F_OTHER"]
    dat["MONOPOLY"] = dat["TMFAT"] + dat["TPFAT"]
    dat["ALLMEAT"] = dat["T_PF_MPS_TOTAL"] + dat["T_PF_EGGS"] + dat["T_PF_NUTSDS"] + dat["T_PF_SOY"]
    dat["SEAPLANT"] = dat["T_PF_SEAFD_HI"] + dat["T_PF_SEAFD_LOW"] + dat["T_PF_NUTSDS"] + dat["T_PF_SOY"]
    dat["ADDSUGC"] = 16 * dat["T_ADD_SUGARS"]
    dat["SOLFATC"] = 9 * dat["T_SOLID_FATS"]
    dat["MAXALCGR"] = 13 * (dat["TKCAL"] / 1000)
    dat["EXALCCAL"] = np.where(dat["TALCO"] <= dat["MAXALCGR"], 0, 7 * (dat["TALCO"] - dat["MAXALCGR"]))

    dat["EMPTYCAL10"] = dat["ADDSUGC"] + dat["SOLFATC"] + dat["EXALCCAL"]

    return dat


if __name__ == "__main__":
    fped = pd.read_csv("data/processed_fped.csv")
    diet = pd.read_csv("data/processed_diet.csv")
    demo = pd.read_csv("data/processed_demo.csv")

    data = hei(fped, diet, demo, days=[1], agethresh=2)
    data.to_csv("data/my_HEI.csv", index=False)
