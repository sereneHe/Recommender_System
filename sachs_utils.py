from os.path import join
import numpy as np
import pandas as pd


def load_data(variant, normalize, data_path):
    if variant == 1:
        filename = '1_cd3cd28'
    elif variant == 2:
        filename = '2_cd3cd28icam2'
    elif variant == 3:
        filename = '3_cd3cd28_aktinhib'
    elif variant == 4:
        filename = '4_cd3cd28_g0076'
    elif variant == 5:
        filename = '5_cd3cd28_psitect'
    elif variant == 6:
        filename = '6_cd3cd28_u0126'
    elif variant == 7:
        filename = '7_cd3cd28_ly'
    elif variant == 8:
        filename = '8_pma'
    elif variant == 9:
        filename = '9_b2camp'
    elif variant == 10:
        filename = '10_cd3cd28icam2_aktinhib'
    else:
        assert False

    df = pd.read_csv(join(data_path, f"{filename}.csv"), header=0)
    if normalize:

        df = df - np.mean(df, axis=0, keepdims=True)
        df = df / np.var(df, axis=0, keepdims=True)
        print('normalizing')

    return df