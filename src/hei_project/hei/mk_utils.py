from scipy.stats import spearmanr, kendalltau
import numpy as np
import pandas as pd

##Data cleaning funcs
def clean(v,u):
    v =v.astype(np.float64)
    u =u.astype(np.float64)
    mask = ~np.isnan(u) & ~np.isnan(v)
    # Apply mask to restrict u and v
    u  = u[mask]
    v  = v[mask]    
    
    #assert len(u)>0
    """
    u -= u.mean()
    v -= v.mean()

    u_norm = np.sqrt( (u**2).sum() )
    v_norm = np.sqrt( (v**2).sum() )

    return u/u_norm,v/v_norm
    """
    return v,u
    

def corr(u,v, corr_type=None):    
    
    
    if corr_type == 'spearmanr':
        return spearmanr(u,v)[0]
    elif corr_type == 'kendalltau':
        return kendalltau(u,v)[0]
    elif corr_type == 'pearson':
        return np.corrcoef(u,v)[0,1]
    else:
        raise Exception('unknowns corr type')


def perm_test_pval(u,v, corr_type = None, n_permutes = 1000, seed = None):
    ref_value = corr(u,v,corr_type = corr_type)

    rng = np.random.default_rng(seed=seed)
    
    res = np.array(
        [ corr(rng.permutation(u),v, corr_type=corr_type) for i in range(n_permutes)]
    )
    cnt = (res>=ref_value).sum() if ref_value>0 else (res<=ref_value).sum()
    return cnt/n_permutes



def split_visits_from_visit_labels(df):
    all_columns = df.columns.tolist()    
    visit_3_columns = [col for col in all_columns if "visit_3" in col]
    
    df1 = df[["ID"] + visit_3_columns]
    other_columns = [col for col in all_columns if "visit_3" not in col and col != "ID"]
    
    df2 = df[["ID"] + other_columns]    

    return df1, df2 


def average_visits(df):
    df = df.groupby("ID").mean()
    df.drop(columns=["VISIT"], inplace=True)
    df.reset_index(inplace=True)    
    return df

def split_visits_from_column(df, lst1, lst2):
    df1 = df[df['VISIT'].isin(lst1)]
    df1 = average_visits(df1) if len(lst1)>1 else df1.drop(columns=["VISIT"]).reset_index()

    df2 = df[df['VISIT'].isin(lst2)]
    df2 = average_visits(df2) if len(lst2)>1 else df2.drop(columns=["VISIT"]).reset_index()

    return df1,df2




######### ML classes

from sklearn.neighbors import KNeighborsRegressor

class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        # Get neighbors
        neigh_dist, neigh_ind = self.kneighbors(X)
        
        # Calculate median instead of mean
        predictions = []
        for indices in neigh_ind:
            neighbor_targets = self._y[indices]
            predictions.append(np.median(neighbor_targets))
            
        return np.array(predictions)

def percentile_mask(X, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(X, lower_percentile)
    upper_bound = np.percentile(X, upper_percentile)
    mask = (X >= lower_bound) & (X <= upper_bound)
    return mask

