import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

def ncd_food_conditioning_analysis(prep_data, food_feats, non_food_feats, target_columns, model_type='gp', n_runs=50, n_select_features=5):
    """
    多模态健康数据特征选择与建模分析
    prep_data: 预处理后的DataFrame
    food_feats: 食物相关特征列表
    non_food_feats: 其他特征列表
    target_columns: 目标变量列表
    model_type: 'gp'（高斯过程）或 'linreg'（线性回归）
    n_runs: 重复次数
    n_select_features: 特征选择数量
    """
    # 构建特征集
    full_feats = food_feats + [
        'age','gender_numeric', 'stress_index','fatigue_index','mean_hrt', 'site_continental',
        'weight', 'height', 'GMWI', 'microbiome_Shannon'
    ] + [c for c in non_food_feats if 'dbs_rbc_lip' in c] + [n for n in prep_data.columns if 'microb_clean15_' in n]
    full_feats = full_feats[:20]  # 可调整

    # 选择目标变量
    target_col = target_columns[0]

    # 构建模型
    if model_type == 'gp':
        kernel = RBF(length_scale=[1.0]*len(full_feats))
        model = Pipeline([
            ("scale", StandardScaler()),
            ("GP", GaussianProcessRegressor(kernel=kernel, alpha=1., normalize_y=True, n_restarts_optimizer=3))
        ])
    else:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("linreg", LinearRegression())
        ])

    # 特征选择与模型评估
    results = []
    for run in range(n_runs):
        X = prep_data[full_feats]
        y = prep_data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results.append((train_score, test_score))

    return results