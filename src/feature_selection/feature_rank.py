import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import rankdata, norm
from boruta import BorutaPy


def info_features(X, y, top=50, return_raw=False):
    n, p = X.shape

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=1605)

    # mann-whitney u (vectorized via ranks, normal approximation)
    X_arr = X.values.astype(float)
    y_arr = np.asarray(y)
    n0, n1 = (y_arr == 0).sum(), (y_arr == 1).sum()
    ranks = rankdata(X_arr, axis=0)
    R1 = ranks[y_arr == 1].sum(axis=0)
    U1 = R1 - n1 * (n1 + 1) / 2
    U_min = np.minimum(U1, n0 * n1 - U1)
    mu_U = n0 * n1 / 2
    sigma_U = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
    z = (U_min - mu_U) / sigma_U
    mw_pvalues = 2 * norm.cdf(z)

    # auc per variable
    auc_scores = np.zeros(p)
    for i in range(p):
        col = X.iloc[:, i]
        auc = roc_auc_score(y, col)
        auc_scores[i] = max(auc, 1 - auc)

    # boruta
    X_tmp = X.values
    y_tmp = y
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    boruta = BorutaPy(rf, random_state=2205, n_estimators="auto")
    boruta.fit(X_tmp, y_tmp)
    boruta_res = boruta.ranking_

    results_raw = pd.DataFrame(
        {
            "variable": [f"V{i}" for i in range(1, p + 1)],
            "auc": auc_scores,
            "mutual_info": mi_scores,
            "mw_pvalue": mw_pvalues,
            "boruta": boruta_res,
        }
    )

    results = results_raw.copy()

    # higher = better
    for col in ["auc", "mutual_info"]:
        results[f"rank_{col}"] = results[col].rank(ascending=False)

    # lower = better
    for col in ["boruta", "mw_pvalue"]:
        results[f"rank_{col}"] = results[col].rank(ascending=True)

    results["mean_rank"] = results[
        [
            "rank_auc",
            "rank_mutual_info",
            "mw_pvalue",
            "rank_boruta",
        ]
    ].mean(axis=1)
    results = results.sort_values("mean_rank").reset_index(drop=True)
    results["overall_rank"] = results.index + 1
    top = results.head(top)

    if return_raw:
        return top, results_raw
    results.to_csv("feature_importance_rank_new.csv", index=False)
    return top


if __name__ == "__main__":
    X = pd.read_csv("/Users/ola/projects/cost-sensitive-marketing/data/x_train.txt", sep = " ")
    y = pd.read_csv("/Users/ola/projects/cost-sensitive-marketing/data/y_train.txt", sep = " ")
    y = y.values.ravel()
    res = info_features(X, y)