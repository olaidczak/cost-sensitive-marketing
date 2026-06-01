import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
import time

SEED = 3105
SEED_MODELS = SEED
SEED_RANDOM_SEARCH = SEED
def get_models():
    models = {
        "XGBoost": (
            XGBClassifier(random_state=SEED_MODELS, verbosity=0, eval_metric="logloss", seed=SEED_MODELS),
            {
                "clf__n_estimators": randint(200, 600),
                "clf__max_depth": randint(3, 7),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=SEED_MODELS, verbose=-1),
            {
                "clf__n_estimators": randint(200, 600),
                "clf__max_depth": randint(3, 7),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__num_leaves": randint(15, 63),
                "clf__subsample": uniform(0.6, 0.4),
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED_MODELS, n_jobs=-1),
            {
                "clf__n_estimators": randint(200, 600),
                "clf__max_depth": randint(4, 12),
                "clf__min_samples_leaf": randint(1, 20),
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=SEED_MODELS, n_jobs=-1),
            {
                "clf__n_estimators": randint(200, 600),
                "clf__max_depth": randint(4, 12),
                "clf__min_samples_leaf": randint(1, 20),
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=SEED_MODELS),
            {
                "clf__n_estimators": randint(100, 400),
                "clf__max_depth": randint(2, 5),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__subsample": uniform(0.6, 0.4),
            },
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True, random_state=SEED_MODELS),
            {"clf__C": loguniform(0.1, 100), "clf__gamma": loguniform(1e-3, 1)},
        ),
        "LogRegl1": (
            LogisticRegression(solver="saga", l1_ratio=1, max_iter=5000, random_state=SEED_MODELS),
            {"clf__C": loguniform(0.01, 10)},
        ),
        "LogRegl2": (
            LogisticRegression(solver="saga", l1_ratio=0, max_iter=5000, random_state=SEED_MODELS),
            {"clf__C": loguniform(0.01, 10)},
        ),
    }
    return models


def custom_score(y_true, y_pred, n_var):
    # when using do:
    # from sklearn.metrics import make_scorer
    # scorer = make_scorer(custom_score, greater_is_better=True)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp * 10 - fp * 5 - n_var * 200


def score_with_thresh(y_true, y_pred, n_var, thresh):
    # returns the score taking threshold into account and max contacts
    max_contact = int(round(len(y_true) * 0.2))
    pred = np.zeros(len(y_true), dtype=int)
    sel = np.where(y_pred >= thresh)[0]
    if len(sel) > max_contact:
        sel = sel[np.argsort(-y_pred[sel])[:max_contact]]
    pred[sel] = 1
    return custom_score(y_true, pred, n_var), int(pred.sum())


# custom scorer to use in cv
def make_cap_scorer(n_vars):
    """Capped competition scorer for a model that uses `n_vars` features."""

    def cap_score(y_true, y_prob):
        n = len(y_true)
        cap = int(round(0.2 * n))
        top = np.argsort(-y_prob)[:cap]
        pred = np.zeros(n, dtype=int)
        pred[top] = 1
        return custom_score(y_true, pred, n_vars)

    return make_scorer(cap_score, response_method="predict_proba")


def perform_random_search_cv(X_train, y_train, seed, k_values, n_iter, subsets):
    # seed relates to StratifiedKFold not Random Search, random serach seed is the same all the time
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results = []
    models = get_models()

    for k in k_values:
        # scoring dict
        scoring = {
            "custom": make_cap_scorer(n_vars=k),
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "precision": make_scorer(precision_score, zero_division=0),
        }
        print(f"----- Evaluating k = {k} -------")
        for subset_id, (kind, features) in enumerate(subsets[k]):
            Xk = X_train[features]
            print(f"Evaluating subset {kind}")
            for name, (estimator, space) in models.items():
                pipeline = Pipeline(
                    [("scaler", StandardScaler()), ("clf", clone(estimator))]
                )
                random_search = RandomizedSearchCV(
                    pipeline,
                    space,
                    n_iter=n_iter,
                    scoring=scoring,
                    refit="custom",
                    cv=cv,
                    random_state=SEED_RANDOM_SEARCH,
                    n_jobs=-1,
                )
                random_search.fit(Xk, y_train)
                bi = random_search.best_index_
                results.append(
                    {
                        "k": k,
                        "kind": kind,
                        "model": name,
                        "cv_score": random_search.best_score_,
                        "features": features,
                        "best_params": random_search.best_params_,
                        "accuracy": random_search.cv_results_["mean_test_accuracy"][bi],
                        "balanced_accuracy": random_search.cv_results_[
                            "mean_test_balanced_accuracy"
                        ][bi],
                        "precision": random_search.cv_results_["mean_test_precision"][
                            bi
                        ],
                    }
                )
                # print(f"    Finished {name}")

    res = (
        pd.DataFrame(results)
        .sort_values("cv_score", ascending=False)
        .reset_index(drop=True)
    )
    ts = int(time.time())
    res.to_csv(f"cv_results_{ts}_seed_{seed}.csv", index=False)
    return res
